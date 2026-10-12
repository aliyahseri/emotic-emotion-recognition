"""
RTX 5070 Optimized Training Script — EMOTIC
=========================================
Fully utilizes RTX 5070 capabilities with:
- Mixed precision (BF16) for maximum speed
- Differential learning rates for fine-tuning
- Gradient accumulation for effective batch scaling
- Proper GPU utilization optimizations
"""

import os
import time
import argparse
import yaml
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.emotic_dataset import build_dataloaders
from src.model.emotic_model import EMOTICModel, EMOTICLoss
from src.evaluation.metrics import compute_ap, compute_vad_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_optimizer(model, cfg):
    """Create optimizer with differential learning rates for backbone vs head."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['body_stream', 'context_stream']):
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg["training"]["backbone_lr"]},
        {"params": head_params, "lr": cfg["training"]["head_lr"]},
    ], weight_decay=cfg["training"]["weight_decay"])
    
    log.info(f"Backbone params: {len(backbone_params)} LR: {cfg['training']['backbone_lr']}")
    log.info(f"Head params: {len(head_params)} LR: {cfg['training']['head_lr']}")
    
    return optimizer


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    model.train()
    running = {"total": 0.0, "cat": 0.0, "vad": 0.0}
    
    # Gradient accumulation setup
    accum_steps = cfg["training"].get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        # Move data to GPU with non_blocking for async transfer
        body = batch["body"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        cat_targets = batch["categories"].to(device, non_blocking=True)
        vad_targets = batch["vad"].to(device, non_blocking=True)

        # Mixed precision forward pass
        amp_context = autocast(device_type="cuda") if cfg["training"].get("mixed_precision", False) else nullcontext()
        with amp_context:
            cat_logits, vad_pred = model(body, context)
            total_loss, cat_loss, vad_loss = criterion(
                cat_logits, vad_pred, cat_targets, vad_targets
            )
            
            # Scale loss for gradient accumulation
            if accum_steps > 1:
                total_loss = total_loss / accum_steps
        
        # Check for NaN
        if torch.isnan(total_loss):
            log.warning(f"NaN loss at step {step}, skipping batch")
            continue
            
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Step optimizer every N steps
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running["total"] += total_loss.item() * accum_steps
        running["cat"] += cat_loss.item() * accum_steps
        running["vad"] += vad_loss.item() * accum_steps

        if (step + 1) % 50 == 0:
            log.info(
                f"Epoch {epoch} | step {step+1}/{len(loader)} "
                f"| loss {total_loss.item():.4f} "
                f"(cat={cat_loss.item():.4f}, vad={vad_loss.item():.4f})"
            )

    n = len(loader)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = {"total": 0.0, "cat": 0.0, "vad": 0.0}

    all_cat_probs, all_cat_targets = [], []
    all_vad_pred, all_vad_targets = [], []

    for batch in loader:
        body = batch["body"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        cat_targets = batch["categories"].to(device, non_blocking=True)
        vad_targets = batch["vad"].to(device, non_blocking=True)

        cat_logits, vad_pred = model(body, context)
        total_loss, cat_loss, vad_loss = criterion(
            cat_logits, vad_pred, cat_targets, vad_targets
        )

        running["total"] += total_loss.item()
        running["cat"] += cat_loss.item()
        running["vad"] += vad_loss.item()

        all_cat_probs.append(torch.sigmoid(cat_logits).cpu())
        all_cat_targets.append(cat_targets.cpu())
        all_vad_pred.append(vad_pred.cpu())
        all_vad_targets.append(vad_targets.cpu())

    n = len(loader)
    losses = {k: v / n for k, v in running.items()}

    cat_probs = torch.cat(all_cat_probs)
    cat_targets = torch.cat(all_cat_targets)
    vad_pred = torch.cat(all_vad_pred)
    vad_targets = torch.cat(all_vad_targets)

    # Check for NaN in predictions
    if torch.isnan(cat_probs).any() or torch.isnan(vad_pred).any():
        log.warning("NaN detected in validation predictions")
        return losses, 0.0, {}

    mAP = compute_ap(cat_probs.numpy(), cat_targets.numpy())
    vad_metrics = compute_vad_metrics(vad_pred.numpy(), vad_targets.numpy())

    return losses, mAP, vad_metrics


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    
    log.info(f"Device: {device}")
    log.info(f"Mixed Precision: {'Enabled' if amp_enabled else 'Disabled'}")
    log.info(f"Batch Size: {cfg['data']['batch_size']}")
    
    # RTX 5070 optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        log.info("Enabled cuDNN benchmark and TF32 for RTX 5070")

    # Data
    loaders, *_ = build_dataloaders(cfg["data"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model
    model = EMOTICModel(
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
    ).to(device)
    
    # Enable torch.compile for RTX 5070 (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == "cuda":
        log.info("Enabling torch.compile for faster training")
        model = torch.compile(model, mode="max-autotune")

    # Loss
    criterion = EMOTICLoss(lambda_vad=cfg["training"].get("lambda_vad", 0.5))

    # Optimizer with differential learning rates
    optimizer = get_optimizer(model, cfg)

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )
    
    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=amp_enabled)

    save_dir = cfg["training"].get("save_dir", "checkpoints")
    epochs = cfg["training"]["epochs"]
    best_map = 0.0

    log.info(f"Starting training for {epochs} epochs")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Effective batch size: {cfg['data']['batch_size'] * cfg['training'].get('gradient_accumulation_steps', 1)}")

    # Training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg
        )
        val_losses, val_mAP, val_vad = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        
        # Get current learning rates
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        
        log.info("=" * 80)
        log.info(f"Epoch {epoch}/{epochs} | {elapsed:.1f}s")
        log.info(f"Train Loss: {train_losses['total']:.4f} (cat={train_losses['cat']:.4f}, vad={train_losses['vad']:.4f})")
        log.info(f"Val Loss:   {val_losses['total']:.4f} (cat={val_losses['cat']:.4f}, vad={val_losses['vad']:.4f})")
        log.info(f"Val mAP:    {val_mAP:.4f}")
        log.info(f"LR: backbone={backbone_lr:.2e}, head={head_lr:.2e}")
        log.info("=" * 80)

        # Save best model
        if val_mAP > best_map:
            best_map = val_mAP
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                },
                os.path.join(save_dir, "best_model_rtx5070.pth"),
            )

        # Periodic checkpoint
        if epoch % cfg["training"].get("save_every", 5) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                },
                os.path.join(save_dir, f"epoch_{epoch:03d}_rtx5070.pth"),
            )

    log.info(f"Training complete. Best val mAP: {best_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx5070.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)

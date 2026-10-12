"""
Robust Training Script — EMOTIC
================================
Proper training for modern GPUs (RTX 5070+):
  - Differential learning rates (backbone vs head)
  - Mixed precision (BF16) via torch.amp
  - Gradient accumulation
  - Per-epoch metric summary
  - NaN-safe training
"""

import os
import time
import argparse
import yaml
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

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


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def get_optimizer(model: EMOTICModel, cfg: dict) -> optim.Optimizer:
    """
    Differential learning rates:
      - backbone (pretrained ResNet/ViT) gets a lower LR to avoid destroying features
      - head (fusion + classifiers) gets a higher LR to learn fast
    Falls back to a single 'learning_rate' key if differential keys are absent.
    """
    t = cfg["training"]
    backbone_lr = t.get("backbone_lr", t.get("learning_rate", 1e-5))
    head_lr     = t.get("head_lr",     t.get("learning_rate", 1e-4))
    weight_decay = t.get("weight_decay", 1e-4)

    # Collect backbone vs head parameter groups
    # Assumes model exposes .backbone_body and .backbone_context for pretrained parts
    backbone_params, head_params = [], []
    backbone_names = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Heuristic: anything named "backbone*" or "body_net*" / "context_net*"
        # goes in the backbone group. Adjust to match your model's attribute names.
        if name.startswith(("backbone", "body_net", "context_net")):
            backbone_params.append(param)
            backbone_names.add(name)
        else:
            head_params.append(param)

    log.info(
        f"Optimizer | backbone params: {len(backbone_params)} "
        f"(lr={backbone_lr}) | head params: {len(head_params)} (lr={head_lr})"
    )

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]
    return optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scaler,
    device, epoch, cfg
):
    model.train()
    running = {"total": 0.0, "cat": 0.0, "vad": 0.0}

    use_amp   = cfg["training"].get("mixed_precision", True)
    accum_steps = max(1, cfg["training"].get("gradient_accumulation_steps", 1))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    optimizer.zero_grad()
    nan_batches = 0

    for step, batch in enumerate(loader):
        body        = batch["body"].to(device, non_blocking=True)
        context     = batch["context"].to(device, non_blocking=True)
        cat_targets = batch["categories"].to(device, non_blocking=True)
        vad_targets = batch["vad"].to(device, non_blocking=True)

        # ── Forward ──────────────────────────────
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            cat_logits, vad_pred = model(body, context)
            total_loss, cat_loss, vad_loss = criterion(
                cat_logits, vad_pred, cat_targets, vad_targets
            )
            # Scale down for gradient accumulation
            total_loss = total_loss / accum_steps

        # NaN guard — skip this batch but don't zero the accumulated gradient
        if torch.isnan(total_loss):
            nan_batches += 1
            log.warning(f"Epoch {epoch} | step {step+1}: NaN loss, skipping batch")
            if (step + 1) % accum_steps == 0:
                optimizer.zero_grad()
            continue

        # ── Backward ─────────────────────────────
        scaler.scale(total_loss).backward()

        # Only update every `accum_steps` mini-batches
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track un-scaled losses for logging
        running["total"] += total_loss.item() * accum_steps
        running["cat"]   += cat_loss.item()
        running["vad"]   += vad_loss.item()

        if (step + 1) % 50 == 0:
            current_lr = optimizer.param_groups[-1]["lr"]   # head LR
            log.info(
                f"Epoch {epoch} | step {step+1}/{len(loader)} "
                f"| loss {total_loss.item() * accum_steps:.4f} "
                f"(cat={cat_loss.item():.4f}, vad={vad_loss.item():.4f}) "
                f"| lr={current_lr:.2e}"
            )

    if nan_batches:
        log.warning(f"Epoch {epoch}: {nan_batches} batches skipped due to NaN")

    n = max(1, len(loader) - nan_batches)
    return {k: v / n for k, v in running.items()}


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    running = {"total": 0.0, "cat": 0.0, "vad": 0.0}

    use_amp   = cfg["training"].get("mixed_precision", True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    all_cat_probs,   all_cat_targets = [], []
    all_vad_pred,    all_vad_targets  = [], []

    for batch in loader:
        body        = batch["body"].to(device, non_blocking=True)
        context     = batch["context"].to(device, non_blocking=True)
        cat_targets = batch["categories"].to(device, non_blocking=True)
        vad_targets = batch["vad"].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            cat_logits, vad_pred = model(body, context)
            total_loss, cat_loss, vad_loss = criterion(
                cat_logits, vad_pred, cat_targets, vad_targets
            )

        running["total"] += total_loss.item()
        running["cat"]   += cat_loss.item()
        running["vad"]   += vad_loss.item()

        all_cat_probs.append(torch.sigmoid(cat_logits).float().cpu())
        all_cat_targets.append(cat_targets.float().cpu())
        all_vad_pred.append(vad_pred.float().cpu())
        all_vad_targets.append(vad_targets.float().cpu())

    n = len(loader)
    losses = {k: v / n for k, v in running.items()}

    cat_probs   = torch.cat(all_cat_probs)
    cat_targets = torch.cat(all_cat_targets)
    vad_pred    = torch.cat(all_vad_pred)
    vad_targets = torch.cat(all_vad_targets)

    if torch.isnan(cat_probs).any() or torch.isnan(vad_pred).any():
        log.warning("NaN detected in validation predictions — metrics unreliable")
        return losses, 0.0, {}

    mAP        = compute_ap(cat_probs.numpy(), cat_targets.numpy())
    vad_metrics = compute_vad_metrics(vad_pred.numpy(), vad_targets.numpy())

    return losses, mAP, vad_metrics


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device        : {device}")
    if device.type == "cuda":
        log.info(f"GPU           : {torch.cuda.get_device_name(0)}")
        log.info(f"BF16 support  : {torch.cuda.is_bf16_supported()}")

    use_amp = cfg["training"].get("mixed_precision", True)
    log.info(f"Mixed precision: {use_amp}")
    log.info(f"Batch size     : {cfg['data']['batch_size']}")
    log.info(
        f"Grad accum     : {cfg['training'].get('gradient_accumulation_steps', 1)} "
        f"(effective batch = "
        f"{cfg['data']['batch_size'] * cfg['training'].get('gradient_accumulation_steps', 1)})"
    )

    # ── Data ─────────────────────────────────
    loaders, *_ = build_dataloaders(cfg["data"])
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Model ────────────────────────────────
    model = EMOTICModel(
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
    ).to(device)
    log.info(f"Model params   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ─────────────────────────────────
    criterion = EMOTICLoss(lambda_vad=cfg["training"].get("lambda_vad", 0.5))

    # ── Optimizer ────────────────────────────
    optimizer = get_optimizer(model, cfg)

    # GradScaler — only meaningful for float16; a no-op for BF16 but harmless
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ── Scheduler ────────────────────────────
    epochs = cfg["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_dir = cfg["training"].get("save_dir", "checkpoints")
    best_map = 0.0

    log.info(f"Training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, cfg
        )
        val_losses, val_mAP, val_vad = validate(
            model, val_loader, criterion, device, cfg
        )
        scheduler.step()

        elapsed = time.time() - t0

        # Per-epoch summary
        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr     = optimizer.param_groups[-1]["lr"]
        log.info(
            f"\n{'='*70}\n"
            f"  Epoch {epoch:3d}/{epochs} | {elapsed:.1f}s\n"
            f"  train loss : total={train_losses['total']:.4f}  "
            f"cat={train_losses['cat']:.4f}  vad={train_losses['vad']:.4f}\n"
            f"  val   loss : total={val_losses['total']:.4f}  "
            f"cat={val_losses['cat']:.4f}  vad={val_losses['vad']:.4f}\n"
            f"  val mAP    : {val_mAP:.4f}  (best so far: {max(best_map, val_mAP):.4f})\n"
            f"  VAD metrics: {val_vad}\n"
            f"  LR         : backbone={backbone_lr:.2e}  head={head_lr:.2e}\n"
            f"{'='*70}"
        )

        # ── Save best ────────────────────────
        if val_mAP > best_map:
            best_map = val_mAP
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                },
                os.path.join(save_dir, "best_model.pth"),
            )

        # ── Periodic checkpoint ───────────────
        if epoch % cfg["training"].get("save_every", 5) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_map": best_map,
                    "cfg": cfg,
                },
                os.path.join(save_dir, f"epoch_{epoch:03d}.pth"),
            )

    log.info(f"Training complete. Best val mAP: {best_map:.4f}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)

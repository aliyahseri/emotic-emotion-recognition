"""
Simple Robust Training Script — EMOTIC
===================================
Simplified training without complex optimizations that might cause issues.
Focus on stability and reliability first.
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


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running = {"total": 0.0, "cat": 0.0, "vad": 0.0}
    
    for step, batch in enumerate(loader):
        body = batch["body"].to(device)
        context = batch["context"].to(device)
        cat_targets = batch["categories"].to(device)
        vad_targets = batch["vad"].to(device)

        optimizer.zero_grad()
        
        # Simple forward pass - no mixed precision for stability
        cat_logits, vad_pred = model(body, context)
        total_loss, cat_loss, vad_loss = criterion(
            cat_logits, vad_pred, cat_targets, vad_targets
        )
        
        # Check for NaN
        if torch.isnan(total_loss):
            log.warning(f"NaN loss at step {step}, skipping batch")
            continue
            
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running["total"] += total_loss.item()
        running["cat"] += cat_loss.item()
        running["vad"] += vad_loss.item()

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
        body = batch["body"].to(device)
        context = batch["context"].to(device)
        cat_targets = batch["categories"].to(device)
        vad_targets = batch["vad"].to(device)

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
    log.info(f"Device: {device}")
    log.info(f"Using batch size: {cfg['data']['batch_size']}")

    # Data
    loaders, *_ = build_dataloaders(cfg["data"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Model
    model = EMOTICModel(
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
    ).to(device)

    # Loss
    criterion = EMOTICLoss(lambda_vad=cfg["training"].get("lambda_vad", 0.5))

    # Optimizer - single LR for simplicity
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4)
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )

    save_dir = cfg["training"].get("save_dir", "checkpoints")
    epochs = cfg["training"]["epochs"]
    best_map = 0.0

    log.info(f"Starting training for {epochs} epochs")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_losses, val_mAP, val_vad = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch}/{epochs} | {elapsed:.1f}s "
            f"| train_loss={train_losses['total']:.4f} "
            f"| val_loss={val_losses['total']:.4f} "
            f"| val_mAP={val_mAP:.4f}"
        )

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
                os.path.join(save_dir, "best_model_simple.pth"),
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
                os.path.join(save_dir, f"epoch_{epoch:03d}_simple.pth"),
            )

    log.info(f"Training complete. Best val mAP: {best_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)

"""
Evaluation Metrics — EMOTIC
============================
Implements all evaluation metrics from the paper:

Classification (26-category multi-label):
    • Average Precision per class
    • Mean Average Precision (mAP)
    • Precision, Recall, F1-score (micro/macro)

Regression (VAD):
    • MAE (per dimension + averaged)
    • RMSE (per dimension + averaged)
    • Pearson Correlation Coefficient (per dimension)
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import pearsonr

from src.dataset.emotic_dataset import EMOTIC_CATEGORIES


# ── Classification metrics ────────────────────────────────────────────────────
def compute_ap(
    probs: np.ndarray,
    targets: np.ndarray,
    verbose: bool = False,
) -> float:
    """
    Compute Mean Average Precision (mAP) over all 26 categories.

    Args:
        probs   : (N, 26) sigmoid probabilities
        targets : (N, 26) binary ground-truth labels
        verbose : if True, print per-class AP

    Returns:
        mAP (float)
    """
    aps = []
    for i, cat in enumerate(EMOTIC_CATEGORIES):
        if targets[:, i].sum() == 0:
            continue  # skip class with no positive samples
        ap = average_precision_score(targets[:, i], probs[:, i])
        aps.append(ap)
        if verbose:
            print(f"  AP[{cat:20s}] = {ap:.4f}")

    mAP = float(np.mean(aps))
    if verbose:
        print(f"\n  mAP = {mAP:.4f}")
    return mAP


def compute_classification_report(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute precision, recall, F1 for multi-label classification.

    Returns:
        dict with keys: mAP, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1
                        per_class_ap (list)
    """
    preds = (probs >= threshold).astype(int)

    per_class_ap = []
    for i in range(len(EMOTIC_CATEGORIES)):
        if targets[:, i].sum() == 0:
            per_class_ap.append(0.0)
        else:
            per_class_ap.append(
                average_precision_score(targets[:, i], probs[:, i])
            )

    return {
        "mAP": float(np.mean([ap for ap in per_class_ap if ap > 0])),
        "per_class_ap": dict(zip(EMOTIC_CATEGORIES, per_class_ap)),
        "micro_precision": precision_score(targets, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(targets, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(targets, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(targets, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
    }


# ── Regression metrics ────────────────────────────────────────────────────────
def compute_vad_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict:
    """
    Compute MAE, RMSE, and Pearson r for VAD regression.

    Args:
        pred   : (N, 3) predicted VAD in [0, 1]
        target : (N, 3) ground-truth VAD in [0, 1]

    Returns:
        dict with per-dimension and mean metrics
    """
    dims = ["valence", "arousal", "dominance"]
    result = {}

    maes, rmses, pearsons = [], [], []
    for i, dim in enumerate(dims):
        p, t = pred[:, i], target[:, i]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        r, _ = pearsonr(p, t)

        result[f"mae_{dim[0]}"] = mae
        result[f"rmse_{dim[0]}"] = rmse
        result[f"pearson_{dim[0]}"] = float(r)
        maes.append(mae)
        rmses.append(rmse)
        pearsons.append(float(r))

    result["mae_mean"] = float(np.mean(maes))
    result["rmse_mean"] = float(np.mean(rmses))
    result["pearson_mean"] = float(np.mean(pearsons))
    return result


# ── Pretty print ──────────────────────────────────────────────────────────────
def print_full_report(cat_probs, cat_targets, vad_pred, vad_targets):
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    cls = compute_classification_report(cat_probs, cat_targets)
    print(f"mAP            : {cls['mAP']:.4f}")
    print(f"Micro Precision: {cls['micro_precision']:.4f}")
    print(f"Micro Recall   : {cls['micro_recall']:.4f}")
    print(f"Micro F1       : {cls['micro_f1']:.4f}")
    print(f"Macro F1       : {cls['macro_f1']:.4f}")

    print("\nPer-class AP:")
    for cat, ap in cls["per_class_ap"].items():
        bar = "█" * int(ap * 30)
        print(f"  {cat:22s} {ap:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("REGRESSION METRICS (VAD)")
    print("=" * 60)
    vad = compute_vad_metrics(vad_pred, vad_targets)
    for dim in ["valence", "arousal", "dominance"]:
        d = dim[0]
        print(
            f"  {dim:10s}: MAE={vad[f'mae_{d}']:.4f}  "
            f"RMSE={vad[f'rmse_{d}']:.4f}  "
            f"r={vad[f'pearson_{d}']:.4f}"
        )
    print(
        f"  {'mean':10s}: MAE={vad['mae_mean']:.4f}  "
        f"RMSE={vad['rmse_mean']:.4f}  "
        f"r={vad['pearson_mean']:.4f}"
    )
    print("=" * 60 + "\n")

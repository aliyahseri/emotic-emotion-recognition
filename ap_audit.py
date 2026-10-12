#!/usr/bin/env python3
"""
Audit the evaluation fix to understand mAP calculation.
"""

import numpy as np
from sklearn.metrics import average_precision_score

# Simulate a simple case we can verify by hand
# 5 samples, 1 category, 3 positives
y_true = np.array([1, 1, 0, 1, 0])
y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

sklearn_ap = average_precision_score(y_true, y_scores)
print(f"sklearn AP: {sklearn_ap:.4f}")  # Should be ~0.8333 for this case

# Check if our evaluate.py is doing macro or micro average
# Load the report and check
import json
with open("results/evaluation_report.json") as f:
    report = json.load(f)

new = report["new_model"]
per_cat = new.get("metrics", {}).get("category_maps", {})
nonzero = {k: v for k, v in per_cat.items() if v > 0}
print(f"\nCategories with mAP > 0: {len(nonzero)}/26")
print(f"Mean of nonzero categories: {np.mean(list(nonzero.values())):.4f}")
print(f"Mean over all 26 categories: {np.mean(list(per_cat.values())):.4f}")
print(f"Reported overall mAP: {new.get('overall_map', 'N/A')}")
print(f"\nAre these consistent? The reported mAP should equal mean over all 26 categories.")

# Show the old vs new AP calculation difference
print("\n=== OLD vs NEW AP CALCULATION ===")
print("Old manual AP calculation had bug - only considered precision at positive points")
print("New sklearn AP calculates properly for binary classification")
print("Check if this explains the jump in mAP")

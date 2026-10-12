#!/usr/bin/env python3
"""
Check if mAP is reasonable and test set size.
"""

import json
import numpy as np

with open("results/evaluation_report.json") as f:
    report = json.load(f)

per_cat = report["new_model"]["metrics"]["category_maps"]
print("=== PER-CATEGORY mAP ANALYSIS ===")
print("Categories with mAP > 0:")
for cat, val in sorted(per_cat.items(), key=lambda x: -x[1]):
    if val > 0:
        print(f"  {cat}: {val:.4f}")

print(f"\nMean over categories with samples: {np.mean([v for v in per_cat.values() if v > 0]):.4f}")
print(f"Number of categories with samples: {sum(1 for v in per_cat.values() if v > 0)}/26")

print("\n=== BENCHMARK COMPARISON ===")
print("EMOTIC state-of-the-art typically reports ~0.27-0.35 mAP")
print("Our 0.76 mAP is very high - verify test set size")

# Check dataset info
dataset_info = report.get("dataset_info", {})
print(f"\nTest set size: {dataset_info.get('num_samples', 'N/A')}")

if dataset_info.get('num_samples', 0) < 100:
    print("WARNING: Small test set may inflate mAP variance")

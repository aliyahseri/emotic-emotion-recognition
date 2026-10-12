#!/usr/bin/env python3
"""
Sanity check for evaluation results to ensure mAP numbers are meaningful.
"""

import json

with open("results/evaluation_report.json") as f:
    report = json.load(f)

print("=== PER-CATEGORY mAP (New Model) ===")
per_cat = report.get("new_model", {}).get("metrics", {}).get("category_maps", {})
sorted_cats = sorted(per_cat.items(), key=lambda x: x[1], reverse=True)
for cat, val in sorted_cats:
    bar = "#" * int(val * 40)
    print(f"{cat:<20} {val:.4f} {bar}")

print("\n=== TOP 5 CATEGORIES ===")
for cat, val in sorted_cats[:5]:
    print(f"  {cat}: {val:.4f}")

print("\n=== BOTTOM 5 CATEGORIES ===")
for cat, val in sorted_cats[-5:]:
    print(f"  {cat}: {val:.4f}")

print("\n=== VAD COMPARISON ===")
for model_name in ["old_model", "new_model"]:
    vad = report.get(model_name, {}).get("metrics", {}).get("vad_mae", {})
    print(f"{model_name}: V={vad.get('valence','N/A'):.4f}  A={vad.get('arousal','N/A'):.4f}  D={vad.get('dominance','N/A'):.4f}")

print("\n=== OVERALL METRICS ===")
for model_name in ["old_model", "new_model"]:
    metrics = report.get(model_name, {}).get("metrics", {})
    print(f"{model_name}:")
    print(f"  Overall mAP: {metrics.get('overall_map', 'N/A'):.4f}")
    print(f"  VAD MAE: {metrics.get('vad_mae', {})}")

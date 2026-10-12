#!/usr/bin/env python3
"""
Diagnose the VAD bug - understand if old model has VAD head.
"""

import json
import pandas as pd
import numpy as np

# Load evaluation report
with open("results/evaluation_report.json") as f:
    report = json.load(f)

print("=== RAW VAD STATS ===")
for model_name in ["old_model", "new_model"]:
    vad = report[model_name].get("metrics", {}).get("vad_mae", {})
    print(f"\n{model_name}:")
    for dim, val in vad.items():
        print(f"  {dim}: {val:.4f}")

# Check val CSV for VAD ground truth scale
print("\n=== VAL CSV ANALYSIS ===")
df = pd.read_csv('data/real_dataset/val.csv')
print(f'Columns: {df.columns.tolist()}')
vad_cols = [c for c in df.columns if any(x in c.lower() for x in ['val', 'aro', 'dom', 'vad', 'v_', 'a_', 'd_'])]
print(f'VAD columns: {vad_cols}')
if vad_cols:
    print(df[vad_cols].describe())
else:
    print("No VAD columns found in val CSV")

# Check if old model has VAD head
print("\n=== OLD MODEL CHECKPOINT ANALYSIS ===")
import torch
try:
    old_ckpt = torch.load("checkpoints/best_model.pth", map_location="cpu")
    
    if isinstance(old_ckpt, dict):
        keys = old_ckpt.get("model_state_dict", old_ckpt).keys()
        vad_keys = [k for k in keys if any(x in k.lower() for x in ["vad", "regression", "continuous"])]
        print("VAD-related keys in OLD model:", vad_keys if vad_keys else "NONE FOUND")
        print("\nFirst 10 keys:", list(keys)[:10])
    else:
        print("Checkpoint format not dict")
except Exception as e:
    print(f"Error loading old checkpoint: {e}")

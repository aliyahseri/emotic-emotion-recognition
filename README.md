# Context-Aware Emotion Recognition (EMOTIC-inspired)

A two-stream PyTorch model that predicts 26 emotion categories and continuous Valence/Arousal/Dominance (VAD) for people in images. It looks at both the person and the surrounding scene.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

A facial expression alone often isn't enough to tell how a person feels. Body pose and the scene around them carry a lot of signal too. This project is **inspired by** the EMOTIC benchmark (Kosti et al.). A body stream sees the person crop and a context stream sees the whole image. Their features are fused to predict:

- **26 discrete emotion categories** (multi-label, e.g. *Engagement*, *Peace*, *Annoyance*)
- **3 continuous dimensions**: Valence, Arousal, Dominance

It is not a faithful replication of the paper. The backbones differ, and the paper's fusion network is replaced by simple concatenation (see [Tech stack](#tech-stack)).

Key features:

- **Two-stream model** (`src/model/emotic_model.py`): ResNet-50 body stream on 128×128 person crops and ResNet-101 context stream on 224×224 full images, both ImageNet-pretrained. The two 2048-d features are concatenated (4096-d) and fed to a categorical head (26 logits) and a VAD head (3 outputs).
- **Joint loss**: `BCEWithLogitsLoss` for categories plus `0.5 × MSE` for VAD (`lambda_vad` is configurable).
- **Training** (`src/training/train.py`): AdamW, cosine LR schedule, mixed precision (BF16 when the GPU supports it), gradient accumulation, gradient clipping, a NaN guard, and best-checkpoint selection by validation mAP. All of it is driven by YAML configs.
- **Inference pipeline** (`src/inference/inference_pipeline.py`): YOLOv5 person detection through `torch.hub` (it falls back to a full-image box). It runs per-person predictions, draws annotations, and has a **two-checkpoint ensemble** that averages probabilities and VAD.
- **Explainability**: Grad-CAM on the last conv block of both streams (`src/explainability/gradcam.py`).
- **Calibration**: temperature scaling fitted on the validation set (`calibration.py`).
- **Evaluation** (`evaluate.py`): compares two checkpoints on per-category AP, mAP, VAD MAE/RMSE, a top-10 confusion matrix and a reliability diagram.
- **Annotation converter** (`scripts/convert_annotations.py`): turns the official EMOTIC `.mat` annotations into the CSV format the dataset loader reads.
- **Streamlit demo** (`app/app.py`): upload an image to see detected people, their top emotions, VAD bars and optional Grad-CAM overlays.

## Tech stack

| Component | Choice | Why (as evident in the code) |
|---|---|---|
| Framework | PyTorch, torchvision | ImageNet-pretrained ResNet-50 / ResNet-101 backbones out of the box |
| Body stream | ResNet-50 @ 128×128 | Person crops are small, so a lighter backbone is enough |
| Context stream | ResNet-101 @ 224×224 | The whole scene needs more capacity |
| Fusion | Concatenation (4096-d) → two MLP heads | Simple multi-task setup sharing one fused representation |
| Training | AdamW, cosine LR, BF16 AMP, grad accumulation | Fits two large backbones on a single consumer GPU (configs target a 12 GB RTX card) |
| Person detection | YOLOv5s via `torch.hub` | No extra install step; can be disabled with `EMOTIC_DISABLE_YOLO=1` |
| Metrics | scikit-learn `average_precision_score` | Standard per-class AP / mAP for multi-label tasks |
| UI | Streamlit | Quick interactive demo |
| Config | YAML (`configs/*.yaml`) | Swap datasets and hyperparameters without code changes |

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png)

*To do for the owner: run `streamlit run app/app.py` with a trained checkpoint, upload a photo with one or more people, and capture the annotated result with the per-person emotion chips and VAD bars. Save it as `docs/screenshot.png`.*

## Dataset

**No data or model weights are included in this repository.** The code supports two data sources, and both must be obtained separately.

### 1. EMOTIC (the intended benchmark)

- **What it is**: images of people in natural, unconstrained scenes. Each annotated person has a body bounding box, any of **26 emotion categories** (multi-label), and **continuous Valence/Arousal/Dominance on a 1–10 scale**. The images come from MSCOCO, ADE20K and web/movie frames.
- **Source**: R. Kosti, J. M. Alvarez, A. Recasens, A. Lapedriza, *EMOTIC: Emotions in Context Dataset* (CVPR Workshops 2017) and *Context Based Emotion Recognition using EMOTIC Dataset* (IEEE TPAMI 2019). Project page: <https://s3.sunai.uoc.edu/emotic/> [TODO: confirm link]. Access must be **requested from the authors**, and their terms of use apply.
- **Format**: images plus a MATLAB `Annotations.mat` with `train` / `val` / `test` splits.
- **Where to put it** (paths used by `configs/default.yaml`, `rtx5070.yaml` and `simple.yaml`):

```
data/
├── raw/
│   └── emotic/
│       ├── Annotations/
│       │   └── Annotations.mat        # official annotations
│       └── emotic/                    # image_root
│           ├── ade20k/ ...
│           ├── emodb_small/ ...
│           ├── framesdb/ ...
│           └── mscoco/ ...
└── annotations/                       # produced by scripts/convert_annotations.py
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Convert the annotations with:

```bash
python scripts/convert_annotations.py --mat_dir data/raw/emotic/Annotations --out_dir data/annotations
```

The CSV columns are `image_path, x1, y1, x2, y2, <26 category columns>, valence, arousal, dominance`. The loader also accepts `cat_0 … cat_25` instead of category names. The converter stores only each image's file name, without its sub-folder. The loader finds images by searching `image_root` recursively for that file name (see Known issues).

### 2. AffectNet-derived set (what the experiments actually used)

- **What it is**: face images from AffectNet (Mollahosseini et al., 2017) with the **8 AffectNet expression classes mapped deterministically onto EMOTIC categories**. VAD values were **generated from emotion-specific ranges** rather than annotated. This turns an 8-class face-expression dataset into the 26-category + VAD format the model expects.
- **Provenance**: the script that built this set is **not in the repository**. The mapping below is the one documented by the original project. [TODO: confirm: mapping, VAD ranges and the exact AffectNet source/subset used]

  | AffectNet class | Mapped EMOTIC categories |
  |---|---|
  | Neutral | Peace |
  | Happy | Happiness, Excitement, Pleasure |
  | Sad | Sadness, Suffering, Pain |
  | Angry | Anger, Annoyance, Disapproval |
  | Fear | Fear, Disquietment |
  | Surprise | Surprise, Anticipation |
  | Disgust | Aversion |
  | Contempt | Disconnection, Disapproval |

- **License**: AffectNet is distributed by its authors for research use under their own agreement. It is not redistributed here.
- **Where to put it** (paths used by `configs/real_dataset.yaml`, `evaluate.py` and `calibration.py`):

```
data/
└── real_dataset/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/                 # image_root
        ├── archive.zip         # optional: extract_images.bat unzips it here (Windows)
        ├── Train/<Class>/<image>.jpg
        └── Test/<Class>/<image>.jpg
```

The loader tries `image_root/<image_path>`, then `Train/` and `Test/` variants with a capitalised class folder, and finally a recursive file-name index. Note that `evaluate.py` passes `images/Train` as the image root, so its test images come from the **Train** folder (see Results).

### Model weights

Checkpoints are produced by training and are git-ignored (`*.pth`):

- `checkpoints/` — from `configs/default.yaml` (EMOTIC)
- `checkpoints_real/` — from `configs/real_dataset.yaml` (AffectNet-derived)

Each run writes `best_model.pth` (best validation mAP) and `epoch_XXX.pth` every `save_every` epochs. Each checkpoint is a dict with keys `model`, `optimizer`, `scaler`, `epoch`, `best_map` and `cfg`. The apps and the evaluation script expect `checkpoints_real/best_model.pth` and/or `checkpoints/best_model.pth` by default.

## Installation

Python 3.10+ is required (the code uses `list[str]` and `str | None` annotations).

```bash
git clone https://github.com/aliyahseri/emotic-emotion-recognition.git
cd emotic-emotion-recognition

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For a CUDA build of PyTorch, install `torch`/`torchvision` from <https://pytorch.org/get-started/locally/> first.

**YOLOv5 note:** person detection is loaded with `torch.hub.load("ultralytics/yolov5", "yolov5s")`. That needs **network access** on first use and pulls YOLOv5's own dependencies. To run fully offline (one full-image box per image), set:

```bash
# macOS / Linux
export EMOTIC_DISABLE_YOLO=1
# Windows (PowerShell)
$env:EMOTIC_DISABLE_YOLO = "1"
```

## Usage

Run all commands from the repository root.

**Train**

```bash
# EMOTIC
python -m src.training.train --config configs/default.yaml
# AffectNet-derived set
python -m src.training.train --config configs/real_dataset.yaml
```

There are two alternative scripts: `src/training/train_rtx5070.py` (default config `configs/rtx5070.yaml`, saves `best_model_rtx5070.pth`) and `src/training/train_simple.py` (plain Adam, no AMP, saves `best_model_simple.pth`).

**Single-image inference (Python)**

```python
from src.inference.inference_pipeline import EMOTICInference

pipeline = EMOTICInference("checkpoints_real/best_model.pth", use_gradcam=False)
for person in pipeline.predict_image("path/to/image.jpg"):
    print(person["box"], person["categories"], person["vad"])
```

**Evaluate / compare two checkpoints** (writes plots and `evaluation_report.json` to `results/`)

```bash
python evaluate.py --old checkpoints/best_model.pth --new checkpoints_real/best_model.pth \
    --test-csv data/real_dataset/test.csv --image-root data/real_dataset/images --device cuda
```

**Temperature scaling** (writes `results/calibration_params.json` and `results/calibrated_model.pth`)

```bash
python calibration.py --checkpoint checkpoints_real/best_model.pth \
    --val-csv data/real_dataset/val.csv --image-root data/real_dataset/images
```

**Streamlit demo**

```bash
streamlit run app/app.py
```

The sidebar sets the checkpoint path, threshold, top-K, ensemble mode and Grad-CAM. `app/app.py` is the recommended entry point. The other files in `app/` are earlier variants with known problems (see Known issues).

**Diagnostic scripts**: `sanity_check.py`, `map_sanity.py`, `ap_audit.py` and `vad_audit.py` read `results/evaluation_report.json` and print per-category AP/VAD summaries for checking the evaluation.

## Results

**There are no trustworthy benchmark numbers yet.** Earlier versions of this README and the demo sidebar quoted mAP and VAD-MAE figures. They have been removed because they don't measure what they appear to measure. [TODO: confirm: re-run evaluation on the official EMOTIC test split]

### Results & evaluation caveats

The earlier figures can't be compared with the EMOTIC literature, where published mAP is roughly **0.27–0.35** (the repo's own `map_sanity.py` notes this). The reasons:

1. **Wrong test set.** `evaluate.py` defaults to `data/real_dataset/test.csv` (the AffectNet-derived set), not the EMOTIC test split. It also loads images from `data/real_dataset/images/Train`.
2. **The label mapping makes the task easy.** Each AffectNet class maps to a fixed group of EMOTIC categories (e.g. Happy → Happiness + Excitement + Pleasure). Categories in a group always fire together, and only about 16 of the 26 categories ever appear. The problem collapses to 8-way face-expression classification, which gives inflated per-category AP. `evaluate.py` also averages only over categories that have positives.
3. **The VAD targets collapse.** The loader normalises VAD from the EMOTIC 1–10 scale and **replaces any value outside [1, 10] (or NaN) with 0.5** (`src/dataset/emotic_dataset.py`, `VAD_RANGE` and `_normalize_vad_values`). If the generated VAD values were not on a 1–10 scale, every target becomes the constant 0.5. That would explain why VAD MAE looked very small.
4. **Detector boxes, not ground-truth boxes.** Evaluation runs the full inference pipeline (YOLOv5 or a full-image box) and keeps only the first detection, rather than using the annotated person box. It also silently skips samples that raise errors.

A meaningful result needs EMOTIC trained with `configs/default.yaml`, evaluated on `data/annotations/test.csv` with ground-truth boxes, and mAP averaged over all 26 categories.

| Model | Train data | Test data | mAP | VAD MAE |
|---|---|---|---|---|
| Two-stream (`configs/default.yaml`) | EMOTIC train | EMOTIC test | [TODO] | [TODO] |

## Project structure

```
emotic-emotion-recognition/
├── src/
│   ├── model/emotic_model.py              # two-stream model + joint loss
│   ├── dataset/emotic_dataset.py          # CSV dataset, transforms, dataloaders
│   ├── training/
│   │   ├── train.py                       # main training script
│   │   ├── train_rtx5070.py               # variant (fixed backbone LR groups)
│   │   └── train_simple.py                # variant (Adam, no AMP)
│   ├── inference/
│   │   ├── inference_pipeline.py          # YOLOv5 + model + ensemble
│   │   └── inference_pipeline_optimized.py  # unused torch.compile variant
│   ├── evaluation/metrics.py              # AP / mAP / VAD metrics
│   └── explainability/gradcam.py          # Grad-CAM for both streams
├── app/                                   # Streamlit apps (app.py is the main one)
├── configs/                               # default / real_dataset / rtx5070 / simple
├── scripts/convert_annotations.py         # EMOTIC .mat -> CSV
├── evaluate.py                            # two-checkpoint comparison
├── calibration.py                         # temperature scaling
├── sanity_check.py, map_sanity.py, ap_audit.py, vad_audit.py  # eval diagnostics
├── extract_images.bat                     # unzip AffectNet-derived images (Windows)
├── deploy_streamlit_cloud.md
├── requirements.txt / requirements_cloud.txt
└── LICENSE
```

## Challenges & what I learned

- **Multi-task learning with two backbones**: training ResNet-50 + ResNet-101 together needed BF16 mixed precision, gradient accumulation and gradient clipping to fit a single consumer GPU and stay stable (with a NaN guard as a safety net).
- **Differential learning rates**: pretrained backbones need a smaller LR than the new heads. Getting the parameter groups right depends on matching the model's real attribute names (`body_stream` / `context_stream`). The bug in `train.py` shows how easily that silently fails.
- **Dataset engineering**: the official `.mat` annotations have nested MATLAB structs, so the converter handles cell arrays, missing fields and per-person rows.
- **Evaluation hygiene**: the diagnostic scripts (`map_sanity.py`, `vad_audit.py`) came from noticing that the numbers looked too good. The investigation showed how label mapping, test-set choice and target normalisation can each inflate metrics. [TODO: confirm]
- **Explainability and calibration**: adding Grad-CAM and temperature scaling to make predictions easier to inspect and trust. [TODO: confirm]

## Known issues & limitations

- **Untrustworthy metrics**: see [Results & evaluation caveats](#results--evaluation-caveats). The default evaluation uses the AffectNet-derived set (`evaluate.py`, `--test-csv` default) and images from its `Train` folder.
- **VAD normalisation collapse**: values outside [1, 10] or NaN become 0.5 (`src/dataset/emotic_dataset.py`, `VAD_RANGE` and `_normalize_vad_values`). This silently masks scale mismatches.
- **Backbone LR ignored in `train.py`**: `get_optimizer` (~line 71) matches the prefixes `backbone`, `body_net` and `context_net`, but the model uses `body_stream` / `context_stream`. Every parameter therefore goes into the head group, and `backbone_lr` is never applied. `train_rtx5070.py` matches the correct names.
- **Evaluation uses detector boxes**: `evaluate.py` runs YOLOv5 (or a full-image box) and takes the first detection instead of the ground-truth box. It also skips failing samples with only a warning (`except Exception: continue`).
- **Misleading "old checkpoint has no VAD head" diagnosis**: `vad_audit.py` reads `model_state_dict`, but checkpoints store weights under `model`, so it lists the top-level keys and reports "NONE FOUND". Every loader calls `load_state_dict` strictly, so a checkpoint without `vad_head.*` weights would fail to load at all.
- **`demo_app.py` crashes on upload**: it calls `compute_uncertainty`, `display_results` and other helpers at module level before they are defined, which raises a `NameError` once an image is processed. It also sets `pipeline.temperature`, which the pipeline never reads, so calibration has no effect. Its sidebar shows a hardcoded "Test mAP" and claims "Eval data: EMOTIC".
- **Missing example images**: `app/demo_app.py` (`EXAMPLE_IMAGES`) points to `assets/examples/*.jpg`. Those files are not in the repo (the placeholders were random noise and were removed). The app shows a warning when one is selected.
- **App variants preprocess differently from training**: `app_cpu.py`, `app_debug.py`, `app_simple.py` and `app_windows.py` feed BGR images scaled to [0, 1] with no ImageNet normalisation and no person crop. They also call `torch.load` again, uncached, at module level on every rerun just to show `best_map`, which raises a `KeyError` if a checkpoint lacks that key. A missing checkpoint is caught earlier by `st.stop()`.
- **Duplication**: six near-duplicate Streamlit apps and three training scripts.
- **Unused code**: `src/inference/inference_pipeline_optimized.py` (torch.compile, batching, warmup) isn't imported anywhere.
- **Package file misnamed**: `src/dataset/__init.py` should be `__init__.py`. Imports still work because Python treats the folder as a namespace package.
- **EMOTIC converter ignores the image sub-folder**: `scripts/convert_annotations.py` saves only `filename`. Images are found through a recursive file-name index, which breaks if two sub-datasets share a file name.
- **`EMOTICEnsembleInference` has no `predict_image`**: only `predict_array` (`src/inference/inference_pipeline.py`).
- **`train_rtx5070.py` AMP dtype**: it uses `autocast(device_type="cuda")` without a dtype (FP16 by default) despite the BF16 comments, and hardcodes `cuda` for AMP and the GradScaler.
- **No early stopping** in any training script. Training runs the full number of epochs and keeps the best-mAP checkpoint.
- **Version notes**: `torch.amp.GradScaler(device=...)` needs a recent PyTorch 2.x. The `requirements.txt` floor (`torch>=2.0.0`) may be too low. [TODO: confirm minimum version]
- **Not a faithful EMOTIC replication**: the model docstring says it "faithfully replicates" Kosti et al., but the backbones and the fusion differ.

## Future improvements

- Train and evaluate on the official EMOTIC splits with ground-truth boxes, and report mAP over all 26 categories (plus VAD on the 1–10 scale).
- Fix the parameter-group prefixes in `train.py` and merge the three training scripts into one.
- Make VAD normalisation fail loudly (or rescale) instead of silently substituting 0.5.
- Merge the Streamlit apps into one on top of `EMOTICInference`, and apply the calibrated temperature to the logits.
- Store the image sub-folder in the converted CSVs.
- Add a fusion network or attention between streams, closer to the original paper.
- Add unit tests for the dataset loader and metrics.

## Credits / Acknowledgements

- **EMOTIC dataset and task**: R. Kosti, J. M. Alvarez, A. Recasens, A. Lapedriza. *EMOTIC: Emotions in Context Dataset*, CVPR Workshops 2017; *Context Based Emotion Recognition using EMOTIC Dataset*, IEEE TPAMI 2019. This project's architecture is inspired by their two-stream design.
- **AffectNet**: A. Mollahosseini, B. Hasani, M. H. Mahoor. *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild*, IEEE Transactions on Affective Computing, 2017.
- **YOLOv5** by Ultralytics (<https://github.com/ultralytics/yolov5>), loaded at runtime through `torch.hub` (AGPL-3.0, not vendored here).
- **Grad-CAM**: R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- **Temperature scaling**: C. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- ImageNet-pretrained ResNet weights from torchvision.

## License

MIT, see [LICENSE](LICENSE). Datasets and third-party models keep their own licenses and terms.

## Author

**Aliyah Seri**

- GitHub: [`aliyahseri`](https://github.com/aliyahseri)

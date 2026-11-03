# Context-Aware Emotion Recognition

## Introduction
This project recognizes people's emotions in images by looking at both the person and the scene around them. A two-stream deep learning model predicts 26 emotion categories and continuous Valence, Arousal and Dominance (VAD) scores for each person. It includes Grad-CAM explanations and a Streamlit demo app.

## Models Used
* ResNet-50 body stream (person crop)
* ResNet-101 context stream (whole image)
* YOLOv5 for person detection
* Grad-CAM for visual explanations

## Training
* `python -m src.training.train --config configs/default.yaml`

## Usage
* Demo app: `streamlit run app/app.py`
* Evaluate a checkpoint: `python evaluate.py --old checkpoints/best_model.pth --new checkpoints_real/best_model.pth`

## Dataset
[EMOTIC](https://s3.sunai.uoc.edu/emotic/index.html) (26 emotion categories + VAD), placed in `data/` as set in `configs/default.yaml`.

## Requirements
* python 3.10+
* pytorch
* torchvision
* opencv
* streamlit
* pandas
* pyyaml

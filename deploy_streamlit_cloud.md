# Deploying the demo to Streamlit Community Cloud

These notes describe how the Streamlit demo could be deployed. It has not been verified end to end.
[TODO: confirm: deployment was never tested]

## Prerequisites

- A trained checkpoint. Checkpoints are **not** in this repository (`*.pth` is git-ignored), so the deployed app
  needs a checkpoint that it can reach. One option is to download it at startup from your own storage, which needs a
  small code change.
- `requirements_cloud.txt` lists the packages the app imports. Streamlit Cloud looks for a file named
  `requirements.txt`, so either point the app's settings at the right file (if your plan allows it) or copy
  `requirements_cloud.txt` over `requirements.txt` in the branch you deploy.

## Steps

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repository.
3. Set **Main file path** to `app/app.py`. It uses the shared inference pipeline, has a checkpoint path field in the sidebar, and shows an error instead of crashing when the checkpoint is missing.
4. Optional: set the environment variable `EMOTIC_DISABLE_YOLO=1` (for example through the app's secrets or settings) to
   skip the YOLOv5 download through `torch.hub` and use the full image as the person box.
5. Deploy.

## Notes

- Free-tier instances are CPU-only with limited memory. The model has two ResNet backbones (ResNet-50 and ResNet-101),
  so expect slow first loads.
- `app/demo_app.py` is **not** a good entry point right now: it calls helper functions before they are defined
  (see Known issues in the README).

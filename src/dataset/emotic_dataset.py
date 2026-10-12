"""
EMOTIC Dataset Loader
=====================
Custom PyTorch Dataset for the EMOTIC dataset.
Handles:
  - Loading images from disk
  - Cropping person bounding boxes
  - Parsing 26-category multi-label annotations
  - Parsing continuous VAD (Valence, Arousal, Dominance) labels

Annotation format (EMOTIC official .mat converted to .csv):
    filename, x1, y1, x2, y2, cat1,...,cat26, valence, arousal, dominance
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Canonical 26 EMOTIC emotion categories ─────────────────────────────────
EMOTIC_CATEGORIES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
]
NUM_CATEGORIES = len(EMOTIC_CATEGORIES)  # 26
VAD_RANGE = (1, 10)                       # continuous labels in [1, 10]
_IMAGE_INDEX_CACHE = {}


# ── Default transforms ──────────────────────────────────────────────────────
def get_transforms(split: str = "train"):
    """
    Returns torchvision transforms for body-crop and context images.

    Args:
        split: 'train' | 'val' | 'test'

    Returns:
        body_transform, context_transform
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        body_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        context_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / test
        body_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize,
        ])
        context_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    return body_transform, context_transform


# ── Dataset class ───────────────────────────────────────────────────────────
class EMOTICDataset(Dataset):
    """
    PyTorch Dataset for EMOTIC.

    Expects a CSV annotation file with columns:
        image_path, x1, y1, x2, y2,
        <cat0> ... <cat25>,   (binary 0/1 per category)
        valence, arousal, dominance

    Args:
        annotation_csv   : path to the split CSV file
        image_root       : root directory of the EMOTIC images
        body_transform   : torchvision transform for person crop
        context_transform: torchvision transform for full scene image
        normalize_vad    : if True, scale VAD from [1,10] → [0,1]
    """

    def __init__(
        self,
        annotation_csv: str,
        image_root: str,
        body_transform=None,
        context_transform=None,
        normalize_vad: bool = True,
    ):
        self.image_root = image_root
        self.body_transform = body_transform
        self.context_transform = context_transform
        self.normalize_vad = bool(normalize_vad)

        self.df = pd.read_csv(annotation_csv)
        self._fix_column_names()
        self._validate_columns()

        self.image_index = self._get_image_index(image_root)

        if len(self.df) > 0:
            try:
                sample_rel = str(self.df.iloc[np.random.randint(0, len(self.df))]["image_path"])
                _ = self._resolve_image_path(sample_rel)
            except Exception as e:
                warnings.warn(f"EMOTICDataset init sanity check failed for '{annotation_csv}': {e}")

    # ── helpers ─────────────────────────────────────────────────────────────
    def _validate_columns(self):
        # Check for either emotion category names OR numeric format
        emotion_name_cols = EMOTIC_CATEGORIES
        emotion_numeric_cols = [f"cat_{i}" for i in range(len(EMOTIC_CATEGORIES))]
        
        # Determine which format we have
        has_emotion_names = any(col in self.df.columns for col in emotion_name_cols)
        has_emotion_numeric = any(col in self.df.columns for col in emotion_numeric_cols)
        
        if not has_emotion_names and not has_emotion_numeric:
            raise ValueError("No emotion columns found in CSV")
        
        # Check for required base columns
        base_required = ["image_path", "x1", "y1", "x2", "y2", "valence", "arousal", "dominance"]
        missing = [c for c in base_required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {missing}")
        
        # Store which format we're using
        self._uses_numeric_emotion_cols = has_emotion_numeric
    
    def _fix_column_names(self):
        """Fix column names to match expected format"""
        # Map common column name variations
        column_mapping = {
            'filename': 'image_path',
            'image': 'image_path',
            'path': 'image_path'
        }
        
        # Rename columns if needed
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
                break

    def _resolve_image_path(self, rel_path: str) -> str:
        rel_path = str(rel_path)
        rel_path = rel_path.lstrip("/\\")

        emotion_part = None
        filename = None

        if "/" in rel_path:
            emotion_part, filename = rel_path.split("/", 1)
        elif "\\" in rel_path:
            emotion_part, filename = rel_path.split("\\", 1)

        candidates: list[str] = []

        def add_candidate(*parts: str):
            candidates.append(os.path.join(*parts))

        add_candidate(self.image_root, rel_path)

        if emotion_part and filename:
            emo_dir = emotion_part[:1].upper() + emotion_part[1:]
            corrected = os.path.join(emo_dir, filename)
            add_candidate(self.image_root, corrected)
            add_candidate(self.image_root, "Train", corrected)
            add_candidate(self.image_root, "Test", corrected)

            add_candidate(self.image_root, "Train", emo_dir, os.path.basename(filename))
            add_candidate(self.image_root, "Test", emo_dir, os.path.basename(filename))
        else:
            add_candidate(self.image_root, "Train", rel_path)
            add_candidate(self.image_root, "Test", rel_path)

        for path in candidates:
            if os.path.isfile(path):
                return path

        indexed_path = self.image_index.get(os.path.basename(rel_path))
        if indexed_path is not None and os.path.isfile(indexed_path):
            return indexed_path

        raise FileNotFoundError(
            "Could not resolve image path. "
            f"rel_path='{rel_path}', image_root='{self.image_root}'. "
            f"Tried: {candidates[:6]}{'...' if len(candidates) > 6 else ''}"
        )

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = self._resolve_image_path(rel_path)
        img = Image.open(full_path).convert("RGB")
        return img

    @staticmethod
    def _get_image_index(image_root: str) -> dict[str, str]:
        abs_root = str(Path(image_root).resolve())
        cached = _IMAGE_INDEX_CACHE.get(abs_root)
        if cached is not None:
            return cached

        root = Path(abs_root)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        index = {
            path.name: str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in exts
        }
        _IMAGE_INDEX_CACHE[abs_root] = index
        return index

    def _crop_person(self, img: Image.Image, x1, y1, x2, y2) -> Image.Image:
        w, h = img.size
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        return img.crop((x1, y1, x2, y2))

    def _normalize_vad_values(self, v, a, d):
        lo, hi = VAD_RANGE
        # Handle invalid values (NaN, inf, or out of range)
        def safe_norm(x):
            if not (lo <= x <= hi) or not (x == x):  # Check range and NaN
                return 0.5  # Return middle value for invalid data
            return (x - lo) / (hi - lo)
        return safe_norm(v), safe_norm(a), safe_norm(d)

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load full scene image ─────────────────────────────────────────
        context_img = self._load_image(row["image_path"])

        # ── Crop person ──────────────────────────────────────────────────
        body_img = self._crop_person(
            context_img, row["x1"], row["y1"], row["x2"], row["y2"]
        )

        # ── Apply transforms ─────────────────────────────────────────────
        if self.body_transform:
            body_tensor = self.body_transform(body_img)
        else:
            body_tensor = transforms.ToTensor()(body_img)

        if self.context_transform:
            context_tensor = self.context_transform(context_img)
        else:
            context_tensor = transforms.ToTensor()(context_img)

        # ── Category labels (multi-hot) ───────────────────────────────────
        if self._uses_numeric_emotion_cols:
            # Use numeric column format (cat_0, cat_1, etc.)
            emotion_cols = [f"cat_{i}" for i in range(len(EMOTIC_CATEGORIES))]
            cat_labels = torch.tensor(
                row[emotion_cols].values.astype(np.float32),
            )
        else:
            # Use emotion name format (Affection, Anger, etc.)
            cat_labels = torch.tensor(
                row[EMOTIC_CATEGORIES].values.astype(np.float32),
            )
        
        cat_labels = cat_labels.float()

        # ── Continuous VAD ───────────────────────────────────────────────
        v, a, d = float(row["valence"]), float(row["arousal"]), float(row["dominance"])
        if self.normalize_vad:
            v, a, d = self._normalize_vad_values(v, a, d)
        vad = torch.tensor([v, a, d], dtype=torch.float32)

        return {
            "body": body_tensor,          # (3, 128, 128)
            "context": context_tensor,    # (3, 224, 224)
            "categories": cat_labels,     # (26,)
            "vad": vad,                   # (3,)
            "image_path": self._resolve_image_path(row["image_path"]),
        }


# ── Factory helper ───────────────────────────────────────────────────────────
def build_dataloaders(cfg: dict):
    """
    Build train / val / test DataLoaders from a config dict.

    Expected cfg keys:
        train_csv, val_csv, test_csv, image_root,
        batch_size, num_workers
    """
    from torch.utils.data import DataLoader

    train_body_tf, train_ctx_tf = get_transforms("train")
    val_body_tf, val_ctx_tf = get_transforms("val")

    train_ds = EMOTICDataset(
        cfg["train_csv"], cfg["image_root"],
        train_body_tf, train_ctx_tf,
    )
    val_ds = EMOTICDataset(
        cfg["val_csv"], cfg["image_root"],
        val_body_tf, val_ctx_tf,
    )
    test_ds = EMOTICDataset(
        cfg["test_csv"], cfg["image_root"],
        val_body_tf, val_ctx_tf,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        ),
    }
    return loaders, train_ds, val_ds, test_ds

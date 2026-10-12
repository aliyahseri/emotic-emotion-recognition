"""
EMOTIC Annotation Converter
============================
Converts the official EMOTIC MATLAB (.mat) annotation files to CSV format
expected by EMOTICDataset.

Supported layouts:
    1. Official EMOTIC layout:
        Annotations/
            Annotations.mat
       where the .mat file contains the splits: train, val, test

    2. Legacy split-per-file layout:
        annotations/
            train_annotations.mat
            val_annotations.mat
            test_annotations.mat

Each .mat file contains a struct array with fields:
    filename        : image file name (relative to the dataset folder)
    npeople         : number of annotated persons in the image
    person          : struct array per person with:
        body_bbox       [x1, y1, x2, y2]
        body_crops      (not used here)
        annotations_categories  : cell array of category names
        annotations_continuous  : struct {valence, arousal, dominance}

Output CSV columns:
    image_path, x1, y1, x2, y2,
    Affection, Anger, ..., Yearning,    (26 binary columns)
    valence, arousal, dominance

Usage:
    python scripts/convert_annotations.py

    python scripts/convert_annotations.py \
        --mat_dir data/raw/emotic/Annotations \
        --out_dir data/annotations
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except ImportError as exc:
    raise ImportError("scipy is required: pip install scipy") from exc


# Keep this list aligned with the dataset loader.
EMOTIC_CATEGORIES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
]

DEFAULT_MAT_DIR = os.path.join("data", "raw", "emotic", "Annotations")
DEFAULT_OUT_DIR = os.path.join("data", "annotations")


def _safe_str(cell) -> str:
    """Extract a string from a MATLAB cell or char array."""
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return ""
        if cell.dtype.kind in {"U", "S"} and cell.ndim == 1:
            return "".join(str(ch) for ch in cell.tolist())
        return _safe_str(cell.flat[0])
    return str(cell)


def _parse_categories(cat_cell) -> list[str]:
    """Parse a MATLAB category cell into a list of category name strings."""
    cats = []
    if cat_cell is None:
        return cats
    if hasattr(cat_cell, "categories"):
        return _parse_categories(cat_cell.categories)
    if isinstance(cat_cell, np.ndarray):
        for item in cat_cell.flat:
            cats.extend(_parse_categories(item))
    else:
        cats.append(_safe_str(cat_cell))
    return cats


def _collect_numeric_values(value) -> list[float]:
    """Flatten nested MATLAB numeric values into a simple Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        values = []
        for item in value.flat:
            values.extend(_collect_numeric_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_collect_numeric_values(item))
        return values
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _mean_or_default(value, default: float = 5.0) -> float:
    """Convert MATLAB numeric values to a scalar mean with a fallback."""
    values = _collect_numeric_values(value)
    if not values:
        return default
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _extract_continuous(cont) -> tuple[float, float, float]:
    """Extract valence/arousal/dominance from one or many MATLAB structs."""
    if cont is None:
        return 5.0, 5.0, 5.0

    if isinstance(cont, np.ndarray):
        return (
            _mean_or_default([getattr(item, "valence", None) for item in cont.flat]),
            _mean_or_default([getattr(item, "arousal", None) for item in cont.flat]),
            _mean_or_default([getattr(item, "dominance", None) for item in cont.flat]),
        )

    return (
        _mean_or_default(getattr(cont, "valence", None)),
        _mean_or_default(getattr(cont, "arousal", None)),
        _mean_or_default(getattr(cont, "dominance", None)),
    )


def _get_split_data(mat: dict, split: str):
    """Extract the annotation array for a split from a loaded .mat file."""
    for key in [split, "data", "ann", "annotations"]:
        if key in mat:
            return mat[key]
    raise KeyError(f"Cannot find annotation key for split '{split}'. Keys: {list(mat.keys())}")


def convert_split_data(data) -> pd.DataFrame:
    """Convert one split's annotation array to a DataFrame."""
    if not hasattr(data, "flat"):
        data = np.array([data], dtype=object)

    rows = []

    for item in data.flat:
        filename = _safe_str(item.filename)

        persons = item.person
        if not hasattr(persons, "flat"):
            persons = np.array([persons], dtype=object)

        for person in persons.flat:
            bbox = np.asarray(person.body_bbox).reshape(-1)
            x1, y1, x2, y2 = map(int, bbox[:4])

            cat_names = _parse_categories(
                getattr(person, "annotations_categories", None)
            )
            cat_dict = {cat: 0 for cat in EMOTIC_CATEGORIES}
            for cat_name in cat_names:
                if cat_name in cat_dict:
                    cat_dict[cat_name] = 1

            cont = getattr(person, "annotations_continuous", None)
            if cont is not None:
                valence, arousal, dominance = _extract_continuous(cont)
            else:
                valence = arousal = dominance = 5.0

            rows.append(
                {
                    "image_path": filename,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    **cat_dict,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                }
            )

    df = pd.DataFrame(rows)
    col_order = (
        ["image_path", "x1", "y1", "x2", "y2"]
        + EMOTIC_CATEGORIES
        + ["valence", "arousal", "dominance"]
    )
    return df[col_order]


def convert_mat(mat_path: str, split: str) -> pd.DataFrame:
    """Convert one .mat annotation file to a DataFrame for a given split."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return convert_split_data(_get_split_data(mat, split))


def _resolve_combined_mat_path(mat_source: str) -> str | None:
    """Return the path to a combined Annotations.mat file if one exists."""
    if os.path.isfile(mat_source):
        return mat_source

    if not os.path.isdir(mat_source):
        return None

    combined_path = os.path.join(mat_source, "Annotations.mat")
    if os.path.isfile(combined_path):
        return combined_path

    return None


def _print_split_summary(df: pd.DataFrame, out_path: str) -> None:
    print(f"  Saved {len(df)} annotations to {out_path}")
    print("     Category distribution (positive labels):")
    cat_sums = df[EMOTIC_CATEGORIES].sum().sort_values(ascending=False)
    for cat, cnt in cat_sums.head(10).items():
        print(f"     {cat:22s}: {int(cnt)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert EMOTIC .mat annotations to CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat_dir",
        default=DEFAULT_MAT_DIR,
        help="Directory containing Annotations.mat or split .mat files",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined_mat_path = _resolve_combined_mat_path(args.mat_dir)
    if combined_mat_path is not None:
        mat = sio.loadmat(combined_mat_path, squeeze_me=True, struct_as_record=False)
        for split in ["train", "val", "test"]:
            if split not in mat:
                print(f"  [SKIP] split '{split}' not found in {combined_mat_path}")
                continue

            print(f"  Converting split '{split}' from {os.path.basename(combined_mat_path)} ...")
            df = convert_split_data(mat[split])
            out_path = os.path.join(args.out_dir, f"{split}.csv")
            df.to_csv(out_path, index=False)
            _print_split_summary(df, out_path)
        return

    for split in ["train", "val", "test"]:
        mat_name = f"{split}_annotations.mat"
        mat_path = os.path.join(args.mat_dir, mat_name)

        if not os.path.isfile(mat_path):
            print(f"  [SKIP] {mat_path} not found")
            continue

        print(f"  Converting {mat_name} ...")
        df = convert_mat(mat_path, split)
        out_path = os.path.join(args.out_dir, f"{split}.csv")
        df.to_csv(out_path, index=False)
        _print_split_summary(df, out_path)


if __name__ == "__main__":
    main()

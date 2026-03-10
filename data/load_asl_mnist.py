"""
ASL MNIST dataset loader and converter.

Loads sign_mnist_train.csv (and optionally sign_mnist_test.csv) from
Kaggle's ASL MNIST dataset and converts them into NumPy arrays ready
for the training pipeline.

CSV format:
  Column 0  - label (0-25, maps 1:1 to A-Z; J=9 and Z=25 never appear)
  Columns 1-784 - grayscale pixel values for a 28x28 image

Usage:
  python -m data.load_asl_mnist          # quick self-test
  from data.load_asl_mnist import load_asl_mnist
"""

import os
import sys
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

ASL_MNIST_DIR = "data/asl_mnist"

# ── Label mapping ────────────────────────────────────────────────────────────

# ASL MNIST labels are 0-25 (A-Z) but J (9) and Z (25) are absent.
LABEL_TO_LETTER = {i: chr(ord("A") + i) for i in range(26)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}

# The 24 classes that actually appear in the dataset
VALID_LABELS = sorted(set(range(26)) - {9, 25})          # [0..8, 10..24]
VALID_LETTERS = [LABEL_TO_LETTER[i] for i in VALID_LABELS]  # A-I, K-Y


def _csv_to_arrays(csv_path: str | Path):
    """Read a sign_mnist CSV file and return (images, labels).

    Returns
    -------
    images : np.ndarray, shape (N, 28, 28), dtype uint8
    labels : np.ndarray, shape (N,), dtype int64
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            rows.append([int(v) for v in row])

    data = np.array(rows, dtype=np.int64)
    labels = data[:, 0]
    pixels = data[:, 1:].astype(np.uint8).reshape(-1, 28, 28)
    return pixels, labels


# ── Public API ───────────────────────────────────────────────────────────────

def load_asl_mnist(
    data_dir: str | Path | None = None,
    split: str = "train",
    flatten: bool = False,
    normalize: bool = False,
):
    """Load the ASL MNIST dataset.

    Parameters
    ----------
    data_dir : path to folder containing the CSV files.
               Defaults to config.ASL_MNIST_DIR.
    split : "train", "test", or "both".
    flatten : If True, return pixel arrays as (N, 784) instead of (N, 28, 28).
    normalize : If True, scale pixel values to 0.0–1.0 float32.

    Returns
    -------
    If split is "train" or "test":
        (images, labels, label_map)
    If split is "both":
        (train_images, train_labels, test_images, test_labels, label_map)

    label_map : dict mapping integer label → letter string
    """
    data_dir = Path(data_dir) if data_dir else Path(ASL_MNIST_DIR)

    file_map = {
        "train": "sign_mnist_train.csv",
        "test": "sign_mnist_test.csv",
    }

    def _load(name):
        imgs, lbls = _csv_to_arrays(data_dir / file_map[name])
        if flatten:
            imgs = imgs.reshape(imgs.shape[0], -1)
        if normalize:
            imgs = imgs.astype(np.float32) / 255.0
        return imgs, lbls

    label_map = {i: LABEL_TO_LETTER[i] for i in VALID_LABELS}

    if split == "both":
        tr_imgs, tr_lbls = _load("train")
        te_imgs, te_lbls = _load("test")
        return tr_imgs, tr_lbls, te_imgs, te_lbls, label_map
    elif split in file_map:
        imgs, lbls = _load(split)
        return imgs, lbls, label_map
    else:
        raise ValueError(f"split must be 'train', 'test', or 'both'; got '{split}'")


def save_as_images(
    data_dir: str | Path | None = None,
    output_dir: str | Path = "data/asl_mnist_images",
    split: str = "train",
):
    """Convert CSV rows to individual .png files organised by letter.

    Creates one subfolder per letter inside output_dir:
        output_dir/A/00000.png, output_dir/B/00001.png, ...

    Useful if you want to feed ASL MNIST through the same image-based
    augmentation pipeline used for the webcam-collected data.
    """
    import cv2

    imgs, lbls, label_map = load_asl_mnist(data_dir=data_dir, split=split)
    output_dir = Path(output_dir)

    for idx, (img, lbl) in enumerate(zip(imgs, lbls)):
        letter = label_map[int(lbl)]
        letter_dir = output_dir / letter.lower()
        letter_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(letter_dir / f"{idx:05d}.png"), img)

    print(f"Saved {len(imgs)} images to {output_dir}")


# ── Self-test ────────────────────────────────────────────────────────────────

def _self_test():
    """Quick sanity check — run with:  python -m data.load_asl_mnist"""
    print("=" * 60)
    print("ASL MNIST Loader – Self-Test")
    print("=" * 60)

    data_dir = Path(ASL_MNIST_DIR)
    train_csv = data_dir / "sign_mnist_train.csv"
    test_csv = data_dir / "sign_mnist_test.csv"

    if not train_csv.exists():
        print(f"\n[!] Training CSV not found at {train_csv}")
        print("    Download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
        print(f"    Place sign_mnist_train.csv (and optionally sign_mnist_test.csv) in {data_dir}")
        return

    # Load training set
    imgs, lbls, label_map = load_asl_mnist(split="train")
    print(f"\nTraining set")
    print(f"  Images shape : {imgs.shape}  (expected: (N, 28, 28))")
    print(f"  Labels shape : {lbls.shape}")
    print(f"  Pixel dtype  : {imgs.dtype}")
    print(f"  Pixel range  : [{imgs.min()}, {imgs.max()}]")
    print(f"  Unique labels: {sorted(set(lbls.tolist()))}")
    print(f"  Num classes  : {len(set(lbls.tolist()))}")

    # Verify no J or Z
    assert 9 not in lbls, "Label 9 (J) should not appear"
    assert 25 not in lbls, "Label 25 (Z) should not appear"
    print("  J/Z check    : PASS (labels 9 and 25 absent)")

    # Label map
    print(f"\nLabel map ({len(label_map)} classes):")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(label_map.items())))

    # Flattened + normalised
    imgs_f, _, _ = load_asl_mnist(split="train", flatten=True, normalize=True)
    print(f"\nFlattened & normalised")
    print(f"  Shape : {imgs_f.shape}  (expected: (N, 784))")
    print(f"  Dtype : {imgs_f.dtype}")
    print(f"  Range : [{imgs_f.min():.2f}, {imgs_f.max():.2f}]")

    # Test set (optional)
    if test_csv.exists():
        te_imgs, te_lbls, _ = load_asl_mnist(split="test")
        print(f"\nTest set")
        print(f"  Images shape : {te_imgs.shape}")
        print(f"  Labels shape : {te_lbls.shape}")
    else:
        print(f"\n[i] Test CSV not found at {test_csv} – skipping.")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
# Data quality verification utility
#
# Validates data/landmarks.pickle for common issues and prints
# summary statistics.  Returns exit code 0 (pass) or 1 (fail)
# for CI integration.
#
# Usage:
#   python -m data.verify_data                    # default path
#   python -m data.verify_data path/to/file.pickle
#   python -m data.verify_data --fix              # remove bad samples & resave


import os
import sys
import pickle
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from config import DATA_DIR
from collections import Counter

DEFAULT_PICKLE = "data/landmarks.pickle"
EXPECTED_FEATURES = 42          # 21 landmarks × 2 (x, y)
IMBALANCE_THRESHOLD = 0.5       # warn if any class has < 50% of the max count

# ── Core verification ────────────────────────────────────────────────────────

def load_pickle(path: str | Path):
    """Load and return the landmark pickle dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Pickle file is empty (0 bytes): {path}")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Pickle file is corrupt: {path} — {e}")
    return data


def verify_structure(data: dict) -> list[str]:
    """Check top-level keys and array types."""
    errors = []
    for key in ("data", "labels"):
        if key not in data:
            errors.append(f"Missing key '{key}' in pickle")
    if errors:
        return errors
    if not isinstance(data["data"], np.ndarray):
        errors.append(f"'data' is {type(data['data'])}, expected np.ndarray")
    if not isinstance(data["labels"], np.ndarray):
        errors.append(f"'labels' is {type(data['labels'])}, expected np.ndarray")
    return errors


def verify_shapes(data: dict) -> list[str]:
    """Verify matching lengths and expected feature dimension."""
    errors = []
    X, y = data["data"], data["labels"]
    if len(X) != len(y):
        errors.append(f"Length mismatch: data={len(X)}, labels={len(y)}")
    if len(X) == 0:
        errors.append("Dataset is empty (0 samples)")
    if X.ndim != 2:
        errors.append(f"Data has {X.ndim} dims, expected 2 (N, {EXPECTED_FEATURES})")
    elif X.shape[1] != EXPECTED_FEATURES:
        errors.append(f"Feature dim is {X.shape[1]}, expected {EXPECTED_FEATURES}")
    return errors


def find_bad_samples(X: np.ndarray) -> np.ndarray:
    """Return boolean mask of bad samples (all-zero, NaN, or Inf)."""
    all_zero = np.all(X == 0, axis=1)
    has_nan = np.any(np.isnan(X), axis=1)
    has_inf = np.any(np.isinf(X), axis=1)
    return all_zero | has_nan | has_inf


def class_statistics(y: np.ndarray) -> dict:
    """Compute per-class sample counts and summary stats."""
    counts = Counter(str(lbl) for lbl in y)
    counts_sorted = dict(sorted(counts.items()))
    values = list(counts.values())
    return {
        "counts": counts_sorted,
        "total": len(y),
        "num_classes": len(counts),
        "min": min(values),
        "max": max(values),
        "mean": np.mean(values),
        "std": np.std(values),
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_report(stats: dict, bad_mask: np.ndarray, warnings: list[str]):
    """Print a formatted summary table."""

    print_header("Per-Class Sample Counts")
    print(f"  {'Class':>6}  {'Count':>7}  {'Status'}")
    print(f"  {'-----':>6}  {'-----':>7}  {'------'}")

    threshold = stats["max"] * IMBALANCE_THRESHOLD
    for cls, count in stats["counts"].items():
        status = ""
        if count == 0:
            status = "!! EMPTY"
        elif count < threshold:
            status = "! LOW"
        print(f"  {cls:>6}  {count:>7}  {status}")

    print_header("Dataset Statistics")
    print(f"  Total samples    : {stats['total']}")
    print(f"  Num classes      : {stats['num_classes']}")
    print(f"  Min per class    : {stats['min']}")
    print(f"  Max per class    : {stats['max']}")
    print(f"  Mean per class   : {stats['mean']:.1f}")
    print(f"  Std dev          : {stats['std']:.1f}")

    bad_count = int(np.sum(bad_mask))
    print(f"\n  Bad samples      : {bad_count}  (all-zero / NaN / Inf)")

    if warnings:
        print_header("Warnings")
        for w in warnings:
            print(f"  ⚠  {w}")


# ── Fix mode ─────────────────────────────────────────────────────────────────

def remove_bad_samples(path: str | Path, data: dict, bad_mask: np.ndarray):
    """Remove bad samples from the pickle and resave."""
    good = ~bad_mask
    cleaned = {
        "data": data["data"][good],
        "labels": data["labels"][good],
    }
    removed = int(np.sum(bad_mask))
    with open(path, "wb") as f:
        pickle.dump(cleaned, f)
    print(f"\n  Removed {removed} bad samples and saved to {path}")
    return cleaned


# ── Main ─────────────────────────────────────────────────────────────────────

def verify(pickle_path: str | Path = DEFAULT_PICKLE, fix: bool = False) -> bool:
    """Run all checks. Returns True if dataset passes, False otherwise."""

    pickle_path = Path(pickle_path)
    passed = True
    warnings: list[str] = []

    print_header("Data Quality Verification")
    print(f"  File: {pickle_path}")

    # 1. Load
    try:
        data = load_pickle(pickle_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  FAIL: {e}")
        return False

    # 2. Structure
    errs = verify_structure(data)
    if errs:
        for e in errs:
            print(f"\n  FAIL: {e}")
        return False
    print("  Structure       : OK")

    # 3. Shapes
    errs = verify_shapes(data)
    if errs:
        for e in errs:
            print(f"\n  FAIL: {e}")
        return False
    print(f"  Shape           : {data['data'].shape}")
    print(f"  Dtype           : data={data['data'].dtype}, labels={data['labels'].dtype}")

    # 4. Bad samples
    bad_mask = find_bad_samples(data["data"])
    bad_count = int(np.sum(bad_mask))
    if bad_count > 0:
        warnings.append(f"{bad_count} bad sample(s) detected (all-zero / NaN / Inf)")
        passed = False

    # 5. Class stats
    stats = class_statistics(data["labels"])

    # 6. Empty classes
    empty = [cls for cls, cnt in stats["counts"].items() if cnt == 0]
    if empty:
        warnings.append(f"Empty classes: {empty}")
        passed = False

    # 7. Imbalance
    threshold = stats["max"] * IMBALANCE_THRESHOLD
    low = {cls: cnt for cls, cnt in stats["counts"].items() if 0 < cnt < threshold}
    if low:
        warnings.append(
            f"Imbalanced classes (< {IMBALANCE_THRESHOLD:.0%} of max): "
            + ", ".join(f"{c}={n}" for c, n in low.items())
        )
        passed = False

    # 8. Value range
    vmin, vmax = float(data["data"].min()), float(data["data"].max())
    print(f"  Value range     : [{vmin:.4f}, {vmax:.4f}]")
    if vmin < -0.5 or vmax > 1.5:
        warnings.append(f"Suspicious value range [{vmin:.4f}, {vmax:.4f}] — expected ~[0, 1]")

    # Report
    print_report(stats, bad_mask, warnings)

    # Fix
    if fix and bad_count > 0:
        data = remove_bad_samples(pickle_path, data, bad_mask)
        print("  Re-verifying after fix...")
        new_bad = find_bad_samples(data["data"])
        if int(np.sum(new_bad)) == 0:
            print("  Bad samples     : 0  (clean)")

    # Result
    print_header("Result")
    if passed:
        print("  PASS — dataset looks good!")
    else:
        print("  FAIL — see warnings above")

    return passed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify data quality of extracted landmarks pickle."
    )
    parser.add_argument(
        "pickle_path",
        nargs="?",
        default=DEFAULT_PICKLE,
        help=f"Path to landmarks pickle (default: {DEFAULT_PICKLE})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove bad samples and resave the pickle file",
    )
    args = parser.parse_args()

    passed = verify(pickle_path=args.pickle_path, fix=args.fix)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
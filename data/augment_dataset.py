"""
Script to apply augmentations to the dataset images.

Uses configuration from config.py to apply transformations like rotation and
brightness changes to increase dataset robustness.
"""
import os
import sys

from utils.augment import augment_dataset
from config import DATA_DIR, AUGMENT_CONFIG

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    augment_dataset(
        input_root=DATA_DIR,
        output_root=DATA_DIR,
        factor=AUGMENT_CONFIG["factor"],
        config=AUGMENT_CONFIG
    )

print(f"Processing class {class_dir.name}, found {len(images)} images")
for img_path in images:
    print(f"  Original: {img_path.name}")
# the purpose of this file is to augment the dataset 

import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.augment import augment_dataset
from config import DATA_DIR, AUGMENTED_DIR, AUGMENT_CONFIG

if __name__ == "__main__":
    augment_dataset(
        input_root=DATA_DIR,
        output_root=AUGMENTED_DIR,
        factor=AUGMENT_CONFIG["factor"],
        config=AUGMENT_CONFIG
    )

print(f"Processing class {class_dir.name}, found {len(images)} images")
for img_path in images:
    print(f"  Original: {img_path.name}")
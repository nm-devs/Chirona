"""
Data augmentation utilities.

Provides image transformation functions like rotation, flip, and zoom using OpenCV.
"""
import cv2
import numpy as np 
import random

from pathlib import Path 

# augmentations

def horizontal_flip(img):
    return cv2.flip(img, 1)

def random_rotation(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def brightness_contrast(img):
    alpha = 1 + random.uniform(-0.2, 0.2)
    beta = random.uniform(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_zoom(img):
    h, w = img.shape[:2]
    zoom = 1+ random.uniform(-0.1, 0.1)
    
    new_h, new_w = int(h / zoom), int(w / zoom)
    y1= (h-new_h) // 2
    x1= (w-new_w) // 2
    
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))
    

# pipeline

def apply_augmentations(img, config):
    out = img.copy()
    
    if config["flip"]:
        out = horizontal_flip(out)
    
    if config["rotate"]:
        out = random_rotation(out)
    
    if config["zoom"]:
        out = random_zoom(out)
    
    if config["brightness"]:
        out = brightness_contrast(out)
    
    return out

# dataset augmentor

def augment_dataset(input_root, output_root, factor, config):
    input_root = Path(input_root)
    output_root = Path(output_root)

    total_original = 0
    total_augmented = 0
    
    for class_dir in sorted(input_root.iterdir()):
        if not class_dir.is_dir():
            continue
        
        out_class_dir = output_root / class_dir.name
        out_class_dir.mkdir(parents= True, exist_ok=True)

        images = list(class_dir.glob("*.jpg"))

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            total_original += 1
            
            # copy original

            cv2.imwrite(str(out_class_dir / img_path.name), img)

            #create augmented images
            for i in range(factor - 1):
                aug = apply_augmentations(img, config)
                new_name = f"{img_path.stem}_aug{i}.png"
                cv2.imwrite(str(out_class_dir / new_name), aug)
                total_augmented += 1
    
    print(f"\n original images : {total_original}")
    print(f"\n augmented images : {total_augmented}")
    print(f"\n total images : {total_original + total_augmented}")
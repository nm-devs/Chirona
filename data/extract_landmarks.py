"""
Batch processes raw image data to extract hand landmarks.

Runs MediaPipe over all collected images and saves the normalized features
into a pickle file for model training.
"""
import os
import sys
import cv2
import mediapipe as mp
import numpy as np 
import pickle 
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from core.sign_classifier import SignClassifier
from core.feature_extractor import FeatureExtractor
from core.hand_detector import HandDetector
from core.feature_extractor import FeatureExtractor
from config import (
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_landmarks_batch(
        raw_data_dir="./data/raw",
        output_path="./data/landmarks.pickle",
):
    logging.info("initializing hand detector...")

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE
    )
    fe =FeatureExtractor(use_z=False)

    all_data = []
    all_labels = []

    total_images = 0
    successful = 0
    skipped = 0

    root = Path(raw_data_dir)

    if not root.exists():
        logging.error(f"Directory not found: {raw_data_dir}")
        return
    
    class_folders = [d for d in root.iterdir() if d.is_dir()]

    logging.info(f"found {len(class_folders)} classes")

    # Switch to static image mode for batch processing
    detector.hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    for class_folder in class_folders:
        class_label = class_folder.name
        logging.info(f"processing class {class_label}...")

        images = list(class_folder.glob("*.jpg")) + \
                    list(class_folder.glob("*.png"))
        
        logging.info(f"found {len(images)} images")

        for image_path in images:

            total_images +=1

            image = cv2.imread(str(image_path))

            if image is None:
                logging.warning(f"failed to read image: {image_path}")
                skipped += 1
                continue

            hands_data = detector.detect(image)

            if not hands_data:
                logging.warning(f"no hand detected in image: {image_path}")
                skipped += 1
                continue

            for hand in hands_data:
                landmarks = hand['landmarks']
                features = fe.extract(landmarks)
                normalized = fe.normalize(features)
                all_data.append(normalized)
                all_labels.append(class_label)
                successful += 1

    # After all classes processed
    if successful == 0:
        logging.error("no valid samples extracted")
        return
    
    logging.info("converting to numpy arrays...")

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels)

    output_dict = {
        "data": all_data,
        "labels": all_labels
    }

    logging.info(f"saving to {output_path}...")

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
    
    logging.info("[OK] Saved successfully!")
    logging.info("=" * 50)
    logging.info(f"Total images: {total_images}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Data shape: {all_data.shape}")
    logging.info(f"Data range: [{np.min(all_data):.4f}, {np.max(all_data):.4f}]")
    logging.info("=" * 50)

if __name__ == "__main__":
    extract_landmarks_batch()
    print("\n[OK] Landmark extraction completed!")
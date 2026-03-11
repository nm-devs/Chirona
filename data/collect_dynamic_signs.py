"""
Interactive script for capturing dynamic sign sequences via webcam.

Provides a UI to capture multi-frame sequences for signs involving motion.
Saves sequences as numpy arrays of shape (SEQUENCE_LENGTH, EXPECTED_FEATURES)
to the dynamic data directory.
"""
import os
import sys
import cv2
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from core.hand_detector import HandDetector
from core.feature_extractor import FeatureExtractor
from config import (
    CAM_HEIGHT, CAM_WIDTH, DYNAMIC_DATA_DIR, CAMERA_INDEX,
    DYNAMIC_CLASSES, SEQUENCE_LENGTH, SEQUENCES_PER_CLASS,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    EXPECTED_FEATURES
)

# Configuration for UI
COUNTDOWN = 3
DELAY_BETWEEN_CLASSES = 50  # milliseconds
COLOR_TEXT = (0, 255, 0)
COLOR_WARN = (0, 0, 255)
COLOR_INFO = (255, 255, 0)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera (index={}).".format(CAMERA_INDEX))
        exit(1)

    # Initialize HandDetector and FeatureExtractor
    detector = HandDetector(max_hands=1)  # Focus on a single hand for collection by default
    fe = FeatureExtractor(use_z=False)

    print("=" * 50)
    print("DYNAMIC SIGN DATA COLLECTION")
    print(f"Classes: {DYNAMIC_CLASSES}")
    print(f"Sequence Length: {SEQUENCE_LENGTH} frames")
    print(f"Samples per class: {SEQUENCES_PER_CLASS}")
    print("=" * 50)

    for sign_class in DYNAMIC_CLASSES:
        class_dir = os.path.join(DYNAMIC_DATA_DIR, sign_class)
        os.makedirs(class_dir, exist_ok=True)

        for sample_num in range(SEQUENCES_PER_CLASS):
            # 1. Wait for user to start the sequence
            while True:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)

                cv2.putText(frame, f'Class: {sign_class.upper()} | Sample {sample_num+1}/{SEQUENCES_PER_CLASS}',(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_INFO, 2)
                cv2.putText(frame, 'Press "S" to start countdown',(20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
                cv2.putText(frame, 'Press "Q" to quit', (20,140), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WARN, 2)
                cv2.imshow('Collect Dynamic Signs', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                if key == ord('s'):
                    break

            # 2. Countdown phase
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)

                elapsed = time.time() - start_time
                remaining = COUNTDOWN - int(elapsed)
                if remaining <= 0:
                    break

                cv2.putText(frame, f'{remaining}', (CAM_WIDTH//2 - 50, CAM_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 4, COLOR_WARN, 4)
                cv2.imshow('Collect Dynamic Signs', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

            # 3. Recording phase
            sequence_data = []
            frames_recorded = 0
            
            while frames_recorded < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)

                # Process hand
                hands_data = detector.detect(frame)
                frame = detector.draw_hands(frame, hands_data)

                # If a hand is found, grab the features. If multiple, grab the first.
                if hands_data:
                    landmarks = hands_data[0]['landmarks']
                    features = fe.extract(landmarks)
                    normalized_features = fe.normalize(features)
                    sequence_data.append(normalized_features)
                    frames_recorded += 1
                    
                    # Overlay Recording Status
                    cv2.putText(frame, f'RECORDING: {frames_recorded}/{SEQUENCE_LENGTH}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WARN, 3)
                else:
                    # Provide an immediate user queue that tracking is lost causing recording to stall
                    cv2.putText(frame, f'RECORDING: {frames_recorded}/{SEQUENCE_LENGTH}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WARN, 3)
                    cv2.putText(frame, 'NO HAND DETECTED - PAUSED', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WARN, 2)

                cv2.imshow('Collect Dynamic Signs', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

            # 4. Save Sequence
            sequence_array = np.array(sequence_data, dtype=np.float32)
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(class_dir, f'{timestamp}.npy')
            np.save(file_path, sequence_array)
            print(f"Saved {file_path} - Shape: {sequence_array.shape}")

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*50)
    print("Dynamic data collection complete!")
    print("="*50)

if __name__ == "__main__":
    main()

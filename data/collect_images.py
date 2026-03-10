"""
Interactive script for capturing dataset images via webcam.

Provides a UI with countdowns to capture samples for each alphabet class defined
in the config. Saves images to raw data directory.
"""
import os
import sys
import cv2
import time

from pathlib import Path
from config import (
    CLASSES,
    IMAGES_PER_CLASS,
    CAM_HEIGHT,
    CAM_WIDTH,
    COUNTDOWN,
    DELAY_BETWEEN_CLASSES,
    DATA_DIR,
    CAMERA_INDEX
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Check if camera opened successfully
if not cap.isOpened():
    print("ERROR: Could not open camera (index={}).\nCheck:".format(CAMERA_INDEX))
    print("  1. Camera is connected and not in use by another app")
    print("  2. CAMERA_INDEX in config.py is correct (0 is usually default)")
    exit(1)

for class_letter in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_letter)
    os.makedirs(class_dir, exist_ok=True)

    # wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera. Camera may be disconnected.")
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
        frame = cv2.flip(frame, 1) # mirror effect
        cv2.putText(frame, f'Letter {class_letter.upper()} - Press "S" to start',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'Press "Q" to quit',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)    
        cv2.imshow('Collect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key== ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        if key== ord('s'):
            break
    start_time = time.time()

    # countdown before starting collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera during countdown.")
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = COUNTDOWN - int(elapsed)
        if remaining <= 0:
            break

        # Show countdown number in center
        cv2.putText(frame, f'{remaining + 1}', (CAM_WIDTH//2 - 50, CAM_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 4)
        cv2.putText(frame, 'Press "Q" to quit', (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Collect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    # Collect images
    for img_num in range(IMAGES_PER_CLASS):
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Failed to read frame {img_num} for letter {class_letter.upper()}. Skipping.")
            continue
        frame = cv2.flip(frame, 1)
        
        # overlay progress
        cv2.putText(frame, f'Letter {class_letter.upper()} - Image {img_num+1}/{IMAGES_PER_CLASS}', (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, 'Press "Q" to quit', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('Collect', frame)
        timestamp = int(time.time() * 1000)
        cv2.imwrite(os.path.join(class_dir, f'{timestamp}.jpg'), frame)
        # Wait for delay, then check for quit key
        key = cv2.waitKey(DELAY_BETWEEN_CLASSES) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

# clean up
cap.release()
cv2.destroyAllWindows()
print("\n" + "="*50)
print("Data collection complete!")
print("="*50)
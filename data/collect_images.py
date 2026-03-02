# dataset collection script
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import time
from pathlib import Path
from config import (
    NUM_CLASSES,
    IMAGES_PER_CLASS,
    CAM_HEIGHT,
    CAM_WIDTH,
    COUNTDOWN,
    DELAY_BETWEEN_CLASSES,
    DATA_DIR,
    CAMERA_INDEX
)

cap=cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

for class_id in range(NUM_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

    # wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1) # mirror effect
        cv2.putText(frame, f'Class {class_id} - Press "S" to start',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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
            continue
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
            continue
        frame = cv2.flip(frame, 1)
        
        #overlay progress
        cv2.putText(frame, f'Class {class_id} - Image {img_num+1}/{IMAGES_PER_CLASS}', (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, 'Press "Q" to quit', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('Collect', frame)
        timestamp = int(time.time() * 1000)
        cv2.imwrite(os.path.join(class_dir, f'{timestamp}.jpg'), frame)
        cv2.waitKey(DELAY_BETWEEN_CLASSES) # small delay to avoid duplicates
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

#clean up
cap.release()
cv2.destroyAllWindows() 
print("Data collection complete!")
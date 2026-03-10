from xml.parsers.expat import model
import pickle
"""
Main entry point for the Chirona application.

Initializes the hardware, machine learning models, and controllers to translate
real-time hand gestures into actionable commands or sign language translation.
"""
import cv2
import time
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from core.hand_detector import HandDetector
from core.feature_extractor import FeatureExtractor
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from core.sign_classifier import SignClassifier
from core.feature_extractor import FeatureExtractor
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE, CONFIDENCE_THRESHOLD,
)

# Initialize controllers
controllers = {
    "mouse": MouseController(alpha=SMOOTHING_ALPHA),
    "sign_language": SignLanguageController(),
}

# Initialize hand detector
detector = HandDetector(MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

# hand detection toggle
current_max_hands = MAX_HANDS

fe=FeatureExtractor(use_z=False)

try: 
    sign_clf = SignClassifier('models/trained_model.pkl')
    logging.warning('sign classifier loaded successfully')
except FileNotFoundError:
    logging.warning("Trained model not found. Sign language recognition will not work until you run 'train_model.py' to create the model file.")
    sign_clf = None
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sign_clf = None

def display_prediction(frame, label, confidence, position):
    if confidence < CONFIDENCE_THRESHOLD:
        return frame# Dont display low confidence predictions
    x, y = position
    text = f'{label} ({confidence:.1%})'
    bg_rext = (x-10, y-30, x+180, y+10)
    # draw background rectangle for text
    cv2.rectangle(frame, (x - 5, y-25), (x + 150, y + 5),(0,0,0), -1)
    #draw text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
try:
    with open(".modelss/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Trained model not found. Please run 'train_model.py' first to create the model file.")
    model=None
# Initialize webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Runtime state
prev_time = 0
mode = "mouse"
# hand detection mode toggle
max_hands_mode = 1  # Start with single hand mode
frame_count = 0     # For prediction frequency optimization

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    hands_data = detector.detect(frame)
    frame = detector.draw_hands(frame, hands_data)

    # Increment frame counter for prediction frequency control
    frame_count += 1

    # Process first detected hand with the active controller
    if hands_data:
        hand = hands_data[0]
        
        # SIGN LANGUAGE MODE - Predict every 3rd frame for performance
        if mode == "sign_language" and sign_clf is not None:
            landmarks = hand['landmarks']
            
            # Only predict every 3rd frame to improve FPS
            if frame_count % 3 == 0:
                # Extract and normalize features
                features = fe.extract(landmarks)
                normalized = fe.normalize(features)
            
                # Make prediction - returns (label, confidence) tuple
                pred_label, pred_confidence = sign_clf.predict(normalized)
            
                # Display prediction on frame if above confidence threshold
                if pred_confidence >= CONFIDENCE_THRESHOLD:
                    # Use landmark bounding box for position 
                    bbox = hand['bbox']
                    position = (bbox[0], bbox[1])  # top-left corner
                    frame = display_prediction(frame, pred_label, pred_confidence, position)
        
        # MOUSE CONTROL MODE
        elif mode == "mouse":
            controllers[mode].process_frame(frame, hand, detector)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Display FPS and mode
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Hands: {max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if mode == 'sign_language' and sign_clf is None:
        cv2.putText(frame, f'Min confidence: {CONFIDENCE_THRESHOLD:.0%}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow(WINDOW_TITLE, frame)

    # Handle keypresses
    """key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break"""
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == ord('h'):
        #toggle between 1 and 2 hand modes
        max_hands_mode = 2 if max_hands_mode == 1 else 1
        detector.hands.max_num_hands = max_hands_mode
        print(f'Hand dectection mode: {max_hands_mode} hand(s)')
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
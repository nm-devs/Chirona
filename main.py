from xml.parsers.expat import model

import cv2
import time
from core.hand_detector import HandDetector
from core.feature_extractor import FeatureExtractor
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE,
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

import pickle 
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
max_hands_mode = 1 #start with single hand mode
# Main loop
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    hands_data = detector.detect(frame)


    # Process first detected hand with the active controller
    if hands_data:
        hand = hands_data[0]
        landmarks = hand['landmarks']
        
        # Test FeatureExtractor
        features = fe.extract(landmarks)
        normalized = fe.normalize(features)
        
        # Print debug info
        print(f"Features shape: {features.shape}")
        print(f"Features sample: {features[:6]}")  # First 3 landmarks (x,y,x,y,x,y)
        print(f"Normalized shape: {normalized.shape}")
        print(f"Normalized sample: {normalized[:6]}")
        print("---")
        
        # Still use original controller logic
        controllers[mode].process_frame(frame, hand, detector)
        #prediction = model.predict(normalized.reshape(1, -1))       

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Display FPS and mode
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Hands: {max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
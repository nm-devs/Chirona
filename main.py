"""
Main entry point for the Chirona application.

Initializes the hardware, machine learning models, and controllers to translate
real-time hand gestures into actionable commands or sign language translation.
"""

import pickle
import cv2
import time
import sys
import logging
import numpy as np

from collections import deque
from core.gesture_detector import GestureDetector
from utils.gesture_display import draw_gesture_feedback, draw_sentence_display
from xml.parsers.expat import model
from core.sign_classifier import SignClassifier
from core.hand_detector import HandDetector
from utils.text_overlay import draw_prediction, draw_sentence_builder_ui
from core.feature_extractor import FeatureExtractor
from core.sentence_builder import SentenceBuilder
from utils.prediction_smoother import PredictionSmoother
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE, CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW_SIZE, SMOOTHING_DOMINANCE_THRESHOLD,
    COLOR_PRIMARY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChironaApp:
    def __init__(self):
        self._setup()
        
    def _setup(self):
        """Initialize models, controllers, and hardware."""
        self.controllers = {
            "mouse": MouseController(alpha=SMOOTHING_ALPHA),
            "sign_language": SignLanguageController(),
        }
        
        # Initialize hand detector (start in single hand mode)
        self.detector = HandDetector(1, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)
        self.fe = FeatureExtractor(use_z=False)  # Must match training config
        
        # Load trained sign language model
        try: 
            self.classifier = SignClassifier('models/trained_model.pkl')
            logging.warning('sign classifier loaded successfully')
        except FileNotFoundError:
            logging.warning("Trained model not found. Sign language recognition will not work until you run 'train_model.py' to create the model file.")
            self.classifier = None
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.classifier = None
            
        # Initialize webcam
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        if not self.cap.isOpened():
            print("Failed to open camera")
            sys.exit(1)
            
        # Runtime state variables
        self.prev_time = 0
        self.mode = "mouse"
        self.prediction_history = deque(maxlen=5)  # For gesture vs sign distinction
        self.max_hands_mode = 1 # start with single hand mode
        
        self.smoother = PredictionSmoother(
            window_size=SMOOTHING_WINDOW_SIZE, 
            dominance_threshold=SMOOTHING_DOMINANCE_THRESHOLD
        )
        self.sentence_builder = SentenceBuilder()
        self.displayed_sign = None
        self.displayed_confidence = None
        self.frame_count = 0
        # initilize gesture detector
        self.gesture_detector = GestureDetector()
        self.sentence_builder = SentenceBuilder()
        self.last_detected_gesture = None
        self.gesture_cooldown = 0.5  # seconds

    def _process_prediction(self, hand, hands_data):
        """Extract features, predict gesture, and smooth the output for the UI.
        Uses OPTION A: Confidence-based filtering to detect gestures vs ASL signs."""
        landmarks = hand['landmarks']
        gesture = self.gesture_detector.detect_gesture(hands_data)
        # detect gestures first (before smoothing) since they rely on raw landmark positions and we want to catch them even if the predicted sign is unstable
        if gesture:
            # handle gesture control immediately
            if gesture == 'space':
                self.sentence_builder.add_space()
            elif gesture == 'backspace':
                self.sentence_builder.backspace()
            elif gesture == 'speak':
                text = self.sentence_builder.speak()
                print(f"Speaking: {text}")
            elif gesture == 'clear':
                self.sentence_builder.clear()
            self.last_detected_gesture = gesture
            # Skip letter detection when gesture is detected to focus on gesture
            return
        else:
            self.last_detected_gesture = None           
        # Only predict every 3rd frame to improve FPS
        if self.frame_count % 3 == 0:
            # Extract and normalize features
            features = self.fe.extract(landmarks)
            normalized_features = self.fe.normalize(features)

            # Predict gesture
            if self.classifier is not None:
                label, confidence = self.classifier.predict(normalized_features)
                
                #Track prediction history for gesture/sign distinction
                if confidence > 0.70:  # Only track confident predictions
                    self.prediction_history.append(label)
                
                # Check if this looks like a gesture (hand moving with 2+ different detections)
                unique_predictions = len(set(self.prediction_history))
                
                # consistent predictions = likely ASL sign, varying predictions = likely gesture
                if confidence > 0.0:
                    self.smoother.add_prediction(label)
                stable = self.smoother.get_stable_prediction()

                # update displayd sign if stabl prediction is available
                if stable is not None:
                    self.displayed_sign = stable
                    self.displayed_confidence = confidence

    def _handle_keypress(self):
        """Handle keyboard input. Returns False if app should exit."""
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle control modes 
        if key == ord('m'):
            self.mode = "sign_language" if self.mode == "mouse" else "mouse"
            self.sentence_builder.clear()
            
        # Manually add space with spacebar when in sign language mode
        if key == ord(' ') and self.mode == "sign_language":
            self.sentence_builder.add_space()
            
        if key == ord('h'):
            #toggle between 1 and 2 hand modes
            self.max_hands_mode = 2 if self.max_hands_mode == 1 else 1
            self.detector.hands.max_num_hands = self.max_hands_mode
            print(f'Hand dectection mode: {self.max_hands_mode} hand(s)')
            
        # Exit on Escape key or window close
        if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            return False
            
        return True

    def run(self):
        """Main application runtime loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            hands_data = self.detector.detect(frame)
            frame = self.detector.draw_hands(frame, hands_data)
            
            self.frame_count += 1

            # Process first detected hand with the active controller
            if hands_data:
                first_hand = hands_data[0]
                
                if self.mode == "sign_language":
                    self._process_prediction(first_hand, hands_data)
                elif self.mode == "mouse":
                    self.controllers[self.mode].process_frame(frame, first_hand, self.detector)
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time

            # Update sentence builder
            if self.mode == "sign_language":
                if hands_data:
                    self.sentence_builder.update(self.displayed_sign, current_time)
                else:
                    self.sentence_builder.update(None, current_time)

            # Display info text overlays
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
            cv2.putText(frame, f'Hands: {self.max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
            cv2.putText(frame, f'Mode: {self.mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
            
            if self.mode == 'sign_language' and self.classifier is None:
                cv2.putText(frame, f'Min confidence: {CONFIDENCE_THRESHOLD:.0%}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 1)

            # Display predicted sign bar if available
            if self.mode == "sign_language" and self.displayed_sign and self.displayed_confidence:
                draw_prediction(frame, self.displayed_sign, self.displayed_confidence)

            # Display sentence builder UI
            if self.mode == "sign_language":
                draw_sentence_builder_ui(frame, self.sentence_builder, current_time)
            # Display gesture feedback
            if self.mode == "sign_language" and self.last_detected_gesture:
                frame = draw_gesture_feedback(frame, self.last_detected_gesture, (10, 250))
            # display current sentence
            if self.mode == "sign_language":
                current_text = self.sentence_builder.get_display_text()
                frame = draw_sentence_display(frame, current_text, (10, 450))   
            cv2.imshow(WINDOW_TITLE, frame)

            # Break loop if _handle_keypress asks to exit
            if not self._handle_keypress():
                break

        self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ChironaApp()
    app.run()
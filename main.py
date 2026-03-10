"""
Main entry point for the Chirona application.

Initializes the hardware, machine learning models, and controllers to translate
real-time hand gestures into actionable commands or sign language translation.
"""
import cv2
import time
import sys

from core.sign_classifier import SignClassifier
from core.hand_detector import HandDetector
from utils.text_overlay import draw_prediction
from core.feature_extractor import FeatureExtractor
from utils.prediction_smoother import PredictionSmoother
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE,
    SMOOTHING_WINDOW_SIZE, SMOOTHING_DOMINANCE_THRESHOLD,
    COLOR_PRIMARY
)

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
        except FileNotFoundError:
            print("Trained model not found. Please run 'train_model.py' first to create the model file.")
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
        self.max_hands_mode = 1 # start with single hand mode
        
        self.smoother = PredictionSmoother(
            window_size=SMOOTHING_WINDOW_SIZE, 
            dominance_threshold=SMOOTHING_DOMINANCE_THRESHOLD
        )
        self.displayed_sign = None
        self.displayed_confidence = None

    def _process_prediction(self, hand):
        """Extract features, predict gesture, and smooth the output for the UI."""
        landmarks = hand['landmarks']
        
        # Extract and normalize features
        features = self.fe.extract(landmarks)
        normalized_features = self.fe.normalize(features)

        # Predict gesture
        if self.classifier is not None:
            label, confidence = self.classifier.predict(normalized_features)
            self.smoother.add_prediction(label)
            stable = self.smoother.get_stable()

            # Update displayed sign if stable prediction is available
            if stable is not None:
                self.displayed_sign = stable
                self.displayed_confidence = confidence

    def _handle_keypress(self):
        """Handle keyboard input. Returns False if app should exit."""
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle control modes 
        if key == ord('m'):
            self.mode = "sign_language" if self.mode == "mouse" else "mouse"
            
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

            # Process first detected hand with the active controller
            if hands_data:
                first_hand = hands_data[0]
                self.controllers[self.mode].process_frame(frame, first_hand, self.detector)
                self._process_prediction(first_hand)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time

            # Display info text overlays
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
            cv2.putText(frame, f'Hands: {self.max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
            cv2.putText(frame, f'Mode: {self.mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)

            # Display predicted sign bar if available
            if self.displayed_sign and self.displayed_confidence:
                draw_prediction(frame, self.displayed_sign, self.displayed_confidence)

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
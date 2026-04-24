"""
Main entry point for the Chirona Sign Language application.

Initializes the hardware, machine learning models, and feature extractors
to translate real-time hand gestures into actionable sign language translation.
"""


import cv2
import sys
import time
import pickle
import logging
import numpy as np

from collections import deque
from xml.parsers.expat import model
from core.hand_detector import HandDetector
from core.sign_classifier import SignClassifier
from core.gesture_detector import GestureDetector
from core.sentence_builder import SentenceBuilder
from core.feature_extractor import FeatureExtractor
from core.motion_detector import MotionDetector
from core.dynamic_classifier import DynamicClassifier
from utils.prediction_smoother import PredictionSmoother
from utils.text_to_speech import TextToSpeech
from utils.hud import draw_hud
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    COLOR_PRIMARY, COLOR_WARNING, COLOR_ACCENT, WINDOW_TITLE,
    TTS_ENABLED, TTS_SPEECH_RATE, TTS_VOLUME,
    MOTION_VELOCITY_THRESHOLD, MOTION_FRAMES_REQUIRED, MOTION_COOLDOWN_FRAMES,
    DYNAMIC_CONFIDENCE_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChironaApp:
    def __init__(self):
        self._setup()
        
    def _setup(self):
        """Initialize models and hardware."""
        # Initialize hand detector (start in single hand mode)
        self.detector = HandDetector(max_hands=1)
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
        
        # initialize text-to-speech system if enabled
        self.tts = None 
        if TTS_ENABLED:
            try:
                self.tts = TextToSpeech(rate=TTS_SPEECH_RATE, volume=TTS_VOLUME)
                logging.info("Text-to-speech system initialized successfully.")
            except Exception as e:
                logging.error(f"Error initializing text-to-speech system: {e}")
                self.tts = None
        
        # Load dynamic (LSTM) classifier
        self.dynamic_classifier = DynamicClassifier()
        if self.dynamic_classifier.is_available:
            logging.info("Dynamic LSTM classifier loaded successfully.")
        else:
            logging.warning("Dynamic LSTM classifier not available. Dynamic sign recognition disabled.")

        # Motion detector for static ↔ dynamic switching
        self.motion_detector = MotionDetector(
            velocity_threshold=MOTION_VELOCITY_THRESHOLD,
            motion_frames_required=MOTION_FRAMES_REQUIRED,
            cooldown_frames=MOTION_COOLDOWN_FRAMES,
        )

        # Runtime state variables
        self.prev_time = 0
        self.mode = "mouse"
        self.prediction_history = deque(maxlen=5)
        self.max_hands_mode = 1
        self.smoother = PredictionSmoother()
        self.sentence_builder = SentenceBuilder(tts=self.tts)
        self.displayed_sign = None
        self.displayed_confidence = None
        self.displayed_source = None
        self.show_reference = False
        self.frame_count = 0

        # initilize gesture detector
        self.gesture_detector = GestureDetector()
        self.last_detected_gesture = None
        self.gesture_cooldown = 0.5

    def _process_prediction(self, hand, hands_data):
        """Extract features, predict gesture, and smooth the output for the UI.
        Uses motion detection to switch between static RF and dynamic LSTM classifiers."""
        landmarks = hand['landmarks']

        # Check for raw gesture first to prevent classifier fallback during cooldown/hold
        raw_gesture = self.gesture_detector.detect_raw_gesture(hands_data)

        # Get smoothed gesture (respects cooldown and consistency)
        gesture = self.gesture_detector.detect_gesture(hands_data)

        if raw_gesture:
            if gesture:
                logging.info(f"Gesture triggered: {gesture}")
                if gesture == 'space':
                    self.sentence_builder.add_space()
                    logging.info("Space gesture: added space")
                elif gesture == 'backspace':
                    self.sentence_builder.backspace()
                    logging.info("Backspace gesture: removed character")
                elif gesture == 'speak':
                    text = self.sentence_builder.speak()
                    logging.info(f"Speak gesture - Sentence: '{self.sentence_builder.sentence}' | Current word: '{self.sentence_builder.current_word}' | Full text: '{text}'")
                elif gesture == 'clear':
                    self.sentence_builder.clear()
                    logging.info("Clear gesture: cleared sentence")
                self.last_detected_gesture = gesture
            else:
                logging.debug(f"Raw gesture '{raw_gesture}' detected but not triggered (cooldown/smoothing)")
            return
        else:
            self.last_detected_gesture = None

        # Extract and normalize features every frame (needed for motion detection)
        features = self.fe.extract(landmarks)
        normalized_features = self.fe.normalize(features)

        # Update motion detector
        motion_state = self.motion_detector.update(normalized_features)

        if motion_state == 'dynamic_ready' and self.dynamic_classifier.is_available:
            sequence = self.motion_detector.get_sequence()
            if sequence is not None:
                label, confidence = self.dynamic_classifier.predict(sequence)
                if label and confidence >= DYNAMIC_CONFIDENCE_THRESHOLD:
                    self.displayed_sign = label
                    self.displayed_confidence = confidence
                    self.displayed_source = 'dynamic'
                    logging.info(f"Dynamic sign detected: {label} ({confidence:.1%})")
                else:
                    logging.debug(f"LSTM prediction below threshold: {label} ({confidence:.1%})")
                self.motion_detector.finish_dynamic(label, confidence)
            return

        if motion_state == 'buffering':
            return

        # Static classification (only every 3rd frame for FPS)
        if self.frame_count % 3 == 0 and self.classifier is not None:
            label, confidence = self.classifier.predict(normalized_features)

            if confidence > 0.70:
                self.prediction_history.append(label)

            if confidence > 0.0:
                self.smoother.add_prediction(label)
            stable = self.smoother.get_stable_prediction()

            if stable is not None:
                self.displayed_sign = stable
                self.displayed_confidence = confidence
                self.displayed_source = 'static'

    def _handle_keypress(self):
        """Handle keyboard input. Returns False if app should exit."""
        key = cv2.waitKey(1) & 0xFF
        
        # Manually add space with spacebar
        if key == ord(' '):
            self.sentence_builder.add_space()

        #triggr speech with 's' key for testing without gesture
        if key == ord('s'):
            text = self.sentence_builder.speak()
            if text:
                logging.info(f"Manually triggered speech: '{text}'")      
        if key == ord('r'):
            self.show_reference = not self.show_reference

        if key == ord('h'):
            self.max_hands_mode = 2 if self.max_hands_mode == 1 else 1
            self.detector.hands.max_num_hands = self.max_hands_mode
            print(f'Hand detection mode: {self.max_hands_mode} hand(s)')
            
        if key == 27:
            return False
        
        # Only check window property if we've processed at least a few frames
        # to avoid false positives during initialization
        if self.frame_count > 10:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                return False
            
        return True

    def run(self):
        """Main application runtime loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                logging.error("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            hands_data = self.detector.detect(frame)
            frame = self.detector.draw_hands(frame, hands_data)
            
            self.frame_count += 1

            # Process first detected hand
            if hands_data:
                first_hand = hands_data[0]
                self._process_prediction(first_hand, hands_data)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time

            # Update sentence builder
            if hands_data:
                self.sentence_builder.update(self.displayed_sign, current_time)
            else:
                self.sentence_builder.update(None, current_time)

            # Determine mode display
            motion_state = self.motion_detector.state if hands_data else 'static'
            buffer_progress = None
            if motion_state == 'buffering':
                mode_text = 'Mode: Dynamic (buffering)'
                mode_color = COLOR_WARNING
                buffer_progress = self.motion_detector.get_buffer_progress()
            elif self.displayed_source == 'dynamic' and self.displayed_sign:
                mode_text = 'Mode: Dynamic (LSTM)'
                mode_color = COLOR_ACCENT
            else:
                mode_text = 'Mode: Static (RF)'
                mode_color = COLOR_PRIMARY

            tts_speaking = bool(self.tts and self.tts.is_currently_speaking())

            draw_hud(
                frame,
                fps=fps,
                hand_count=self.max_hands_mode,
                mode_text=mode_text,
                mode_color=mode_color,
                label=self.displayed_sign,
                confidence=self.displayed_confidence,
                source=self.displayed_source,
                sentence_builder=self.sentence_builder,
                current_time=current_time,
                tts_speaking=tts_speaking,
                buffer_progress=buffer_progress,
                show_reference=self.show_reference,
            )
            
            cv2.imshow(WINDOW_TITLE, frame)

            # Break loop if _handle_keypress asks to exit
            if not self._handle_keypress():
                break
                
        self.cleanup()

    def cleanup(self):
        """Release resources."""
        #shutdown TTS system if initialized
        if self.tts:
            self.tts.shutdown()

        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ChironaApp()
    app.run()
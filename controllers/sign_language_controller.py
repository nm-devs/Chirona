"""
Controller for processing visual feedback in sign language translation mode.

Handles the rendering of hand tracking points and skeletons on the camera feed
when the user is spelling out signs.
"""
import cv2

from utils.drawing_utils import draw_hand_points, draw_hand_skeleton


class SignLanguageController:
    def __init__(self):
        pass

    def process_frame(self, frame, hand, detector):
        """Process one frame in sign-language mode."""
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]

        # Draw visuals
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)

        # Placeholder until classifier is loaded
        cv2.putText(
            frame,
            "Sign Language Mode - classifier not loaded yet",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
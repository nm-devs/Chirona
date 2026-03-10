"""
Drawing utilities for overlaying MediaPipe hand tracking visualizations.

Handles drawing landmarks, bounding boxes, and hand skeletons for UI feedback.
"""
import cv2

# Fingertip landmark indices
TIP_IDS = [4, 8, 12, 16, 20]


def draw_hand_points(frame, positions, tip_ids=TIP_IDS):
    """
    Draw circles on all 21 hand landmarks.
    Fingertips get larger circles.
    
    Args:
        frame: The image to draw on
        positions: List of (idx, px, py) tuples
        tip_ids: List of fingertip landmark indices
    """
    for idx, px, py in positions:
        radius = 8 if idx in tip_ids else 5
        cv2.circle(frame, (px, py), radius, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)  # Black outline


def draw_bounding_box(frame, positions, label, color):
    """
    Draw a bounding box with label around a hand.
    
    Args:
        frame: The image to draw on
        positions: List of (idx, px, py) tuples
        label: Text label to display (e.g., "Left Hand")
        color: BGR color tuple for the box
    """
    x_coords = [p[1] for p in positions]
    y_coords = [p[2] for p in positions]
    
    x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
    y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
    
    # Draw bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw label background and text
    cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 120, y_min), color, cv2.FILLED)
    cv2.putText(frame, label, (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_hand_skeleton(frame, hand_landmarks, mp_hands, mp_drawing):
    """
    Draw skeleton connections between hand landmarks.
    
    Args:
        frame: The image to draw on
        hand_landmarks: MediaPipe hand landmarks object
        mp_hands: MediaPipe hands module (mp.solutions.hands)
        mp_drawing: MediaPipe drawing utils (mp.solutions.drawing_utils)
    """
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
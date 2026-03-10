"""
Text overlay and UI utilities for the main application.

Provides helper methods to render prediction labels, confidence scores,
and progress bars on top of webcam frames.
"""
import cv2

from config import (
    COLOR_PRIMARY, COLOR_WARNING, COLOR_DANGER, COLOR_BLACK_BG,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
)

def draw_prediction(frame, label, confidence, position="top-right"):
    
    if confidence >= CONFIDENCE_HIGH:
        color = COLOR_PRIMARY      # green
    elif confidence >= CONFIDENCE_MEDIUM:       # implicitly < CONFIDENCE_HIGH because of the if above
        color = COLOR_WARNING      # yellow
    else:                          # < CONFIDENCE_MEDIUM
        color = COLOR_DANGER       # red
    
    # h = height in pixels (e.g., 720)
    # w = width in pixels  (e.g., 1280)
    # _ = channels (3 for BGR color, we ignore this)
    h, w, _ = frame.shape
    padding = 20
    box_width = 250
    box_height = 120

    if position == "top-right":
        x = w - box_width - padding    # e.g., 1280 - 250 - 20 = 1010
        y = padding                    # 20px from top
    elif position == "top-left":
        x = padding                    # 20px from left
        y = padding                    # 20px from top
    elif position == "bottom-right":
        x = w - box_width - padding    # e.g., 1280 - 250 - 20 = 1010
        y = h - box_height - padding   # e.g., 720 - 120 - 20 = 580
    elif position == "bottom-left":
        x = padding                    # 20px from left
        y = h - box_height - padding   # e.g., 720 - 120 - 20 = 580
    else:
        raise ValueError("Invalid position. Must be 'top-right', 'top-left', 'bottom-right', or 'bottom-left'.")
    
    # Semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Position variables for text and bar
    text_x = x + 15
    text_y = y + 40
    bar_x = x + 15
    bar_y = y + 70
    bar_max_width = box_width - 30
    bar_height = 20

    # Draw label and confidence text
    cv2.putText(frame, label.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"{int(confidence * 100)}%", (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Gray background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), COLOR_BLACK_BG, cv2.FILLED)
    
    # Colored fill bar
    fill_width = int(bar_max_width * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, cv2.FILLED)

    return frame
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


def draw_sentence_builder_ui(frame, sentence_builder, current_time):
    """Draws the sentence builder output and progress bar."""
    h, w, _ = frame.shape

    # 1. Calculate progress if there's a current sign
    progress = 0.0
    if sentence_builder.current_sign:
        elapsed = current_time - sentence_builder.start_time
        progress = max(0.0, min(1.0, elapsed / sentence_builder.confirm_duration))

    # 2. Render progress bar (bottom center)
    bar_width = 400
    bar_height = 20
    bar_x = (w - bar_width) // 2
    bar_y = h - 60

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLOR_BLACK_BG, cv2.FILLED)
    
    if progress > 0:
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), COLOR_WARNING, cv2.FILLED)
    
    # Text for "Hold to confirm {letter}"
    if sentence_builder.current_sign:
        text = f"Holding: {sentence_builder.current_sign.upper()}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y - 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WARNING, 2)

    # 3. Render Current Word at Top
    word_text = sentence_builder.current_word.upper()
    in_progress_char = sentence_builder.current_sign.upper() if sentence_builder.current_sign else ""
    
    word_scale = 2.0
    word_thickness = 4
    
    word_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, word_scale, word_thickness)[0]
    in_progress_size = cv2.getTextSize(in_progress_char, cv2.FONT_HERSHEY_SIMPLEX, word_scale, word_thickness)[0]
    
    space_between = 5 if in_progress_char and word_text else 0
    total_width = word_size[0] + in_progress_size[0] + space_between
    
    start_x = (w - total_width) // 2
    top_y = 60
    
    if word_text:
        # Draw confirmed letters block (with black outline for readability)
        cv2.putText(frame, word_text, (start_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, word_scale, (0, 0, 0), word_thickness + 2)
        cv2.putText(frame, word_text, (start_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, word_scale, COLOR_PRIMARY, word_thickness)
    
    if in_progress_char:
        in_progress_x = start_x + word_size[0] + space_between
        # Draw in-progress letter block (with black outline for readability)
        cv2.putText(frame, in_progress_char, (in_progress_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, word_scale, (0, 0, 0), word_thickness + 2)
        cv2.putText(frame, in_progress_char, (in_progress_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, word_scale, COLOR_WARNING, word_thickness)

    # 4. Render Full sentence below it
    sentence_text = sentence_builder.sentence.upper()
    if sentence_text:
        sent_scale = 1.0
        sent_thickness = 2
        sent_size = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, sent_scale, sent_thickness)[0]
        sent_x = (w - sent_size[0]) // 2
        sent_y = top_y + 50
        # Draw sentence block (with black outline for readability)
        cv2.putText(frame, sentence_text, (sent_x, sent_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (0, 0, 0), sent_thickness + 2)
        cv2.putText(frame, sentence_text, (sent_x, sent_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, COLOR_PRIMARY, sent_thickness)

    return frame
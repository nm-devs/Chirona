import cv2


def draw_gesture_feedback(frame, gesture_name, position=(50, 120)):
    """Draw visual feedback for detected gesture on frame."""
    if not gesture_name:
        return frame
    
    gesture_messages = {
        'space': ('SPACE', (0, 255, 0)),  # green
        'backspace': ('BACKSPACE', (0, 165, 255)),  # orange
        'speak': ('SPEAK', (255, 0, 0)),  # red
        'clear': ('CLEAR', (255, 0, 255))  # magenta
    }

    if gesture_name not in gesture_messages:
        return frame
    
    text, color = gesture_messages[gesture_name]
    x, y = position

    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - 25), (x + 200, y + 10), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, f'GESTURE: {text}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame


def draw_sentence_display(frame, sentence, position=(10, 400)):
    """Draw current sentence at bottom of frame."""
    if not sentence:
        return frame
    
    x, y = position
    h, w, _ = frame.shape
    
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - 30), (w - 10, y + 10), (0, 0, 0), -1)
    # Draw sentence text
    cv2.putText(frame, sentence, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def draw_speaking_feedback(frame, is_speaking, position=(50, 120)):
    """Draw 'SPEAKING...' feedback while TTS is actively synthesizing."""
    if not is_speaking:
        return frame
    
    x, y = position
    
    # Draw background rectangle for visibility
    cv2.rectangle(frame, (x - 5, y - 25), (x + 250, y + 10), (0, 100, 0), -1)
    
    # Draw "SPEAKING..." text in bright green
    cv2.putText(frame, 'SPEAKING...', (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    return frame

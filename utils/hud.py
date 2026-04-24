"""
Professional heads-up display overlay for the webcam feed.

Renders all UI elements (prediction, sentence, status, ASL reference)
with semi-transparent backgrounds and clean typography.
"""
import cv2
import numpy as np
from config import (
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    COLOR_PRIMARY, COLOR_WARNING, COLOR_DANGER, COLOR_ACCENT,
)

_COLOR_BG = (20, 20, 20)
_COLOR_BG_ACCENT = (40, 40, 40)
_COLOR_WHITE = (255, 255, 255)
_COLOR_GRAY = (160, 160, 160)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN
_BG_ALPHA = 0.65


def _draw_transparent_rect(frame, x, y, w, h, color=_COLOR_BG, alpha=_BG_ALPHA):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _confidence_color(confidence):
    if confidence >= CONFIDENCE_HIGH:
        return COLOR_PRIMARY
    elif confidence >= CONFIDENCE_MEDIUM:
        return COLOR_WARNING
    return COLOR_DANGER


# ── Top-left status panel ────────────────────────────────

def draw_status_panel(frame, fps, hand_count, mode_text, mode_color,
                      tts_speaking=False, buffer_progress=None):
    h, w = frame.shape[:2]
    panel_w = 320
    panel_h = 110
    px, py = 8, 8

    _draw_transparent_rect(frame, px, py, panel_w, panel_h)

    y_off = py + 24
    cv2.putText(frame, f'FPS: {int(fps)}', (px + 10, y_off), _FONT, 0.6, COLOR_PRIMARY, 1)
    cv2.putText(frame, f'Hands: {hand_count}', (px + 130, y_off), _FONT, 0.6, _COLOR_WHITE, 1)

    y_off += 28
    cv2.putText(frame, mode_text, (px + 10, y_off), _FONT, 0.55, mode_color, 1)

    if buffer_progress is not None:
        y_off += 24
        bar_x = px + 10
        bar_w = panel_w - 20
        bar_h = 12
        cv2.rectangle(frame, (bar_x, y_off), (bar_x + bar_w, y_off + bar_h), _COLOR_BG_ACCENT, cv2.FILLED)
        fill = int(bar_w * buffer_progress)
        cv2.rectangle(frame, (bar_x, y_off), (bar_x + fill, y_off + bar_h), COLOR_WARNING, cv2.FILLED)
        pct = f'{int(buffer_progress * 100)}%'
        cv2.putText(frame, pct, (bar_x + bar_w + 6, y_off + 11), _FONT, 0.4, COLOR_WARNING, 1)

    if tts_speaking:
        y_off += 24
        cv2.putText(frame, 'SPEAKING...', (px + 10, y_off), _FONT, 0.55, COLOR_PRIMARY, 1)


# ── Top-right prediction panel ───────────────────────────

def draw_prediction_panel(frame, label, confidence, source=None):
    if not label or confidence is None:
        return

    h, w = frame.shape[:2]
    panel_w = 260
    panel_h = 130
    px = w - panel_w - 8
    py = 8

    _draw_transparent_rect(frame, px, py, panel_w, panel_h)

    color = _confidence_color(confidence)

    if source:
        tag = 'STATIC' if source == 'static' else 'DYNAMIC'
        tag_color = COLOR_PRIMARY if source == 'static' else COLOR_ACCENT
        cv2.putText(frame, tag, (px + 10, py + 20), _FONT, 0.45, tag_color, 1)

    cv2.putText(frame, label.upper(), (px + 10, py + 58), _FONT, 1.6, color, 3)
    cv2.putText(frame, f'{int(confidence * 100)}%', (px + 10, py + 85), _FONT, 0.7, color, 1)

    bar_x = px + 10
    bar_y = py + 95
    bar_w = panel_w - 20
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _COLOR_BG_ACCENT, cv2.FILLED)
    fill = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, cv2.FILLED)


# ── Center: current word + sentence ──────────────────────

def draw_sentence_panel(frame, sentence_builder, current_time):
    h, w = frame.shape[:2]

    current_word = sentence_builder.current_word.upper()
    in_progress = sentence_builder.current_sign.upper() if sentence_builder.current_sign else ''
    sentence = sentence_builder.sentence.upper()

    word_scale = 1.8
    word_thick = 3

    word_sz = cv2.getTextSize(current_word, _FONT, word_scale, word_thick)[0]
    ip_sz = cv2.getTextSize(in_progress, _FONT, word_scale, word_thick)[0]
    gap = 4 if in_progress and current_word else 0
    total_w = word_sz[0] + ip_sz[0] + gap

    x = (w - total_w) // 2
    y = 60

    if current_word:
        cv2.putText(frame, current_word, (x, y), _FONT, word_scale, (0, 0, 0), word_thick + 2)
        cv2.putText(frame, current_word, (x, y), _FONT, word_scale, COLOR_PRIMARY, word_thick)

    if in_progress:
        ipx = x + word_sz[0] + gap
        cv2.putText(frame, in_progress, (ipx, y), _FONT, word_scale, (0, 0, 0), word_thick + 2)
        cv2.putText(frame, in_progress, (ipx, y), _FONT, word_scale, COLOR_WARNING, word_thick)

    if sentence:
        sent_scale = 0.8
        sent_thick = 2
        max_w = w - 400
        words = sentence.split()
        lines = []
        cur_line = ''
        for word in words:
            if not word:
                continue
            test = (cur_line + ' ' + word).strip()
            if cv2.getTextSize(test, _FONT, sent_scale, sent_thick)[0][0] <= max_w:
                cur_line = test
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)

        sy = y + 40
        for line in lines:
            sz = cv2.getTextSize(line, _FONT, sent_scale, sent_thick)[0]
            sx = (w - sz[0]) // 2
            cv2.putText(frame, line, (sx, sy), _FONT, sent_scale, (0, 0, 0), sent_thick + 2)
            cv2.putText(frame, line, (sx, sy), _FONT, sent_scale, _COLOR_WHITE, sent_thick)
            sy += 30

    # Hold-to-confirm progress bar
    if sentence_builder.current_sign:
        elapsed = current_time - sentence_builder.start_time
        progress = max(0.0, min(1.0, elapsed / sentence_builder.confirm_duration))

        bar_w = 360
        bar_h = 16
        bar_x = (w - bar_w) // 2
        bar_y = h - 55

        _draw_transparent_rect(frame, bar_x - 4, bar_y - 22, bar_w + 8, bar_h + 28, alpha=0.5)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _COLOR_BG_ACCENT, cv2.FILLED)
        fill = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), COLOR_WARNING, cv2.FILLED)

        hold_text = f'Holding: {sentence_builder.current_sign.upper()}'
        tsz = cv2.getTextSize(hold_text, _FONT, 0.55, 1)[0]
        tx = bar_x + (bar_w - tsz[0]) // 2
        cv2.putText(frame, hold_text, (tx, bar_y - 6), _FONT, 0.55, COLOR_WARNING, 1)


# ── Bottom-left: ASL reference card ──────────────────────

_ASL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def draw_asl_reference(frame):
    h, w = frame.shape[:2]

    cols = 7
    rows = 4
    cell_w = 32
    cell_h = 28
    pad = 8

    grid_w = cols * cell_w + pad * 2
    grid_h = rows * cell_h + pad * 2 + 22

    gx = 8
    gy = h - grid_h - 8

    _draw_transparent_rect(frame, gx, gy, grid_w, grid_h, alpha=0.75)

    cv2.putText(frame, 'ASL Reference  [R] to hide', (gx + pad, gy + 16), _FONT, 0.4, _COLOR_GRAY, 1)

    for i, letter in enumerate(_ASL_LETTERS):
        r = i // cols
        c = i % cols
        cx = gx + pad + c * cell_w + cell_w // 2 - 6
        cy = gy + 26 + pad + r * cell_h + cell_h // 2 + 5
        cv2.putText(frame, letter, (cx, cy), _FONT, 0.5, _COLOR_WHITE, 1)
        cv2.rectangle(frame,
                      (gx + pad + c * cell_w, gy + 24 + pad + r * cell_h),
                      (gx + pad + (c + 1) * cell_w, gy + 24 + pad + (r + 1) * cell_h),
                      _COLOR_BG_ACCENT, 1)


# ── Full HUD render call ─────────────────────────────────

def draw_hud(frame, *, fps, hand_count, mode_text, mode_color,
             label, confidence, source,
             sentence_builder, current_time,
             tts_speaking=False, buffer_progress=None,
             show_reference=False):

    draw_status_panel(frame, fps, hand_count, mode_text, mode_color,
                      tts_speaking=tts_speaking, buffer_progress=buffer_progress)

    draw_prediction_panel(frame, label, confidence, source=source)

    draw_sentence_panel(frame, sentence_builder, current_time)

    if show_reference:
        draw_asl_reference(frame)

    return frame

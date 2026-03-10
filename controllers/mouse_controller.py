"""
Translates hand landmarks into mouse movement and click actions on the user's OS.

Features precise cursor control, jitter reduction mapping, and specific gestures
for left-click, right-click, and scrolling.
"""
import time
import math
import cv2
import pyautogui
import numpy as np

from utils.drawing_utils import draw_hand_points, draw_hand_skeleton
from config import (
    CAM_WIDTH, CAM_HEIGHT, FRAME_REDUCTION,
    PINCH_DISTANCE, SCROLL_JITTER_THRESHOLD, SCROLL_SPEED_MULTIPLIER,
    CLICK_COOLDOWN, FINGER_CIRCLE_RADIUS,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_DANGER
)

# Make pyautogui fast (VERY important)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

class MouseController:
    def __init__(self, alpha=0.25, move_interval=0.01, dead_zone=10):
        """
        Mouse controller with smoothing, safety, and throttling.

        alpha         -> smoothing strength (0.2–0.35 is good)
        move_interval -> max mouse update rate (seconds)
        dead_zone     -> ignore tiny movements (pixels)
        """

        # Screen size
        self.screen_w, self.screen_h = pyautogui.size()

        # Current mouse position
        self.current_x, self.current_y = pyautogui.position()

        # Smoothing + stability settings
        self.alpha = alpha
        self.dead_zone = dead_zone

        # Timing (prevents FPS drops)
        self.move_interval = move_interval
        self.last_move_time = 0

        # Scroll tracking state
        self.prev_y1 = 0

        # Pinch / click-and-drag state
        self.is_pinching = False
        self.pinch_start_time = 0
        self.click_fired = False
        self.is_dragging = False
        self.last_click_time = 0
        self.drag_threshold = 0.3  # seconds: pinch longer than this becomes a drag
        self.double_click_threshold = 0.35 # seconds: max time between pinches for double click
        self.pending_tap_time = 0

    def process_frame(self, frame, hand, detector):
        """Process one frame in mouse-control mode."""
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]

        # Draw visuals
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)

        if not positions:
            return

        # Extract landmarks
        x1, y1 = positions[8][1], positions[8][2]    # Index Finger Tip
        x2, y2 = positions[4][1], positions[4][2]    # Thumb Tip
        x3, y3 = positions[12][1], positions[12][2]  # Middle Finger Tip
        x4, y4 = positions[16][1], positions[16][2]  # Ring Finger Tip

        anchor_x, anchor_y = positions[5][1], positions[5][2]  # Index Finger Knuckle

        # Check scroll gesture (Thumb + Ring) first
        dist_scroll = math.hypot(x2 - x4, y2 - y4)

        if dist_scroll < PINCH_DISTANCE:
            # SCROLL MODE
            cv2.circle(frame, (x4, y4), FINGER_CIRCLE_RADIUS, COLOR_ACCENT, cv2.FILLED)

            if self.prev_y1 == 0:
                self.prev_y1 = y1
            delta_y = y1 - self.prev_y1

            if abs(delta_y) > SCROLL_JITTER_THRESHOLD:
                scroll_amount = int(-delta_y * SCROLL_SPEED_MULTIPLIER)
                self.scroll(scroll_amount)

        else:
            # NORMAL MOUSE MODE (Move + Click)
            cv2.circle(frame, (anchor_x, anchor_y), FINGER_CIRCLE_RADIUS, COLOR_SECONDARY, cv2.FILLED)
            cv2.circle(frame, (x2, y2), FINGER_CIRCLE_RADIUS, COLOR_SECONDARY, cv2.FILLED)

            # Determine pinch state
            distance = math.hypot(x2 - x1, y2 - y1)
            pinching_now = distance < PINCH_DISTANCE

            # Move cursor — freeze during click pinch, allow during drag
            if not pinching_now or self.is_dragging:
                x_screen = np.interp(anchor_x, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, self.screen_w))
                y_screen = np.interp(anchor_y, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, self.screen_h))
                self.move(x_screen, y_screen)

            now = time.time()

            # Fire any pending single tap if the double-click window has expired
            if self.pending_tap_time > 0 and (now - self.pending_tap_time) > self.double_click_threshold:
                pyautogui.click(button='left')
                self.pending_tap_time = 0
                self.last_click_time = now

            if pinching_now and not self.is_pinching:
                # Pinch just started
                self.is_pinching = True
                self.pinch_start_time = now
                self.click_fired = False
                
                # If we had a pending tap from a very recent release, this is a DOUBLE CLICK!
                if self.pending_tap_time > 0:
                    pyautogui.doubleClick(button='left')
                    self.pending_tap_time = 0  # Consume it
                    self.click_fired = True    # Prevent a single click on release
                    self.last_click_time = now

                cv2.circle(frame, (anchor_x, anchor_y), FINGER_CIRCLE_RADIUS, COLOR_PRIMARY, cv2.FILLED)

            elif pinching_now and self.is_pinching:
                # Sustained pinch — check if it should become a drag
                cv2.circle(frame, (anchor_x, anchor_y), FINGER_CIRCLE_RADIUS, COLOR_PRIMARY, cv2.FILLED)
                if not self.click_fired and (now - self.pinch_start_time) > self.drag_threshold:
                    pyautogui.mouseDown(button='left')
                    self.is_dragging = True
                    self.click_fired = True

            elif not pinching_now and self.is_pinching:
                # Pinch just released
                if self.is_dragging:
                    # End drag
                    pyautogui.mouseUp(button='left')
                    self.is_dragging = False
                elif not self.click_fired and (now - self.last_click_time) > CLICK_COOLDOWN:
                    # Short pinch released — queue it to see if it becomes a double-click!
                    self.pending_tap_time = now

                self.is_pinching = False
                self.click_fired = False

            # Right click (Thumb + Middle)
            distance_right = math.hypot(x2 - x3, y2 - y3)
            if distance_right < PINCH_DISTANCE:
                cv2.circle(frame, (x3, y3), FINGER_CIRCLE_RADIUS, COLOR_DANGER, cv2.FILLED)
                self.click('right')
                time.sleep(CLICK_COOLDOWN)

        # Update previous y1 for next frame
        self.prev_y1 = y1

    def move(self, x, y):
        """Move mouse smoothly to (x, y)"""

        # Rate limit mouse updates
        now = time.time()
        if now - self.last_move_time < self.move_interval:
            return
        self.last_move_time = now

        # Clamp target to screen bounds
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))

        # Ignore tiny jitter
        if abs(x - self.current_x) < self.dead_zone and abs(y - self.current_y) < self.dead_zone:
            return

        # Exponential smoothing (stable + responsive)
        self.current_x = self.current_x * (1 - self.alpha) + x * self.alpha
        self.current_y = self.current_y * (1 - self.alpha) + y * self.alpha

        # Move real OS cursor
        pyautogui.moveTo(int(self.current_x), int(self.current_y))

    def click(self, button="left"):
        pyautogui.click(button=button)

    def scroll(self, dy):
        pyautogui.scroll(dy)
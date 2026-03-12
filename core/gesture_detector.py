import numpy as np
import mediapipe as mp
import time
import logging

logger = logging.getLogger(__name__)

class GestureDetector:
    def __init__(self):
        self.last_gesture_time = {}  # Cooldown tracking for each gesture
        # Different cooldowns for different gestures
        self.cooldown_seconds = 0.5  # Default cooldown for space, backspace, clear
        self.speak_cooldown_seconds = 0.2  # Shorter cooldown for speak gesture (200ms)
        self.landmark_threshold_spread = 0.16  # normalized distance for spread (open palm) - MUST be open
        self.landmark_threshold_curl = 0.18  # normalized distance for curl (fist) - tighter
        self.landmark_threshold_thumbs = 0.25  # normalized vertical distance for thumbs up
        
        # Frame consistency for smoother detection
        self.gesture_frame_buffer = []  # Track gestures detected across frames
        self.consistency_frames = 2  # Require gesture to be detected in 2+ consecutive frames
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
    
    def _detect_raw_gesture(self, hands_data):
        '''Raw gesture detection without frame smoothing.'''
        if len(hands_data) == 0:
            return None
        
        # Two hand gestures FIRST (check before single hand)
        if self.is_two_open_palms(hands_data):
            return 'clear'
        
        # Single hand gestures
        if len(hands_data) >= 1:
            hand = hands_data[0]['landmarks']

            if self.is_open_palm(hand):
                return 'space'
            if self.is_fist(hand):
                return 'backspace'
            if self.is_thumbs_up(hand):
                return 'speak'
        
        return None
    
    def is_open_palm(self, landmarks):
        '''Detect open palm gesture based on landmark positions.'''
        # fingertip indices in mediapipe
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky fingertips
        wrist = landmarks.landmark[0]

        # Calculate hand center 
        hand_center_y = np.mean([landmarks.landmark[i].y for i in fingertip_indices])
        hand_center_x = np.mean([landmarks.landmark[i].x for i in fingertip_indices])
        wrist_y = wrist.y

        # Check 1: fingers extended (need 4+ above hand center for true open palm)
        fingers_extended = sum(1 for idx in fingertip_indices if landmarks.landmark[idx].y < hand_center_y) >= 4

        # Check 2: fingers spread apart
        spread = max([landmarks.landmark[idx].x for idx in fingertip_indices]) - min([landmarks.landmark[idx].x for idx in fingertip_indices])
        spread_ok = spread > self.landmark_threshold_spread

        # Check 3: palm facing camera (z-depth consistent with open palm)
        z_values = [landmarks.landmark[idx].z for idx in fingertip_indices]
        z_consistent = max(z_values) - min(z_values) < 0.12

        return fingers_extended and spread_ok and z_consistent
    
    def is_fist(self, landmarks):
        '''Detect fist gesture based on landmark positions.'''
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky fingertips
        wrist = landmarks.landmark[0]

        # Calculate distances from fingertip to wrist
        distances = []
        for idx in fingertip_indices:
            fingertip = landmarks.landmark[idx]
            dist = self._distance(fingertip, wrist)
            distances.append(dist)
        
        # All fingers curled if distances are below threshold
        all_curled = all(d < self.landmark_threshold_curl for d in distances)
        return all_curled
    
    def is_thumbs_up(self, landmarks):
        '''Detect thumbs up gesture.'''
        thumb_tip = landmarks.landmark[4]
        wrist = landmarks.landmark[0]
        other_fingertip_indices = [8, 12, 16, 20]
        
        # Check 1: thumb extended upward (y coordinate decreases upward in MediaPipe)
        thumb_up = (wrist.y - thumb_tip.y) > self.landmark_threshold_thumbs

        # Check 2: other fingers curled (distances from wrist below threshold)
        other_curled = all(self._distance(landmarks.landmark[idx], wrist) < self.landmark_threshold_curl for idx in other_fingertip_indices) 
        return thumb_up and other_curled
    
    def is_two_open_palms(self, hands_data):
        '''Detect two open palms.'''
        if len(hands_data) != 2:
            return False
        
        return self.is_open_palm(hands_data[0]['landmarks']) and self.is_open_palm(hands_data[1]['landmarks'])
    
    def detect_gesture(self, hands_data):
        '''Detect which gesture is being made with frame-based smoothing.'''
        current_time = time.time()
        
        # Get raw gesture detection
        raw_gesture = self._detect_raw_gesture(hands_data)
        
        if raw_gesture:
            logger.debug(f"Raw gesture detected: {raw_gesture}")
        
        # Use different cooldown depending on gesture type
        cooldown = self.speak_cooldown_seconds if raw_gesture == 'speak' else self.cooldown_seconds
        
        # Check cooldown for THIS specific gesture only
        if raw_gesture and raw_gesture in self.last_gesture_time:
            if current_time - self.last_gesture_time[raw_gesture] < cooldown:
                logger.debug(f"Gesture {raw_gesture} in cooldown ({cooldown}s)")
                return None  # Still in cooldown for this gesture
        
        # Update frame buffer for consistency checking
        if raw_gesture:
            self.gesture_frame_buffer.append(raw_gesture)
        else:
            self.gesture_frame_buffer.append(None)
        
        # Keep only last N frames
        if len(self.gesture_frame_buffer) > self.consistency_frames:
            self.gesture_frame_buffer.pop(0)
        
        # Check if gesture is consistent across frames
        recent_gestures = [g for g in self.gesture_frame_buffer if g is not None]
        
        if recent_gestures:
            logger.debug(f"Recent gestures: {recent_gestures} (need {self.consistency_frames} matches)")
        
        # Require consistent gesture detection across frames
        if len(recent_gestures) >= self.consistency_frames:
            # All recent detections must be the same gesture
            if all(g == recent_gestures[0] for g in recent_gestures):
                gesture = recent_gestures[0]
                self.last_gesture_time[gesture] = current_time
                self.gesture_frame_buffer = []  # Reset buffer after triggering
                logger.info(f"Gesture triggered: {gesture}")
                return gesture
        
        return None
    

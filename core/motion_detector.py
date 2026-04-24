"""
Detects hand motion by computing landmark velocity between frames.

Used to automatically switch between the static (RandomForest) classifier
for still signs and the dynamic (LSTM) classifier for motion-based signs.
"""
import numpy as np
import logging
from collections import deque
from config import SEQUENCE_LENGTH, EXPECTED_FEATURES

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, velocity_threshold=0.012, motion_frames_required=5,
                 cooldown_frames=15):
        self.velocity_threshold = velocity_threshold
        self.motion_frames_required = motion_frames_required
        self.cooldown_frames = cooldown_frames

        self.prev_features = None
        self.velocity_history = deque(maxlen=10)
        self.sequence_buffer = []
        self.state = 'static'
        self.cooldown_counter = 0
        self.last_dynamic_label = None
        self.last_dynamic_confidence = None

    def compute_velocity(self, features):
        if self.prev_features is None:
            self.prev_features = features.copy()
            return 0.0

        diff = features - self.prev_features
        velocity = float(np.sqrt(np.mean(diff ** 2)))
        self.prev_features = features.copy()
        return velocity

    def update(self, features):
        """Feed a new frame of features and return the current state.

        Returns one of:
            'static'        — hand is still, use RF classifier
            'buffering'     — motion detected, collecting frames for LSTM
            'dynamic_ready' — 30-frame buffer is full, ready for LSTM prediction
        """
        if features is None:
            self._reset_buffer()
            return self.state

        velocity = self.compute_velocity(features)
        self.velocity_history.append(velocity)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return 'static'

        if self.state == 'static':
            recent_motion = sum(
                1 for v in self.velocity_history if v > self.velocity_threshold
            )
            if recent_motion >= self.motion_frames_required:
                self.state = 'buffering'
                self.sequence_buffer = list(self.velocity_history)[-self.motion_frames_required:]
                self.sequence_buffer = []
                logger.info(f"Motion detected (velocity={velocity:.4f}), starting sequence buffer")

        if self.state == 'buffering':
            self.sequence_buffer.append(features.copy())

            if len(self.sequence_buffer) >= SEQUENCE_LENGTH:
                self.state = 'dynamic_ready'
                logger.info("Sequence buffer full — ready for LSTM prediction")

        return self.state

    def get_sequence(self):
        """Return the buffered sequence as a numpy array for LSTM inference."""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None
        seq = np.array(self.sequence_buffer[:SEQUENCE_LENGTH], dtype=np.float32)
        return seq

    def finish_dynamic(self, label=None, confidence=None):
        """Called after LSTM prediction to reset back to static mode."""
        self.last_dynamic_label = label
        self.last_dynamic_confidence = confidence
        self._reset_buffer()
        self.cooldown_counter = self.cooldown_frames

    def _reset_buffer(self):
        self.sequence_buffer = []
        self.state = 'static'

    def get_buffer_progress(self):
        """Return buffering progress as 0.0–1.0 for UI."""
        if self.state != 'buffering':
            return 0.0
        return len(self.sequence_buffer) / SEQUENCE_LENGTH

    def get_velocity(self):
        if not self.velocity_history:
            return 0.0
        return self.velocity_history[-1]

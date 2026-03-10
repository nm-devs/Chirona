"""
FeatureExtractor: Landmark extraction and normalization.

This module provides a reusable class to convert MediaPipe hand landmarks
into numerical feature arrays suitable for machine learning. Features
can include x, y coordinates (42 features) or x, y, z (63 features).

The module ensures:
- Consistent extraction of features
- Position-invariant normalization
- Safe handling of edge cases (NaN, Inf, or degenerate hand sizes)
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class FeatureExtractor:
    """
    Extracts and normalizes hand landmark features from MediaPipe output.

    Modes:
        use_z=False → 42 features (21 landmarks × x, y)
        use_z=True  → 63 features (21 landmarks × x, y, z)

    Provides:
        - extract(): converts landmarks to raw feature vector
        - normalize(): makes features position-invariant
        - get_feature_count(): helper to know expected array size
        - get_info(): returns configuration info
    """

    def __init__(self, use_z=False):
        """
        Initialize FeatureExtractor configuration.

        Args:
            use_z (bool): If True, include z-coordinates (depth)
        """
        self.use_z = use_z
        self.num_features = 63 if use_z else 42
        logging.info(f"FeatureExtractor initialized (use_z={self.use_z})")

    def extract(self, hand_landmarks):
        """
        Convert MediaPipe hand landmarks to a flat numpy array.

        Compatible with HandDetector output:
            hand_data = detector.detect(image)
            features = fe.extract(hand_data[0]['landmarks'])

        Args:
            hand_landmarks: MediaPipe Landmarks object with 21 points.
                Each point must have .x, .y, (.z optional)

        Returns:
            numpy.ndarray: Shape (42,) or (63,), dtype float32
                           Invalid landmarks are zero-filled to preserve shape.
        """
        assert len(hand_landmarks.landmark) == 21, \
            f"Expected 21 landmarks, got {len(hand_landmarks.landmark)}"

        features = []
        for idx, lm in enumerate(hand_landmarks.landmark):
            # FIX: Fill with zeros instead of skipping to preserve array shape
            if np.isnan(lm.x) or np.isnan(lm.y) or np.isinf(lm.x) or np.isinf(lm.y):
                logging.warning(f"Invalid landmark at index {idx}: {lm} — filling with zeros")
                features.append(0.0)
                features.append(0.0)
                if self.use_z:
                    features.append(0.0)
            else:
                features.append(lm.x)
                features.append(lm.y)
                if self.use_z:
                    z_val = lm.z if hasattr(lm, 'z') else 0.0
                    features.append(z_val)

        return np.array(features, dtype=np.float32)

    def normalize(self, features):
        """
        Make hand features position-invariant by subtracting min x, min y.

        Keeps shape consistent; z-coordinates remain unchanged.

        Args:
            features: numpy array from extract()

        Returns:
            numpy.ndarray: Normalized features (same shape)
        """
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        coords_per_lm = 3 if self.use_z else 2

        reshaped = features.reshape(-1, coords_per_lm)

        xs = reshaped[:, 0]
        ys = reshaped[:, 1]

        min_x = np.min(xs)
        min_y = np.min(ys)
        epsilon = 1e-6

        reshaped[:, 0] -= min_x - epsilon
        reshaped[:, 1] -= min_y - epsilon

        return reshaped.flatten().astype(np.float32)

    def get_feature_count(self):
        """Return expected number of features (42 or 63)."""
        return self.num_features

    def get_info(self):
        """Return configuration info for debugging/documentation."""
        return {
            'use_z': self.use_z,
            'feature_count': self.num_features,
            'landmarks_count': 21
        }
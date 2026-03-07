"""
FeatureExtractor: Landmark extraction and normalization.

This module provides a reusable class to convert MediaPipe hand landmarks
into numerical feature arrays suitable for machine learning. Features
can include x, y coordinates (42 features) or x, y, z (63 features).

The module ensures:
- Consistent extraction of features
- Position-invariant normalization
- Safe handling of edge cases (NaN, Inf, or degenerate hand sizes)

Example Usage:

    fe = FeatureExtractor(use_z=False)
    features = fe.extract(hand_landmarks)
    normalized = fe.normalize(features)

Compatible with HandDetector output:
    hands_data = detector.detect(image)
    features = fe.extract(hands_data[0]['landmarks'])
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


# -----------------------------
# Mock Data & Unit Tests Below
# -----------------------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockLandmark:
        x: float
        y: float
        z: float = 0.0

    class MockLandmarks:
        def __init__(self, points):
            self.landmark = [MockLandmark(*p) for p in points]

    def create_mock_landmarks(num_points=21, position=(0, 0)):
        """Generate random mock hand landmarks for testing."""
        points = []
        for i in range(num_points):
            x = np.random.rand() * 0.3 + position[0]
            y = np.random.rand() * 0.3 + position[1]
            z = np.random.rand() * 0.1
            points.append((x, y, z))
        return MockLandmarks(points)

    def create_hand_with_fixed_shape(position=(0, 0)):
        """
        Create a deterministic hand shape at a given position offset.
        Used to verify position-invariance in normalization.
        """
        base_shape = [
            (0.50, 0.30), (0.50, 0.35), (0.50, 0.40),
            (0.52, 0.25), (0.52, 0.30), (0.52, 0.35), (0.52, 0.40),
            (0.54, 0.24), (0.54, 0.29), (0.54, 0.34), (0.54, 0.39),
            (0.56, 0.25), (0.56, 0.30), (0.56, 0.35), (0.56, 0.40),
            (0.57, 0.27), (0.57, 0.32), (0.57, 0.37), (0.57, 0.42),
            (0.48, 0.35), (0.46, 0.38)
        ]
        offset_shape = [(x + position[0], y + position[1], 0.0) for x, y in base_shape]
        return MockLandmarks(offset_shape)

    # -----------------------------
    # Tests
    # -----------------------------
    def test_extract_without_z():
        fe = FeatureExtractor(use_z=False)
        hand = create_mock_landmarks()
        features = fe.extract(hand)
        assert features.shape == (42,), f"Shape mismatch: {features.shape}"
        assert features.dtype == np.float32
        print("✓ test_extract_without_z passed.")

    def test_extract_with_z():
        fe = FeatureExtractor(use_z=True)
        hand = create_mock_landmarks()
        features = fe.extract(hand)
        assert features.shape == (63,), f"Shape mismatch: {features.shape}"
        print("✓ test_extract_with_z passed.")

    def test_nan_fill_preserves_shape():
        """Verify NaN landmarks are zero-filled, not skipped."""
        fe = FeatureExtractor(use_z=False)
        points = [(float('nan'), float('nan'), 0.0)] + \
                 [(0.5 + i * 0.01, 0.3 + i * 0.01, 0.0) for i in range(20)]
        hand = MockLandmarks(points)
        features = fe.extract(hand)
        assert features.shape == (42,), f"NaN caused shape mismatch: {features.shape}"
        assert features[0] == 0.0 and features[1] == 0.0, "NaN not zero-filled correctly"
        print("✓ test_nan_fill_preserves_shape passed.")

    def test_normalize_position_invariance():
        """Verify same hand shape at different positions gives identical normalized features."""
        fe = FeatureExtractor(use_z=False)

        hand1 = create_hand_with_fixed_shape(position=(0.1, 0.2))
        hand2 = create_hand_with_fixed_shape(position=(0.7, 0.8))

        norm1 = fe.normalize(fe.extract(hand1))
        norm2 = fe.normalize(fe.extract(hand2))

        assert norm1.shape == norm2.shape, "Shape mismatch"
        assert np.allclose(norm1, norm2, atol=1e-6), \
            f"Position-invariance failed: max diff = {np.max(np.abs(norm1 - norm2))}"

        print("✓ test_normalize_position_invariance passed.")
        print(f"  Max difference between positions: {np.max(np.abs(norm1 - norm2)):.2e}")

    def test_normalize_range():
        fe = FeatureExtractor(use_z=False)
        features = np.array([0.5, 0.3, 0.6, 0.4, 0.7, 0.2], dtype=np.float32)
        normalized = fe.normalize(features)
        xs = normalized[0::2]
        ys = normalized[1::2]
        assert np.abs(np.min(xs)) < 1e-3
        assert np.abs(np.min(ys)) < 1e-3
        print("✓ test_normalize_range passed.")

    # Run all tests
    test_extract_without_z()
    test_extract_with_z()
    test_nan_fill_preserves_shape()
    test_normalize_position_invariance()
    test_normalize_range()
    print("\n✓ All tests passed. FeatureExtractor ready to use.")
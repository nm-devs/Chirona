import numpy as np

from core.feature_extractor import FeatureExtractor

# -----------------------------
# Mock Data
# -----------------------------
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

def test_extract_with_z():
    fe = FeatureExtractor(use_z=True)
    hand = create_mock_landmarks()
    features = fe.extract(hand)
    assert features.shape == (63,), f"Shape mismatch: {features.shape}"

def test_nan_fill_preserves_shape():
    """Verify NaN landmarks are zero-filled, not skipped."""
    fe = FeatureExtractor(use_z=False)
    points = [(float('nan'), float('nan'), 0.0)] + \
             [(0.5 + i * 0.01, 0.3 + i * 0.01, 0.0) for i in range(20)]
    hand = MockLandmarks(points)
    features = fe.extract(hand)
    assert features.shape == (42,), f"NaN caused shape mismatch: {features.shape}"
    assert features[0] == 0.0 and features[1] == 0.0, "NaN not zero-filled correctly"

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

def test_normalize_range():
    fe = FeatureExtractor(use_z=False)
    features = np.array([0.5, 0.3, 0.6, 0.4, 0.7, 0.2], dtype=np.float32)
    normalized = fe.normalize(features)
    xs = normalized[0::2]
    ys = normalized[1::2]
    assert np.abs(np.min(xs)) < 1e-3
    assert np.abs(np.min(ys)) < 1e-3

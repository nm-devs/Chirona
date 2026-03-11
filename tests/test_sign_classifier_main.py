"""
Quick test of SignClassifier without needing webcam/hand detector.
Tests the exact flow used in main.py.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from core.sign_classifier import SignClassifier
from core.feature_extractor import FeatureExtractor

# Mock MediaPipe landmark objects (same as in feature_extractor.py)
@dataclass
class MockLandmark:
    x: float
    y: float
    z: float = 0.0

class MockLandmarks:
    def __init__(self, points):
        self.landmark = [MockLandmark(*p) for p in points]

def create_mock_hand_landmarks(position=(0.0, 0.0)):
    """Create fake MediaPipe hand landmarks for testing."""
    base_shape = [
        (0.50, 0.30), (0.50, 0.35), (0.50, 0.40),
        (0.52, 0.25), (0.52, 0.30), (0.52, 0.35), (0.52, 0.40),
        (0.54, 0.24), (0.54, 0.29), (0.54, 0.34), (0.54, 0.39),
        (0.56, 0.25), (0.56, 0.30), (0.56, 0.35), (0.56, 0.40),
        (0.57, 0.27), (0.57, 0.32), (0.57, 0.37), (0.57, 0.42),
        (0.48, 0.35), (0.46, 0.38)
    ]
    # Add position offset and z-coordinate
    offset_shape = [(x + position[0], y + position[1], 0.0) for x, y in base_shape]
    return MockLandmarks(offset_shape)

print("=" * 80)
print("TESTING SIGNCLASSIFIER INTEGRATION (as used in main.py)")
print("=" * 80)

# STEP 1: Create the same components as main.py
print("\n[1] Initializing components...")
try:
    fe = FeatureExtractor(use_z=False)
    classifier = SignClassifier('models/trained_model.pkl')
    print("  ✓ Components initialized successfully")
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    print("  First run: python models/mockmodel.py")
    exit(1)

# STEP 2: Create fake landmarks (what HandDetector would return)
print("\n[2] Creating dummy hand landmarks...")
landmarks = create_mock_hand_landmarks(position=(0.1, 0.1))
print(f"  Created MockLandmarks with {len(landmarks.landmark)} points")
print(f"  Sample landmark[0]: x={landmarks.landmark[0].x}, y={landmarks.landmark[0].y}")

# STEP 3: Extract features (what main.py does at line ~45)
print("\n[3] Extracting features...")
features = fe.extract(landmarks)
print(f"  Extracted shape: {features.shape}")
print(f"  First 5 values: {features[:5]}")

# STEP 4: Normalize features (what main.py does at line ~46)
print("\n[4] Normalizing features...")
normalized_features = fe.normalize(features)
print(f"  Normalized shape: {normalized_features.shape}")
print(f"  Value range: [{np.min(normalized_features):.3f}, {np.max(normalized_features):.3f}]")

# STEP 5: Predict gesture (what main.py does at line ~49)
print("\n[5] Predicting gesture...")
label, confidence = classifier.predict(normalized_features)
print(f"  Predicted label: {label}")
print(f"  Confidence: {confidence:.4f} ({confidence:.1%})")

# STEP 6: Display as it would appear in main.py
print("\n[6] Output as it appears in main.py:")
print(f"  cv2.putText(frame, \"{label} ({confidence:.1%})\", ...)")

print("\n" + "=" * 80)
print("✓ SUCCESS! SignClassifier works correctly in main.py flow")
print("=" * 80)
print("\nNext steps:")
print("  1. Create test model: python models/mockmodel.py")
print("  2. Run main with webcam: python main.py")
print("  3. Point hand at camera to see predictions")
print("  4. Press 'm' to switch modes")
print("  5. Press ESC to exit")
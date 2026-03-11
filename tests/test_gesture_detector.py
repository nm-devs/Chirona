"""
Unit tests for GestureDetector class.
Tests individual gesture detection functions with realistic test data.
"""

import pytest
import time
from core.gesture_detector import GestureDetector


# ============= Mock Landmark Classes =============

class MockLandmark:
    """Simulates a MediaPipe landmark point."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockHandLandmarks:
    """Simulates MediaPipe's hand landmarks object."""
    def __init__(self, points):
        """
        Args:
            points: List of (x, y, z) tuples for 21 landmarks
        """
        self.landmark = [MockLandmark(x, y, z) for x, y, z in points]


# ============= Test Hand Factories =============

def create_open_palm():
    """Create mock hand in open palm position - all 5 fingers extended upward."""
    return MockHandLandmarks([
        (0.5, 0.9, 0.0),    # 0: Wrist (at bottom)
        # Thumb - extended to left and up (but shorter than others)
        (0.35, 0.6, 0.0),   # 1: Thumb base
        (0.3, 0.4, 0.0),    # 2: Thumb middle
        (0.27, 0.2, 0.0),   # 3: Thumb pip
        (0.25, 0.25, 0.0),  # 4: Thumb tip (shorter reach, brings up average)
        # Index - center, extended up
        (0.5, 0.6, 0.0),    # 5: Index base
        (0.5, 0.4, 0.0),    # 6: Index middle
        (0.5, 0.2, 0.0),    # 7: Index pip
        (0.5, 0.0, 0.0),    # 8: Index tip (extended up)
        # Middle - center, extended up slightly higher
        (0.55, 0.6, 0.0),   # 9: Middle base
        (0.55, 0.4, 0.0),   # 10: Middle middle
        (0.55, 0.2, 0.0),   # 11: Middle pip
        (0.55, 0.0, 0.0),   # 12: Middle tip (extended up, tallest)
        # Ring - right, extended up
        (0.65, 0.6, 0.0),   # 13: Ring base
        (0.65, 0.4, 0.0),   # 14: Ring middle
        (0.65, 0.2, 0.0),   # 15: Ring pip
        (0.65, 0.0, 0.0),   # 16: Ring tip (extended right-up)
        # Pinky - far right, extended up
        (0.75, 0.6, 0.0),   # 17: Pinky base
        (0.75, 0.4, 0.0),   # 18: Pinky middle
        (0.75, 0.2, 0.0),   # 19: Pinky pip
        (0.75, 0.0, 0.0),   # 20: Pinky tip (extended far right-up)
    ])

def create_fist():
    """Create mock hand in fist position - all fingers curled close to wrist."""
    return MockHandLandmarks([
        (0.5, 0.85, 0.0),   # 0: Wrist
        # Thumb - curled tight
        (0.48, 0.8, 0.0),   # 1: Thumb base
        (0.48, 0.82, 0.0),  # 2: Thumb middle
        (0.48, 0.84, 0.0),  # 3: Thumb pip
        (0.48, 0.855, 0.0), # 4: Thumb tip (curled very close to wrist ~0.005 distance)
        # Index - curled tight
        (0.5, 0.8, 0.0),    # 5: Index base
        (0.5, 0.82, 0.0),   # 6: Index middle
        (0.5, 0.84, 0.0),   # 7: Index pip
        (0.5, 0.855, 0.0),  # 8: Index tip (curled close to wrist)
        # Middle - curled tight
        (0.52, 0.8, 0.0),   # 9: Middle base
        (0.52, 0.82, 0.0),  # 10: Middle middle
        (0.52, 0.84, 0.0),  # 11: Middle pip
        (0.52, 0.855, 0.0), # 12: Middle tip (curled close)
        # Ring - curled tight
        (0.54, 0.8, 0.0),   # 13: Ring base
        (0.54, 0.82, 0.0),  # 14: Ring middle
        (0.54, 0.84, 0.0),  # 15: Ring pip
        (0.54, 0.855, 0.0), # 16: Ring tip (curled close)
        # Pinky - curled tight
        (0.56, 0.8, 0.0),   # 17: Pinky base
        (0.56, 0.82, 0.0),  # 18: Pinky middle
        (0.56, 0.84, 0.0),  # 19: Pinky pip
        (0.56, 0.855, 0.0), # 20: Pinky tip (curled close)
    ])


def create_thumbs_up():
    """Create mock hand in thumbs up position - thumb extended up, rest curled."""
    return MockHandLandmarks([
        (0.5, 0.85, 0.0),   # 0: Wrist (at bottom)
        # Thumb - extended WAY up
        (0.48, 0.6, 0.0),   # 1: Thumb base
        (0.48, 0.4, 0.0),   # 2: Thumb middle
        (0.48, 0.2, 0.0),   # 3: Thumb pip
        (0.48, 0.0, 0.0),   # 4: Thumb tip (extended way up, 0.85 distance from wrist!)
        # Index - curled tight
        (0.5, 0.8, 0.0),    # 5: Index base
        (0.5, 0.82, 0.0),   # 6: Index middle
        (0.5, 0.84, 0.0),   # 7: Index pip
        (0.5, 0.855, 0.0),  # 8: Index tip (curled close)
        # Middle - curled tight
        (0.52, 0.8, 0.0),   # 9: Middle base
        (0.52, 0.82, 0.0),  # 10: Middle middle
        (0.52, 0.84, 0.0),  # 11: Middle pip
        (0.52, 0.855, 0.0), # 12: Middle tip (curled close)
        # Ring - curled tight
        (0.54, 0.8, 0.0),   # 13: Ring base
        (0.54, 0.82, 0.0),  # 14: Ring middle
        (0.54, 0.84, 0.0),  # 15: Ring pip
        (0.54, 0.855, 0.0), # 16: Ring tip (curled close)
        # Pinky - curled tight
        (0.56, 0.8, 0.0),   # 17: Pinky base
        (0.56, 0.82, 0.0),  # 18: Pinky middle
        (0.56, 0.84, 0.0),  # 19: Pinky pip
        (0.56, 0.855, 0.0), # 20: Pinky tip (curled close)
    ])


# ============= Test Functions =============

def test_is_open_palm():
    """Test open palm detection."""
    detector = GestureDetector()
    
    # TEST 1: Open palm should be detected
    open_palm = create_open_palm()
    result = detector.is_open_palm(open_palm)
    assert result is True, f"Open palm should be detected, got {result}"
    
    # TEST 2: Fist should NOT be detected as open palm
    fist = create_fist()
    result = detector.is_open_palm(fist)
    assert result is False, f"Fist should NOT be open palm, got {result}"


def test_is_fist():
    """Test fist detection."""
    detector = GestureDetector()
    
    # TEST 1: Fist should be detected
    fist = create_fist()
    result = detector.is_fist(fist)
    assert result is True, f"Fist should be detected, got {result}"
    
    # TEST 2: Open palm should NOT be detected as fist
    open_palm = create_open_palm()
    result = detector.is_fist(open_palm)
    assert result is False, f"Open palm should NOT be fist, got {result}"


def test_is_thumbs_up():
    """Test thumbs up detection."""
    detector = GestureDetector()
    
    # TEST 1: Thumbs up should be detected
    thumbs_up = create_thumbs_up()
    result = detector.is_thumbs_up(thumbs_up)
    assert result is True, f"Thumbs up should be detected, got {result}"
    
    # TEST 2: Open palm should NOT be thumbs up
    open_palm = create_open_palm()
    result = detector.is_thumbs_up(open_palm)
    assert result is False, f"Open palm should NOT be thumbs up, got {result}"
    
    # TEST 3: Fist should NOT be thumbs up
    fist = create_fist()
    result = detector.is_thumbs_up(fist)
    assert result is False, f"Fist should NOT be thumbs up, got {result}"


def test_is_two_open_palms():
    """Test two open palms detection."""
    detector = GestureDetector()
    
    # TEST 1: Two open palms should be detected
    hands_both_open = [
        {'landmarks': create_open_palm()},
        {'landmarks': create_open_palm()},
    ]
    result = detector.is_two_open_palms(hands_both_open)
    assert result is True, f"Two open palms should be detected, got {result}"
    
    # TEST 2: One hand should NOT be detected
    hands_one = [{'landmarks': create_open_palm()}]
    result = detector.is_two_open_palms(hands_one)
    assert result is False, f"Single hand should NOT be detected, got {result}"
    
    # TEST 3: Two hands, one open one fist should NOT be detected
    hands_mixed = [
        {'landmarks': create_open_palm()},
        {'landmarks': create_fist()},
    ]
    result = detector.is_two_open_palms(hands_mixed)
    assert result is False, f"Mixed hands should NOT be detected, got {result}"
    
    # TEST 4: Zero hands should NOT be detected
    hands_none = []
    result = detector.is_two_open_palms(hands_none)
    assert result is False, f"No hands should NOT be detected, got {result}"


def test_cooldown():
    """Test cooldown mechanism prevents double-triggers."""
    detector = GestureDetector()
    hands = [{'landmarks': create_open_palm()}]
    
    # TEST 1: First gesture should be detected
    # Requires 2 consecutive frames due to consistency filtering
    detector.detect_gesture(hands)
    gesture1 = detector.detect_gesture(hands)
    assert gesture1 == 'space', f"First gesture should be 'space', got {gesture1}"
    
    # TEST 2: Second gesture immediately after should be blocked (cooldown)
    gesture2 = detector.detect_gesture(hands)
    assert gesture2 is None, f"Second gesture should be None (cooldown), got {gesture2}"
    
    # TEST 3: Wait for cooldown to expire
    time.sleep(0.6)
    # Requires 2 frames again because the buffer resets after successful detection
    detector.detect_gesture(hands)
    gesture3 = detector.detect_gesture(hands)
    assert gesture3 == 'space', f"After cooldown, gesture should work, got {gesture3}"


if __name__ == '__main__':
    # Run with: pytest tests/test_gesture_detector.py -v
    # Or run directly: python tests/test_gesture_detector.py
    pytest.main([__file__, '-v', '-s'])

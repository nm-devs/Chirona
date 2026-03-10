import pytest

from utils.prediction_smoother import PredictionSmoother


def test_empty_history():
    """get_stable_prediction should return None when no predictions have been added."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    assert smoother.get_stable_prediction() is None


def test_clear_dominance():
    """Should return the dominant prediction when one sign fills the entire window."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    for _ in range(5):
        smoother.add_prediction("A")
    assert smoother.get_stable_prediction() == "A"


def test_no_dominance():
    """Should return None when no single prediction exceeds the threshold."""
    smoother = PredictionSmoother(window_size=6, dominance_threshold=0.6)
    for sign in ["A", "B", "C", "A", "B", "C"]:
        smoother.add_prediction(sign)
    assert smoother.get_stable_prediction() is None


def test_threshold_boundary_met():
    """Should return the dominant sign when it hits exactly 60%."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    for sign in ["A", "A", "A", "B", "B"]:  # A = 60%
        smoother.add_prediction(sign)
    assert smoother.get_stable_prediction() == "A"


def test_threshold_boundary_not_met():
    """Should return None when the top sign is just below the threshold."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    for sign in ["A", "A", "B", "B", "C"]:  # A = 40%
        smoother.add_prediction(sign)
    assert smoother.get_stable_prediction() is None


def test_sliding_window():
    """Old predictions should slide out of the window as new ones are added."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    for _ in range(5):
        smoother.add_prediction("A")
    assert smoother.get_stable_prediction() == "A"

    # Push all A's out by adding 5 B's
    for _ in range(5):
        smoother.add_prediction("B")
    assert smoother.get_stable_prediction() == "B"


def test_clear():
    """Clearing the history should reset to None."""
    smoother = PredictionSmoother(window_size=5, dominance_threshold=0.6)
    for _ in range(5):
        smoother.add_prediction("A")
    assert smoother.get_stable_prediction() == "A"

    smoother.clear()
    assert smoother.get_stable_prediction() is None


def test_custom_window_size():
    """Smoother should respect a custom window size."""
    smoother = PredictionSmoother(window_size=3, dominance_threshold=0.6)
    for sign in ["X", "X", "Y"]:  # X = 66%
        smoother.add_prediction(sign)
    assert smoother.get_stable_prediction() == "X"
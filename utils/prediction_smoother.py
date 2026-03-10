"""
Temporal smoothing for machine learning predictions.

Applies a moving window over recent predictions to filter out single-frame
flickers and report a stable, dominant gesture prediction.
"""
from collections import Counter, deque
from config import SMOOTHING_WINDOW_SIZE, SMOOTHING_DOMINANCE_THRESHOLD

class PredictionSmoother:
    def __init__(self, window_size = SMOOTHING_WINDOW_SIZE, dominance_threshold = SMOOTHING_DOMINANCE_THRESHOLD):
        self.window_size = window_size
        self.dominance_threshold = dominance_threshold
        self.prediction_history = deque(maxlen=window_size)

    def add_prediction(self, new_prediction):
        self.prediction_history.append(new_prediction)
    
    def get_stable_prediction(self):
        if not self.prediction_history:
            return None
        prediction, count = Counter(self.prediction_history).most_common(1)[0]
        ratio = count / len(self.prediction_history)
        if ratio >= self.dominance_threshold:
            return prediction
        return None

    def clear(self):
        self.prediction_history.clear()
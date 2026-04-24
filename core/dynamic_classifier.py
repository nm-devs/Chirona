"""
DynamicClassifier loads a trained LSTM model to classify dynamic hand gestures.

Wraps a Keras LSTM model and provides an inference API matching the style
of SignClassifier, but for temporal sequences instead of single frames.
"""
import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DynamicClassifier:
    def __init__(self, model_path='models/saved/model_lstm.h5',
                 label_map_path='models/saved/lstm_labels.pickle'):
        self.model = None
        self.label_map = {}
        self.is_available = False

        self._load_model(model_path, label_map_path)

    def _load_model(self, model_path, label_map_path):
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            logger.info(f"LSTM model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model from {model_path}: {e}")
            return

        try:
            with open(label_map_path, 'rb') as f:
                self.label_map = pickle.load(f)
            logger.info(f"LSTM label map loaded: {self.label_map}")
        except Exception as e:
            logger.warning(f"Could not load LSTM label map from {label_map_path}: {e}")
            self.model = None
            return

        self.is_available = True

    def predict(self, sequence):
        """Predict a dynamic sign from a landmark sequence.

        Args:
            sequence: numpy array of shape (30, 42)

        Returns:
            tuple: (label, confidence) or (None, 0.0) if unavailable
        """
        if not self.is_available or self.model is None:
            return None, 0.0

        try:
            input_data = np.expand_dims(sequence, axis=0)
            proba = self.model.predict(input_data, verbose=0)[0]

            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            label = self.label_map.get(best_idx, f"class_{best_idx}")

            return label, confidence
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None, 0.0

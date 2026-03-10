"""
SignClassifier loads a trained scikit-learn model to predict hand gestures.

Provides inference methods to calculate the most likely gesture label and
confidence score based on extracted hand landmark features.
"""
import os
import pickle
import logging
import numpy as np

from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SignClassifier:
    """
    SignClassifier: Load and inference wrapper for hand gesture classification.

    Loads a trained sklearn classifier model (pickle format) and provides a
    clean API for predicting hand gesture labels from feature vectors.

    Example Usage:
        clf = SignClassifier("models/trained_model.pkl")
        features = feature_extractor.normalize(...)  # shape (42,)
        label, confidence = clf.predict(features)
        print(f"Detected: {label} ({confidence:.2%})")

    Model Requirements:
      - Must be pickle-serialized sklearn classifier
      - Must have predict_proba() method (returns confidence scores)
      - Must have classes_ attribute (gesture label names)
    """

    def __init__(self, model_path):
        """
        Initialize SignClassifier by loading a model from disk.

        Args:
            model_path (str): Path to the trained pickle model file.
                            Example: "models/trained_model.pkl"

        Raises:
            FileNotFoundError: If file doesn't exist at path
            ValueError: If file is not a valid model
        """
        # STEP 1: Store the path for reference
        self.model_path = model_path
        
        # STEP 2: Initialize attributes as None
        self.model = None           # Will hold the actual trained model object
        self.classes_ = None        # Will hold gesture label names
        
        # STEP 3: Load the model from disk
        self._load_model()

    def _load_model(self):
        """Load and validate the pickle model file."""
        
        # STEP A: Check if file exists before trying to open it
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}\n"
                f"Expected location: {os.path.abspath(self.model_path)}\n"
                f"Make sure you ran: python models/mockmodel.py"
            )
        
        # STEP B: Wrap loading in try/except for error handling
        try:
            # Open file in binary read mode ('rb')
            with open(self.model_path, 'rb') as f:
                # pickle.load() deserializes the binary data back into Python object
                self.model = pickle.load(f)
            
            # STEP C: Validate model has predict_proba() method
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError(
                    "Model is missing predict_proba() method!\n"
                    "This method gives us confidence scores.\n"
                    "Make sure model was trained with sklearn (RandomForest, SVM, etc.)"
                )
            
            # STEP D: Validate model has classes_ attribute
            if not hasattr(self.model, 'classes_'):
                raise ValueError(
                    "Model is missing classes_ attribute!\n"
                    "This maps indices to gesture names (e.g., 0→'A', 1→'B').\n"
                    "Ensure model was properly trained and saved."
                )
            
            # STEP E: Store classes_ for later use in predict()
            self.classes_ = self.model.classes_
            
            # STEP F: Log success message for debugging
            num_classes = len(self.classes_)
            logging.info(f"Model loaded successfully! Found {num_classes} gesture classes")
        
        # STEP G: Catch pickle-specific errors
        except pickle.UnpicklingError as e:
            raise ValueError(
                f"Failed to unpickle model file!\n"
                f"Error: {e}\n"
                f"The file might be corrupted or not a valid pickle.\n"
                f"Try retraining and resaving the model."
            )
        
        # STEP H: Catch any other unexpected errors
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading model: {e}"
            )

    def predict(self, features):
        """
        Predict gesture label from a feature array.

        Args:
            features (np.ndarray): Feature array of shape (42,)
                                  From FeatureExtractor.normalize()
                                  dtype: float32

        Returns:
            tuple: (label, confidence)
                - label (str): Gesture name, e.g., "A", "B", "peace"
                - confidence (float): Between 0.0 and 1.0
                Example: ("A", 0.87) means 87% confident it's "A"

        Raises:
            ValueError: If feature shape is wrong
        """
        
        # STEP 1: Convert to numpy array if it's not already
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        # STEP 2: Ensure data type is float32
        features = features.astype(np.float32)
        
        # STEP 3: Validate shape is exactly (42,)
        # 42 = 21 hand landmarks × 2 coordinates (x, y)
        if features.shape != (42,):
            raise ValueError(
                f"ERROR: Feature shape is wrong!\n"
                f"Expected: (42,)    [21 landmarks × 2 coords]\n"
                f"Got: {features.shape}\n"
                f"Make sure you used FeatureExtractor.normalize() output"
            )
        
        # STEP 4: Reshape from (42,) to (1, 42)
        # sklearn models expect 2D input: (n_samples, n_features)
        features_reshaped = features.reshape(1, -1)
        
        # STEP 5: Call predict_proba to get confidence scores
        # Returns probabilities for EVERY class
        # Shape: (1, n_classes)
        proba = self.model.predict_proba(features_reshaped)
        proba = proba[0]  # Get just the probabilities for our single sample
        
        # STEP 6: Find which class has highest probability
        best_idx = np.argmax(proba)
        
        # STEP 7: Convert index to actual gesture label
        label = str(self.classes_[best_idx])
        
        # STEP 8: Get the confidence value
        confidence = float(proba[best_idx])
        
        # STEP 9: Return both label and confidence
        return label, confidence

    def get_classes(self):
        """Return list of all possible gesture labels."""
        return list(self.classes_)
    
    def is_ready(self):
        """Check if model loaded successfully and is ready to predict."""
        return self.model is not None
    
    def predict_batch(self, features_array):
        """
        Predict for multiple feature vectors at once (faster than loop).

        Args:
            features_array: numpy array shape (n_samples, 42)

        Returns:
            list of (label, confidence) tuples
        """
        # Get probabilities for ALL samples at once
        # Shape: (n_samples, n_classes)
        proba = self.model.predict_proba(features_array)
        
        # Process each sample's probabilities
        results = []
        for p in proba:
            # For each sample, find the best prediction
            best_idx = np.argmax(p)
            label = str(self.classes_[best_idx])
            confidence = float(p[best_idx])
            results.append((label, confidence))
        
        return results


# Unit tests - only run when executing this file directly
if __name__ == "__main__":
    print("=" * 80)
    print("SIGNCLASSIFIER UNIT TESTS")
    print("=" * 80)
    
    # TEST 1: Missing model file
    def test_init_with_missing_model():
        """Test that FileNotFoundError is raised for missing files."""
        print("\nTEST 1: Loading from missing path...")
        try:
            clf = SignClassifier("nonexistent/model.pkl")
            print("  [FAIL] FAILED: Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            print("  [OK] PASSED: FileNotFoundError raised as expected")
    
    # TEST 2: Valid model loading
    def test_init_with_valid_model():
        """Test that valid model loads successfully."""
        print("\nTEST 2: Loading valid model...")
        try:
            clf = SignClassifier("models/trained_model.pkl")
            assert clf.model is not None, "Model should be loaded"
            assert clf.classes_ is not None, "Classes should be set"
            print("  [OK] PASSED: Model loaded successfully")
            print(f"  Found {len(clf.classes_)} gesture classes")
        except FileNotFoundError:
            print("  [WARN] SKIPPED: No trained model found yet")
            print("    Run: python models/mockmodel.py")
    
    # TEST 3: Valid prediction
    def test_predict_with_valid_features():
        """Test predict() with correct feature shape."""
        print("\nTEST 3: Predicting with valid features...")
        try:
            clf = SignClassifier("models/trained_model.pkl")
            
            # Create dummy features (shape 42,)
            features = np.random.rand(42).astype(np.float32)
            
            # Call predict
            label, confidence = clf.predict(features)
            
            # Verify return types
            assert isinstance(label, str), "Label should be string"
            assert isinstance(confidence, float), "Confidence should be float"
            assert 0.0 <= confidence <= 1.0, "Confidence should be 0.0-1.0"
            
            print("  [OK] PASSED: predict() returned valid results")
            print(f"  Predicted: {label} with {confidence:.1%} confidence")
        except FileNotFoundError:
            print("  [WARN] SKIPPED: No trained model found yet")
    
    # TEST 4: Invalid shape rejection
    def test_predict_with_invalid_shape():
        """Test that predict() rejects wrong feature shapes."""
        print("\nTEST 4: Rejecting invalid feature shape...")
        try:
            clf = SignClassifier("models/trained_model.pkl")
            
            # Create features with WRONG shape (50 instead of 42)
            bad_features = np.random.rand(50).astype(np.float32)
            
            try:
                clf.predict(bad_features)
                print("  [FAIL] FAILED: Should have raised ValueError")
            except ValueError as e:
                print("  [OK] PASSED: ValueError raised for wrong shape")
        except FileNotFoundError:
            print("  [WARN] SKIPPED: No trained model found yet")
    
    # TEST 5: Batch prediction
    def test_predict_batch():
        """Test batch prediction."""
        print("\nTEST 5: Batch prediction...")
        try:
            clf = SignClassifier("models/trained_model.pkl")
            
            # Create batch of 5 samples
            batch_features = np.random.rand(5, 42).astype(np.float32)
            results = clf.predict_batch(batch_features)
            
            assert len(results) == 5, "Should return 5 results"
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
            
            print("  [OK] PASSED: Batch prediction works")
            for i, (label, conf) in enumerate(results):
                print(f"    Sample {i}: {label} ({conf:.1%})")
        except FileNotFoundError:
            print("  [WARN] SKIPPED: No trained model found yet")
    
    # RUN ALL TESTS
    test_init_with_missing_model()
    test_init_with_valid_model()
    test_predict_with_valid_features()
    test_predict_with_invalid_shape()
    test_predict_batch()
    
    print("\n" + "=" * 80)
    print("To create a test model run: python models/mockmodel.py")
    print("Then run tests again: python core/sign_classifier.py")
    print("=" * 80)
"""
Generates dummy training data and trains a quick mock RandomForest model.

Useful for testing the inference pipeline and UI without needing a full dataset.
"""
import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs

# Create dummy training data
X, y = make_blobs(n_samples=100, n_features=42, centers=5, random_state=42)
    
# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
    
# Map integers to string labels
# This step is important! model.classes_ should contain strings
model.classes_ = np.array(['A', 'B', 'C', 'D', 'E'])
    
# Save model
os.makedirs("models", exist_ok=True)
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
print("[OK] Mock model saved to models/trained_model.pkl")
print(f"  Model has {len(model.classes_)} classes: {list(model.classes_)}")
print("\nNow you can test with: python core/sign_classifier.py")
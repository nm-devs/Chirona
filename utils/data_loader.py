import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import BASE_DIR

def load_landmark_data(test_size=0.2, random_state=42):
    # Loads landmark data and returns train/test splits.
    data_path = os.path.join(BASE_DIR, "data", "landmarks.pickle")

    if not os.path.exists(data_path):
        print("Error: Run python data/extract_landmarks.py first.")
        exit(1)

    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    X = np.array(dataset["data"])
    y = np.array(dataset["labels"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
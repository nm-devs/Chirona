# training script
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.data_loader import load_landmark_data
from config import BASE_DIR

X_train, X_test, y_train, y_test = load_landmark_data()

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")

# save to models/saved/model_rf.p — loaded by sign_classifier at runtime
model_dir = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_rf.p")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
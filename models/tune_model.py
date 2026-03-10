"""
Hyperparameter tuning for the sign language Random Forest classifier.

Runs an exhaustive grid search using Cross-Validation to find the best tree parameters,
and saves the most optimal model.
"""
import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.data_loader import load_landmark_data
from config import BASE_DIR, RF_PARAM_GRID, CV_FOLDS

X_train, X_test, y_train, y_test, label_map = load_landmark_data()

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=RF_PARAM_GRID,
    cv=CV_FOLDS,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

# Print best parameters and accuracy
print(f"\n{'='*60}")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"{'='*60}\n")

# Print comparison table of all combinations
results = grid_search.cv_results_
print(f"{'Rank':<6} {'n_estimators':<15} {'max_depth':<12} {'Mean CV Acc':<15} {'Std':<10}")
print("-" * 58)

for i in range(len(results["params"])):
    rank = results["rank_test_score"][i]
    params = results["params"][i]
    mean_score = results["mean_test_score"][i]
    std_score = results["std_test_score"][i]
    print(f"{rank:<6} {params['n_estimators']:<15} {str(params['max_depth']):<12} {mean_score:<15.4f} {std_score:<10.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy (best model): {test_accuracy:.4f}")

model_dir = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_rf_tuned.p")

# Save label map
label_map_path = os.path.join(model_dir, "labels.pickle")
with open(label_map_path, "wb") as f:
    pickle.dump(label_map, f)

print(f"Label map saved to {label_map_path}")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Best tuned model saved to {model_path}")
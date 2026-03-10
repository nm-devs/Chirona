"""
Evaluates the performance of a trained model on the test set.

Parameters:
- model: The trained model to evaluate.
- X_test: Test features.
- y_test: True labels for the test set.
- class_labels: List of class labels (optional).
- save_path: Path to save the evaluation results (optional).

Returns:
- A dictionary containing evaluation metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, class_labels=None, save_path=None):

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Determine unique sorted labels present in the data
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))

    # Map class_labels to only those present in the data
    if class_labels is not None:
        label_set = set(str(l) for l in unique_labels)
        display_labels = [l for l in class_labels if l in label_set]
        if not display_labels:
            display_labels = [str(l) for l in unique_labels]
    else:
        display_labels = [str(l) for l in unique_labels]

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    # Generate classification report
    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # Print the classification report
    print("Classification Report:")
    print(report)

    # Find top 5 most-confused class pairs
    labels = display_labels
    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((labels[i], labels[j], cm[i][j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confused_pairs[:5]

    print("Top 5 Most-Confused Class Pairs:")
    for true_label, pred_label, count in top_confused:
        print(f"  True: {true_label} -> Predicted: {pred_label} ({count} times)")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # Return evaluation metrics as a dictionary
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'top_confused_pairs': top_confused
    }
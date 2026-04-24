"""
Training script for the LSTM dynamic gesture classifier.

Loads temporal sequence data from data/sequences/ and trains an LSTM model
to classify dynamic signs (hello, thank you, help, etc.).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BASE_DIR, DYNAMIC_DATA_DIR, DYNAMIC_CLASSES, SEQUENCE_LENGTH, EXPECTED_FEATURES


def load_sequence_data():
    sequences = []
    labels = []
    label_map = {}

    for idx, sign_class in enumerate(DYNAMIC_CLASSES):
        label_map[idx] = sign_class
        class_dir = os.path.join(DYNAMIC_DATA_DIR, sign_class)

        if not os.path.isdir(class_dir):
            print(f"WARNING: No data directory for '{sign_class}' at {class_dir}")
            continue

        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"WARNING: No .npy files found for '{sign_class}'")
            continue

        loaded = 0
        for filename in npy_files:
            filepath = os.path.join(class_dir, filename)
            try:
                seq = np.load(filepath)
                if seq.shape == (SEQUENCE_LENGTH, EXPECTED_FEATURES):
                    sequences.append(seq)
                    labels.append(idx)
                    loaded += 1
                else:
                    print(f"  Skipping {filename}: shape {seq.shape} != ({SEQUENCE_LENGTH}, {EXPECTED_FEATURES})")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

        print(f"  Loaded {loaded} samples for '{sign_class}'")

    if not sequences:
        print("\nERROR: No sequence data found. Run data/collect_dynamic_signs.py first.")
        sys.exit(1)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, label_map


def build_model(num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQUENCE_LENGTH, EXPECTED_FEATURES)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_training_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'lstm_training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")


def main():
    print("=" * 50)
    print("LSTM DYNAMIC GESTURE TRAINING")
    print("=" * 50)

    print(f"\nLoading sequence data from {DYNAMIC_DATA_DIR}...")
    X, y, label_map = load_sequence_data()

    num_classes = len(set(y))
    print(f"\nDataset: {X.shape[0]} samples, {num_classes} classes")
    print(f"Sequence shape: {X.shape[1:]} (frames x features)")
    print(f"Classes: {label_map}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    print("\nBuilding LSTM model...")
    model = build_model(num_classes)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True,
    )

    print("\nTraining...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    if accuracy < 0.80:
        print("\nWARNING: Accuracy is below 80%. Consider collecting more data or tuning hyperparameters.")

    save_dir = os.path.join(BASE_DIR, 'models', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'model_lstm.h5')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    import pickle
    label_map_path = os.path.join(save_dir, 'lstm_labels.pickle')
    with open(label_map_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Label map saved to {label_map_path}")

    plot_training_history(history, save_dir)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"Model: {model_path}")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()

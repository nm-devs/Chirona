# 🤟 Chirona
**Real-time Sign Language Interpreter using Computer Vision & Machine Learning**

A Python application that uses MediaPipe, OpenCV, and scikit-learn to detect hands in real-time, extract landmark features, and classify ASL (American Sign Language) signs. It also doubles as a gesture-based mouse controller.

---

## ✨ Features

- 🖐️ **Real-time Hand Detection** — Tracks up to 2 hands simultaneously using MediaPipe's 21-point landmark model
- 🔤 **ASL Sign Classification** — Recognizes static ASL alphabet signs (A–Z, excluding J and Z) via a trained RandomForest classifier
- 🖱️ **Gesture Mouse Control** — Move, click, right-click, and scroll using hand gestures
- 📊 **Prediction Smoothing** — Sliding-window voting system prevents flickering sign displays
- 🎯 **Confidence Display** — Color-coded prediction overlay with confidence bar (green / yellow / red)
- 📸 **Data Collection Pipeline** — Built-in webcam script to capture sign images with countdown timer
- 🔬 **Feature Extraction** — Position-invariant landmark normalization for robust classification
- 📈 **Model Evaluation & Tuning** — Confusion matrix visualization, classification report, and GridSearchCV hyperparameter tuning
- 🔄 **Data Augmentation** — Horizontal flip, rotation, brightness/contrast, and zoom augmentations
- 📦 **ASL MNIST Support** — Loader/converter for the Kaggle ASL MNIST dataset
- 🪞 **Mirror-mode Webcam** — Live feed with FPS counter and mode indicator

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Computer Vision** | OpenCV ≥ 4.8, MediaPipe ≥ 0.10 |
| **Machine Learning** | scikit-learn ≥ 1.3 (RandomForest, GridSearchCV) |
| **Numerical** | NumPy ≥ 1.24 |
| **Mouse Control** | PyAutoGUI |
| **Visualization** | Matplotlib, Seaborn |
| **Language** | Python 3.9–3.11 |

## 📁 Project Structure

```text
chirona/
├── core/                # Core ML & detection logic (classifiers, extractors)
├── controllers/         # Application modes (mouse control, sign language)
├── utils/               # Shared utilities (drawing, smoothing, text overlay)
├── data/                # Data collection, extraction, and dataset loaders
├── models/              # Training, evaluation scripts, and saved weights
├── tests/               # Unit and integration tests
├── main.py              # Application entry point
├── config.py            # Global hyperparameters and UI constants
├── requirements.txt     # Python dependencies
└── ROADMAP.md           # Development roadmap & technical reference
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9–3.11 (recommended for MediaPipe compatibility)
- A webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/nm-devs/hand-2-cursor.git
cd hand-2-cursor

# Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# 1. Collect sign images via webcam
python data/collect_images.py

# 2. Extract hand landmarks from collected images
python data/extract_landmarks.py

# 3. Train the classifier
python models/train_model.py

# (Optional) Tune hyperparameters with GridSearchCV
python models/tune_model.py
```

### Running the App

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `M` | Toggle between Mouse Control and Sign Language mode |
| `ESC` | Exit the application |

### Gesture Mouse Controls

| Gesture | Action |
|---------|--------|
| ☝️ Index finger | Move cursor |
| 🤏 Thumb + Index pinch | Left click |
| 🤏 Thumb + Middle pinch | Right click |
| 🤏 Thumb + Ring pinch | Scroll mode |

## 🧪 Running Tests

```bash
pytest tests/
```

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan. Current progress:

- [x] Project restructure & modular architecture
- [x] Data collection pipeline
- [x] Feature extraction & normalization
- [x] RandomForest model training & evaluation
- [x] Hyperparameter tuning (GridSearchCV)
- [x] Real-time sign classification with confidence display
- [x] Prediction smoothing (anti-flicker)
- [x] Data augmentation utilities
- [x] ASL MNIST dataset support
- [ ] Text-to-speech output
- [ ] Sentence building from individual signs
- [ ] Dynamic gesture recognition (LSTM)
- [ ] Desktop GUI (PyQt5)
- [ ] Web-based version (Flask/FastAPI)

---

> ⚠️ Project is actively in development — model training is ongoing and a live demo will be added upon completion.

Made with ❤️ and Python

Done By [Michael Musallam](https://github.com/michealmou) and [Nadim Baboun](https://github.com/nadeemtsf)

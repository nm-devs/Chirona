import os

# ── Base Directory ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Camera ────────────────────────────────────────────
CAMERA_INDEX = 0              # Default webcam index
CAM_WIDTH = 1280              # Capture width
CAM_HEIGHT = 720              # Capture height

# ── Hand Detection (MediaPipe) ────────────────────────
MAX_HANDS = 2                 # Max simultaneous hands
DETECTION_CONFIDENCE = 0.7    # Min detection confidence
TRACKING_CONFIDENCE = 0.7     # Min tracking confidence

# ── Mouse Controller ──────────────────────────────────
SMOOTHING_ALPHA = 0.3         # Exponential smoothing (0.2–0.35)
FRAME_REDUCTION = 150         # Edge padding for coordinate mapping

# ── Gesture Thresholds ────────────────────────────────
PINCH_DISTANCE = 40           # Pixels to register a pinch
SCROLL_JITTER_THRESHOLD = 5   # Min Y-delta to trigger scroll
SCROLL_SPEED_MULTIPLIER = 1.5 # Scroll speed factor
CLICK_COOLDOWN = 0.1          # Seconds to wait after a click

# ── Drawing ───────────────────────────────────────────
FINGER_CIRCLE_RADIUS = 15     # Radius for fingertip circles

# ── Display ───────────────────────────────────────────
WINDOW_TITLE = "Hand Tracking"

# ── Data Collection ───────────────────────────────────
CLASSES = list("abcdefghiklmnopqrstuvwxy")  # A-Z minus J and Z
IMAGES_PER_CLASS = 50
COUNTDOWN = 3
DELAY_BETWEEN_CLASSES = 50 # milliseconds

# ── Model ─────────────────────────────────────────────
NUM_CLASSES = len(CLASSES)
DATA_DIR= "./data/raw"

# augmentation paths
# AUGMENTED_DIR = "data/augmented"

# ── ASL MNIST ─────────────────────────────────────────
ASL_MNIST_DIR = "data/asl_mnist"

# ── Augmentation Settings ─────────────────────────────
AUGMENT_CONFIG = {
    "flip": True,
    "rotate": True,
    "brightness": True,
    "zoom": False,
    "factor": 5
}

# ── Hyperparameter Tuning ─────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
}
CV_FOLDS = 5
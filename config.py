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

# ── Prediction Smoothing ──────────────────────────────
SMOOTHING_WINDOW_SIZE = 15       # Number of recent predictions to keep
SMOOTHING_DOMINANCE_THRESHOLD = 0.6  # 60% dominance required to update display

# ── UI Colors (BGR Format) ────────────────────────────
COLOR_PRIMARY = (0, 255, 0)       # Green (Text, Left Click)
COLOR_SECONDARY = (255, 0, 255)   # Magenta (Mouse tracking)
COLOR_ACCENT = (255, 255, 0)      # Cyan (Scroll tracking)
COLOR_WARNING = (0, 220, 255)     # Yellow (Medium confidence)
COLOR_DANGER = (0, 0, 220)        # Red (Low confidence / Right click)
COLOR_WHITE = (255, 255, 255)     # White text
COLOR_BLACK_BG = (50, 50, 50)     # Gray background bars

# ── Prediction Confidence Thresholds ──────────────────
CONFIDENCE_HIGH = 0.8             # Threshold for green prediction UI
CONFIDENCE_MEDIUM = 0.5           # Threshold for yellow prediction UI
CONFIDENCE_THRESHOLD=0.70          # Minimum confidence to display prediction (70%)

# ── Sentence Builder ──────────────────────────────────
CONFIRM_DURATION = 1.5            # Seconds to wait before confirming a word

# ── Data Processing ───────────────────────────────────
DEFAULT_PICKLE = "data/landmarks.pickle"
EXPECTED_FEATURES = 42          # 21 landmarks × 2 (x, y)
IMBALANCE_THRESHOLD = 0.5       # warn if any class has < 50% of the max count

# ── Text-to-Speech (pyttsx3) ──────────────────────
TTS_ENABLED = True                    # Enable/disable TTS system
TTS_SPEECH_RATE = 150                 # Words per minute (default 150)
TTS_VOLUME = 0.9                      # Volume level 0-1 (default 0.9)
TTS_SPEAK_LETTERS = True             # Speak each letter as confirmed (optional)
TTS_SPEAK_WORDS = True               # Speak each word when confirmed
TTS_SPEAK_SENTENCES = True           # Speak full sentence when triggered
TTS_FALLBACK_SILENT = True           # Silent fallback if TTS unavailable
TTS_QUEUE_MAX_SIZE = 50              # Max queued speech requests
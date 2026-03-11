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

# ── Display ───────────────────────────────────────────
WINDOW_TITLE = "Hand Tracking"

# ── Data ─────────────────────────────────────────────
DATA_DIR = "./data/raw"
DYNAMIC_DATA_DIR = "./data/sequences"

# ── Dynamic Signs ────────────────────────────────────
DYNAMIC_CLASSES = ['hello', 'thank you', 'help', 'sorry', 'yes', 'no', 'please']
SEQUENCE_LENGTH = 30
SEQUENCES_PER_CLASS = 50

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

CLICK_COOLDOWN = 0.5 # 500ms cooldown between clicks to prevent multiple triggers from one gesture
SPREAD_THRESHOLD = 30 # Minimum normalized distance between thumb and pinky for "spread" gesture
CURL_THRESHOLD = 40 # Threshold for determining if fingers are curled (0 to 1)
THUMBS_THRESHOLD = 80 # Threshold for determining if thumb is up
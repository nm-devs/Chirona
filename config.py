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

# ── Data ─────────────────────────────────────────────
DATA_DIR = "./data/raw"

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
CONFIDENCE_THRESHOLD = 0.70       # Minimum confidence to display prediction (70%)
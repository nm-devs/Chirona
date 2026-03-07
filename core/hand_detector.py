import cv2
import mediapipe as mp
from config import(
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE
)

class HandDetector:
    def __init__(self, max_hands=MAX_HANDS, detection_confidence=DETECTION_CONFIDENCE, tracking_confidence=TRACKING_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def detect(self, frame):
        """
        Detect hands in frame and return list of hand data with pixel coordinates.
        Each hand dict contains: label, landmarks (raw), positions (list of (id, x, y))
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = hand_info.classification[0].label
                
                # Convert landmarks to pixel positions
                positions = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    positions.append((idx, px, py))
                
                hands_data.append({
                    "label": hand_label,
                    "landmarks": hand_landmarks,
                    "positions": positions
                })
        def draw_hands(self, frame, hands_data):
            for hand in hands_data:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand['landmarks'],
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
                #draw label
                if hand['positions']:
                    #wrist is landmark 0
                    wrist_x, wrist_y = hand['positions'][0][1], hand['positions'][0][2]
                    cv2.putText(frame, hand['label'], (wrist_x - 10, wrist_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                return frame
        return hands_data
"""
Controller for processing visual feedback in sign language translation mode.

Handles the rendering of hand tracking points and skeletons on the camera feed
when the user is spelling out signs.
"""
import cv2
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from config import (
    TTS_ENABLED,
    TTS_SPEAK_LETTERS,
    TTS_SPEAK_WORDS
)
from utils.text_to_speech import TextToSpeech
from utils.drawing_utils import draw_hand_points, draw_hand_skeleton


class SignLanguageController:
    def __init__(self, tts: TextToSpeech = None):
        if tts:
            self.tss = tts
        else:
            self.tss = TextToSpeech() if TTS_ENABLED else None
        
        self.last_spoken_sign = None
        self.last_spoken_word = None
        logging.info(f"SignLanguageController initialized with TTS enabled: {TTS_ENABLED}")

    def process_frame(self, frame, hand, detector):
        """Process one frame in sign-language mode."""
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]

        # Draw visuals
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)

        # Placeholder until classifier is loaded
        cv2.putText(
            frame,
            "Sign Language Mode - classifier not loaded yet",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    def speak_sign(self, sign: str):
        # speak a recognized sign using tts
        if not sign or not self.tss or not TTS_ENABLED:
            return
        #avoid repeating the same sign too often
        if sign == self.last_spoken_sign and TTS_SPEAK_LETTERS:
            return
        
        if TTS_SPEAK_LETTERS:
            self.tss.speak(sign)
            self.last_spoken_sign = sign
            logging.debug(f"Spoken sign: {sign}")
    
    def speak_word(self, word: str):
        # speak a confirmed word using tss

        if not word or not self.tss or not TTS_ENABLED:
            return
        # avoid repeating the same word too often
        if word == self.last_spoken_word and TTS_SPEAK_WORDS:
            return
        if TTS_SPEAK_WORDS:
            self.tss.speak(word)
            self.last_spoken_word = word
            logging.debug(f"Spoken word: {word}")
    def speak_sentence(self, sentence: str):
        # speak a confirmed sentence using tts
        if not sentence or not self.tss or not TTS_ENABLED:
            return
        self.tss.speak(sentence)
        logging.debug(f"Spoken sentence: {sentence}")
    
    def handle_thumbs_up_gesture(self, current_sentence: str):
        # handle thumgs gesture to trigger speech:
        if current_sentence:
            self.speak_sentence(current_sentence)
            logging.info("Thumbs up detected - triggered sentence speech")
    def clear_speech_memory(self):
        # clear last spoken signs and words to allow repeating them
        self.last_spoken_sign = None
        self.last_spoken_word = None
        logging.info("Cleared speech memory for signs and words")
    
    def shutdown(self):
        # shutdown the tts engine
        if self.tts:
            self.tts.shutdown()
            logging.info('signlanguagecontroller shutdown')
            
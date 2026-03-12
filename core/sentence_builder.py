"""
Accumulates individual recognized signs into words and sentences.

Implements hold-to-confirm logic to prevent jitter and accidental typing.
"""
import logging
from utils.text_to_speech import TextToSpeech
from config import CONFIRM_DURATION,TTS_ENABLED, TTS_SPEAK_LETTERS, TTS_SPEAK_WORDS, TTS_SPEAK_SENTENCES

logger = logging.getLogger(__name__)
# The SentenceBuilder class manages the construction of words and sentences from recognized signs. It uses a hold-to-confirm mechanism to ensure that only deliberate gestures are added to the current word. The class also provides methods for adding spaces, backspacing, clearing the sentence, and retrieving the text for display or speech synthesis.
class SentenceBuilder:
    def __init__(self, tts=None):
        
        self.current_word = ""
        self.sentence = ""
        self.history = []  # Keep track of recent predictions for smoothing
        self.current_sign = None
        self.start_time = 0.0
        self.confirm_duration = CONFIRM_DURATION
        self.tts = tts if tts else (TextToSpeech() if TTS_ENABLED else None)
        self.last_spoken_letter = None # prevent repeating same letter if held for multiple seconds
        self.last_spoken_word = None # prevent repeating same word if held for multiple seconds
    
    def add_letter(self, letter):
        '''Adds a letter to the current word.'''
        self.current_word += letter
        logger.debug(f"Letter added: {letter} | Current word: '{self.current_word}'")
        self._speak_letter(letter)
    def _speak_letter(self, letter: str):
        # Disabled to prevent TTS queue congestion - only speak on thumbs-up
        # Individual letter speech was causing the full sentence speech to get lost
        return

    def update(self, sign, timestamp):
        if not sign:
            self.current_sign = None
            self.start_time = 0.0
            return
            
        if sign != self.current_sign:
            self.current_sign = sign
            self.start_time = timestamp
            logger.debug(f"New sign detected: {sign}")
        elif timestamp - self.start_time >= self.confirm_duration:
            logger.info(f"Sign confirmed after {self.confirm_duration}s hold: {sign}")
            self.add_letter(sign)
            self.current_sign = None
            self.start_time = 0.0

    def add_space(self):
        """Moves current_word to sentence with a space."""
        if self.current_word:
            self.history.append((self.sentence, self.current_word))
            self.sentence += self.current_word + " "
            logger.info(f"Word confirmed: '{self.current_word}' | Sentence so far: '{self.sentence}'")
            self._speak_word(self.current_word)  # Speak the confirmed word
            self.current_word = ""
        return self.sentence
    
    def _speak_word(self, word: str):
        # Disabled to prevent TTS queue congestion - only speak on thumbs-up
        # Individual word speech was causing the full sentence speech to get lost
        return

    def get_display_text(self):
        """Returns the full accumulated text."""
        return self.sentence + self.current_word
    
    def backspace(self):
        # Remove last character from current word, or if empty, remove last word from sentence
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence: # if word empty then remove last word from sentence
            words = self.sentence.strip().split()
            if words:
                self.current_word = words[-1][:-1] #remove first letter
                self.sentence = " ".join(words[:-1]) + " "
            return self.get_display_text()
        
    def clear(self):
        """Resets everything."""
        self.current_word = ""
        self.sentence = ""
        self.current_sign = None
        self.start_time = 0.0
        self.last_spoken_letter = None
        self.last_spoken_word = None
        return self.sentence
    
    def speak(self):
        """Speak the full accumulated text (called by thumbs-up gesture)."""
        # Get all text: confirmed words (sentence) + current working word
        text = (self.sentence + self.current_word).strip()
        
        logger.info(f"THUMBS-UP TRIGGER - Speaking: '{text}'")
        
        if text:
            if TTS_ENABLED and self.tts:
                logger.info(f"Sending to TTS: {len(text)} characters")
                self.tts.speak(text)
            else:
                logger.warning("TTS disabled or not initialized")
        else:
            logger.warning("No text to speak")
        
        return text 
    
"""
Accumulates individual recognized signs into words and sentences.

Implements hold-to-confirm logic to prevent jitter and accidental typing.
"""
from controllers.sign_language_controller import SignLanguageController
from config import CONFIRM_DURATION

class SentenceBuilder:
    def __init__(self, sign_controller: SignLanguageController = None):
        self.sign_controller = sign_controller
        self.current_word = ""
        self.sentence = ""
        self.current_sign = None
        self.start_time = 0.0
        self.confirm_duration = CONFIRM_DURATION
    
    def add_sign(self, sign: str):
        #add a sign to the current word, using hold-to-confirm logic
        self.current_word+= sign
        #speak the letter if tts is enabled
        if self.sign_controller:
            self.sign_controller.speak_sign(sign)
    
    def confirm_word(self):
        #confirm word and speak it if tts is enabled
        if self.current_word:
            self.words.appead(self.current_word)
            #speak the confrimed word if tts is enabled
            if self.sign_controller:
                self.sign_controller.speak_word(self.current_word)
            self.current_word = ""
    
    def update(self, sign, timestamp):
        if not sign:
            self.current_sign = None
            self.start_time = 0.0
            return
            
        if sign != self.current_sign:
            self.current_sign = sign
            self.start_time = timestamp
        elif timestamp - self.start_time >= self.confirm_duration:
            self.current_word += sign
            self.current_sign = None
            self.start_time = 0.0

    def add_space(self):
        """Moves current_word to sentence with a space."""
        if self.current_word:
            self.sentence += self.current_word + " "
            self.current_word = ""
        elif not self.sentence.endswith(" ") and self.sentence != "":
            self.sentence += " "

    def get_display_text(self):
        """Returns the full accumulated text."""
        return self.sentence + self.current_word

    def clear(self):
        """Resets everything."""
        self.current_word = ""
        self.sentence = ""
        self.current_sign = None
        self.start_time = 0.0
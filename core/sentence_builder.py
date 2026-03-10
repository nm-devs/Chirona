"""
Accumulates individual recognized signs into words and sentences.

Implements hold-to-confirm logic to prevent jitter and accidental typing.
"""

CONFIRM_DURATION = 1.5  # Seconds to wait before confirming a word

class SentenceBuilder:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""
        self.current_sign = None
        self.start_time = 0.0
        self.confirm_duration = CONFIRM_DURATION
    
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
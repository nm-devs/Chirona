"""
Accumulates individual recognized signs into words and sentences.

Implements hold-to-confirm logic to prevent jitter and accidental typing.
"""

CONFIRM_DURATION = 1.5  # Seconds to wait before confirming a word

class SentenceBuilder:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""
        self.history = []  # Keep track of recent predictions for smoothing
        self.current_sign = None
        self.start_time = 0.0
        self.confirm_duration = CONFIRM_DURATION
    def add_letter(self, letter):
        '''Adds a letter to the current word.'''
        self.current_word += letter

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
            self.history.append((self.sentence, self.current_word))
            self.sentence += self.current_word + " "
            self.current_word = ""
        return self.sentence
    

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
        return self.sentence
    
    def speak(self): 
        # return text to be spoken 
        text = (self.sentence + self.current_word).strip()
        return text 
    
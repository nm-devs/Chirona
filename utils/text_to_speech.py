import pyttsx3
import queue
import threading
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextToSpeech:
    def __init__(self, rate=150, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.engine = None
        self.is_available = False
        self.speech_queue = queue.Queue(maxsize=50)  # Limit queue size to prevent memory issues
        self.should_stop = False
        self.worker_thread = None
        self.lock = threading.Lock()  # To protect access to the engine
        self.is_speaking = False  # Track if TTS is currently speaking (for UI feedback)
        
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)

            # try to set a preferred voice (optional)
            voices = self.engine.getProperty('voices')
            if len(voices) > 1:
                # prefer female voice usually index 1
                self.engine.setProperty('voice', voices[1].id)
            elif len(voices) > 0:
                self.engine.setProperty('voice', voices[0].id)
            
            self.is_available = True
            logging.info(f'ENGINE INITIALIZED: {self.engine} '
                        f'RATE: {self.rate}, VOLUME: {self.volume}, VOICES: {len(voices)}')
            # start worker thread

            self.should_stop = False
            self.worker_thread = threading.Thread(
                target=self._process_speech_queue,
                daemon=True
            )
            self.worker_thread.start()
            logging.info("TextToSpeech worker thread started.")
        except Exception as e:
            logging.error(f"Failed to initialize TextToSpeech engine: {e} "
                          "Text-to-speech functionality will be disabled.")

    def speak(self, text: str) -> bool:
        if not text or not self.is_available:
            return False
        try:
            # dont block on queue.put() to avoid freezing the app if tts is slow
            self.speech_queue.put_nowait(text)
            logging.info(f"Queued text for speech: '{text}'")
            return True
        except queue.Full:
            logging.warning(f"Speech queue is full. Dropping text: '{text}'")
            return False

    def _process_speech_queue(self):
        # background thread to process speech queue
        logging.debug('speech worker thread running')
        while not self.should_stop:
            try:
                # wait for a speech request, with timeout to allow checking should_stop
                text = self.speech_queue.get(timeout=0.5)

                if text and self.engine:
                    try:
                        self.is_speaking = True  # Set flag: TTS is actively speaking
                        logging.info(f"Starting speech: '{text}'")
                        with self.lock:
                            self.engine.say(text)
                            self.engine.runAndWait()
                        logging.info(f"Spoken text: '{text}'")
                    except Exception as e:
                        logging.error(f"Error during speech synthesis: {e}")
                    finally:
                        self.is_speaking = False  # Clear flag: TTS speech complete
                self.speech_queue.task_done()
            except queue.Empty:
                # timeout - continue loop to check should_stop
                continue
            except Exception as e:
                logging.error(f"Unexpected error in speech worker thread: {e}")
        logging.debug('speech worker thread stopped')
    
    def set_speech_rate(self, rate: int):
        self.rate = max(50, min(300, rate))  # clamp rate to reasonable range
        if self.engine and self.is_available:
            with self.lock:
                self.engine.setProperty('rate', self.rate)
            logging.info(f"Speech rate set to: {self.rate}")
    
    def set_volume(self, volume: float):
        # set volume between 0.0 and 1.0
        self.volume = max(0.0, min(1.0, volume)) # clamp volume to valid range
        if self.engine and self.is_available:
            with self.lock:
                self.engine.setProperty('volume', self.volume)
    
    def is_currently_speaking(self) -> bool:
        """Check if TTS is currently synthesizing speech. Used for UI feedback."""
        return self.is_speaking
    
    def get_available_voices(self):
        
        if self.engine and self.is_available:
            try:
                return self.engine.getProperty('voices')
            except Exception as e:
                logging.error(f"Error retrieving voices: {e}")
        return []
    
    def set_voice(self, voice_id:str):
        if self.engine and self.is_available:
            try:
                with self.lock:
                    self.engine.setProperty('voice', voice_id)
                logging.info(f"Voice set to: {voice_id}")
            except Exception as e:
                logging.error(f"Error setting voice: {e}")
    
    def clear_queue(self):
        #clear all pending speech requests
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        logging.info("Speech queue cleared.")
    
    def shutdown(self):
        # cleanly shutdown the TTS engine and worker thread
        logging.info("Shutting down TextToSpeech engine...")
        self.should_stop = True
        #wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            if self.worker_thread.is_alive():
                logging.warning("TextToSpeech worker thread did not exit cleanly.")
        
        # cleanup pyttsx3 engine resources
        if self.engine:
            try:
                self.engine.stop()  # stop any ongoing speech
            except Exception as e:
                logging.error(f"Error stopping TTS engine: {e}")
            self.engine = None
        self.is_available = False
        logging.info("TextToSpeech engine shutdown complete.")
    
    def __del__(self):
        #ensure resources are cleaned up if object is deleted
        self.shutdown()
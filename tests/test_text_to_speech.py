import unittest
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.text_to_speech import TextToSpeech
from unittest.mock import patch, MagicMock, Mock 

class TestTextToSpeech(unittest.TestCase):
    # test text to speech initialization

    def tearDown(self):
        #intentionally left blank since TextToSpeech class handles its own cleanup
        pass    
    def test_initiliazation_success(self):
        # test successful initialization of TextToSpeech
        tts = TextToSpeech()
        try:
            self.assertIsNotNone(tts.engine)
            self.assertEqual(tts.rate, 150)
            self.assertEqual(tts.volume, 1.0)
            self.assertTrue(tts.is_available)
        finally:
            tts.shutdown()  # ensure resources are cleaned up after test
    def test_initialization_custom_parameeters(self):
        # test initialization with custom parameters
        tts = TextToSpeech(rate=200, volume=0.5)
        try:
            self.assertEqual(tts.rate, 200)
            self.assertEqual(tts.volume, 0.5)
        finally:
            tts.shutdown()
    def test_worker_thread_starts(self):
        #test that the worker thread starts successfully
        tts = TextToSpeech()
        try:
            self.assertTrue(tts.worker_thread.is_alive())
        finally:
            tts.shutdown()
class TestTextToSpeechBasicFunctionality(unittest.TestCase):
    # test basic speak functionality
    def setUp(self):
        self.tts = TextToSpeech()
    def tearDown(self):
        self.tts.shutdown()
    def test_speak_with_valid_text(self):
        # test that speak method queues text successfully
        result = self.tts.speak("Hello, world!")
        self.assertTrue(result)
    
    def test_speak_with_empty_text(self):
        # test that speak method returns False for empty text
        result = self.tts.speak("")
        self.assertFalse(result)
    
    def test_speak_with_none(self):
        # test that speak method returns False for None input
        result = self.tts.speak(None)
        self.assertFalse(result)
    def test_speak_non_blocking(self):
        # test that speak() trturns immediately
        start_time = time.time()
        self.tts.speak("Testing non-blocking speak")
        elapsed_time = time.time() - start_time

        # speak should return immediately, so elapsed time should be very short (e.g. < 0.1 seconds)
        self.assertLess(elapsed_time, 0.1, "speak() method is blocking, took too long to return")
class TestTextToSpeechConfiguration(unittest.TestCase):
    # test text to speech configuration methods
    def setUp(self):
        self.tts = TextToSpeech()
    def tearDown(self):
        self.tts.shutdown()
    def test_set_speech_rate(self):
        # test that set_speech_rate updates the rate property
        self.tts.set_speech_rate(200)
        self.assertEqual(self.tts.rate, 200)
    def test_set_speech_rate_clamping(self):
        # test that set_speech_rate clamps values to a reasonable range (e.g. 50-400 wpm)
        self.tts.set_speech_rate(999)  # above max
        self.assertEqual(self.tts.rate, 300)  # should be clamped to 300
        self.tts.set_speech_rate(20)   # below min
        self.assertEqual(self.tts.rate, 50)   # should be clamped to 50

    def test_set_volume(self):
        # test that set_volume updates the volume property
        self.tts.set_volume(0.5)
        self.assertEqual(self.tts.volume, 0.5)
    def test_set_volume_clamping(self):
        # test that set_volume clamps values to 0.0 - 1.0
        self.tts.set_volume(1.5)  # above max
        self.assertEqual(self.tts.volume, 1.0)  # should be clamped to 1.0
        self.tts.set_volume(-0.5) # below min
        self.assertEqual(self.tts.volume, 0.0)  # should be clamped to 0.0
    
class TestTextToSpeechQueueManagement(unittest.TestCase):
    # test that the speech queue is managed correctly
    def setUp(self):
        self.tts = TextToSpeech()
    def tearDown(self):
        self.tts.shutdown()
    def test_clear_queue(self):
        #test clearing the queue removes all pending speech requests
        # add multiple items to the queue
        self.tts.speak("First message")
        self.tts.speak("Second message")
        self.tts.speak("Third message")
        # clear the queue
        self.tts.clear_queue()
        #queue should be empty
        self.assertTrue(self.tts.speech_queue.empty())
    def test_queue_size_limit(self):
        #test that queue respects the maximum size limit
        # fill the queue to its max size
        for i in range(50):
            self.tts.speak(f"Message {i}")
        #trying to add more should return False
        result = self.tts.speak("This message should not be added")
        self.assertFalse(result)

class TestTextToSpeechShutdown(unittest.TestCase):
    #test that shutdown method properly stops the worker thread and releases resources
    def test_shutdown_stops_worker_thread(self):
        tts = TextToSpeech()
        worker = tts.worker_thread

        self.assertTrue(worker.is_alive())
        tts.shutdown()

        # give thread a moment to stop
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
    def test_shutdown_marks_unavailable(self):
        tts = TextToSpeech()
        self.assertTrue(tts.is_available)
        tts.shutdown()
        self.assertFalse(tts.is_available)

class TestTextToSpeechErrorHandling(unittest.TestCase):
    #test text to speech error handling during initialization and speaking
    @patch('utils.text_to_speech.pyttsx3.init')
    def test_graceful_fallback_on_init_failure(self,mock_init):
        #test gravefull fallback
        mock_init.side_effect = Exception("TTS engine initialization failed")
        tts = TextToSpeech()

        #should not crash but should mark TTS as unavailable
        self.assertFalse(tts.is_available)
        self.assertIsNone(tts.engine)

        # speak() should return False since TTS is unavailable
        result = tts.speak("This should not be spoken")
        self.assertFalse(result)
    
    def test_speak_when_unavailable(self):
        #test that speak() returns False if TTS is unavailable
        tts = TextToSpeech()
        tts.is_available = False  # simulate TTS being unavailable

        result = tts.speak("This should not be spoken")
        self.assertFalse(result)

        tts.shutdown()

class TestTextToSpeechThreadSafety(unittest.TestCase):
    # test thread safety of text to speech operations
    def setUp(self):
        self.tts = TextToSpeech()

    def tearDown(self):
        self.tts.shutdown()
    def test_concurrent_speak_calls(self):
        #test multiple concurrent calls to speak() do not cause errors or
        import threading
        results = []
        def call_speak(text):
            result = self.tts.speak(text)
            results.append(result)
        threads = [
            threading.Thread(target=call_speak, args=(f"Message {i}",))
            for i in range(10)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # all speak calls should succeed (or fail gracefully without crashing)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
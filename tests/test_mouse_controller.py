import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
from controllers.mouse_controller import MouseController

class TestMouseController(unittest.TestCase):
    
    @patch('controllers.mouse_controller.pyautogui')
    def test_initialization(self, mock_pyautogui):
        """Test that the controller initializes state cleanly without errors."""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        
        mouse = MouseController(alpha=0.3)
        self.assertEqual(mouse.screen_w, 1920)
        self.assertEqual(mouse.screen_h, 1080)
        self.assertFalse(mouse.is_dragging)
        
    @patch('controllers.mouse_controller.pyautogui')
    def test_move_mouse(self, mock_pyautogui):
        """Test the move logic correctly invokes the OS move command."""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        
        # alpha=1.0 means no exponential smoothing, so it snaps directly to coordinates
        mouse = MouseController(alpha=1.0) 
        
        # Simulate moving past the deadzone
        mouse.move(1000, 800)
        
        # Assert PyAutoGUI was called with the correct integer casted coordinates
        mock_pyautogui.moveTo.assert_called_with(1000, 800)
        
    @patch('controllers.mouse_controller.pyautogui')
    def test_click_delegation(self, mock_pyautogui):
        """Test that the direct left/right clicks are passed to PyAutoGUI."""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        
        mouse = MouseController()
        mouse = MouseController()
        mouse.click('left')
        
        mock_pyautogui.click.assert_called_with(button='left')

    def create_mock_hand(self, is_pinched=False):
        """Helper to create a dummy hand dataset with varying pinch states"""
        # Base coordinates
        thumb_x, thumb_y = 100, 100
        
        # If pinched, index finger is at the same location as thumb. Otherwise, far away.
        index_x, index_y = (100, 100) if is_pinched else (200, 200)
        
        positions = [
            [i, 0, 0, 0] for i in range(21)
        ]
        
        # Override specific landmarks
        positions[4] = [4, thumb_x, thumb_y, 0]    # Thumb tip
        positions[5] = [5, 120, 120, 0]            # Index knuckle
        positions[8] = [8, index_x, index_y, 0]    # Index tip
        positions[12] = [12, 300, 300, 0]          # Middle tip (far)
        positions[16] = [16, 400, 400, 0]          # Ring tip (far)

        return {
            "positions": positions,
            "landmarks": None
        }

    @patch('controllers.mouse_controller.pyautogui')
    @patch('controllers.mouse_controller.time.time')
    @patch('controllers.mouse_controller.draw_hand_points')
    @patch('controllers.mouse_controller.draw_hand_skeleton')
    @patch('controllers.mouse_controller.cv2.circle')
    def test_single_click(self, mock_cv2, mock_skel, mock_pts, mock_time, mock_pyautogui):
        """Test a short pinch eventually fires a single click"""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        mouse = MouseController()
        class MockDetector: pass
        detector = MockDetector()
        detector.mp_hands = None
        detector.mp_drawing = None
        dummy_frame = None
        
        # Frame 1: Pinch starts at t=1.0
        mock_time.return_value = 1.0
        mouse.process_frame(dummy_frame, self.create_mock_hand(is_pinched=True), detector)
        
        self.assertTrue(mouse.is_pinching)
        self.assertFalse(mouse.is_dragging) # Drag shouldn't start yet
        
        # Frame 2: Pinch releases at t=1.2 (0.2s elapsed > CLICK_COOLDOWN)
        mock_time.return_value = 1.2
        mouse.process_frame(dummy_frame, self.create_mock_hand(is_pinched=False), detector)
        
        self.assertFalse(mouse.is_pinching)
        self.assertGreater(mouse.pending_tap_time, 0) # Tap is queued!
        
        # Frame 3: Time passes, threshold expires at t=1.6
        mock_time.return_value = 1.6
        mouse.process_frame(dummy_frame, self.create_mock_hand(is_pinched=False), detector)
        
        mock_pyautogui.click.assert_called_with(button='left')
        self.assertEqual(mouse.pending_tap_time, 0)
        
    @patch('controllers.mouse_controller.pyautogui')
    @patch('controllers.mouse_controller.time.time')
    @patch('controllers.mouse_controller.draw_hand_points')
    @patch('controllers.mouse_controller.draw_hand_skeleton')
    @patch('controllers.mouse_controller.cv2.circle')
    def test_double_click(self, mock_cv2, mock_skel, mock_pts, mock_time, mock_pyautogui):
        """Test two rapid pinches fire a double click without firing a single click"""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        mouse = MouseController()
        class MockDetector: pass
        detector = MockDetector()
        detector.mp_hands = None
        detector.mp_drawing = None
        dummy_frame = None
        
        # 1. Pinch 1
        mock_time.return_value = 1.0
        mouse.process_frame(dummy_frame, self.create_mock_hand(True), detector)
        
        # 2. Release 1 (Queues tap)
        mock_time.return_value = 1.2
        mouse.process_frame(dummy_frame, self.create_mock_hand(False), detector)
        
        # 3. Pinch 2 (Before threshold expires, e.g. t=1.3)
        mock_time.return_value = 1.3
        mouse.process_frame(dummy_frame, self.create_mock_hand(True), detector)
        
        # Assert double click fired, and tap is consumed
        mock_pyautogui.doubleClick.assert_called_with(button='left')
        mock_pyautogui.click.assert_not_called()
        self.assertEqual(mouse.pending_tap_time, 0)

    @patch('controllers.mouse_controller.pyautogui')
    @patch('controllers.mouse_controller.time.time')
    @patch('controllers.mouse_controller.draw_hand_points')
    @patch('controllers.mouse_controller.draw_hand_skeleton')
    @patch('controllers.mouse_controller.cv2.circle')
    def test_drag(self, mock_cv2, mock_skel, mock_pts, mock_time, mock_pyautogui):
        """Test holding a pinch fires mouseDown and mouseUp"""
        mock_pyautogui.size.return_value = (1920, 1080)
        mock_pyautogui.position.return_value = (500, 500)
        mouse = MouseController()
        class MockDetector: pass
        detector = MockDetector()
        detector.mp_hands = None
        detector.mp_drawing = None
        dummy_frame = None
        
        # 1. Pinch starts
        mock_time.return_value = 1.0
        mouse.process_frame(dummy_frame, self.create_mock_hand(True), detector)
        
        # 2. Hold pinch past drag threshold (0.4s) -> t=1.4
        mock_time.return_value = 1.4
        mouse.process_frame(dummy_frame, self.create_mock_hand(True), detector)
        
        mock_pyautogui.mouseDown.assert_called_with(button='left')
        self.assertTrue(mouse.is_dragging)
        
        # 3. Release pinch
        mock_time.return_value = 1.6
        mouse.process_frame(dummy_frame, self.create_mock_hand(False), detector)
        
        mock_pyautogui.mouseUp.assert_called_with(button='left')
        self.assertFalse(mouse.is_dragging)
        mock_pyautogui.click.assert_not_called() # Drag should overrule single click

if __name__ == '__main__':
    unittest.main()
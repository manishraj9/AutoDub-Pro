import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import subprocess

# Set mock env keys for safety
os.environ["GEMINI_API_KEY"] = "test_key_1"
os.environ["GEMINI_API_KEY_1"] = "test_key_1"
os.environ["GEMINI_API_KEY_2"] = "test_key_2"
os.environ["ELEVENLABS_API_KEY"] = "mock_eleven_key"


# Ensure package import paths are set correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dubber.config import TEMP_DIR, CACHE_DIR
from dubber.validator import (
    validate_youtube_url,
    validate_eq_gains,
    validate_duck_level,
    validate_segment_intervals,
    validate_write_permissions
)
from dubber.cache import (
    clear_translation_cache,
    get_cached_segments_count,
    add_to_translation_cache,
    get_cached_translation
)
from dubber.utils import (
    calculate_speaking_rate,
    check_api_keys_validity
)

class TestCommandLineInterfaceAndOrchestration(unittest.TestCase):
    """
    Test suite validating command-line entry points, argument shapes,
    boundary validation limits, and overall orchestration logic.
    """
    
    def setUp(self):
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        clear_translation_cache()
        
    def tearDown(self):
        clear_translation_cache()
        # Clean up temp folder files
        if os.path.exists(TEMP_DIR):
            for f in os.listdir(TEMP_DIR):
                try:
                    os.remove(os.path.join(TEMP_DIR, f))
                except Exception:
                    pass

    # ==========================================
    # CLI PARSER & PARAMETER VALIDATIONS
    # ==========================================

    def test_youtube_url_validators_cli(self):
        """Validates YouTube URL formats including mobile and Shorts layouts."""
        self.assertTrue(validate_youtube_url("https://www.youtube.com/watch?v=ABCDEFGHIJK"))
        self.assertTrue(validate_youtube_url("https://youtu.be/ABCDEFGHIJK"))
        self.assertTrue(validate_youtube_url("https://youtube.com/shorts/ABCDEFGHIJK"))
        self.assertTrue(validate_youtube_url("https://youtube.com/embed/ABCDEFGHIJK"))
        
        self.assertFalse(validate_youtube_url("https://google.com"))
        self.assertFalse(validate_youtube_url("https://youtube.com/watch?v=too_short"))
        self.assertFalse(validate_youtube_url(""))

    def test_parametric_eq_gain_limits(self):
        """Checks EQ gain thresholds to prevent digital clipping."""
        # Safe thresholds
        ok, msg = validate_eq_gains(0.0, 0.0, 0.0)
        self.assertTrue(ok)
        ok, msg = validate_eq_gains(-10.0, 5.0, 12.0)
        self.assertTrue(ok)
        
        # Out of bounds
        bad, msg = validate_eq_gains(15.0, 0.0, 0.0)
        self.assertFalse(bad)
        self.assertIn("Bass", msg)
        
        bad_mid, msg = validate_eq_gains(0.0, -13.0, 0.0)
        self.assertFalse(bad_mid)
        self.assertIn("Mid", msg)
        
        bad_treble, msg = validate_eq_gains(0.0, 0.0, 12.5)
        self.assertFalse(bad_treble)
        self.assertIn("Treble", msg)

    def test_audio_ducking_volume_bounds(self):
        """Verifies background audio ducking parameters."""
        # Safe limits
        ok, msg = validate_duck_level(0.0) # No ducking
        self.assertTrue(ok)
        ok, msg = validate_duck_level(-25.0)
        self.assertTrue(ok)
        
        # Out of bounds
        bad_pos, msg = validate_duck_level(5.0)
        self.assertFalse(bad_pos)
        
        bad_neg, msg = validate_duck_level(-70.0)
        self.assertFalse(bad_neg)

    def test_segment_timing_safety_intervals(self):
        """Verifies time offsets for dialogue alignments."""
        good_segs = [
            {"start_time": 0.0, "end_time": 1.5},
            {"start_time": 1.5, "end_time": 4.2}
        ]
        ok, msg = validate_segment_intervals(good_segs, max_duration=10.0)
        self.assertTrue(ok)
        
        # Segment starts after it ends
        bad_segs = [{"start_time": 3.0, "end_time": 2.0}]
        bad, msg = validate_segment_intervals(bad_segs)
        self.assertFalse(bad)
        
        # Negative bounds
        bad_neg = [{"start_time": -1.0, "end_time": 2.0}]
        bad, msg = validate_segment_intervals(bad_neg)
        self.assertFalse(bad)
        
        # Exceeds max video length limits
        bad_len = [{"start_time": 1.0, "end_time": 15.0}]
        bad, msg = validate_segment_intervals(bad_len, max_duration=10.0)
        self.assertFalse(bad)

    def test_directory_write_permissions(self):
        """Checks directory availability for saving output assets."""
        self.assertTrue(validate_write_permissions(TEMP_DIR))
        # Checks fake drive mapping (standard permissions lock failure)
        self.assertFalse(validate_write_permissions("X:\\non_existent_folder_path_test"))

    # ==========================================
    # DUBBING CACHE & PERFORMANCE ESTIMATOR
    # ==========================================

    def test_cache_persistence_integrity(self):
        """Validates JSON cache persistence on disk."""
        clear_translation_cache()
        self.assertEqual(get_cached_segments_count(), 0)
        
        # Add values
        add_to_translation_cache("Original de prueba", "Translated test text", "es")
        self.assertEqual(get_cached_segments_count(), 1)
        self.assertEqual(get_cached_translation("Original de prueba", "es"), "Translated test text")
        
        # Clear and check
        clear_translation_cache()
        self.assertEqual(get_cached_segments_count(), 0)

    def test_speaking_rate_wpm_monitor(self):
        """Validates conversation pace metrics."""
        # 10 words in 3 seconds -> 200 WPM (should trigger fast speech alert)
        wpm = calculate_speaking_rate("one two three four five six seven eight nine ten", duration_sec=3.0)
        self.assertEqual(wpm, 200.0)
        
        # Short limits
        wpm_zero = calculate_speaking_rate("test", duration_sec=0.01)
        self.assertEqual(wpm_zero, 0.0)

    @patch('google.genai.Client')
    def test_api_keys_validation_check(self, mock_client):
        """Ensures the key validation checker correctly detects validity."""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock models list call
        mock_instance.models.list.return_value = []
        
        keys = ["mock_key_1", ""]
        validity = check_api_keys_validity(keys)
        self.assertTrue(validity[0])
        self.assertFalse(validity[1])

    # ==========================================
    # CLI COMMAND LAUNCH SIMULATIONS
    # ==========================================

    def test_cli_help_menu_arguments(self):
        """Simulates running the main CLI launcher with --help parameter."""
        cmd = [sys.executable, 'main.py', '--help']
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        
        # Check standard CLI parameters exist in help output
        self.assertIn("--tts-engine", res.stdout)
        self.assertIn("--eq-low", res.stdout)
        self.assertIn("--noise-gate", res.stdout)
        self.assertIn("--burn-subtitles", res.stdout)
        self.assertIn("--compress", res.stdout)

    def test_cli_invalid_youtube_url_exit(self):
        """Simulates command line crash behavior when an invalid YouTube URL is passed."""
        cmd = [sys.executable, 'main.py', 'https://invalid_url.com', '--tts-engine', 'edge-tts']
        
        # Should fail with exit code 1 due to validation error
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        self.assertEqual(res.returncode, 1)

if __name__ == "__main__":
    unittest.main()

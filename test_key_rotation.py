import os
import unittest
from unittest.mock import patch, MagicMock
import httpx
from google.genai import errors

# Configure environment with dummy keys before importing config
os.environ["GEMINI_API_KEY_1"] = "mock_key_1"
os.environ["GEMINI_API_KEY_2"] = "mock_key_2"
os.environ["GEMINI_API_KEY_3"] = "mock_key_3"

from dubber.config import GeminiClientManager

class TestKeyRotation(unittest.TestCase):
    def test_key_rotation_flow(self):
        manager = GeminiClientManager()
        self.assertEqual(len(manager.api_keys), 3)
        self.assertEqual(manager.current_index, 0)
        
        # Test rotating manually
        self.assertTrue(manager.rotate_key())
        self.assertEqual(manager.current_index, 1)
        
        self.assertTrue(manager.rotate_key())
        self.assertEqual(manager.current_index, 2)
        
        self.assertTrue(manager.rotate_key())
        self.assertEqual(manager.current_index, 0)

    @patch('google.genai.Client')
    def test_execute_with_retry_rotation(self, mock_client_class):
        # Configure the mock Client class to instantiate with a mock object containing the key
        def mock_init(api_key):
            client = MagicMock()
            client.api_key = api_key # Add a custom attribute for tracking in the test
            return client
            
        mock_client_class.side_effect = mock_init
        
        manager = GeminiClientManager()
        manager.current_index = 0
        
        calls = []
        def mock_operation(client, *args, **kwargs):
            calls.append(client.api_key)
            if len(calls) == 1:
                # Throw a 429 Rate Limit error on the first call
                resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
                raise errors.APIError(429, resp)
            return "success_result"

        result = manager.execute_with_retry(mock_operation)
        
        self.assertEqual(result, "success_result")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], "mock_key_1")
        self.assertEqual(calls[1], "mock_key_2")
        self.assertEqual(manager.current_index, 1)

if __name__ == "__main__":
    unittest.main()

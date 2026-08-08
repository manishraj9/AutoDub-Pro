import os
import httpx
from google import genai
from google.genai import errors
from dotenv import load_dotenv

# Load workspace .env if it exists
load_dotenv()

# Default models (can be overridden via environment variables)
# gemini-3.1-flash-lite is recommended due to higher daily limits (500 RPD vs 20 RPD for flash)
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.1-flash-lite")
DEFAULT_TRANSLATION_MODEL = os.getenv("GEMINI_TRANSLATION_MODEL", "gemini-3.1-flash-lite")

# Default speech synthesis voice
# edge-tts voice list includes high-quality options like en-US-EmmaMultilingualNeural, en-GB-SoniaNeural, etc.
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "en-US-EmmaMultilingualNeural")

# Temporary directories
TEMP_DIR = os.path.join(os.getcwd(), "temp_dubbing")

# ElevenLabs API configuration for voice cloning
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# F5-TTS API server URL for voice cloning
F5_TTS_URL = os.getenv("F5_TTS_URL", "")

# Caching directories for translation serialization
CACHE_DIR = os.path.join(os.getcwd(), "cache_dubbing")

# Default video compression Constant Rate Factor (CRF) - lower is better quality (e.g., 18-28)
DEFAULT_CRF = int(os.getenv("DEFAULT_CRF", "24"))

# Audio Equalizer default bands (in dB)
DEFAULT_EQ_LOW = float(os.getenv("DEFAULT_EQ_LOW", "0.0"))
DEFAULT_EQ_MID = float(os.getenv("DEFAULT_EQ_MID", "0.0"))
DEFAULT_EQ_HIGH = float(os.getenv("DEFAULT_EQ_HIGH", "0.0"))

# Noise Gate default threshold (in dB)
DEFAULT_NOISE_GATE = float(os.getenv("DEFAULT_NOISE_GATE", "-45.0"))

class GeminiClientManager:
    def __init__(self):
        self.api_keys = []
        
        # Check standard GEMINI_API_KEY first
        std_key = os.getenv("GEMINI_API_KEY")
        if std_key:
            self.api_keys.append(std_key)
            
        # Check GEMINI_API_KEY_1 to GEMINI_API_KEY_10
        for i in range(1, 11):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key and key not in self.api_keys:
                self.api_keys.append(key)
            
        self.current_index = 0
        if not self.api_keys:
            print("[Warning] No Gemini API keys found in environment. Please set GEMINI_API_KEY or GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in your .env file.")

    def get_client(self) -> genai.Client:
        if not self.api_keys:
            raise ValueError("No Gemini API keys configured. Set them in your environment or .env file.")
        
        key = self.api_keys[self.current_index]
        return genai.Client(api_key=key)

    def rotate_key(self) -> bool:
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        print(f"[Info] Rotating Gemini API key to key index {self.current_index + 1}/{len(self.api_keys)}...")
        return True

    def execute_with_retry(self, operation_func, *args, **kwargs):
        """
        Executes a GenAI operation. If it fails with a rate limit, quota limit, or other
        transient API error, rotates the key and retries (up to the number of keys available).
        """
        max_retries = max(1, len(self.api_keys))
        last_error = None
        
        for attempt in range(max_retries):
            try:
                client = self.get_client()
                return operation_func(client, *args, **kwargs)
            except errors.APIError as e:
                status_code = getattr(e, 'code', None)
                message = getattr(e, 'message', str(e)).lower()
                
                is_quota_or_rate_limit = (
                    status_code == 429 or 
                    "quota" in message or 
                    "rate limit" in message or 
                    "resource exhausted" in message
                )
                
                if is_quota_or_rate_limit:
                    print(f"[Warning] Gemini API Key {self.current_index + 1} hit rate/quota limit: {e}")
                    if self.rotate_key():
                        last_error = e
                        continue
                raise e
            except httpx.RequestError as e:
                print(f"[Warning] Network error using Gemini API Key {self.current_index + 1}: {e}")
                if self.rotate_key():
                    last_error = e
                    continue
                raise e
        
        raise ValueError(f"All Gemini API keys exhausted or rate-limited. Last error: {last_error}")

# Global instance of client manager
client_manager = GeminiClientManager()

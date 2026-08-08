import os
import requests
from typing import Optional

def synthesize_f5_tts(
    server_url: str,
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    output_path: str
) -> bool:
    """
    Sends a request to the tunneled F5-TTS FastAPI server to perform voice cloning.
    Saves the generated WAV response to output_path.
    """
    if not server_url:
        print("[Error] F5-TTS Server URL is not configured.")
        return False
        
    # Clean up the URL format (strip trailing slash)
    server_url = server_url.rstrip("/")
    endpoint = f"{server_url}/synthesize"
    
    if not os.path.exists(ref_audio_path):
        print(f"[Error] Reference audio sample not found at: {ref_audio_path}")
        return False
        
    print(f"[Info] Calling F5-TTS synthesis at: {endpoint}...")
    print(f"  - Ref Text: \"{ref_text}\"")
    print(f"  - Gen Text: \"{gen_text}\"")
    
    try:
        with open(ref_audio_path, "rb") as f:
            files = {
                "ref_audio": (os.path.basename(ref_audio_path), f, "audio/wav")
            }
            data = {
                "ref_text": ref_text,
                "gen_text": gen_text
            }
            
            # Send POST request to F5-TTS server
            response = requests.post(endpoint, files=files, data=data, timeout=60.0)
            
        if response.status_code != 200:
            print(f"[Error] F5-TTS synthesis server returned status {response.status_code}: {response.text}")
            return False
            
        # Write response content to output file
        with open(output_path, "wb") as out_f:
            out_f.write(response.content)
            
        print(f"[Success] F5-TTS synthesized segment successfully. Saved to {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to connect to F5-TTS server at {endpoint}: {e}")
        return False
    except Exception as e:
        print(f"[Error] Unexpected error during F5-TTS voice synthesis: {e}")
        return False

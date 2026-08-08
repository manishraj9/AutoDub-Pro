import os
import json
import time
import urllib.request
import subprocess
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from pydub import AudioSegment, silence
from google import genai
from google.genai import errors
import edge_tts

def calculate_speaking_rate(text: str, duration_sec: float) -> float:
    """
    Calculates the speaking rate of a segment in Words Per Minute (WPM).
    Standard conversational rate is 110-150 WPM. Above 180 is flagged as fast.
    """
    if duration_sec <= 0.05:
        return 0.0
    words = len(text.strip().split())
    wpm = (words / duration_sec) * 60.0
    return round(wpm, 1)

def detect_silence_ranges(audio_path: str, silence_thresh_dbfs: int = -40, min_silence_len_ms: int = 1000) -> List[Tuple[float, float]]:
    """
    Analyzes an audio file and detects periods of silence.
    Returns:
        List[Tuple[float, float]]: A list of (start_seconds, end_seconds) silence blocks.
    """
    print(f"[Info] Scanning {audio_path} for silence (Threshold: {silence_thresh_dbfs} dBFS, Min Length: {min_silence_len_ms}ms)...")
    if not os.path.exists(audio_path):
        return []
        
    audio = AudioSegment.from_wav(audio_path)
    silent_ranges = silence.detect_silence(
        audio, 
        min_silence_len=min_silence_len_ms, 
        silence_thresh=silence_thresh_dbfs
    )
    
    # Convert milliseconds to seconds
    result = []
    for start, end in silent_ranges:
        result.append((round(start / 1000.0, 2), round(end / 1000.0, 2)))
        
    print(f"[Info] Found {len(result)} silence gaps.")
    return result

def compress_video_for_web(video_path: str, output_path: str, crf: int = 24) -> None:
    """
    Re-encodes the final output video track using libx264 with Constant Rate Factor (CRF)
    to compress file size for easier sharing while keeping high visual fidelity.
    """
    print(f"[Info] Compressing video {video_path} to {output_path} (CRF Level: {crf})...")
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-c:a', 'copy',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg video compression failed: {result.stderr}")
    print("[Info] Compression completed successfully.")

def check_api_keys_validity(api_keys: List[str]) -> List[bool]:
    """
    Validates a list of Gemini API keys by running cheap pings.
    Returns:
        List[bool]: True for each valid key, False for invalid.
    """
    results = []
    print(f"[Info] Testing validity of {len(api_keys)} Gemini API keys...")
    for idx, key in enumerate(api_keys):
        if not key or key.startswith("your_"):
            results.append(False)
            continue
            
        try:
            client = genai.Client(api_key=key)
            client.models.list()
            results.append(True)
            print(f"  - Key {idx + 1}: Valid")
        except errors.APIError as e:
            print(f"  - Key {idx + 1}: Invalid (APIError: {e})")
            results.append(False)
        except Exception as e:
            print(f"  - Key {idx + 1}: Invalid (Error: {e})")
            results.append(False)
    return results

async def get_available_edge_voices_async() -> List[Dict[str, str]]:
    """Pulls all voices from Edge-TTS asynchronously."""
    try:
        voices = await edge_tts.list_voices()
        result = []
        for v in voices:
            result.append({
                "name": v["Name"],
                "short_name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"]
            })
        result.sort(key=lambda x: x["short_name"])
        return result
    except Exception as e:
        print(f"[Warning] Failed to fetch voices from Edge-TTS service: {e}")
        return []

def get_available_edge_voices() -> List[Dict[str, str]]:
    """Helper wrapper to execute async voices list in sync environments."""
    return asyncio.run(get_available_edge_voices_async())

def download_youtube_thumbnail(url: str, output_path: str) -> str:
    """
    Fetches the video thumbnail image from YouTube to display in our Web dashboard.
    """
    video_id = ""
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
        
    if not video_id:
        print("[Warning] Could not extract YouTube video ID for thumbnail download.")
        return ""
        
    thumb_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(thumb_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"[Info] YouTube thumbnail saved to: {output_path}")
        return output_path
    except Exception:
        fallback_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        try:
            req = urllib.request.Request(fallback_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(output_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"[Info] YouTube thumbnail (fallback) saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"[Warning] Failed to download YouTube thumbnail: {e}")
            return ""

def check_internet_speed() -> Dict[str, Any]:
    """
    Runs a cheap ping to verify network latency.
    """
    host = "8.8.8.8"
    print(f"[Info] Pinging DNS host {host} to verify latency...")
    try:
        start_t = time.time()
        cmd = ["ping", "-n", "2", host]
        subprocess.run(cmd, capture_output=True, check=True)
        latency = (time.time() - start_t) / 2.0 * 1000.0
        return {
            "status": "online",
            "latency_ms": round(latency, 1)
        }
    except Exception:
        return {
            "status": "offline",
            "latency_ms": 9999.0
        }

def analyze_system_ffmpeg_codecs() -> Dict[str, List[str]]:
    """
    Queries local FFmpeg installation configurations to identify supported
    audio encoders and video decoders, ensuring full capability.
    """
    encoders = []
    decoders = []
    try:
        # Get encoders list
        res_enc = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, check=True)
        for line in res_enc.stdout.splitlines():
            line = line.strip()
            if line.startswith("A") or line.startswith("V"):
                parts = line.split()
                if len(parts) >= 2:
                    encoders.append(parts[1])
                    
        # Get decoders list
        res_dec = subprocess.run(['ffmpeg', '-decoders'], capture_output=True, text=True, check=True)
        for line in res_dec.stdout.splitlines():
            line = line.strip()
            if line.startswith("A") or line.startswith("V"):
                parts = line.split()
                if len(parts) >= 2:
                    decoders.append(parts[1])
    except Exception as e:
        print(f"[Warning] Failed to query FFmpeg codecs configuration: {e}")
        
    return {
        "audio_video_encoders": sorted(encoders),
        "audio_video_decoders": sorted(decoders)
    }

def verify_audio_sample_rate_conversion(input_wav: str, output_wav: str, target_rate: int = 16000) -> bool:
    """
    Checks if a WAV audio track requires sampling rate conversion (e.g. converting 44.1kHz to 16kHz
    for Gemini transcription models) and converts it using FFmpeg standard resamplers.
    """
    if not os.path.exists(input_wav):
        return False
        
    cmd = [
        'ffmpeg', '-y',
        '-i', input_wav,
        '-ar', str(target_rate),
        '-ac', '1',  # Force mono layout for transcription efficiency
        output_wav
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[Warning] Resampling conversion failed: {e}")
        return False

def generate_diagnostic_report(temp_dir: str = "temp_dubbing", cache_dir: str = "cache_dubbing") -> Dict[str, Any]:
    """
    Scans the system environment and directories to generate a diagnostic report.
    Checks:
    - FFmpeg availability
    - FFprobe availability
    - Network status
    - Write permissions in temp and cache dirs
    - Available Gemini keys count
    - Estimated cache entries
    """
    from dubber.config import client_manager
    from dubber.validator import validate_ffmpeg_installed, validate_write_permissions
    
    print("[Info] Compiling system diagnosis report...")
    
    # 1. Check FFmpeg/FFprobe
    ffmpeg_ok, ffmpeg_msg = validate_ffmpeg_installed()
    
    # 2. Check Network
    net = check_internet_speed()
    
    # 3. Check Write Perms
    temp_ok = validate_write_permissions(temp_dir)
    cache_ok = validate_write_permissions(cache_dir)
    
    # 4. Check API Keys
    keys_count = len(client_manager.api_keys)
    
    # 5. Check Cache File Sizes
    cache_files = []
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            p = os.path.join(cache_dir, f)
            if os.path.isfile(p):
                cache_files.append({
                    "filename": f,
                    "size_bytes": os.path.getsize(p)
                })
                
    # 6. Analyze codecs
    codecs = analyze_system_ffmpeg_codecs()
    
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ffmpeg_status": {
            "installed": ffmpeg_ok,
            "message": ffmpeg_msg
        },
        "network_status": net,
        "directories": {
            "temp_dir_writable": temp_ok,
            "cache_dir_writable": cache_ok
        },
        "api_keys": {
            "configured_count": keys_count
        },
        "cache_files": cache_files,
        "codec_info": {
            "has_aac_encoder": "aac" in codecs["audio_video_encoders"],
            "has_libx264_encoder": "libx264" in codecs["audio_video_encoders"]
        }
    }

def detect_source_language(audio_path: str, model: str = "gemini-3.1-flash-lite") -> str:
    """
    Analyzes a short sample of the audio file using Gemini to detect the spoken language.
    """
    from dubber.config import client_manager, TEMP_DIR
    
    if not os.path.exists(audio_path):
        return "Unknown"
        
    print(f"[Info] Detecting source language of audio {audio_path}...")
    
    # Extract first 15 seconds of audio to detect language
    sample_path = os.path.join(TEMP_DIR, "lang_detect_sample.wav")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-t', '15',
        '-ar', '16000',
        '-ac', '1',
        sample_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    if not os.path.exists(sample_path):
        return "Unknown"
        
    try:
        def op(client):
            file_ref = client.files.upload(file=sample_path)
            while file_ref.state.name == "PROCESSING":
                time.sleep(1)
                file_ref = client.files.get(name=file_ref.name)
            
            prompt = (
                "Identify the primary spoken language in the audio. "
                "Return ONLY the standard English name of the language (e.g., 'French', 'German', 'Spanish', 'Hindi', 'Japanese'). "
                "Do not include any other words, punctuation, or formatting."
            )
            
            res = client.models.generate_content(
                model=model,
                contents=[file_ref, prompt]
            )
            
            try:
                client.files.delete(name=file_ref.name)
            except Exception:
                pass
                
            return res.text.strip().strip('"').strip("'")
            
        detected = client_manager.execute_with_retry(op)
        print(f"[Success] Detected language: {detected}")
        return detected
    except Exception as e:
        print(f"[Warning] Language auto-detection failed: {e}. Defaulting to 'Unknown'.")
        return "Unknown"
    finally:
        if os.path.exists(sample_path):
            os.remove(sample_path)

def benchmark_api_keys(api_keys: List[str]) -> List[Dict[str, Any]]:
    """
    Tests and benchmarks latency (round-trip time) of all configured Gemini API keys.
    Returns:
        List[Dict[str, Any]]: List of results with key index, status, and latency.
    """
    results = []
    print(f"[Info] Benchmarking RTT latency for {len(api_keys)} Gemini API keys...")
    for idx, key in enumerate(api_keys):
        if not key or key.startswith("your_"):
            results.append({"index": idx + 1, "status": "unconfigured", "latency_ms": -1.0})
            continue
            
        try:
            client = genai.Client(api_key=key)
            start_t = time.time()
            client.models.list()
            latency = (time.time() - start_t) * 1000.0
            results.append({
                "index": idx + 1,
                "status": "valid",
                "latency_ms": round(latency, 1)
            })
            print(f"  - Key {idx + 1}: Valid, Latency: {latency:.1f}ms")
        except Exception as e:
            results.append({
                "index": idx + 1,
                "status": f"error: {str(e)}",
                "latency_ms": -1.0
            })
            print(f"  - Key {idx + 1}: Invalid/Error: {e}")
    return results

def save_custom_voice_preset(preset_name: str, voice_map: Dict[str, str], filepath: str = None) -> bool:
    """
    Saves a mapping of Speaker IDs to Voice IDs/names to a JSON preset file.
    """
    if not filepath:
        preset_dir = os.path.join(os.getcwd(), "cache_dubbing", "presets")
        os.makedirs(preset_dir, exist_ok=True)
        filepath = os.path.join(preset_dir, f"{preset_name}.json")
        
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(voice_map, f, indent=4)
        print(f"[Success] Voice preset '{preset_name}' saved to {filepath}")
        return True
    except Exception as e:
        print(f"[Error] Failed to save voice preset: {e}")
        return False

def load_custom_voice_preset(preset_name: str, filepath: str = None) -> Optional[Dict[str, str]]:
    """
    Loads a voice mapping preset JSON file.
    """
    if not filepath:
        preset_dir = os.path.join(os.getcwd(), "cache_dubbing", "presets")
        filepath = os.path.join(preset_dir, f"{preset_name}.json")
        
    if not os.path.exists(filepath):
        print(f"[Warning] Voice preset '{preset_name}' not found at {filepath}")
        return None
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Success] Loaded voice preset '{preset_name}' with {len(data)} mapping(s).")
        return data
    except Exception as e:
        print(f"[Error] Failed to load voice preset: {e}")
        return None



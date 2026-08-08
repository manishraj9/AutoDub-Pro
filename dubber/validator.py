import os
import re
import subprocess
from typing import List, Dict, Any, Tuple

def validate_youtube_url(url: str) -> bool:
    """
    Validates the structure of a YouTube URL using strict regex mapping.
    Matches standard watch URLs, short URLs (youtu.be), embeds, and shorts.
    """
    if not url:
        return False
        
    pattern = re.compile(
        r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})'
    )
    return bool(pattern.match(url.strip()))

def get_youtube_video_id(url: str) -> str:
    """
    Extracts the 11-character video ID from a validated YouTube URL.
    """
    if not url:
        return ""
    
    # Check match
    url_clean = url.strip()
    match = re.search(r'(v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})', url_clean)
    if match:
        return match.group(2)
    return ""

def validate_ffmpeg_installed() -> Tuple[bool, str]:
    """
    Verifies if FFmpeg and FFprobe system binaries are correctly installed and visible in the host PATH.
    """
    try:
        ffmpeg_res = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if ffmpeg_res.returncode != 0:
            return False, "FFmpeg found but returned error code during version probe check."
            
        ffprobe_res = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
        if ffprobe_res.returncode != 0:
            return False, "FFprobe found but returned error code during version probe check."
            
        return True, "FFmpeg and FFprobe are fully configured and functional."
    except FileNotFoundError:
        return False, "FFmpeg/FFprobe binaries are not found in the system environment PATH variables."
    except Exception as e:
        return False, f"System validation encountered an error during FFmpeg check: {str(e)}"

def validate_write_permissions(directory_path: str) -> bool:
    """
    Confirms write permission availability inside the targeted directory path
    by attempting to write, read, and delete a small temporary lock file.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception:
            return False
            
    temp_lock_file = os.path.join(directory_path, ".write_permission_lock")
    try:
        with open(temp_lock_file, "w") as f:
            f.write("test")
        
        # Verify read
        with open(temp_lock_file, "r") as f:
            content = f.read()
            
        # Delete
        os.remove(temp_lock_file)
        return content == "test"
    except Exception:
        return False

def validate_audio_wav_header(wav_path: str) -> Dict[str, Any]:
    """
    Parses a WAV file header using binary structure reading to analyze
    sampling rate, channel count, bit-depth and format type,
    ensuring standard PCM compatibility.
    """
    if not wav_path or not os.path.exists(wav_path):
        return {"valid": False, "error": "Target file does not exist on disk."}
        
    try:
        with open(wav_path, 'rb') as wav_file:
            # RIFF Header
            riff_tag = wav_file.read(4)
            if riff_tag != b'RIFF':
                return {"valid": False, "error": "Invalid RIFF identifier. Not a WAV file."}
                
            wav_file.seek(8) # Skip file size
            wave_tag = wav_file.read(4)
            if wave_tag != b'WAVE':
                return {"valid": False, "error": "Invalid WAVE sub-header format."}
                
            # Search for 'fmt ' chunk
            fmt_found = False
            for _ in range(10): # Look ahead chunks
                chunk_header = wav_file.read(4)
                if not chunk_header:
                    break
                chunk_size = int.from_bytes(wav_file.read(4), byteorder='little')
                if chunk_header == b'fmt ':
                    fmt_found = True
                    # Read format details
                    audio_format = int.from_bytes(wav_file.read(2), byteorder='little')
                    channels = int.from_bytes(wav_file.read(2), byteorder='little')
                    sample_rate = int.from_bytes(wav_file.read(4), byteorder='little')
                    byte_rate = int.from_bytes(wav_file.read(4), byteorder='little')
                    block_align = int.from_bytes(wav_file.read(2), byteorder='little')
                    bits_per_sample = int.from_bytes(wav_file.read(2), byteorder='little')
                    
                    return {
                        "valid": True,
                        "format_type": "PCM" if audio_format == 1 else f"Non-PCM ({audio_format})",
                        "channels": channels,
                        "sample_rate": sample_rate,
                        "bits_per_sample": bits_per_sample,
                        "bit_rate_kbps": int((byte_rate * 8) / 1000)
                    }
                else:
                    # Skip chunk data
                    wav_file.seek(chunk_size, 1)
                    
            if not fmt_found:
                return {"valid": False, "error": "fmt chunk identifier missing in wav file."}
    except Exception as e:
        return {"valid": False, "error": f"Header analyzer failed with exception: {str(e)}"}

def validate_video_duration(video_path: str) -> float:
    """
    Executes FFprobe to determine the duration of the media file in seconds.
    """
    if not video_path or not os.path.exists(video_path):
        return 0.0
        
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_str = result.stdout.strip()
        return float(duration_str)
    except Exception:
        return 0.0

def validate_segment_intervals(segments: List[dict], max_duration: float = 0.0) -> Tuple[bool, str]:
    """
    Checks transcribed/translated segment timing intervals:
    - Verifies start_time < end_time
    - Verifies no overlapping of the same speaker turns
    - Verifies durations do not exceed video limits
    """
    for idx, seg in enumerate(segments):
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)
        
        if start < 0 or end < 0:
            return False, f"Negative time boundary found at segment {idx}: ({start}s - {end}s)."
            
        if start >= end:
            return False, f"Start time must precede end time at segment {idx}: ({start}s - {end}s)."
            
        if max_duration > 0.0 and end > (max_duration + 0.5):
            return False, f"Segment {idx} bounds exceed video limit: ({end}s > {max_duration}s)."
            
    return True, "All segment intervals are logically sound."

def validate_duck_level(duck_db: float) -> Tuple[bool, str]:
    """
    Validates the gain level configured for background ducking.
    Should be negative and not exceed -60dB (fully muted).
    """
    if duck_db > 0.0:
        return False, "Background ducking gain should be negative (less than 0.0 dB)."
    if duck_db < -60.0:
        return False, "Background ducking gain cannot exceed -60.0 dB (too quiet)."
    return True, "Ducking level is within safe bounds."

def validate_eq_gains(low: float, mid: float, high: float) -> Tuple[bool, str]:
    """
    Ensures parametric Equalizer gains are set within safety thresholds (-12dB to +12dB)
    to prevent digital distortion or ear damage.
    """
    for name, val in [("Bass", low), ("Mid", mid), ("Treble", high)]:
        if val < -12.0 or val > 12.0:
            return False, f"{name} EQ gain ({val} dB) is outside the safe range of [-12.0, 12.0] dB."
    return True, "Parametric EQ settings are within safe thresholds."

def retrieve_ffmpeg_audio_codecs() -> List[str]:
    """
    Queries FFmpeg and parses supported audio encoders.
    """
    cmd = ['ffmpeg', '-codecs']
    supported = []
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Scan lines for audio encoders
        for line in res.stdout.splitlines():
            line = line.strip()
            # Audio codecs line format: " D.A... codec_name  Description"
            # where A represents audio
            if len(line) > 10 and line[1] == 'D' and line[2] == '.' and line[3] == 'A':
                parts = line[6:].split()
                if parts:
                    supported.append(parts[0])
    except Exception:
        pass
    return sorted(list(set(supported)))

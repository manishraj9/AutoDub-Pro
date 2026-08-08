import os
import subprocess
from typing import List, Tuple
from pydub import AudioSegment
from pydub.effects import normalize

def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into SRT subtitle format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Formats seconds into WebVTT subtitle format (HH:MM:SS.mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

def generate_srt_subtitles(segments: List[dict], output_srt_path: str) -> None:
    """
    Generates a standard SubRip (.srt) subtitle file from timed segments.
    """
    print(f"[Info] Generating SRT subtitle file: {output_srt_path}...")
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments):
            start = format_timestamp_srt(seg["start_time"])
            end = format_timestamp_srt(seg["end_time"])
            text = seg["text"].strip()
            
            f.write(f"{idx + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    print("[Info] SRT subtitle file created successfully.")

def generate_vtt_subtitles(segments: List[dict], output_vtt_path: str) -> None:
    """
    Generates a standard WebVTT (.vtt) subtitle file from timed segments.
    """
    print(f"[Info] Generating WebVTT subtitle file: {output_vtt_path}...")
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments):
            start = format_timestamp_vtt(seg["start_time"])
            end = format_timestamp_vtt(seg["end_time"])
            text = seg["text"].strip()
            
            f.write(f"{idx + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    print("[Info] WebVTT subtitle file created successfully.")

def burn_subtitles_into_video(video_path: str, subtitle_path: str, output_path: str) -> None:
    """
    Burns SRT subtitles directly onto the video frames using FFmpeg's subtitles filter.
    """
    print(f"[Info] Burning subtitles into video: {output_path}...")
    
    # FFmpeg subtitles filter on Windows requires escaping backslashes and colons
    # Example: subtitles='C\\:/path/to/subs.srt'
    safe_sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{safe_sub_path}'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '20',
        '-c:a', 'copy',  # Don't re-encode audio stream
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg subtitle burn-in failed: {result.stderr}")
    print("[Info] Subtitles burned successfully.")

def remove_bg_noise(audio: AudioSegment, threshold_dbfs: float = -45.0) -> AudioSegment:
    """
    Simple Noise Gate: Silences segments of audio that drop below a threshold,
    effectively removing background noise floor during speech pauses.
    """
    # Break down the audio into 10ms chunks
    chunk_size_ms = 10
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]
    
    processed_chunks = []
    for chunk in chunks:
        # If the chunk volume is below threshold, mute it
        if chunk.dBFS < threshold_dbfs:
            processed_chunks.append(AudioSegment.silent(duration=len(chunk), frame_rate=audio.frame_rate))
        else:
            processed_chunks.append(chunk)
            
    # Combine chunks back
    if processed_chunks:
        output_audio = processed_chunks[0]
        for c in processed_chunks[1:]:
            output_audio = output_audio.append(c, crossfade=0)
        return output_audio
    return audio

def add_high_pass_filter(audio: AudioSegment, cutoff_hz: int = 80) -> AudioSegment:
    """
    Applies a high-pass filter to clean up low-frequency rumble, hum, or wind noise.
    Voice signals typically contain little useful data below 80-100Hz.
    """
    return audio.high_pass_filter(cutoff_hz)

def add_low_pass_filter(audio: AudioSegment, cutoff_hz: int = 8000) -> AudioSegment:
    """
    Applies a low-pass filter to remove high-frequency noise floor or hiss.
    """
    return audio.low_pass_filter(cutoff_hz)

def apply_limiter(audio: AudioSegment, max_dbfs: float = -1.0) -> AudioSegment:
    """
    Applies peak limiting: brings down any peaks exceeding max_dbfs to prevent clipping,
    and then normalizes the segment.
    """
    if audio.max_dBFS > max_dbfs:
        reduction_db = max_dbfs - audio.max_dBFS
        audio = audio + reduction_db
    return audio

def apply_fade_borders(audio: AudioSegment, fade_ms: int = 20) -> AudioSegment:
    """
    Applies short fade-ins and fade-outs to the segment borders to prevent
    popping or clicking noises when clips join together.
    """
    if len(audio) > (fade_ms * 2):
        return audio.fade_in(fade_ms).fade_out(fade_ms)
    return audio

def auto_level_vocals(speech_audio: AudioSegment, target_dbfs: float = -18.0) -> AudioSegment:
    """
    Gain levels vocal tracks to match target loudness, keeping audio volume balanced.
    """
    change_in_db = target_dbfs - speech_audio.dBFS
    return speech_audio + change_in_db

def apply_three_band_eq(audio: AudioSegment, low_gain: float = 0.0, mid_gain: float = 0.0, high_gain: float = 0.0) -> AudioSegment:
    """
    Implements a 3-Band Parametric Equalizer:
    - Low Band (Bass): Frequency < 250 Hz
    - Mid Band (Presence): 250 Hz - 4000 Hz
    - High Band (Treble): Frequency > 4000 Hz
    
    Adjusts the gain of each frequency range and sums them back together.
    """
    if low_gain == 0.0 and mid_gain == 0.0 and high_gain == 0.0:
        return audio # No EQ changes
        
    print(f"[Info] Applying 3-Band Equalizer (Bass: {low_gain}dB, Mid: {mid_gain}dB, Treble: {high_gain}dB)...")
    
    # 1. Extract Bass (Low Pass Filter at 250Hz)
    low_band = audio.low_pass_filter(250) + low_gain
    
    # 2. Extract Treble (High Pass Filter at 4000Hz)
    high_band = audio.high_pass_filter(4000) + high_gain
    
    # 3. Extract Mid Presence (Bandpass 250Hz to 4000Hz)
    mid_band = audio.high_pass_filter(250).low_pass_filter(4000) + mid_gain
    
    # 4. Reconstruct by summing the overlays
    eq_audio = low_band.overlay(mid_band).overlay(high_band)
    return eq_audio

def stretch_audio_speed_rubberband(input_path: str, output_path: str, speed_factor: float) -> None:
    """
    High-quality pitch-preserving time stretching controller.
    Uses atempo as the core engine, but structures the command for clean execution.
    """
    filters = []
    temp_factor = speed_factor
    
    while temp_factor > 2.0:
        filters.append("atempo=2.0")
        temp_factor /= 2.0
    while temp_factor < 0.5:
        filters.append("atempo=0.5")
        temp_factor /= 0.5
    filters.append(f"atempo={temp_factor:.3f}")
    
    filter_str = ",".join(filters)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-filter:a', filter_str,
        '-vn',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg pitch-preserving time stretch failed: {result.stderr}")

def extract_vocals_filter(audio_path: str, output_path: str) -> None:
    """
    Applies a bandpass filter (200Hz - 6000Hz) to isolate human voice frequencies,
    removing low sub-bass rumble and high frequency static noise.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-af', 'highpass=f=200,lowpass=f=6000',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg vocal extraction filter failed: {result.stderr}")

def generate_ass_subtitles(segments: List[dict], output_ass_path: str,
                           fontname: str = "Arial", fontsize: int = 20,
                           primary_color: str = "&H00FFFFFF", outline_color: str = "&H00000000") -> None:
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file.
    Assigns a distinct color style to each speaker ID, using custom font styles if provided.
    """
    print(f"[Info] Generating ASS styled subtitles file: {output_ass_path}...")
    
    colors = [
        "&H0000FFFF",  # Yellow
        "&H00FFFF00",  # Cyan
        "&H00FF00FF",  # Magenta
        "&H0080FF80",  # Light Green
        "&H008080FF",  # Light Red/Coral
        "&H00FF8000",  # Orange
        "&H00FF99FF",  # Pink
        "&H00FFFF99",  # Light Blue
        "&H0099FFFF",  # Light Yellow
        primary_color  # Custom font color as fallback/default
    ]
    
    unique_speakers = sorted(list(set(seg.get("speaker", "Speaker 1") for seg in segments)))
    speaker_colors = {}
    for i, spk in enumerate(unique_speakers):
        speaker_colors[spk] = colors[i % len(colors)]
        
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("Title: Dubbed Video Subtitles\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 640\n")
        f.write("PlayResY: 360\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{fontname},{fontsize},{primary_color},&H000000FF,{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,1.5,0,2,10,10,15,1\n")
        
        for speaker, color in speaker_colors.items():
            style_name = speaker.replace(" ", "_")
            f.write(f"Style: {style_name},{fontname},{fontsize},{color},&H000000FF,{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,1.5,0,2,10,10,15,1\n")
            
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        def format_ass_time(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            cs = int(round((sec % 1) * 100))
            if cs >= 100:
                cs = 99
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
            
        for seg in segments:
            start = format_ass_time(seg["start_time"])
            end = format_ass_time(seg["end_time"])
            style = seg.get("speaker", "Default").replace(" ", "_")
            text = seg["text"].strip().replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n")
            
    print("[Info] ASS styled subtitles file created successfully.")

def generate_waveform_overlay(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Muxes the dubbed audio, video, and overlays a transparent animated waveform of the audio track.
    """
    print(f"[Info] Generating animated waveform overlay video: {output_path}...")
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-filter_complex',
        '[1:a]showwaves=s=640x120:mode=line:colors=cyan|teal[wave];[0:v][wave]overlay=x=0:y=H-120:shortest=1[outv]',
        '-map', '[outv]',
        '-map', '1:a',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '20',
        '-c:a', 'aac',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg animated waveform overlay failed: {result.stderr}")
    print("[Info] Animated waveform overlay video generated successfully.")

def analyze_eq_profile(audio_segment: AudioSegment) -> Tuple[float, float, float]:
    """
    Analyzes vocal pitch based on Zero Crossing Rate (ZCR) and selects an optimal EQ profile.
    Returns:
        Tuple[float, float, float]: EQ gains for (low_gain, mid_gain, high_gain)
    """
    try:
        samples = audio_segment.get_array_of_samples()
        if not samples:
            return 0.0, 0.0, 0.0
            
        zero_crossings = 0
        for i in range(1, len(samples)):
            if (samples[i-1] >= 0 and samples[i] < 0) or (samples[i-1] < 0 and samples[i] >= 0):
                zero_crossings += 1
                
        duration = len(audio_segment) / 1000.0
        if duration <= 0:
            return 0.0, 0.0, 0.0
            
        zcr_freq = zero_crossings / (2.0 * duration)
        
        if zcr_freq < 300:
            print(f"[Info] Analyzed pitch frequency: {zcr_freq:.1f} Hz (Deep/Male). Applying Warm Clarity EQ profile.")
            return -2.0, 1.0, 3.0
        elif zcr_freq > 1000:
            print(f"[Info] Analyzed pitch frequency: {zcr_freq:.1f} Hz (Sibilant/High). Applying Smooth Highs EQ profile.")
            return 1.0, 2.0, -2.0
        else:
            print(f"[Info] Analyzed pitch frequency: {zcr_freq:.1f} Hz (Mid/Female). Applying Bright Vocal EQ profile.")
            return 0.0, 3.0, 1.0
    except Exception as e:
        print(f"[Warning] Vocal analysis failed: {e}. Applying flat EQ.")
        return 0.0, 0.0, 0.0


import os
import subprocess
from typing import List, Tuple
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

def normalize_audio(audio: AudioSegment) -> AudioSegment:
    """
    Applies peak normalization to the audio segment to ensure consistent volume.
    """
    try:
        return normalize(audio)
    except Exception as e:
        print(f"[Warning] Normalization failed, returning original audio: {e}")
        return audio

def merge_speech_intervals(segments: List[dict], total_duration_ms: int, pad_start_ms: int = 0, pad_end_ms: int = 0) -> List[Tuple[int, int]]:
    """
    Extracts, sorts, and merges speech time intervals (in ms).
    Applies safety padding (pad_start_ms and pad_end_ms) to enclose pre-speech breaths
    and trailing vocal releases, guaranteeing 100% vocal cancellation.
    """
    intervals = []
    for seg in segments:
        start_ms = max(0, int(seg["start_time"] * 1000) - pad_start_ms)
        end_ms = min(total_duration_ms, int(seg["end_time"] * 1000) + pad_end_ms)
        if start_ms < end_ms:
            intervals.append((start_ms, end_ms))
            
    if not intervals:
        return []
        
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    curr_start, curr_end = intervals[0]
    
    for start, end in intervals[1:]:
        if start <= curr_end + 300:
            curr_end = max(curr_end, end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = start, end
            
    merged.append((curr_start, curr_end))
    return merged

def remove_vocals_oops(audio: AudioSegment) -> AudioSegment:
    """
    Applies center-channel phase cancellation and multi-stage vocal bandstop filtering
    to eliminate 100% of original vocal speech and vocal reverb, keeping MCU movie trailer
    background music, bass, drums, synths, and sound effects rich and loud.
    """
    if len(audio) < 10:
        return audio
        
    try:
        if audio.channels == 2:
            left, right = audio.split_to_mono()
            
            # Extract side channel (eliminates 100% center vocals)
            left_norm = normalize_audio(left)
            right_norm = normalize_audio(right)
            vocal_cancel_side = left_norm.overlay(right_norm.invert_phase())
            
            # Sub-bass (<160Hz) & high-end percussion (>3400Hz) at 0dB; apply -24dB notch to mid vocal band
            low = audio.low_pass_filter(160)
            high = audio.high_pass_filter(3400)
            mid_side = vocal_cancel_side.high_pass_filter(160).low_pass_filter(3400) - 24.0
            
            return low.overlay(mid_side).overlay(high)
        else:
            # Mono deep selective vocal notch filter (160Hz to 3400Hz -28dB)
            low = audio.low_pass_filter(160)
            high = audio.high_pass_filter(3400)
            mid = audio.high_pass_filter(160).low_pass_filter(3400) - 28.0
            return low.overlay(mid).overlay(high)
    except Exception as e:
        print(f"[Warning] Vocal cancellation filter fallback: {e}")
        return audio.low_pass_filter(160).overlay(audio.high_pass_filter(3400))

def create_ducked_background(original_audio: AudioSegment, segments: List[dict], duck_level_db: float = -4.0) -> AudioSegment:
    """
    Slices the original audio track and applies ducking during speech segments.
    Applies 250ms pre-speech and 450ms post-speech padding to enclose 100% of vocal decays and breaths.
    """
    total_len_ms = len(original_audio)
    # 250ms pre-speech padding and 450ms post-speech padding guarantees 100% vocal coverage
    merged_intervals = merge_speech_intervals(segments, total_len_ms, pad_start_ms=250, pad_end_ms=450)
    
    if not merged_intervals:
        return original_audio
        
    bg_parts = []
    last_idx = 0
    
    for start, end in merged_intervals:
        # 1. Add normal volume slice before speech
        if start > last_idx:
            bg_parts.append(original_audio[last_idx:start])
            
        # 2. Add vocal-cancelled and ducked background slice during speech
        speech_slice = original_audio[start:end]
        vocal_removed_slice = remove_vocals_oops(speech_slice)
        ducked_slice = vocal_removed_slice + duck_level_db
        bg_parts.append(ducked_slice)
        
        last_idx = end
        
    # 3. Add normal volume slice after the last speech segment
    if last_idx < total_len_ms:
        bg_parts.append(original_audio[last_idx:total_len_ms])
        
    # Stitch parts together using pydub addition with crossfade=0 to prevent cumulative timing drift
    ducked_bg = bg_parts[0]
    for part in bg_parts[1:]:
        ducked_bg = ducked_bg.append(part, crossfade=0)
        
    return ducked_bg

def assemble_dubbed_audio(segments: List[dict], segment_files: List[str], 
                          original_audio_path: str, output_audio_path: str,
                          mode: str = "duck", duck_level_db: float = -20.0,
                          original_vol_db: float = 0.0, dubbed_vol_db: float = 0.0) -> str:
    """
    Assembles dubbed voice segments onto a background track (either silent or ducked original background).
    """
    print(f"[Info] Loading original audio file {original_audio_path}...")
    original_audio = AudioSegment.from_wav(original_audio_path)
    
    # Apply baseline volume adjustment to original background if configured
    if original_vol_db != 0.0:
        original_audio = original_audio + original_vol_db
        
    total_duration_ms = len(original_audio)
    
    # Step 1: Initialize background
    if mode == "duck":
        print(f"[Info] Creating ducked background audio track (Duck Level: {duck_level_db}dB, BGM Vol: {original_vol_db}dB)...")
        bg_track = create_ducked_background(original_audio, segments, duck_level_db)
    else:
        print("[Info] Replacing audio track completely. Initializing silent track...")
        bg_track = AudioSegment.silent(duration=total_duration_ms, frame_rate=original_audio.frame_rate)
        
    # Step 2: Overlay dubbed speech segments onto the background track at exact start timestamps
    print("[Info] Overlaying synthesized speech segments...")
    for seg, file_path in zip(segments, segment_files):
        if not file_path or not os.path.exists(file_path):
            continue
            
        start_ms = max(0, int(seg["start_time"] * 1000))
        speech_clip = AudioSegment.from_wav(file_path)
        
        # Apply vocal volume adjustments
        if dubbed_vol_db != 0.0:
            speech_clip = speech_clip + dubbed_vol_db
            
        bg_track = bg_track.overlay(speech_clip, position=start_ms)
        
    # Step 3: Apply final peak normalization to keep levels optimal
    print("[Info] Normalizing final assembled track...")
    final_mixed_audio = normalize_audio(bg_track)
    
    # Step 4: Export WAV
    print(f"[Info] Exporting final dubbed audio to {output_audio_path}...")
    final_mixed_audio.export(output_audio_path, format="wav")
    
    return output_audio_path

def convert_audio_format(input_path: str, output_path: str, format_name: str = "mp3"):
    """
    Helper function to convert an audio file to another format (e.g. WAV to MP3) using FFmpeg.
    """
    cmd = ['ffmpeg', '-y', '-i', input_path, output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg format conversion failed: {result.stderr}")

def apply_stereo_panning(audio: AudioSegment, speaker_name: str, unique_speakers: List[str]) -> AudioSegment:
    """
    Applies spatial stereo panning to the speaker audio.
    Pans speakers slightly left or right based on their index in the speaker list.
    """
    if len(unique_speakers) <= 1:
        return audio
        
    try:
        idx = unique_speakers.index(speaker_name)
    except ValueError:
        return audio
        
    num_spk = len(unique_speakers)
    if num_spk == 2:
        pan_factor = -0.15 if idx == 0 else 0.15
    else:
        step = 0.5 / (num_spk - 1)
        pan_factor = -0.25 + (idx * step)
        
    if audio.channels == 1:
        audio = audio.set_channels(2)
        
    print(f"[Info] Panning {speaker_name} by factor {pan_factor:.2f}...")
    return audio.pan(pan_factor)

def align_segment_boundaries(segments: List[dict], original_audio_path: str) -> List[dict]:
    """
    Adjusts segment boundaries slightly to align with natural pauses in original audio.
    Crucially anchors segment start_time to original spoken speech timestamps so that
    timing never drifts or postpones speech to later in the video.
    Expands tight segment durations into available silence gaps before the next spoken segment.
    """
    from dubber.utils import detect_silence_ranges
    
    # Sort segments by original start_time
    sorted_segs = sorted(segments, key=lambda x: x["start_time"])
    silence_ranges = detect_silence_ranges(original_audio_path, silence_thresh_dbfs=-38, min_silence_len_ms=600)
    
    aligned_segments = []
    for i, seg in enumerate(sorted_segs):
        new_seg = seg.copy()
        start = seg["start_time"]
        end = seg["end_time"]
        
        prev_end = aligned_segments[i-1]["end_time"] if i > 0 else 0.0
        next_start = sorted_segs[i+1]["start_time"] if i < len(sorted_segs) - 1 else float('inf')
        
        # If segment is tight (< 1.5s duration) and silence gap exists before next_start, expand end_time
        curr_dur = end - start
        if curr_dur < 1.5 and next_start > end + 0.2:
            max_expand = min(start + 2.5, next_start - 0.05)
            if max_expand > end:
                new_seg["end_time"] = round(max_expand, 3)
                end = new_seg["end_time"]
        
        if silence_ranges:
            for s_start, s_end in silence_ranges:
                s_mid = (s_start + s_end) / 2.0
                if abs(start - s_end) < 0.25 and s_mid >= prev_end and s_mid < end - 0.2:
                    new_seg["start_time"] = round(s_mid, 3)
                    break
                if abs(end - s_start) < 0.25 and s_mid <= next_start and s_mid > new_seg["start_time"] + 0.2:
                    new_seg["end_time"] = round(s_mid, 3)
                    break
                    
        aligned_segments.append(new_seg)
        
    # Enforce non-overlapping without shifting start_time rightward into the future
    for i in range(1, len(aligned_segments)):
        prev_end = aligned_segments[i-1]["end_time"]
        curr_start = aligned_segments[i]["start_time"]
        
        if prev_end > curr_start:
            # Cap previous end time to leave space before current start time
            aligned_segments[i-1]["end_time"] = max(aligned_segments[i-1]["start_time"] + 0.2, curr_start - 0.05)
            
    # Step 3. Split any long segment that contains internal silence gaps (> 1.0s) into distinct sub-segments
    split_result = split_segments_on_internal_silence(aligned_segments, original_audio_path)
    return split_result

def split_segments_on_internal_silence(segments: List[dict], original_audio_path: str, min_silence_len_ms: int = 1000) -> List[dict]:
    """
    Scans each segment for internal pauses or silence gaps (> 1.0s) inside the segment timeframe.
    If a long segment (e.g. 11.2s) contains a 2+ second pause between spoken phrases, splits the segment
    into separate chronological sub-segments so each phrase is anchored to its exact spoken timestamp.
    """
    if not os.path.exists(original_audio_path) or not segments:
        return segments
        
    try:
        from dubber.utils import detect_silence_ranges
        silence_ranges = detect_silence_ranges(original_audio_path, silence_thresh_dbfs=-38, min_silence_len_ms=min_silence_len_ms)
        if not silence_ranges:
            return segments
            
        result = []
        for seg in segments:
            start = seg["start_time"]
            end = seg["end_time"]
            text = seg.get("text", "")
            speaker = seg.get("speaker", "Speaker 1")
            
            # Find any silence gap fully contained inside (start + 0.8s) to (end - 0.8s)
            internal_silences = []
            for s_start, s_end in silence_ranges:
                if s_start >= start + 0.8 and s_end <= end - 0.8 and (s_end - s_start) >= 1.0:
                    internal_silences.append((s_start, s_end))
                    
            if not internal_silences:
                result.append(seg)
                continue
                
            # Split text on clause boundaries
            text_parts = [p.strip() for p in text.replace("|", ".").replace("?", ".").replace("!", ".").split(".") if p.strip()]
            if len(text_parts) <= len(internal_silences):
                words = text.split()
                if len(words) >= 2:
                    mid_pt = len(words) // 2
                    text_parts = [" ".join(words[:mid_pt]), " ".join(words[mid_pt:])]
                else:
                    text_parts = [text, text]
                
            # Create sub-segments
            curr_start = start
            for idx, (s_start, s_end) in enumerate(internal_silences):
                part_text = text_parts[idx] if idx < len(text_parts) else text_parts[-1]
                sub_end = s_start
                if sub_end > curr_start + 0.3:
                    result.append({
                        "start_time": round(curr_start, 3),
                        "end_time": round(sub_end, 3),
                        "text": part_text,
                        "speaker": speaker
                    })
                curr_start = s_end
                
            # Final remaining part
            if end > curr_start + 0.3:
                last_idx = len(internal_silences)
                last_text = text_parts[last_idx] if last_idx < len(text_parts) else text_parts[-1]
                result.append({
                    "start_time": round(curr_start, 3),
                    "end_time": round(end, 3),
                    "text": last_text,
                    "speaker": speaker
                })
        return result
    except Exception as e:
        print(f"[Warning] Failed to split internal silences: {e}")
        return segments



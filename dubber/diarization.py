import json
import os
import numpy as np
from typing import List, Dict
from pydantic import BaseModel, Field
from pydub import AudioSegment
from google import genai
from google.genai import types
from dubber.config import client_manager, DEFAULT_TRANSLATION_MODEL

# Expanded catalog of distinct neural dubbing voices
AVAILABLE_DUBBING_VOICES = [
    "en-US-EmmaMultilingualNeural",   # Multilingual Female (soft, natural)
    "en-US-BrianNeural",              # US Male (deep, clear)
    "en-US-AvaNeural",                # US Female (bright, informative)
    "en-US-AndrewNeural",             # US Male (conversational)
    "en-GB-SoniaNeural",              # UK Female (professional)
    "en-GB-RyanNeural",               # UK Male (natural)
    "en-AU-NatashaNeural",            # Australian Female
    "en-AU-WilliamNeural"             # Australian Male
]

class DiarizedSegment(BaseModel):
    start_time: float = Field(description="Must match the input segment start_time exactly")
    end_time: float = Field(description="Must match the input segment end_time exactly")
    text: str = Field(description="The text content of the segment")
    speaker: str = Field(description="Identify who is speaking this segment, suffixed with their gender in parentheses, e.g., 'Speaker 1 (Male)', 'Speaker 2 (Female)', or 'Speaker 3 (Male)'.")

class DiarizationResponse(BaseModel):
    segments: List[DiarizedSegment]

def estimate_pitch_f0(audio_segment: AudioSegment) -> float:
    """
    Estimates the fundamental frequency (F0 pitch in Hz) of an audio segment
    using autocorrelation to accurately identify speaker gender (Male < 165Hz, Female >= 165Hz).
    """
    try:
        samples = np.array(audio_segment.get_array_of_samples(), dtype=float)
        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)
            
        sr = audio_segment.frame_rate
        if len(samples) < int(sr * 0.15):
            return 0.0
            
        win_len = min(len(samples), int(sr * 1.5))
        signal = samples[:win_len]
        signal = signal - np.mean(signal)
        
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr)//2:]
        
        min_lag = int(sr / 350)
        max_lag = int(sr / 70)
        
        if max_lag >= len(corr):
            return 0.0
            
        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        f0 = sr / float(peak_lag)
        return round(f0, 1)
    except Exception:
        return 0.0

def acoustic_speaker_clustering(segments: List[dict], audio_path: str, max_speakers: int = 3) -> List[dict]:
    """
    Analyzes physical acoustic properties (F0 pitch & spectrum) from the WAV audio file
    to assign distinct speaker labels (Speaker 1, Speaker 2, Speaker 3) and gender tags.
    Caps max speakers to max_speakers (default 3) to prevent ghost speaker fragmentation.
    """
    if not os.path.exists(audio_path) or not segments:
        return segments
        
    try:
        audio = AudioSegment.from_wav(audio_path)
        total_len = len(audio)
        
        # Pre-screen pitch frequencies across all segments to determine dominant voice gender mode
        f0_all = []
        for seg in segments:
            start_ms = max(0, int(seg["start_time"] * 1000))
            end_ms = min(total_len, int(seg["end_time"] * 1000))
            if end_ms - start_ms >= 200:
                clip = audio[start_ms:end_ms]
                f0_val = estimate_pitch_f0(clip)
                if f0_val > 0.0:
                    f0_all.append(f0_val)
                    
        overall_female_ratio = (sum(1 for f in f0_all if f >= 168.0) / float(len(f0_all))) if f0_all else 0.0
        force_male_mode = (overall_female_ratio < 0.25)
        
        speaker_profiles = [] # list of (pitch, speaker_id, gender)
        next_spk_id = 1
        
        clustered = []
        for seg in segments:
            start_ms = max(0, int(seg["start_time"] * 1000))
            end_ms = min(total_len, int(seg["end_time"] * 1000))
            
            if end_ms - start_ms < 150:
                clustered.append(seg)
                continue
                
            clip = audio[start_ms:end_ms]
            f0 = estimate_pitch_f0(clip)
            
            gender = "Male"
            if not force_male_mode and f0 >= 168.0:
                gender = "Female"
                
            # Match pitch to existing profiles (within 35 Hz threshold)
            matched_spk = None
            if f0 > 0.0:
                for p_f0, p_id, p_gen in speaker_profiles:
                    if p_gen == gender and abs(f0 - p_f0) < 35.0:
                        matched_spk = p_id
                        break
                        
            if not matched_spk:
                if len(speaker_profiles) < max_speakers:
                    matched_spk = f"Speaker {next_spk_id} ({gender})"
                    if f0 > 0.0:
                        speaker_profiles.append((f0, matched_spk, gender))
                    next_spk_id += 1
                else:
                    # Map to closest existing profile
                    if speaker_profiles:
                        best_p = min(speaker_profiles, key=lambda p: abs(f0 - p[0]))
                        matched_spk = best_p[1]
                    else:
                        matched_spk = f"Speaker 1 ({gender})"
                        
            new_seg = seg.copy()
            new_seg["speaker"] = matched_spk
            clustered.append(new_seg)
            
        print(f"[Info] Acoustic pitch analysis identified {len(set(s['speaker'] for s in clustered))} unique speaker profile(s) (Force Male Mode: {force_male_mode}).")
        return clustered
    except Exception as e:
        print(f"[Warning] Acoustic speaker clustering failed: {e}")
        return segments

def diarize_batch(client: genai.Client, segments: List[dict], model: str) -> List[dict]:
    """
    Asks Gemini to analyze speech turns and label each segment with a speaker ID.
    """
    prompt = (
        "You are an expert audio analyst and speaker diarization assistant. "
        "Analyze the following conversation transcription segments. Identify the speaker boundaries "
        "and label each segment with a distinct speaker identifier suffixed with their gender in parentheses "
        "(e.g., 'Speaker 1 (Male)', 'Speaker 2 (Male)', 'Speaker 3 (Male)', etc.) based on conversational context, dialogue turns, and tone structure. "
        "Limit the total number of distinct speakers to at most 3. "
        "You MUST keep the exact same start_time and end_time for each segment as provided in the input. "
        "Return the segments in the same order.\n\n"
        f"Input segments to diarize:\n{json.dumps(segments, indent=2)}"
    )
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DiarizationResponse,
            temperature=0.1,
            max_output_tokens=35000,
        ),
    )
    
    raw_text = response.text
    data = json.loads(raw_text)
    diarized_raw = data.get("segments", [])
    
    result = []
    for idx, input_seg in enumerate(segments):
        existing_speaker = input_seg.get("speaker", "Speaker 1 (Male)").strip()
        speaker = existing_speaker if existing_speaker else "Speaker 1 (Male)"
        if idx < len(diarized_raw):
            raw_spk = diarized_raw[idx].get("speaker", "").strip()
            if raw_spk:
                speaker = raw_spk
            
        result.append({
            "start_time": input_seg["start_time"],
            "end_time": input_seg["end_time"],
            "text": input_seg["text"],
            "speaker": speaker
        })
        
    return result

def diarize_segments(segments: List[dict], model: str = DEFAULT_TRANSLATION_MODEL, audio_path: str = None) -> List[dict]:
    """
    Runs hybrid speaker diarization (acoustic pitch clustering + Gemini context analysis).
    Caps max speakers to 3 and enforces clean speaker turns.
    """
    if not segments:
        return []
        
    # Step 1: Run acoustic pitch feature clustering if audio file is provided
    working_segments = segments
    if audio_path and os.path.exists(audio_path):
        working_segments = acoustic_speaker_clustering(segments, audio_path, max_speakers=3)
        
    # Step 2: Run Gemini dialogue context diarization
    batch_size = 50
    diarized_segments = []
    
    total_batches = (len(working_segments) + batch_size - 1) // batch_size
    print(f"[Info] Running speaker diarization on {len(working_segments)} segments in {total_batches} batch(es)...")
    
    for i in range(0, len(working_segments), batch_size):
        batch = working_segments[i:i + batch_size]
        batch_idx = (i // batch_size) + 1
        print(f"[Info] Diarizing batch {batch_idx}/{total_batches}...")
        
        def op(client):
            return diarize_batch(client, batch, model)
            
        diarized_batch_res = client_manager.execute_with_retry(op)
        diarized_segments.extend(diarized_batch_res)
        
    print(f"[Info] Speaker diarization completed. Total unique speakers: {len(set(s['speaker'] for s in diarized_segments))}")
    return diarized_segments

def assign_voices_to_speakers(segments: List[dict], custom_voice_map: Dict[str, str] = None) -> tuple[List[dict], Dict[str, str]]:
    """
    Maps each unique speaker ID to a distinct neural voice.
    Separates male and female voices dynamically based on speaker gender hints.
    """
    if not segments:
        return [], {}
        
    unique_speakers = sorted(list(set(seg.get("speaker", "Speaker 1") for seg in segments)))
    voice_map = {}
    
    female_voices = [
        "en-US-EmmaMultilingualNeural",
        "en-US-AvaNeural",
        "en-GB-SoniaNeural",
        "en-AU-NatashaNeural"
    ]
    male_voices = [
        "en-US-BrianNeural",
        "en-US-AndrewNeural",
        "en-GB-RyanNeural",
        "en-AU-WilliamNeural"
    ]
    
    female_idx = 0
    male_idx = 0
    
    for i, speaker in enumerate(unique_speakers):
        if custom_voice_map and speaker in custom_voice_map:
            voice_map[speaker] = custom_voice_map[speaker]
        else:
            speaker_lower = speaker.lower()
            is_female = "female" in speaker_lower
            is_male = "male" in speaker_lower and not is_female
            
            if is_female:
                voice_map[speaker] = female_voices[female_idx % len(female_voices)]
                female_idx += 1
            elif is_male:
                voice_map[speaker] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            else:
                voice_map[speaker] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            
    updated_segments = []
    for seg in segments:
        spk = seg.get("speaker", "Speaker 1")
        seg_voice = voice_map.get(spk, AVAILABLE_DUBBING_VOICES[0])
        
        new_seg = seg.copy()
        new_seg["voice"] = seg_voice
        updated_segments.append(new_seg)
        
    print(f"[Info] Speaker voice mapping assigned:")
    for speaker, voice in voice_map.items():
        print(f"  - {speaker} -> {voice}")
        
    return updated_segments, voice_map

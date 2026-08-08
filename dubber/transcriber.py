import os
import json
import math
import time
from typing import List
from pydantic import BaseModel, Field
from pydub import AudioSegment
from google import genai
from google.genai import types
from dubber.config import client_manager, DEFAULT_TRANSCRIPTION_MODEL, TEMP_DIR
class Segment(BaseModel):
    start_time: float = Field(description="Start time of the segment in seconds from the beginning of this audio chunk")
    end_time: float = Field(description="End time of the segment in seconds from the beginning of this audio chunk")
    text: str = Field(description="The exact text transcribed in the original language during this timeframe")
    speaker: str = Field(description="Unique speaker identifier, suffixed with their gender in parentheses, e.g., 'Speaker 1 (Male)' or 'Speaker 2 (Female)'. Analyze pitch and voice characteristics carefully.")

class TranscriptionResponse(BaseModel):
    segments: List[Segment]

def chunk_audio(audio_path: str, chunk_length_sec: int = 90) -> List[tuple[str, float]]:
    """
    Chunks a large WAV audio file into smaller 90-second pieces to guarantee high-precision transcription
    and prevent Gemini from skipping dialogue segments or hallucinating large gaps.
    Returns:
        List[tuple[str, float]]: A list of (chunk_file_path, offset_seconds)
    """
    print(f"[Info] Chunking audio file {audio_path} into {chunk_length_sec}-second high-precision pieces...")
    audio = AudioSegment.from_wav(audio_path)
    
    chunk_length_ms = chunk_length_sec * 1000
    total_length_ms = len(audio)
    
    chunks = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    for i, start_ms in enumerate(range(0, total_length_ms, chunk_length_ms)):
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk = audio[start_ms:end_ms]
        
        chunk_file_name = f"{base_name}_chunk_{i}.wav"
        chunk_path = os.path.join(TEMP_DIR, chunk_file_name)
        chunk.export(chunk_path, format="wav")
        
        offset_sec = start_ms / 1000.0
        chunks.append((chunk_path, offset_sec))
        
    print(f"[Info] Created {len(chunks)} audio chunk(s).")
    return chunks

def transcribe_chunk(client: genai.Client, chunk_path: str, offset_sec: float, model: str) -> List[dict]:
    """
    Transcribes a single audio chunk using Gemini API with Structured Output.
    """
    print(f"[Info] Uploading audio chunk {os.path.basename(chunk_path)} to Gemini File API...")
    file_ref = client.files.upload(file=chunk_path)
    
    while file_ref.state.name == "PROCESSING":
        print("[Info] Waiting for audio chunk file processing...")
        time.sleep(2)
        file_ref = client.files.get(name=file_ref.name)
        
    if file_ref.state.name == "FAILED":
        raise RuntimeError(f"Gemini file upload failed for {chunk_path}")
        
    print(f"[Info] Transcribing chunk using {model}...")
    prompt = (
        "You are an expert transcriber and audio analyst. Listen to the provided audio with extreme precision from start to finish.\n\n"
        "CRITICAL INSTRUCTION: You MUST scan every second of the timeline continuously without skipping any section. "
        "Even when there is background music, sound effects, action scenes, or quiet dialogue between 00:45 and 01:45, "
        "you MUST listen carefully and transcribe every single spoken line, sentence, or whisper in the original language.\n"
        "Do NOT omit any dialogue turn. Do NOT leave long gaps (such as 30+ seconds) without checking if words were spoken.\n"
        "If a speaker pauses for more than 1.0 second between phrases, you MUST split them into separate segments with their exact start_time and end_time.\n\n"
        "For each segment, capture:\n"
        "1. The exact start_time and end_time in seconds relative to the start of this audio chunk.\n"
        "2. The exact transcribed text in the speaker's original spoken language (e.g. Hindi, English, Spanish, etc.). Do not perform translation yet.\n"
        "3. The speaker identifier, suffixed with their gender in parentheses, e.g., 'Speaker 1 (Male)' or 'Speaker 2 (Female)', based on pitch and tone."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[file_ref, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranscriptionResponse,
                temperature=0.0,
                max_output_tokens=35000,
            ),
        )
        
        raw_text = response.text
        data = json.loads(raw_text)
        raw_segments = data.get("segments", [])
        
        adjusted_segments = []
        for seg in raw_segments:
            adjusted_segments.append({
                "start_time": round(offset_sec + seg["start_time"], 3),
                "end_time": round(offset_sec + seg["end_time"], 3),
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", "Speaker 1").strip()
            })
            
        return adjusted_segments
    finally:
        try:
            client.files.delete(name=file_ref.name)
        except Exception as e:
            print(f"[Warning] Failed to delete Gemini file {file_ref.name}: {e}")

def chunk_audio(audio_path: str, chunk_length_sec: int = 35, overlap_sec: float = 0.0) -> List[tuple[str, float]]:
    """
    Chunks a WAV audio file into smaller 35-second pieces with overlap to guarantee high-precision
    timestamping and prevent Gemini from skipping dialogue segments or hallucinating timing drift.
    Returns:
        List[tuple[str, float]]: A list of (chunk_file_path, offset_seconds)
    """
    print(f"[Info] Chunking audio file {audio_path} into {chunk_length_sec}-second high-precision pieces...")
    audio = AudioSegment.from_wav(audio_path)
    
    chunk_length_ms = chunk_length_sec * 1000
    overlap_ms = int(overlap_sec * 1000)
    step_ms = max(500, chunk_length_ms - overlap_ms)
    total_length_ms = len(audio)
    
    chunks = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    start_ms = 0
    i = 0
    while start_ms < total_length_ms:
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk = audio[start_ms:end_ms]
        
        chunk_file_name = f"{base_name}_chunk_{i}.wav"
        chunk_path = os.path.join(TEMP_DIR, chunk_file_name)
        chunk.export(chunk_path, format="wav")
        
        offset_sec = start_ms / 1000.0
        chunks.append((chunk_path, offset_sec))
        
        if end_ms >= total_length_ms:
            break
        start_ms += step_ms
        i += 1
        
    print(f"[Info] Created {len(chunks)} audio chunk(s).")
    return chunks

def transcribe_chunk(client: genai.Client, chunk_path: str, offset_sec: float, model: str) -> List[dict]:
    """
    Transcribes a single audio chunk using Gemini API with Structured Output.
    """
    print(f"[Info] Uploading audio chunk {os.path.basename(chunk_path)} to Gemini File API...")
    file_ref = client.files.upload(file=chunk_path)
    
    while file_ref.state.name == "PROCESSING":
        print("[Info] Waiting for audio chunk file processing...")
        time.sleep(1)
        file_ref = client.files.get(name=file_ref.name)
        
    if file_ref.state.name == "FAILED":
        raise RuntimeError(f"Gemini file upload failed for {chunk_path}")
        
    print(f"[Info] Transcribing chunk using {model}...")
    prompt = (
        "You are an expert audio transcriber and diarizer. Listen to the provided audio chunk with extreme precision.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Transcribe every spoken phrase in its original spoken language (e.g. Hindi, English, Spanish, etc.). Do NOT perform translation.\n"
        "2. Provide exact start_time and end_time in seconds relative to the start of THIS audio chunk (0.0 to 35.0 seconds).\n"
        "3. Do NOT hallucinate long timestamp gaps. Align start_time precisely to when the first spoken word of the phrase begins.\n"
        "4. Assign a consistent speaker identifier suffixed with gender, e.g. 'Speaker 1 (Male)' or 'Speaker 2 (Female)'.\n"
        "5. If a speaker pauses for > 1.0 second between phrases, split them into separate segments."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[file_ref, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TranscriptionResponse,
                temperature=0.0,
            ),
        )
        
        raw_text = response.text
        data = json.loads(raw_text)
        raw_segments = data.get("segments", [])
        
        adjusted_segments = []
        for seg in raw_segments:
            adjusted_segments.append({
                "start_time": round(offset_sec + float(seg["start_time"]), 3),
                "end_time": round(offset_sec + float(seg["end_time"]), 3),
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", "Speaker 1").strip()
            })
            
        return adjusted_segments
    finally:
        try:
            client.files.delete(name=file_ref.name)
        except Exception as e:
            print(f"[Warning] Failed to delete Gemini file {file_ref.name}: {e}")

def transcribe_audio(audio_path: str, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> List[dict]:
    """
    Transcribes the entire audio file using high-precision chunking (35-second chunks).
    Deduplicates overlapping boundary segments and guarantees accurate real-time timestamps.
    """
    chunks = chunk_audio(audio_path, chunk_length_sec=35, overlap_sec=2.0)
    all_segments = []
    
    for i, (chunk_path, offset_sec) in enumerate(chunks):
        print(f"[Info] Processing chunk {i+1}/{len(chunks)} (Offset: {offset_sec:.1f}s)...")
        
        def op(client):
            return transcribe_chunk(client, chunk_path, offset_sec, model)
            
        try:
            chunk_segments = client_manager.execute_with_retry(op)
            all_segments.extend(chunk_segments)
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                
    # Sort segments chronologically by start time
    all_segments.sort(key=lambda x: x["start_time"])
    
    # Deduplicate overlapping boundary segments from chunk transitions
    dedup_segments = []
    for seg in all_segments:
        if not seg["text"]:
            continue
        if not dedup_segments:
            dedup_segments.append(seg)
        else:
            last = dedup_segments[-1]
            # Skip if start times are within 1.2s AND text is highly similar or identical
            time_diff = abs(seg["start_time"] - last["start_time"])
            if time_diff < 1.2 and (seg["text"].lower() in last["text"].lower() or last["text"].lower() in seg["text"].lower()):
                continue
            dedup_segments.append(seg)
            
    print(f"[Info] Transcription complete. Total accurate segments: {len(dedup_segments)}")
    return dedup_segments

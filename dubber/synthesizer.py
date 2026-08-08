import os
import asyncio
import subprocess
import requests
from pydub import AudioSegment
import edge_tts
from dubber.config import TEMP_DIR, DEFAULT_TTS_VOICE, ELEVENLABS_API_KEY

def change_audio_speed(input_path: str, output_path: str, speed_factor: float):
    """
    Adjusts the playback speed of an audio file without changing its pitch
    using the FFmpeg atempo audio filter. Handles speed_factor > 2.0 or < 0.5
    by chaining filters.
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
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg speed adjustment failed: {result.stderr}")

async def synthesize_text_async(text: str, voice: str, output_path: str, pitch: str = "+0Hz", rate: str = "+0%"):
    """
    Asynchronously calls edge-tts to synthesize text to an MP3 file with pitch and rate adjustments.
    """
    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
    await communicate.save(output_path)

def extract_speaker_sample(original_audio_path: str, segments: list[dict], speaker_name: str, output_wav_path: str) -> bool:
    """
    Slices the original audio track to extract a clean 10-20 second audio sample
    of the speaker to be used for voice cloning.
    """
    # Find all segments matching the speaker name
    speaker_segs = [s for s in segments if s.get("speaker") == speaker_name]
    if not speaker_segs:
        print(f"[Warning] No segments found for speaker {speaker_name} to clone voice.")
        return False
        
    # Sort segments by length (descending) to get the longest clean utterances
    speaker_segs.sort(key=lambda x: x["end_time"] - x["start_time"], reverse=True)
    
    original_audio = AudioSegment.from_wav(original_audio_path)
    combined_sample = AudioSegment.silent(duration=0, frame_rate=original_audio.frame_rate)
    
    for seg in speaker_segs:
        start_ms = max(0, int(seg["start_time"] * 1000))
        end_ms = min(len(original_audio), int(seg["end_time"] * 1000))
        
        if end_ms - start_ms > 500: # ignore very short noises
            clip = original_audio[start_ms:end_ms]
            combined_sample += clip
            
        # Target 15 seconds of clean speech for optimal voice cloning
        if len(combined_sample) >= 15000:
            break
            
    # Fallback to first segment if total combined is too short
    if len(combined_sample) < 1000:
        first_seg = speaker_segs[0]
        start_ms = max(0, int(first_seg["start_time"] * 1000))
        end_ms = min(len(original_audio), int(first_seg["end_time"] * 1000))
        combined_sample = original_audio[start_ms:end_ms]
        
    print(f"[Info] Extracted {len(combined_sample)/1000.0:.2f}s sample for cloning speaker: {speaker_name}")
    combined_sample.export(output_wav_path, format="wav")
    return True

def add_elevenlabs_voice(api_key: str, name: str, sample_wav_path: str) -> str:
    """
    Uploads a WAV sample to ElevenLabs to create a cloned voice.
    """
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {
        "xi-api-key": api_key
    }
    data = {
        "name": name,
        "description": f"Auto-cloned speaker voice for {name}"
    }
    
    with open(sample_wav_path, "rb") as f:
        files = {
            "files": (os.path.basename(sample_wav_path), f, "audio/wav")
        }
        response = requests.post(url, headers=headers, data=data, files=files)
        
    response.raise_for_status()
    return response.json()["voice_id"]

def delete_elevenlabs_voice(api_key: str, voice_id: str):
    """
    Deletes a cloned voice from ElevenLabs.
    """
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {
        "xi-api-key": api_key
    }
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        print(f"[Info] Cleaned up ElevenLabs cloned voice: {voice_id}")
    except Exception as e:
        print(f"[Warning] Failed to delete ElevenLabs voice {voice_id}: {e}")

def synthesize_elevenlabs_speech(api_key: str, text: str, voice_id: str, output_path: str):
    """
    Synthesizes speech using an ElevenLabs voice ID. Saves output as MP3.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)

def synthesize_segment(text: str, target_duration: float, voice: str, segment_id: int,
                       tts_engine: str = "edge-tts", eleven_api_key: str = None,
                       f5_tts_url: str = None, ref_audio_path: str = None, ref_text: str = None,
                       pitch: str = "+0Hz", rate: str = "+0%") -> str:
    """
    Synthesizes speech for a segment using the chosen TTS engine,
    then speed-matches the audio to the target duration.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_mp3 = os.path.join(TEMP_DIR, f"seg_{segment_id}_raw.mp3")
    temp_wav = os.path.join(TEMP_DIR, f"seg_{segment_id}_raw.wav")
    
    # Generate silence if text is empty/whitespace
    if not text.strip():
        print(f"[Warning] Segment {segment_id} text is empty. Generating silence.")
        silence_dur = max(100, int(target_duration * 1000)) if target_duration > 0.1 else 500
        AudioSegment.silent(duration=silence_dur).export(temp_wav, format="wav")
    # 1. Generate speech based on chosen engine
    elif tts_engine == "elevenlabs" and eleven_api_key and voice:
        # For ElevenLabs, the 'voice' parameter is the cloned voice_id string
        try:
            synthesize_elevenlabs_speech(eleven_api_key, text, voice, temp_mp3)
        except Exception as e:
            print(f"[Warning] ElevenLabs synthesis failed: {e}. Falling back to Edge-TTS.")
            from dubber.config import DEFAULT_TTS_VOICE
            try:
                asyncio.run(synthesize_text_async(text, DEFAULT_TTS_VOICE, temp_mp3, pitch, rate))
            except Exception as ex:
                print(f"[Warning] Edge-TTS fallback also failed: {ex}. Generating silence.")
                silence_dur = max(100, int(target_duration * 1000)) if target_duration > 0.1 else 500
                AudioSegment.silent(duration=silence_dur).export(temp_wav, format="wav")
    elif tts_engine == "f5-tts" and f5_tts_url and ref_audio_path and ref_text:
        from dubber.f5_tts_client import synthesize_f5_tts
        success = synthesize_f5_tts(f5_tts_url, ref_audio_path, ref_text, text, temp_wav)
        if not success:
            print("[Warning] F5-TTS synthesis failed. Falling back to Edge-TTS.")
            from dubber.config import DEFAULT_TTS_VOICE
            fallback_voice = DEFAULT_TTS_VOICE
            try:
                asyncio.run(synthesize_text_async(text, fallback_voice, temp_mp3, pitch, rate))
            except Exception as ex:
                print(f"[Warning] Edge-TTS fallback failed: {ex}. Generating silence.")
                silence_dur = max(100, int(target_duration * 1000)) if target_duration > 0.1 else 500
                AudioSegment.silent(duration=silence_dur).export(temp_wav, format="wav")
    else:
        # Fallback/Default: Edge-TTS
        if not voice or voice == "cloned":
            from dubber.config import DEFAULT_TTS_VOICE
            voice = DEFAULT_TTS_VOICE
        try:
            asyncio.run(synthesize_text_async(text, voice, temp_mp3, pitch, rate))
        except Exception as e:
            print(f"[Warning] Edge-TTS synthesis failed for voice '{voice}': {e}. Trying fallback voice.")
            from dubber.config import DEFAULT_TTS_VOICE
            try:
                asyncio.run(synthesize_text_async(text, DEFAULT_TTS_VOICE, temp_mp3, pitch, rate))
            except Exception as ex:
                print(f"[Warning] Fallback voice also failed: {ex}. Generating silence.")
                silence_dur = max(100, int(target_duration * 1000)) if target_duration > 0.1 else 500
                AudioSegment.silent(duration=silence_dur).export(temp_wav, format="wav")
        
    # Convert output MP3 to WAV if it exists
    if os.path.exists(temp_mp3):
        cmd_conv = ['ffmpeg', '-y', '-i', temp_mp3, temp_wav]
        subprocess.run(cmd_conv, capture_output=True)
        os.remove(temp_mp3)
        
    if not os.path.exists(temp_wav):
        # Last-resort fallback to ensure process never crashes
        print(f"[Warning] WAV not generated for segment {segment_id}. Generating emergency silence.")
        silence_dur = max(100, int(target_duration * 1000)) if target_duration > 0.1 else 500
        AudioSegment.silent(duration=silence_dur).export(temp_wav, format="wav")
        
    # 2. Measure synthesized duration
    audio = AudioSegment.from_wav(temp_wav)
    synth_duration = len(audio) / 1000.0
    
    output_wav = os.path.join(TEMP_DIR, f"seg_{segment_id}_final.wav")
    
    # 3. Apply natural speed matching & duration alignment
    if synth_duration > target_duration and target_duration > 0.1:
        speed_factor = synth_duration / target_duration
        # Cap speedup factor to 1.20x max to preserve 100% natural spoken dialogue pace
        if speed_factor > 1.20:
            speed_factor = 1.20
            
        if speed_factor > 1.02:
            print(f"[Info] Speeding up segment {segment_id} by {speed_factor:.2f}x ({synth_duration:.2f}s -> {target_duration:.2f}s)...")
            change_audio_speed(temp_wav, output_wav, speed_factor)
            os.remove(temp_wav)
        else:
            os.rename(temp_wav, output_wav)
    else:
        os.rename(temp_wav, output_wav)
        
    return output_wav


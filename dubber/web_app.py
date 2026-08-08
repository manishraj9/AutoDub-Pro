import os
import json
import asyncio
import threading
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
from pydub import AudioSegment


from dubber.config import (
    TEMP_DIR, 
    DEFAULT_TRANSCRIPTION_MODEL, 
    DEFAULT_TTS_VOICE,
    DEFAULT_CRF,
    DEFAULT_EQ_LOW,
    DEFAULT_EQ_MID,
    DEFAULT_EQ_HIGH,
    DEFAULT_NOISE_GATE,
    client_manager
)
from dubber.downloader import download_youtube_video
from dubber.transcriber import transcribe_audio
from dubber.translator import translate_segments, condense_translation_if_fast, SUPPORTED_LANGUAGES
from dubber.diarization import diarize_segments, assign_voices_to_speakers, AVAILABLE_DUBBING_VOICES
from dubber.synthesizer import (
    synthesize_segment, 
    extract_speaker_sample, 
    add_elevenlabs_voice, 
    delete_elevenlabs_voice,
    synthesize_text_async
)
from dubber.audio_processor import assemble_dubbed_audio, align_segment_boundaries, apply_stereo_panning
from dubber.processor import mux_video_audio
from dubber.enhancer import (
    apply_three_band_eq, 
    remove_bg_noise, 
    apply_fade_borders, 
    apply_limiter,
    generate_srt_subtitles,
    generate_vtt_subtitles,
    generate_ass_subtitles,
    burn_subtitles_into_video,
    generate_waveform_overlay,
    analyze_eq_profile
)
from dubber.cache import get_cached_translation, add_to_translation_cache, clear_translation_cache, get_cached_segments_count
from dubber.profiler import profiler, estimate_api_characters, generate_html_report
from dubber.utils import (
    check_api_keys_validity, 
    get_available_edge_voices, 
    download_youtube_thumbnail, 
    compress_video_for_web,
    detect_source_language,
    benchmark_api_keys,
    save_custom_voice_preset,
    load_custom_voice_preset
)

app = FastAPI(title="Enterprise Automated Video Dubbing Dashboard")

# Save/load presets endpoints
@app.get("/api/load-preset")
async def load_preset_endpoint(name: str):
    preset = load_custom_voice_preset(name)
    if preset is not None:
        return JSONResponse(content={"success": True, "preset": preset})
    return JSONResponse(content={"success": False, "message": "Preset not found"}, status_code=404)

@app.post("/api/save-preset")
async def save_preset_endpoint(data: dict):
    name = data.get("name")
    voice_map = data.get("voice_map")
    if not name or not voice_map:
        return JSONResponse(content={"success": False, "message": "Invalid request parameters"}, status_code=400)
    success = save_custom_voice_preset(name, voice_map)
    return JSONResponse(content={"success": success})

@app.delete("/api/delete-preset")
async def delete_preset_endpoint(name: str):
    preset_dir = os.path.join(os.getcwd(), "presets")
    preset_file = os.path.join(preset_dir, f"{name}.json")
    if os.path.exists(preset_file):
        try:
            os.remove(preset_file)
            return JSONResponse(content={"success": True})
        except Exception as e:
            return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)
    return JSONResponse(content={"success": False, "message": "Preset not found"}, status_code=404)

@app.get("/api/benchmark-keys")
async def benchmark_keys_endpoint():
    keys = client_manager.api_keys
    if not keys:
        return JSONResponse(content={"configured": False, "keys": []})
    
    results = benchmark_api_keys(keys)
    return JSONResponse(content={"configured": True, "keys": results})


# Ensure static folders and static outputs directory exist
STATIC_DIR = os.path.join(os.getcwd(), "static")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Thread-safe tracker to push events to WebSocket client
class ProgressTracker:
    def __init__(self):
        self.websocket: WebSocket = None
        self.loop = None

    def set_websocket(self, websocket: WebSocket):
        self.websocket = websocket
        self.loop = asyncio.get_running_loop()

    async def send_json_async(self, data: dict):
        if self.websocket:
            try:
                await self.websocket.send_json(data)
            except Exception:
                pass

    def send_log(self, message: str, log_type: str = "info"):
        """Sends a console log to the client."""
        data = {"log": message, "type": log_type}
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_json_async(data), self.loop)

    def send_progress(self, progress: float, status_text: str):
        """Updates the client's progress bar and state label."""
        data = {
            "progress": progress,
            "status_text": status_text
        }
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_json_async(data), self.loop)

    def send_step_status(self, step: str, status: str):
        """Updates step states (pending, active, completed, failed)."""
        data = {
            "step_status": {
                "step": step,
                "status": status
            }
        }
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_json_async(data), self.loop)

    def send_event(self, event_name: str, payload: dict):
        """Sends arbitrary state events to the frontend client."""
        data = {"event": event_name, **payload}
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.send_json_async(data), self.loop)

# Global tracker instance
tracker = ProgressTracker()

# Temporary store to keep state between Phase 1 and Phase 2
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>static/index.html not found</h3>", status_code=404)

# Cache management API
@app.get("/api/clear-cache")
async def clear_cache_endpoint():
    try:
        clear_translation_cache()
        return JSONResponse(content={"success": True, "message": "Translation cache folder cleared."})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.get("/api/cache-stats")
async def get_cache_stats():
    count = get_cached_segments_count()
    return JSONResponse(content={"cached_segments": count})

# Key validity validator API
@app.get("/api/ping-keys")
async def ping_keys_endpoint():
    keys = client_manager.api_keys
    if not keys:
        return JSONResponse(content={"configured": False, "keys": []})
    
    validities = check_api_keys_validity(keys)
    results = [{"index": idx + 1, "valid": val} for idx, val in enumerate(validities)]
    return JSONResponse(content={"configured": True, "keys": results})

# Edge TTS voices lookup API
@app.get("/api/voices")
async def get_voices():
    voices = get_available_edge_voices()
    return JSONResponse(content=voices)

@app.get("/api/languages")
async def get_languages():
    """Returns supported target languages and their locale prefixes."""
    return JSONResponse(content=SUPPORTED_LANGUAGES)

@app.get("/api/preview-voice")
async def preview_voice(voice: str):
    """
    Generates a 3-second preview audio file for an Edge-TTS voice and returns its static URL.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filename = f"preview_{voice}.mp3"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    
    if not os.path.exists(filepath):
        name_clean = voice.split('-')[-1].replace('Neural', '')
        sample_text = f"Hello! This is a preview of the {name_clean} voice."
        try:
            await synthesize_text_async(sample_text, voice, filepath)
        except Exception as e:
            return JSONResponse(content={"success": False, "message": f"Failed to synthesize voice preview: {e}"}, status_code=500)
            
    return JSONResponse(content={"success": True, "url": f"/static/outputs/{filename}"})

# Subtitle download routers
@app.get("/api/download-subs")
async def download_subs(filename: str = Query(..., description="Target file name"), format: str = Query("srt", description="srt or vtt")):
    filepath = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(filename)[0]}.{format}")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="text/plain", filename=os.path.basename(filepath))
    return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.get("/api/download-report")
async def download_report(filename: str = Query(..., description="Target file name")):
    filepath = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(filename)[0]}_report.html")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="text/html", filename=os.path.basename(filepath))
    return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.post("/api/preview-segment-tts")
async def preview_segment_tts(data: dict):
    """
    Synthesizes and returns audio URL for a single segment text preview.
    """
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_TTS_VOICE)
    pitch = data.get("pitch", "+0Hz")
    rate = data.get("rate", "+0%")
    
    if not text.strip():
        return JSONResponse(content={"success": False, "message": "Text is empty"}, status_code=400)
        
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    import hashlib
    h = hashlib.sha256(f"{text}_{voice}_{pitch}_{rate}".encode('utf-8')).hexdigest()[:12]
    filename = f"preview_seg_{h}.mp3"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    
    try:
        await synthesize_text_async(text, voice, filepath, pitch=pitch, rate=rate)
        return JSONResponse(content={"success": True, "url": f"/static/outputs/{filename}"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

@app.get("/api/download-audio")
async def download_audio(filename: str = Query(..., description="Target video filename")):
    """
    Extracts or returns dubbed audio WAV/MP3 track for the requested video.
    """
    base_name = os.path.splitext(filename)[0]
    wav_path = os.path.join(OUTPUTS_DIR, f"{base_name}_audio.mp3")
    vid_path = os.path.join(OUTPUTS_DIR, filename)
    
    if not os.path.exists(wav_path) and os.path.exists(vid_path):
        import subprocess
        cmd = ['ffmpeg', '-y', '-i', vid_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2', wav_path]
        subprocess.run(cmd, capture_output=True)
        
    if os.path.exists(wav_path):
        return FileResponse(wav_path, media_type="audio/mpeg", filename=os.path.basename(wav_path))
    return JSONResponse(content={"error": "Audio file not found"}, status_code=404)

def run_phase_1(video_url: str, model: str, audio_mode: str, duck_level: float, output_filename: str, 
                use_cache: bool, burn_subtitles: bool, compress_video: bool, crf_value: int,
                eq_low: float, eq_mid: float, eq_high: float, noise_gate: float, waveform_overlay: bool = False,
                target_language: str = "English", download_quality: str = "1080p"):
    """
    Executes Ingestion & Alignment: Download, Transcribe, Diarize, and Translate.
    """
    session_id = str(hash(video_url))
    tracker.send_log(f"[System] Initiating Session: {session_id}", "system")
    
    profiler.timers = {}
    profiler.durations = {}
    profiler.order = []
    
    try:
        # Download video thumbnail
        thumb_path = os.path.join(STATIC_DIR, f"{session_id}_thumb.jpg")
        download_youtube_thumbnail(video_url, thumb_path)
        
        # 1. Download
        tracker.send_step_status("download", "active")
        tracker.send_progress(5, f"Downloading video ({download_quality}) from YouTube...")
        tracker.send_log(f"[Info] Running yt-dlp download (Quality: {download_quality}) on URL: {video_url}...")
        
        profiler.start("download")
        video_path, audio_path = download_youtube_video(video_url, TEMP_DIR, quality=download_quality)
        profiler.stop("download")
        
        tracker.send_step_status("download", "completed")
        tracker.send_progress(15, "Video downloaded. Extracting audio...")
        tracker.send_log(f"[Success] Extracted audio WAV file to: {audio_path}", "success")

        
        # 1.5. Detect Language
        tracker.send_progress(20, "Detecting spoken language...")
        detected_lang = detect_source_language(audio_path, model=model)
        tracker.send_log(f"[Success] Auto-detected video language: {detected_lang}", "success")
        
        # 2. Transcribe
        tracker.send_step_status("transcribe", "active")
        tracker.send_progress(25, "Transcribing speech using Gemini API...")
        tracker.send_log(f"[Info] Uploading WAV audio to Gemini for speech recognition...")
        
        profiler.start("transcribe")
        segments = transcribe_audio(audio_path, model=model)
        profiler.stop("transcribe")
        
        tracker.send_step_status("transcribe", "completed")
        tracker.send_progress(45, "Transcription complete. Aligning boundaries...")
        tracker.send_log(f"[Success] Transcribed {len(segments)} segments.", "success")
        
        # 2.5. Align Boundaries
        segments = align_segment_boundaries(segments, audio_path)
        
        # 3. Diarize
        tracker.send_step_status("diarize", "active")
        tracker.send_progress(50, "Analyzing speaker identities...")
        tracker.send_log(f"[Info] Diarizing voice turns using Gemini & acoustic pitch analysis...")
        
        profiler.start("diarize")
        diarized_segments = diarize_segments(segments, model=model, audio_path=audio_path)
        profiler.stop("diarize")
        
        tracker.send_step_status("diarize", "completed")
        tracker.send_progress(60, "Speaker separation complete. Translating...")
        
        # Find unique speakers
        speakers = sorted(list(set(seg.get("speaker", "Speaker 1") for seg in diarized_segments)))
        tracker.send_log(f"[Success] Detected {len(speakers)} speakers: {', '.join(speakers)}", "success")

        # 4. Translate (with Caching support)
        tracker.send_step_status("translate", "active")
        tracker.send_progress(65, f"Translating speech segments to {target_language}...")
        
        profiler.start("translate")
        
        translated_segments = []
        uncached_segments = []
        uncached_indices = []
        
        for idx, seg in enumerate(diarized_segments):
            cached_text = None
            if use_cache:
                cached_text = get_cached_translation(seg["text"], source_lang=detected_lang, target_lang=target_language)
                
            if cached_text:
                seg_copy = seg.copy()
                seg_copy["text"] = cached_text
                translated_segments.append(seg_copy)
            else:
                uncached_segments.append(seg)
                uncached_indices.append(idx)
                translated_segments.append(None)
                
        if uncached_segments:
            tracker.send_log(f"[Info] Querying Gemini for {len(uncached_segments)} translations to {target_language}...", "info")
            api_translated = translate_segments(uncached_segments, model=model, target_language=target_language)
            
            for idx, trans_seg in zip(uncached_indices, api_translated):
                translated_segments[idx] = trans_seg
                if use_cache:
                    add_to_translation_cache(diarized_segments[idx]["text"], trans_seg["text"], source_lang=detected_lang, target_lang=target_language)
        else:
            tracker.send_log(f"[Info] Loaded all translations from local persistent {target_language} cache.", "info")
            
        profiler.stop("translate")
        
        # 4.2. Condense translations
        tracker.send_log(f"[Info] Checking and condensing translations for fast segments in {target_language}...", "info")
        condensed_segments = []
        for seg in translated_segments:
            cond_seg = condense_translation_if_fast(seg, max_wpm=170.0, model=model, target_language=target_language)
            condensed_segments.append(cond_seg)
        translated_segments = condensed_segments
        
        tracker.send_step_status("translate", "completed")
        tracker.send_progress(80, "Translation complete. Awaiting human verification...")
        tracker.send_log("[Success] Translation finished.", "success")
        
        # Store state for Phase 2
        active_sessions[session_id] = {
            "video_path": video_path,
            "audio_path": audio_path,
            "original_segments": diarized_segments,
            "audio_mode": audio_mode,
            "duck_level": duck_level,
            "output_filename": output_filename,
            "burn_subtitles": burn_subtitles,
            "compress_video": compress_video,
            "crf_value": crf_value,
            "eq_low": eq_low,
            "eq_mid": eq_mid,
            "eq_high": eq_high,
            "noise_gate": noise_gate,
            "waveform_overlay": waveform_overlay
        }
        
        # Pack and send segments to editor
        payload_segments = []
        for orig, trans in zip(diarized_segments, translated_segments):
            payload_segments.append({
                "start_time": orig["start_time"],
                "end_time": orig["end_time"],
                "speaker": orig["speaker"],
                "text": orig["text"],
                "translated_text": trans["text"]
            })
            
        edge_voices = get_available_edge_voices()
        # Send full voice objects with gender and locale metadata
        available_voices = edge_voices if edge_voices else [{"short_name": v, "gender": "Male", "locale": "en-US"} for v in AVAILABLE_DUBBING_VOICES]
        
        # Determine target locale prefix for voice filtering
        target_locale = SUPPORTED_LANGUAGES.get(target_language, "en")
        
        tracker.send_event("EDITING_READY", {
            "video_id": session_id,
            "segments": payload_segments,
            "speakers": speakers,
            "available_voices": available_voices,
            "target_locale": target_locale,
            "target_language": target_language,
            "detected_language": detected_lang,
            "thumbnail_url": f"/static/{session_id}_thumb.jpg"
        })
        tracker.send_log("[System] Workspace ready! Please edit translation cells in the table above.", "system")
        
    except Exception as e:
        tracker.send_log(f"[Error] Phase 1 failed: {e}", "error")
        steps = ["download", "transcribe", "diarize", "translate"]
        for s in steps:
            tracker.send_step_status(s, "failed")

def run_phase_2(session_id: str, edited_segments: List[dict], tts_engine: str = "edge-tts", 
                eleven_api_key: str = None, f5_tts_url: str = None,
                sub_fontname: str = "Arial", sub_fontsize: int = 20,
                sub_fontcolor: str = "#ffffff", sub_outlinecolor: str = "#000000",
                pronunciation_dict: dict = None,
                tts_vol_db: float = 0.0, bg_vol_db: float = 0.0):
    """
    Executes Voice Synthesis & Audio-Video Muxing.
    Applies audio enhancements (EQ, Noise Gate), generates subtitles, and compresses.
    """
    session = active_sessions.get(session_id)
    if not session:
        tracker.send_log("[Error] Session not found. Please reload and try again.", "error")
        return
        
    cloned_voices = {}
    if pronunciation_dict is None:
        pronunciation_dict = {}
        
    # Apply custom pronunciation dictionary word/phrase replacements
    for seg in edited_segments:
        seg_text = seg["text"]
        for word, replacement in pronunciation_dict.items():
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            seg_text = pattern.sub(replacement, seg_text)
        seg["tts_text"] = seg_text
        
    try:
        video_path = session["video_path"]
        audio_path = session["audio_path"]
        audio_mode = session["audio_mode"]
        duck_level = session["duck_level"]
        output_filename = session["output_filename"]
        burn_subtitles = session["burn_subtitles"]
        compress_video = session["compress_video"]
        crf_value = session["crf_value"]
        eq_low = session["eq_low"]
        eq_mid = session["eq_mid"]
        eq_high = session["eq_high"]
        noise_gate = session["noise_gate"]
        
        # Phase 2.5: Clone voices on ElevenLabs if configured
        if tts_engine == "elevenlabs" and eleven_api_key:
            tracker.send_log("[Info] Initiating ElevenLabs Voice Cloning for speakers...", "system")
            unique_speakers = sorted(list(set(seg["speaker"] for seg in edited_segments)))
            
            for speaker in unique_speakers:
                sample_wav = os.path.join(TEMP_DIR, f"sample_{speaker.replace(' ', '_')}.wav")
                tracker.send_log(f"[Info] Extracting speech print sample for {speaker}...", "info")
                if extract_speaker_sample(audio_path, edited_segments, speaker, sample_wav):
                    tracker.send_log(f"[Info] Uploading sample for {speaker} to ElevenLabs...", "info")
                    try:
                        voice_id = add_elevenlabs_voice(eleven_api_key, speaker, sample_wav)
                        cloned_voices[speaker] = voice_id
                        tracker.send_log(f"[Success] Cloned voice created for {speaker}: {voice_id}", "success")
                    except Exception as e:
                        tracker.send_log(f"[Error] Failed to clone voice for {speaker}: {e}. Will fallback to Edge-TTS.", "error")
                
            # Map segment voice parameters to ElevenLabs voice IDs
            for seg in edited_segments:
                spk = seg["speaker"]
                if spk in cloned_voices:
                    seg["voice"] = cloned_voices[spk]
                    
        # Extract speaker voice references for F5-TTS if enabled
        speaker_refs = {}
        if tts_engine == "f5-tts" and f5_tts_url:
            tracker.send_log("[Info] Extracting speaker references for F5-TTS...", "system")
            unique_speakers = sorted(list(set(seg["speaker"] for seg in edited_segments)))
            for speaker in unique_speakers:
                raw_ref_wav = os.path.join(TEMP_DIR, f"raw_ref_{speaker.replace(' ', '_')}.wav")
                ref_wav = os.path.join(TEMP_DIR, f"ref_{speaker.replace(' ', '_')}.wav")
                if extract_speaker_sample(audio_path, edited_segments, speaker, raw_ref_wav):
                    from dubber.enhancer import extract_vocals_filter
                    extract_vocals_filter(raw_ref_wav, ref_wav)
                    
                    # Find reference text
                    speaker_segs = [s for s in edited_segments if s.get("speaker") == speaker]
                    speaker_segs.sort(key=lambda x: x["end_time"] - x["start_time"], reverse=True)
                    ref_text_val = "Hello"
                    if speaker_segs:
                        longest_seg = speaker_segs[0]
                        matching_orig = [s for s in session["original_segments"] if s["start_time"] == longest_seg["start_time"]]
                        ref_text_val = matching_orig[0]["text"] if matching_orig else longest_seg["text"]
                        
                    speaker_refs[speaker] = {
                        "audio_path": ref_wav,
                        "text": ref_text_val
                    }
                    tracker.send_log(f"[Success] F5-TTS speaker reference extracted for {speaker}.", "success")
        
        # 5. Synthesize Speech
        tracker.send_step_status("synthesize", "active")
        tracker.send_progress(85, "Synthesizing English speech segments...")
        tracker.send_log("[Info] Starting voice generation...", "info")
        
        profiler.start("synthesize")
        segment_files = []
        for idx, seg in enumerate(edited_segments):
            target_dur = seg["end_time"] - seg["start_time"]
            tracker.send_log(f"[Info] Synthesizing segment {idx+1}/{len(edited_segments)}...")
            
            ref_path = None
            ref_txt = None
            if tts_engine == "f5-tts" and seg["speaker"] in speaker_refs:
                ref_path = speaker_refs[seg["speaker"]]["audio_path"]
                ref_txt = speaker_refs[seg["speaker"]]["text"]
                
            seg_wav = synthesize_segment(
                text=seg.get("tts_text", seg["text"]), 
                target_duration=target_dur, 
                voice=seg["voice"], 
                segment_id=idx,
                tts_engine=tts_engine,
                eleven_api_key=eleven_api_key,
                f5_tts_url=f5_tts_url,
                ref_audio_path=ref_path,
                ref_text=ref_txt,
                pitch=seg.get("pitch", "+0Hz"),
                rate=seg.get("rate", "+0%")
            )
            
            # Apply Vocal EQ, Noise Gate, Fades, and Peak Limiting
            raw_seg_audio = AudioSegment.from_wav(seg_wav)
            enhanced_audio = apply_fade_borders(raw_seg_audio, fade_ms=20)
            
            # Parametric EQ (Automatic Pitch-based or Manual)
            if eq_low == 0.0 and eq_mid == 0.0 and eq_high == 0.0:
                auto_low, auto_mid, auto_high = analyze_eq_profile(raw_seg_audio)
                enhanced_audio = apply_three_band_eq(enhanced_audio, auto_low, auto_mid, auto_high)
            else:
                enhanced_audio = apply_three_band_eq(enhanced_audio, eq_low, eq_mid, eq_high)
                
            # Spatial Stereo Panning
            unique_speakers = sorted(list(set(s["speaker"] for s in edited_segments)))
            enhanced_audio = apply_stereo_panning(enhanced_audio, seg["speaker"], unique_speakers)
            
            enhanced_audio = remove_bg_noise(enhanced_audio, noise_gate)
            enhanced_audio = apply_limiter(enhanced_audio, max_dbfs=-1.0)
            enhanced_audio.export(seg_wav, format="wav")
            
            segment_files.append(seg_wav)
            
        profiler.stop("synthesize")
        tracker.send_step_status("synthesize", "completed")
        tracker.send_progress(92, "Speech synthesis complete. Assembling track...")
        tracker.send_log("[Success] All speech segments synthesized successfully.", "success")
        
        # Save Subtitles (SRT, WebVTT, ASS Styled)
        srt_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(output_filename)[0]}.srt")
        vtt_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(output_filename)[0]}.vtt")
        ass_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(output_filename)[0]}.ass")
        
        def rgb_to_ass_color(hex_color: str) -> str:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                return f"&H00{b}{g}{r}"
            return "&H00FFFFFF"
            
        ass_fontcolor = rgb_to_ass_color(sub_fontcolor)
        ass_outlinecolor = rgb_to_ass_color(sub_outlinecolor)
        
        generate_srt_subtitles(edited_segments, srt_path)
        generate_vtt_subtitles(edited_segments, vtt_path)
        generate_ass_subtitles(
            edited_segments, 
            ass_path, 
            fontname=sub_fontname, 
            fontsize=sub_fontsize, 
            primary_color=ass_fontcolor, 
            outline_color=ass_outlinecolor
        )
        
        # Assemble dubbed WAV
        dubbed_wav = os.path.join(TEMP_DIR, f"{session_id}_dubbed.wav")
        assemble_dubbed_audio(
            segments=edited_segments,
            segment_files=segment_files,
            original_audio_path=audio_path,
            output_audio_path=dubbed_wav,
            mode=audio_mode,
            duck_level_db=duck_level,
            original_vol_db=bg_vol_db,
            dubbed_vol_db=tts_vol_db
        )
        
        # 6. Muxing
        tracker.send_step_status("mux", "active")
        tracker.send_progress(95, "Merging dubbed audio track into video stream...")
        
        profiler.start("mux")
        temp_mux_path = os.path.join(TEMP_DIR, f"{session_id}_muxed.mp4")
        output_video_path = os.path.join(OUTPUTS_DIR, output_filename)
        
        waveform_overlay = session.get("waveform_overlay", False)
        if waveform_overlay:
            tracker.send_log("[Info] Invoking FFmpeg animated waveform overlay...", "info")
            generate_waveform_overlay(video_path, dubbed_wav, temp_mux_path)
        else:
            tracker.send_log("[Info] Invoking FFmpeg stream copy for lossless visual output...", "info")
            mux_video_audio(video_path, dubbed_wav, temp_mux_path)
        
        if burn_subtitles:
            tracker.send_log("[Info] Burning WebVTT captions into visual frames...", "info")
            burn_subtitles_into_video(temp_mux_path, srt_path, output_video_path)
        else:
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(temp_mux_path, output_video_path)
            
        if compress_video:
            tracker.send_log(f"[Info] Executing H.264 compression (CRF: {crf_value})...", "info")
            comp_path = os.path.splitext(output_video_path)[0] + "_compressed.mp4"
            compress_video_for_web(output_video_path, comp_path, crf_value)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(comp_path, output_video_path)
            
        profiler.stop("mux")
        tracker.send_step_status("mux", "completed")
        tracker.send_progress(100, "Dubbing pipeline complete!")
        tracker.send_log(f"[Success] Dubbed video generated successfully: {output_filename}", "success")
        
        # Generate Performance Report
        report_data = profiler.generate_report_data()
        char_stats = estimate_api_characters(edited_segments)
        report_html_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(output_filename)[0]}_report.html")
        generate_html_report(report_data, char_stats, report_html_path)
        
        # Clean up session
        active_sessions.pop(session_id, None)
        
        # Clean up temp folder (except final static assets)
        try:
            if os.path.exists(TEMP_DIR):
                for f in os.listdir(TEMP_DIR):
                    os.remove(os.path.join(TEMP_DIR, f))
        except Exception as e:
            tracker.send_log(f"[Warning] Temporary clean-up failed: {e}")
            
        # Clean up thumbnail
        thumb_img = os.path.join(STATIC_DIR, f"{session_id}_thumb.jpg")
        if os.path.exists(thumb_img):
            os.remove(thumb_img)
            
        # Send completed event
        web_output_url = f"/static/outputs/{output_filename}"
        tracker.send_event("PIPELINE_COMPLETE", {
            "output_url": web_output_url
        })
        tracker.send_log("[System] All processes completed. Video ready for review.", "system")
        
    except Exception as e:
        tracker.send_log(f"[Error] Phase 2 failed: {e}", "error")
        tracker.send_step_status("synthesize", "failed")
        tracker.send_step_status("mux", "failed")
        
    finally:
        # Clean up temporary cloned voices from ElevenLabs account
        if cloned_voices and eleven_api_key:
            tracker.send_log("[Info] Cleaning up temporary cloned voices from ElevenLabs account...", "system")
            for speaker, voice_id in cloned_voices.items():
                delete_elevenlabs_voice(eleven_api_key, voice_id)

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    tracker.set_websocket(websocket)
    tracker.send_log("[System] WebSocket logger channel connected.", "system")
    
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            action = data.get("action")
            
            if action == "start_pipeline":
                config = data.get("config", {})
                url = config.get("url")
                model = config.get("model", "gemini-3.1-flash-lite")
                mode = config.get("mode", "duck")
                duck_level = config.get("duck_level", -20.0)
                output_name = config.get("output_name", "dubbed_output.mp4")
                
                # Fetch quality and caching flags
                use_cache = config.get("use_cache", True)
                burn_subtitles = config.get("burn_subtitles", False)
                compress_video = config.get("compress_video", False)
                crf_value = config.get("crf_value", DEFAULT_CRF)
                
                # EQ and Noise gate parameters
                eq_low = config.get("eq_low", DEFAULT_EQ_LOW)
                eq_mid = config.get("eq_mid", DEFAULT_EQ_MID)
                eq_high = config.get("eq_high", DEFAULT_EQ_HIGH)
                noise_gate = config.get("noise_gate", DEFAULT_NOISE_GATE)
                waveform_overlay = config.get("waveform_overlay", False)
                target_language = config.get("target_language", "English")
                download_quality = config.get("download_quality", "1080p")
                
                # Run Phase 1 in a background thread to keep socket connection responsive
                threading.Thread(
                    target=run_phase_1,
                    args=(url, model, mode, duck_level, output_name, 
                          use_cache, burn_subtitles, compress_video, crf_value,
                          eq_low, eq_mid, eq_high, noise_gate, waveform_overlay,
                          target_language, download_quality)
                ).start()
                
            elif action == "synthesize_and_mux":
                session_id = data.get("video_id")
                segments = data.get("segments", [])
                tts_engine = data.get("tts_engine", "edge-tts")
                eleven_api_key = data.get("eleven_api_key", None)
                f5_tts_url = data.get("f5_tts_url", None)
                
                sub_fontname = data.get("sub_fontname", "Arial")
                sub_fontsize = data.get("sub_fontsize", 20)
                sub_fontcolor = data.get("sub_fontcolor", "#ffffff")
                sub_outlinecolor = data.get("sub_outlinecolor", "#000000")
                pronunciation_dict = data.get("pronunciation_dict", {})
                tts_vol_db = float(data.get("tts_vol_db", 0.0))
                bg_vol_db = float(data.get("bg_vol_db", 0.0))
                
                # Run Phase 2 in a background thread
                threading.Thread(
                    target=run_phase_2,
                    args=(session_id, segments, tts_engine, eleven_api_key, f5_tts_url,
                          sub_fontname, sub_fontsize, sub_fontcolor, sub_outlinecolor,
                          pronunciation_dict, tts_vol_db, bg_vol_db)
                ).start()
                
    except WebSocketDisconnect:
        tracker.set_websocket(None)
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        tracker.set_websocket(None)
def start_web_server(port: int = 8000):
        """
        Launches the FastAPI web server using Uvicorn.
        """
        port = int(os.environ.get("PORT", port))

        print(f"\n[Info] Launching Web Server dashboard at: http://0.0.0.0:{port}\n")

        uvicorn.run(
          app,
          host="0.0.0.0",
          port=port,
          log_level="info"
        )
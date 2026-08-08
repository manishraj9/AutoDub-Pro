import os
import shutil
import time
from tqdm import tqdm
from dubber.config import (
    TEMP_DIR, 
    DEFAULT_TRANSCRIPTION_MODEL, 
    DEFAULT_TRANSLATION_MODEL, 
    DEFAULT_TTS_VOICE,
    DEFAULT_CRF,
    DEFAULT_EQ_LOW,
    DEFAULT_EQ_MID,
    DEFAULT_EQ_HIGH,
    DEFAULT_NOISE_GATE
)
from dubber.downloader import download_youtube_video
from dubber.transcriber import transcribe_audio
from dubber.translator import translate_segments, condense_translation_if_fast
from dubber.diarization import diarize_segments, assign_voices_to_speakers
from dubber.synthesizer import (
    synthesize_segment, 
    extract_speaker_sample, 
    add_elevenlabs_voice, 
    delete_elevenlabs_voice
)
from dubber.audio_processor import assemble_dubbed_audio, align_segment_boundaries, apply_stereo_panning
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
from dubber.cache import get_cached_translation, add_to_translation_cache
from dubber.profiler import profiler, estimate_api_characters, generate_html_report
from dubber.utils import compress_video_for_web, calculate_speaking_rate, detect_source_language
from pydub import AudioSegment
import subprocess

def mux_video_audio(video_path: str, audio_path: str, output_path: str):
    """
    Combines the original video stream with the dubbed audio track using FFmpeg.
    Copies the video stream without re-encoding to save time and retain quality.
    """
    print(f"[Info] Muxing video and audio into final file: {output_path}...")
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-map', '0:v',      # Video from 1st input (original video)
        '-map', '1:a',      # Audio from 2nd input (dubbed audio)
        '-c:v', 'copy',     # Stream copy video (no re-encoding, extremely fast)
        '-c:a', 'aac',      # AAC audio encoding
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg muxing failed: {result.stderr}")
    print("[Info] Video and audio muxed successfully.")

def process_dubbing(url: str, output_video_path: str, voice: str = DEFAULT_TTS_VOICE, 
                    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
                    translation_model: str = DEFAULT_TRANSLATION_MODEL,
                    target_language: str = "English",
                    download_quality: str = "1080p",
                    audio_mode: str = "duck", duck_level_db: float = -20.0,
                    tts_engine: str = "edge-tts", eleven_api_key: str = None, f5_tts_url: str = None,
                    eq_low: float = DEFAULT_EQ_LOW, eq_mid: float = DEFAULT_EQ_MID, eq_high: float = DEFAULT_EQ_HIGH,
                    noise_gate: float = DEFAULT_NOISE_GATE, burn_subtitles: bool = False,
                    use_cache: bool = True, compress_video: bool = False, crf_value: int = DEFAULT_CRF,
                    keep_temp: bool = False, waveform_overlay: bool = False) -> dict:
    """
    Runs the complete dubbing pipeline in CLI mode, incorporating Equalizer bands, noise gates,
    caching layers, profiling metrics, and subtitle generation/burn-in.
    """
    global_start = time.time()
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("="*60)
    print("        STARTING AUTOMATED VIDEO DUBBING PIPELINE        ")
    print("="*60)
    print(f"YouTube URL:     {url}")
    print(f"Output Path:     {output_video_path}")
    print(f"Target Language: {target_language}")
    print(f"Video Quality:   {download_quality}")
    print(f"Audio Mode:      {audio_mode} (Duck Level: {duck_level_db}dB if ducking)")
    print(f"TTS Engine:      {tts_engine}")
    if f5_tts_url:
        print(f"F5-TTS URL:      {f5_tts_url}")
    print(f"Gemini Model:    {transcription_model}")
    print(f"Equalizer:       Bass: {eq_low}dB, Mid: {eq_mid}dB, Treble: {eq_high}dB")
    print(f"Noise Gate:      {noise_gate} dBFS")
    print(f"Subtitles:       Burn-in: {burn_subtitles}")
    print(f"Cache:           Use local cache: {use_cache}")
    print("="*60)
    
    cloned_voices = {}
    
    try:
        # Step 1: Download video & extract original audio
        print(f"\n>>> Step 1: Downloading video ({download_quality}) & extracting audio...")
        profiler.start("download")
        video_path, audio_path = download_youtube_video(url, TEMP_DIR, quality=download_quality)
        profiler.stop("download")
        
        # Step 1.5: Auto-detect source language
        print("\n>>> Step 1.5: Running source language auto-detection...")
        detected_lang = detect_source_language(audio_path, model=transcription_model)
        print(f"[Info] Video language auto-detected as: {detected_lang}")
        
        # Step 2: Transcribe using Gemini API
        print("\n>>> Step 2: Transcribing speech (original language) via Gemini API...")
        profiler.start("transcribe")
        segments = transcribe_audio(audio_path, model=transcription_model)
        profiler.stop("transcribe")
        
        if not segments:
            raise ValueError("No speech segments detected in the video.")
            
        print(f"[Info] Transcribed {len(segments)} segments.")
        
        # Step 2.5: Align segment boundaries using silence ranges
        print("\n>>> Step 2.5: Aligning segment boundaries to silence gaps...")
        segments = align_segment_boundaries(segments, audio_path)
        
        # Step 3: Diarization using Gemini API & Acoustic Pitch Clustering
        print("\n>>> Step 3: Running speaker separation (diarization) via Gemini & Acoustic Pitch Analysis...")
        profiler.start("diarize")
        diarized_segments = diarize_segments(segments, model=transcription_model, audio_path=audio_path)
        profiler.stop("diarize")
        
        # Step 4: Translate using Gemini API (with Caching)
        print(f"\n>>> Step 4: Translating speech segments to {target_language}...")
        profiler.start("translate")
        
        translated_segments = []
        uncached_segments = []
        uncached_indices = []
        
        # Look for translations in cache
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
                # Placeholder for ordering
                translated_segments.append(None)
                
        # Send only uncached elements to translation API
        if uncached_segments:
            print(f"[Info] Querying Gemini for {len(uncached_segments)} uncached translations to {target_language}...")
            api_translated = translate_segments(uncached_segments, model=translation_model, target_language=target_language)
            
            # Map back and add to cache
            for idx, trans_seg in zip(uncached_indices, api_translated):
                translated_segments[idx] = trans_seg
                if use_cache:
                    add_to_translation_cache(diarized_segments[idx]["text"], trans_seg["text"], source_lang=detected_lang, target_lang=target_language)
        else:
            print(f"[Info] All translation segments successfully loaded from local {target_language} cache.")
            
        profiler.stop("translate")
        
        # Step 4.2: Preserve complete translation without aggressive shortening
        condensed_segments = []
        for seg in translated_segments:
            cond_seg = condense_translation_if_fast(seg, max_wpm=260.0, model=translation_model, target_language=target_language)
            condensed_segments.append(cond_seg)
        translated_segments = condensed_segments
        
        # Map translated segments and assign distinct voices to unique speakers
        mapped_segments, voice_map = assign_voices_to_speakers(translated_segments)

        
        # Generate Subtitle files: SRT, WebVTT, and ASS Styled Subtitles
        srt_path = os.path.splitext(output_video_path)[0] + ".srt"
        vtt_path = os.path.splitext(output_video_path)[0] + ".vtt"
        ass_path = os.path.splitext(output_video_path)[0] + ".ass"
        generate_srt_subtitles(mapped_segments, srt_path)
        generate_vtt_subtitles(mapped_segments, vtt_path)
        generate_ass_subtitles(mapped_segments, ass_path)
        
        # Step 4.5: Clone voices on ElevenLabs if configured
        if tts_engine == "elevenlabs" and eleven_api_key:
            print("\n>>> Step 4.5: Creating cloned voices on ElevenLabs...")
            unique_speakers = sorted(list(set(seg["speaker"] for seg in mapped_segments)))
            
            for speaker in unique_speakers:
                sample_wav = os.path.join(TEMP_DIR, f"sample_{speaker.replace(' ', '_')}.wav")
                if extract_speaker_sample(audio_path, mapped_segments, speaker, sample_wav):
                    print(f"[Info] Uploading sample for {speaker} to ElevenLabs...")
                    try:
                        voice_id = add_elevenlabs_voice(eleven_api_key, speaker, sample_wav)
                        cloned_voices[speaker] = voice_id
                        print(f"[Success] Cloned voice created for {speaker}: {voice_id}")
                    except Exception as e:
                        print(f"[Error] Failed to clone voice for {speaker}: {e}. Will fallback to Edge-TTS.")
                
            # Map the voice parameters in mapped_segments to ElevenLabs voice IDs
            for seg in mapped_segments:
                spk = seg["speaker"]
                if spk in cloned_voices:
                    seg["voice"] = cloned_voices[spk]
                    
        # Extract speaker voice reference samples for F5-TTS if enabled
        speaker_refs = {}
        if tts_engine == "f5-tts" and f5_tts_url:
            print("\n>>> Step 4.5: Extracting speaker references for F5-TTS...")
            unique_speakers = sorted(list(set(seg["speaker"] for seg in mapped_segments)))
            for speaker in unique_speakers:
                raw_ref_wav = os.path.join(TEMP_DIR, f"raw_ref_{speaker.replace(' ', '_')}.wav")
                ref_wav = os.path.join(TEMP_DIR, f"ref_{speaker.replace(' ', '_')}.wav")
                if extract_speaker_sample(audio_path, mapped_segments, speaker, raw_ref_wav):
                    from dubber.enhancer import extract_vocals_filter
                    extract_vocals_filter(raw_ref_wav, ref_wav)
                    
                    # Find reference text (original spoken text for reference audio)
                    speaker_segs = [s for s in mapped_segments if s.get("speaker") == speaker]
                    speaker_segs.sort(key=lambda x: x["end_time"] - x["start_time"], reverse=True)
                    ref_text_val = "Hello"
                    if speaker_segs:
                        longest_seg = speaker_segs[0]
                        matching_orig = [s for s in diarized_segments if s["start_time"] == longest_seg["start_time"]]
                        ref_text_val = matching_orig[0]["text"] if matching_orig else longest_seg["text"]
                        
                    speaker_refs[speaker] = {
                        "audio_path": ref_wav,
                        "text": ref_text_val
                    }
        
        # Step 5: Synthesize translated segments with speed matching
        print("\n>>> Step 5: Synthesizing English audio segments...")
        profiler.start("synthesize")
        segment_files = []
        
        for idx, seg in enumerate(tqdm(mapped_segments, desc="Speech Synthesis")):
            target_dur = seg["end_time"] - seg["start_time"]
            
            wpm = calculate_speaking_rate(seg["text"], target_dur)
            if wpm > 180.0:
                print(f"[Warning] Segment {idx} speaking rate is extremely fast: {wpm} WPM. Speech may sound compressed.")
            
            ref_path = None
            ref_txt = None
            if tts_engine == "f5-tts" and seg["speaker"] in speaker_refs:
                ref_path = speaker_refs[seg["speaker"]]["audio_path"]
                ref_txt = speaker_refs[seg["speaker"]]["text"]
                
            seg_wav = synthesize_segment(
                text=seg["text"], 
                target_duration=target_dur, 
                voice=seg["voice"], 
                segment_id=idx,
                tts_engine=tts_engine,
                eleven_api_key=eleven_api_key,
                f5_tts_url=f5_tts_url,
                ref_audio_path=ref_path,
                ref_text=ref_txt
            )
            
            # Apply Vocal Quality Enhancements: EQ, noise gate, limiter, and fades
            raw_seg_audio = AudioSegment.from_wav(seg_wav)
            enhanced_audio = apply_fade_borders(raw_seg_audio, fade_ms=20)
            
            # Parametric EQ (Automatic Pitch-based or Manual)
            if eq_low == 0.0 and eq_mid == 0.0 and eq_high == 0.0:
                auto_low, auto_mid, auto_high = analyze_eq_profile(raw_seg_audio)
                enhanced_audio = apply_three_band_eq(enhanced_audio, auto_low, auto_mid, auto_high)
            else:
                enhanced_audio = apply_three_band_eq(enhanced_audio, eq_low, eq_mid, eq_high)
            
            # Spatial Stereo Panning
            unique_speakers = sorted(list(set(s["speaker"] for s in mapped_segments)))
            enhanced_audio = apply_stereo_panning(enhanced_audio, seg["speaker"], unique_speakers)
            
            # Noise Gate & Limiter
            enhanced_audio = remove_bg_noise(enhanced_audio, noise_gate)
            enhanced_audio = apply_limiter(enhanced_audio, max_dbfs=-1.0)
            
            # Re-export enhanced WAV
            enhanced_audio.export(seg_wav, format="wav")
            segment_files.append(seg_wav)
            
        profiler.stop("synthesize")
        
        # Step 6: Assemble segments into one unified audio track
        print("\n>>> Step 6: Assembling dubbed audio track...")
        profiler.start("mux")
        dubbed_audio_path = os.path.join(TEMP_DIR, "dubbed_final.wav")
        assemble_dubbed_audio(
            segments=mapped_segments,
            segment_files=segment_files,
            original_audio_path=audio_path,
            output_audio_path=dubbed_audio_path,
            mode=audio_mode,
            duck_level_db=duck_level_db
        )
        
        # Step 7: Mux video and dubbed audio (Standard or Waveform overlay)
        temp_mux_path = os.path.join(TEMP_DIR, "temp_muxed.mp4")
        if waveform_overlay:
            generate_waveform_overlay(video_path, dubbed_audio_path, temp_mux_path)
        else:
            mux_video_audio(video_path, dubbed_audio_path, temp_mux_path)
        
        # Optional: Burn subtitles into video
        if burn_subtitles:
            print("\n>>> Step 7.1: Burning subtitles into video track...")
            burn_subtitles_into_video(temp_mux_path, srt_path, output_video_path)
        else:
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(temp_mux_path, output_video_path)
            
        # Optional: Web compression re-encode
        if compress_video:
            print("\n>>> Step 7.2: Compressing final video for web sharing...")
            compressed_path = os.path.splitext(output_video_path)[0] + "_compressed.mp4"
            compress_video_for_web(output_video_path, compressed_path, crf_value)
            
        profiler.stop("mux")
        
        processing_duration = time.time() - global_start
        print("\n" + "="*60)
        print("           DUBBING PIPELINE COMPLETED SUCCESSFULLY          ")
        print("="*60)
        print(f"Final output: {output_video_path}")
        print(f"Total time:   {processing_duration:.2f} seconds ({processing_duration/60.0:.2f} minutes)")
        print("="*60)
        
        # Generate Performance Report
        report_data = profiler.generate_report_data()
        char_stats = estimate_api_characters(mapped_segments)
        report_html_path = os.path.splitext(output_video_path)[0] + "_report.html"
        generate_html_report(report_data, char_stats, report_html_path)
        
        return {
            "success": True,
            "output_path": output_video_path,
            "duration_sec": processing_duration,
            "segments_count": len(segments),
            "speakers_count": len(voice_map)
        }
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        raise e
        
    finally:
        # Clean up temporary cloned voices from ElevenLabs to prevent clogging account
        if cloned_voices and eleven_api_key:
            print("\n>>> Step 7.5: Cleaning up cloned voices from ElevenLabs account...")
            for speaker, voice_id in cloned_voices.items():
                delete_elevenlabs_voice(eleven_api_key, voice_id)
                
        # Step 8: Cleanup local temp folder
        if not keep_temp:
            print("\n>>> Step 8: Cleaning up temporary files...")
            try:
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)
                    print("[Info] Temporary folder cleaned up.")
            except Exception as e:
                print(f"[Warning] Failed to clean up temporary folder {TEMP_DIR}: {e}")

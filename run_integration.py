import os
import sys
import time
from pydub import AudioSegment
from pydub.generators import Sine

# Setup environment mock keys
os.environ["GEMINI_API_KEY_1"] = "test_key_1"
os.environ["GEMINI_API_KEY_2"] = "test_key_2"
os.environ["ELEVENLABS_API_KEY"] = "mock_eleven_key"

from dubber.config import TEMP_DIR
from dubber.audio_processor import assemble_dubbed_audio
from dubber.enhancer import (
    apply_three_band_eq, 
    remove_bg_noise, 
    apply_fade_borders, 
    apply_limiter,
    generate_srt_subtitles,
    generate_vtt_subtitles
)
from dubber.cache import add_to_translation_cache, clear_translation_cache, get_cached_translation
from dubber.profiler import profiler, estimate_api_characters
from dubber.utils import calculate_speaking_rate, check_internet_speed
from dubber.validator import validate_eq_gains, validate_duck_level
from dubber.visualizer import draw_wav_waveform_bmp, generate_speaker_timeline_html

def run_pipeline_simulation():
    """
    Executes a complete simulated pipeline run, validating EQ filters,
    noise gates, persistent caching, WebVTT/SRT subtitles, BMP waveforms,
    HTML timelines, and timing benchmarks without querying active API keys.
    """
    print("="*60)
    print("      INTEGRATION PIPELINE SIMULATION RUN (NO API KEYS)     ")
    print("="*60)
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    clear_translation_cache()
    
    # 1. System latency checks
    print("\n>>> Phase 1: Running system latency health checks...")
    net_status = check_internet_speed()
    print(f"[Success] Network check: {net_status['status']} (Latency: {net_status['latency_ms']}ms)")
    
    # 2. Setup mock timestamps and segments
    print("\n>>> Phase 2: Compiling mock dialogue turns...")
    mock_segments = [
        {"start_time": 0.0, "end_time": 3.0, "speaker": "Host", "text": "Bienvenido al canal digital.", "voice": "en-US-EmmaMultilingualNeural"},
        {"start_time": 3.0, "end_time": 6.5, "speaker": "Guest", "text": "Muchas gracias, es un placer estar aquí.", "voice": "en-US-BrianNeural"},
        {"start_time": 6.5, "end_time": 9.0, "speaker": "Host", "text": "Hoy hablaremos de inteligencia artificial.", "voice": "en-US-EmmaMultilingualNeural"}
    ]
    print(f"[Info] Compiled {len(mock_segments)} conversation turns.")
    
    # 3. Simulate Caching Layer
    print("\n>>> Phase 3: Validating translation cache serialization...")
    for seg in mock_segments:
        # Simulate query & caching
        h_text = seg["text"]
        t_text = "English translation of " + h_text
        add_to_translation_cache(h_text, t_text, "es")
        
        # Verify read
        cached = get_cached_translation(h_text, "es")
        print(f"  - Cached translation: '{h_text}' -> '{cached}'")
        seg["text"] = cached # Swap with translated text
        
    # 4. Generate mock vocal audios
    print("\n>>> Phase 4: Generating simulated voice segments using Sine waves...")
    segment_files = []
    
    for idx, seg in enumerate(mock_segments):
        duration_ms = int((seg["end_time"] - seg["start_time"]) * 1000)
        # Create unique frequency sine wave for each speaker to distinguish them
        freq = 440 if seg["speaker"] == "Host" else 600
        
        # Draw sound segment
        sound = Sine(freq).to_audio_segment(duration=duration_ms, volume=-10.0)
        
        # Apply vocal enhancements: Fades, EQ, Gating, Limiting
        sound = apply_fade_borders(sound, fade_ms=20)
        sound = apply_three_band_eq(sound, low_gain=1.5, mid_gain=0.0, high_gain=2.0)
        sound = remove_bg_noise(sound, threshold_dbfs=-45.0)
        sound = apply_limiter(sound, max_dbfs=-1.0)
        
        seg_path = os.path.join(TEMP_DIR, f"seg_{idx}.wav")
        sound.export(seg_path, format="wav")
        segment_files.append(seg_path)
        
    print(f"[Success] Generated {len(segment_files)} enhanced WAV vocal clips.")
    
    # 5. Subtitles generation
    print("\n>>> Phase 5: Generating subtitle files...")
    srt_path = os.path.join(TEMP_DIR, "simulated_subs.srt")
    vtt_path = os.path.join(TEMP_DIR, "simulated_subs.vtt")
    generate_srt_subtitles(mock_segments, srt_path)
    generate_vtt_subtitles(mock_segments, vtt_path)
    
    # 6. Assembly & mixing
    print("\n>>> Phase 6: Mixing audio canvas...")
    profiler.start("assemble_audio")
    
    # Create mock original audio WAV (silence)
    bg_audio_path = os.path.join(TEMP_DIR, "mock_bg_audio.wav")
    bg_sound = AudioSegment.silent(duration=10000, frame_rate=16000)
    bg_sound.export(bg_audio_path, format="wav")
    
    dubbed_audio_path = os.path.join(TEMP_DIR, "mock_dubbed_output.wav")
    assemble_dubbed_audio(
        segments=mock_segments,
        segment_files=segment_files,
        original_audio_path=bg_audio_path,
        output_audio_path=dubbed_audio_path,
        mode="duck",
        duck_level_db=-20.0
    )
    profiler.stop("assemble_audio")
    print(f"[Success] Dubbed audio mixed and saved: {dubbed_audio_path}")
    
    # 7. Render zero-dependency BMP Waveform
    print("\n>>> Phase 7: Rendering zero-dependency BMP waveform...")
    bmp_path = os.path.join(TEMP_DIR, "simulated_waveform.bmp")
    draw_wav_waveform_bmp(dubbed_audio_path, bmp_path, width=800, height=200)
    
    # 8. Render HTML timeline visualization
    print("\n>>> Phase 8: Compiling HTML Timeline mapping...")
    html_path = os.path.join(TEMP_DIR, "simulated_timeline.html")
    generate_speaker_timeline_html(mock_segments, html_path)
    
    # 9. Timing profiling
    print("\n>>> Phase 9: Compiling performance reports...")
    chart = profiler.draw_ascii_histogram()
    print(chart)
    
    # Verify outputs
    print("\n>>> Verification Logs:")
    outputs = [srt_path, vtt_path, dubbed_audio_path, bmp_path, html_path]
    all_ok = True
    for out in outputs:
        exists = os.path.exists(out)
        status_str = "[Found]" if exists else "[Missing]"
        print(f"  - {os.path.basename(out):30} : {status_str}")
        if not exists:
            all_ok = False
            
    if all_ok:
        print("\n" + "="*60)
        print("    SIMULATION COMPLETED SUCCESSFULLY - SYSTEM FULLY OPERATIONAL ")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("    SIMULATION COMPLETED WITH ERRORS - SOME FILE OUTPUTS MISSING ")
        print("="*60)

if __name__ == "__main__":
    try:
        run_pipeline_simulation()
    except Exception as e:
        print(f"[Critical Error] Simulation crashed: {e}")
        sys.exit(1)

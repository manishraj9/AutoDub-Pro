import os
import time
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pydub import AudioSegment
from pydub.generators import Sine
import csv

# Set mock env keys for safety
os.environ["GEMINI_API_KEY_1"] = "test_key_1"
os.environ["GEMINI_API_KEY_2"] = "test_key_2"
os.environ["ELEVENLABS_API_KEY"] = "mock_eleven_key"

from dubber.config import GeminiClientManager, TEMP_DIR, CACHE_DIR
from dubber.transcriber import chunk_audio
from dubber.translator import translate_segments
from dubber.diarization import diarize_segments, assign_voices_to_speakers
from dubber.synthesizer import (
    change_audio_speed,
    extract_speaker_sample,
    add_elevenlabs_voice,
    delete_elevenlabs_voice,
    synthesize_elevenlabs_speech
)
from dubber.audio_processor import merge_speech_intervals, create_ducked_background

# Core modules
from dubber.enhancer import (
    format_timestamp_srt,
    format_timestamp_vtt,
    generate_srt_subtitles,
    generate_vtt_subtitles,
    apply_three_band_eq,
    remove_bg_noise,
    add_high_pass_filter,
    add_low_pass_filter,
    apply_limiter,
    apply_fade_borders,
    auto_level_vocals
)
from dubber.cache import (
    generate_text_hash,
    get_cache_filepath,
    load_translation_cache,
    save_translation_cache,
    get_cached_translation,
    add_to_translation_cache,
    clear_translation_cache,
    get_cached_segments_count,
    export_cache_to_csv,
    import_cache_from_csv
)
from dubber.profiler import profiler, estimate_api_characters
from dubber.utils import (
    calculate_speaking_rate,
    detect_silence_ranges,
    check_api_keys_validity,
    check_internet_speed,
    analyze_system_ffmpeg_codecs,
    verify_audio_sample_rate_conversion
)
from dubber.validator import (
    validate_youtube_url,
    get_youtube_video_id,
    validate_ffmpeg_installed,
    validate_write_permissions,
    validate_audio_wav_header,
    validate_video_duration,
    validate_segment_intervals,
    validate_duck_level,
    validate_eq_gains,
    retrieve_ffmpeg_audio_codecs
)
from dubber.visualizer import (
    write_bmp_header,
    draw_wav_waveform_bmp,
    generate_speaker_timeline_html
)

class TestVideoDubberPipeline(unittest.TestCase):
    
    def setUp(self):
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        clear_translation_cache()
        
    def tearDown(self):
        clear_translation_cache()

    # ==========================================
    # CORE PIPELINE AUDIO TESTS (6 tests)
    # ==========================================

    def test_audio_chunking_logic(self):
        mock_audio = AudioSegment.silent(duration=10000, frame_rate=16000)
        with patch('pydub.AudioSegment.from_wav', return_value=mock_audio):
            with patch.object(AudioSegment, 'export') as mock_export:
                chunks = chunk_audio("dummy_path.wav", chunk_length_sec=3)
                self.assertEqual(len(chunks), 4)
                self.assertEqual(chunks[0][1], 0.0)
                self.assertEqual(chunks[1][1], 3.0)
                self.assertEqual(chunks[2][1], 6.0)
                self.assertEqual(chunks[3][1], 9.0)
                self.assertEqual(mock_export.call_count, 4)

    def test_translation_batching(self):
        segments = [{"start_time": float(i), "end_time": i + 0.5, "text": f"speech_{i}"} for i in range(45)]
        def mock_translate_batch(client, batch, model, target_language="English"):
            return [{"start_time": seg["start_time"], "end_time": seg["end_time"], "text": f"english_{seg['start_time']}"} 
                    for seg in batch]

        with patch('dubber.translator.translate_batch', side_effect=mock_translate_batch):
            with patch('dubber.config.client_manager.execute_with_retry', side_effect=lambda f, *a, **k: f(MagicMock())):
                results = translate_segments(segments, model="gemini-3.1-flash-lite")
                self.assertEqual(len(results), 45)
                self.assertEqual(results[0]["text"], "english_0.0")

    def test_speaker_diarization_mapping(self):
        diarized_segments = [
            {"start_time": 0.0, "end_time": 2.0, "text": "Hello", "speaker": "Speaker A"},
            {"start_time": 2.0, "end_time": 4.0, "text": "Hi there", "speaker": "Speaker B"}
        ]
        updated_segs, voice_map = assign_voices_to_speakers(diarized_segments)
        self.assertEqual(len(voice_map), 2)
        self.assertEqual(updated_segs[0]["voice"], voice_map["Speaker A"])
        self.assertEqual(updated_segs[1]["voice"], voice_map["Speaker B"])

    def test_speed_adjustment_filter_chaining(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            change_audio_speed("input.wav", "output.wav", 1.5)
            args1 = mock_run.call_args[0][0]
            self.assertIn("atempo=1.500", args1)
            
            change_audio_speed("input.wav", "output.wav", 2.5)
            args2 = mock_run.call_args[0][0]
            filter_str = args2[args2.index("-filter:a") + 1]
            self.assertEqual(filter_str, "atempo=2.0,atempo=1.250")

    def test_speech_intervals_merging(self):
        segments = [
            {"start_time": 1.0, "end_time": 3.0},
            {"start_time": 3.2, "end_time": 5.0}
        ]
        intervals = merge_speech_intervals(segments, total_duration_ms=10000)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0], (1000, 5000))

    def test_ducked_background_construction(self):
        mock_audio = AudioSegment.silent(duration=10000, frame_rate=16000)
        segments = [{"start_time": 2.0, "end_time": 4.0}]
        with patch.object(AudioSegment, 'append', return_value=mock_audio) as mock_append:
            ducked = create_ducked_background(mock_audio, segments, duck_level_db=-20.0)
            self.assertTrue(mock_append.called)

    # ==========================================
    # ELEVENLABS CLONING TESTS (4 tests)
    # ==========================================

    def test_speaker_audio_slice_extraction(self):
        mock_audio = AudioSegment.silent(duration=15000, frame_rate=16000)
        segments = [
            {"start_time": 1.0, "end_time": 4.0, "speaker": "Host"},
            {"start_time": 4.0, "end_time": 6.0, "speaker": "Guest"}
        ]
        with patch('pydub.AudioSegment.from_wav', return_value=mock_audio):
            with patch.object(AudioSegment, 'export') as mock_export:
                extracted = extract_speaker_sample("dummy_audio.wav", segments, "Host", "output_sample.wav")
                self.assertTrue(extracted)
                self.assertTrue(mock_export.called)

    @patch('requests.post')
    def test_elevenlabs_voice_add_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"voice_id": "mock_voice_123"}
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        
        dummy_file_path = os.path.join(TEMP_DIR, "dummy_sample.wav")
        with open(dummy_file_path, "wb") as f:
            f.write(b"WAV DATA")
            
        try:
            voice_id = add_elevenlabs_voice("dummy_api_key", "Speaker A", dummy_file_path)
            self.assertEqual(voice_id, "mock_voice_123")
        finally:
            if os.path.exists(dummy_file_path):
                os.remove(dummy_file_path)

    @patch('requests.delete')
    def test_elevenlabs_voice_delete_request(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_delete.return_value = mock_resp
        
        delete_elevenlabs_voice("dummy_api_key", "voice_abc_123")
        call_kwargs = mock_delete.call_args[1]
        self.assertEqual(call_kwargs["headers"]["xi-api-key"], "dummy_api_key")

    @patch('requests.post')
    def test_elevenlabs_speech_synthesis_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.content = b"AUDIO MP3 CONTENT"
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        
        output_file_path = os.path.join(TEMP_DIR, "output_seg.mp3")
        try:
            synthesize_elevenlabs_speech(
                api_key="dummy_api_key",
                text="Test speech text",
                voice_id="voice_xyz",
                output_path=output_file_path
            )
            self.assertTrue(os.path.exists(output_file_path))
        finally:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    # ==========================================
    # AUDIO ENHANCEMENT TESTS (5 tests)
    # ==========================================

    def test_parametric_equalizer_application(self):
        mock_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        with patch.object(AudioSegment, 'low_pass_filter', return_value=mock_audio) as mock_lp:
            with patch.object(AudioSegment, 'high_pass_filter', return_value=mock_audio) as mock_hp:
                eq_audio = apply_three_band_eq(mock_audio, low_gain=2.0, mid_gain=-1.0, high_gain=3.0)
                self.assertTrue(mock_lp.called)
                self.assertTrue(mock_hp.called)

    def test_noise_gating(self):
        sound = Sine(440).to_audio_segment(duration=500, volume=-10.0)
        silence_segment = AudioSegment.silent(duration=500, frame_rate=sound.frame_rate)
        mixed = sound + silence_segment
        
        gated = remove_bg_noise(mixed, threshold_dbfs=-40.0)
        self.assertEqual(len(gated), 1000)
        self.assertTrue(gated[750].dBFS < -60)

    def test_low_and_high_pass_filters(self):
        mock_audio = AudioSegment.silent(duration=500, frame_rate=16000)
        with patch.object(AudioSegment, 'high_pass_filter', return_value=mock_audio) as mock_hp:
            add_high_pass_filter(mock_audio, cutoff_hz=100)
            mock_hp.assert_called_with(100)
            
        with patch.object(AudioSegment, 'low_pass_filter', return_value=mock_audio) as mock_lp:
            add_low_pass_filter(mock_audio, cutoff_hz=7000)
            mock_lp.assert_called_with(7000)

    def test_fade_borders(self):
        mock_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        with patch.object(AudioSegment, 'fade_in', return_value=mock_audio) as mock_fi:
            apply_fade_borders(mock_audio, fade_ms=25)
            mock_fi.assert_called_with(25)

    def test_limiter_and_vocal_leveler(self):
        mock_audio = Sine(440).to_audio_segment(duration=500, volume=-10.0)
        limited = apply_limiter(mock_audio, max_dbfs=-12.0)
        self.assertTrue(limited.max_dBFS <= -11.9)
        
        leveled = auto_level_vocals(mock_audio, target_dbfs=-15.0)
        self.assertAlmostEqual(leveled.dBFS, -15.0, places=1)

    # ==========================================
    # CACHING LAYER TESTS (4 tests)
    # ==========================================

    def test_cache_serialization_and_retrieval(self):
        text = "Hola amigo como estas"
        translation = "Hello friend how are you"
        
        self.assertIsNone(get_cached_translation(text, "es"))
        add_to_translation_cache(text, translation, "es")
        self.assertEqual(get_cached_translation(text, "es"), translation)
        self.assertEqual(get_cached_segments_count(), 1)

    def test_cache_wiping(self):
        add_to_translation_cache("Frase 1", "Sentence 1", "es")
        add_to_translation_cache("Frase 2", "Sentence 2", "es")
        self.assertEqual(get_cached_segments_count(), 2)
        
        clear_translation_cache()
        self.assertEqual(get_cached_segments_count(), 0)

    def test_text_hash_collisions(self):
        h1 = generate_text_hash("Bonjour")
        h2 = generate_text_hash("bonjour")
        self.assertNotEqual(h1, h2)

    def test_csv_export_and_import(self):
        add_to_translation_cache("Texto de prueba", "Test text", "es")
        csv_path = os.path.join(TEMP_DIR, "cache_export.csv")
        
        try:
            exported = export_cache_to_csv(csv_path, "es")
            self.assertTrue(exported)
            self.assertTrue(os.path.exists(csv_path))
            
            clear_translation_cache()
            self.assertEqual(get_cached_segments_count(), 0)
            
            imported = import_cache_from_csv(csv_path, "es")
            self.assertEqual(imported, 1)
            self.assertEqual(get_cached_translation("Texto de prueba", "es"), "Test text")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    # ==========================================
    # INPUT VALIDATION TESTS (5 tests)
    # ==========================================

    def test_youtube_url_validators(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ"
        ]
        for url in valid_urls:
            self.assertTrue(validate_youtube_url(url))
            self.assertEqual(get_youtube_video_id(url), "dQw4w9WgXcQ")
            
        invalid_urls = [
            "https://google.com",
            "http://youtube.com/watch?v=too_short",
            "invalid_text"
        ]
        for url in invalid_urls:
            self.assertFalse(validate_youtube_url(url))

    def test_ffmpeg_and_duration_probe_check(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="125.43\n")
            installed, msg = validate_ffmpeg_installed()
            self.assertTrue(installed)
            
            duration = validate_video_duration("test_suite.py")
            self.assertEqual(duration, 125.43)

    def test_write_permissions_validator(self):
        self.assertTrue(validate_write_permissions(TEMP_DIR))
        self.assertFalse(validate_write_permissions("Z:\\invalid_folder_location_test"))

    def test_eq_and_duck_bounds_validator(self):
        ok, msg = validate_eq_gains(2.0, -1.0, 3.0)
        self.assertTrue(ok)
        
        bad, msg = validate_eq_gains(15.0, 0.0, 0.0)
        self.assertFalse(bad)
        
        ok_duck, msg = validate_duck_level(-15.0)
        self.assertTrue(ok_duck)
        
        bad_duck, msg = validate_duck_level(5.0)
        self.assertFalse(bad_duck)

    def test_segment_intervals_validator(self):
        good_segs = [
            {"start_time": 0.0, "end_time": 2.5},
            {"start_time": 2.5, "end_time": 5.0}
        ]
        ok, msg = validate_segment_intervals(good_segs, max_duration=10.0)
        self.assertTrue(ok)
        
        bad_segs = [
            {"start_time": 4.0, "end_time": 2.0}
        ]
        bad, msg = validate_segment_intervals(bad_segs, max_duration=10.0)
        self.assertFalse(bad)

    # ==========================================
    # SYSTEM UTILITIES TESTS (3 tests)
    # ==========================================

    def test_subtitle_formatting_timestamps(self):
        srt_t = format_timestamp_srt(1.5)
        vtt_t = format_timestamp_vtt(1.5)
        self.assertEqual(srt_t, "00:00:01,500")
        self.assertEqual(vtt_t, "00:00:01.500")
        
        srt_large = format_timestamp_srt(3665.25)
        self.assertEqual(srt_large, "01:01:05,250")

    def test_speaking_rate_and_profiler(self):
        wpm = calculate_speaking_rate("hello my dear friend", duration_sec=2.0)
        self.assertEqual(wpm, 120.0)
        
        profiler.start("stage_a")
        time.sleep(0.01)
        profiler.stop("stage_a")
        report = profiler.generate_report_data()
        self.assertTrue(report["total_duration_sec"] > 0.0)
        self.assertEqual(report["stages"][0]["stage"], "stage_a")

    @patch('subprocess.run')
    def test_latency_checker_utility(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        status = check_internet_speed()
        self.assertEqual(status["status"], "online")
        self.assertTrue(status["latency_ms"] >= 0.0)

    # ==========================================
    # VISUALIZER TESTS (3 tests)
    # ==========================================

    def test_bmp_header_generation(self):
        headers = write_bmp_header(100, 50)
        # Check size and BM identifier
        self.assertEqual(len(headers), 54)
        self.assertEqual(headers[0:2], b'BM')

    def test_bmp_waveform_generation(self):
        # Generate short sine wav file
        wav_path = os.path.join(TEMP_DIR, "test_waveform_src.wav")
        bmp_path = os.path.join(TEMP_DIR, "test_waveform_out.bmp")
        
        sound = Sine(440).to_audio_segment(duration=200, volume=-10.0)
        sound.export(wav_path, format="wav")
        
        try:
            drawn = draw_wav_waveform_bmp(wav_path, bmp_path, width=200, height=50)
            self.assertTrue(drawn)
            self.assertTrue(os.path.exists(bmp_path))
            
            # Check BMP file size is correct (54 header + (200width * 3BGR = 600 bytes row * 50height = 30000 bytes pixel data))
            # BMP row size is already multiple of 4 (200*3 = 600, which is divisible by 4), so no row padding
            self.assertEqual(os.path.getsize(bmp_path), 30054)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(bmp_path):
                os.remove(bmp_path)

    def test_html_speaker_timeline_generation(self):
        segments = [
            {"start_time": 0.0, "end_time": 2.0, "speaker": "Alice", "text": "Hi Bob!"},
            {"start_time": 2.0, "end_time": 4.5, "speaker": "Bob", "text": "Hey Alice, how is it going?"}
        ]
        html_path = os.path.join(TEMP_DIR, "timeline_output.html")
        try:
            generate_speaker_timeline_html(segments, html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Alice", content)
                self.assertIn("Bob", content)
        finally:
            if os.path.exists(html_path):
                os.remove(html_path)

    def test_developer_guide_loading(self):
        """Verifies that the built-in system manual text loads correctly."""
        from dubber.developer_guide import get_developer_guide
        guide = get_developer_guide()
        self.assertIn("IDEALABS DUBBER PRO", guide)
        self.assertIn("3-BAND PARAMETRIC EQUALIZER", guide)

    def test_generate_diagnostic_report_check(self):
        """Verifies compile diagnostics report format and completeness."""
        from dubber.utils import generate_diagnostic_report
        report = generate_diagnostic_report(TEMP_DIR, CACHE_DIR)
        self.assertIn("timestamp", report)
        self.assertIn("ffmpeg_status", report)
        self.assertIn("network_status", report)
        self.assertIn("directories", report)
        self.assertIn("codec_info", report)

    def test_edge_case_empty_timeline(self):
        """Verifies that visual timeline builder handles empty segment logs safely."""
        html_path = os.path.join(TEMP_DIR, "empty_timeline.html")
        try:
            generate_speaker_timeline_html([], html_path)
            self.assertFalse(os.path.exists(html_path))
        finally:
            if os.path.exists(html_path):
                os.remove(html_path)

    def test_diarization_unassigned_voices(self):
        """Verifies assigning edge-voices handles fallback formats."""
        segments = [{"start_time": 0.0, "end_time": 2.0, "speaker": "A"}]
        updated, vmap = assign_voices_to_speakers(segments)
        self.assertEqual(updated[0]["voice"], vmap["A"])

    def test_ffmpeg_codec_retrievals_validator(self):
        """Verifies standard formats exist in the system ffmpeg codec scans."""
        codecs = retrieve_ffmpeg_audio_codecs()
        self.assertIsInstance(codecs, list)

    # ==========================================
    # ADVANCED FEATURE TESTS (10 tests)
    # ==========================================
    def test_condense_translation_if_fast(self):
        from dubber.translator import condense_translation_if_fast
        seg = {"start_time": 0.0, "end_time": 1.0, "text": "This is a very long segment that will exceed the WPM limit of 170. It needs to be condensed."}
        with patch('dubber.config.client_manager.execute_with_retry', return_value="Short condensed version."):
            res = condense_translation_if_fast(seg, max_wpm=10, model="gemini")
            self.assertEqual(res["text"], "Short condensed version.")

    def test_apply_stereo_panning(self):
        from dubber.audio_processor import apply_stereo_panning
        sound = AudioSegment.silent(duration=1000, frame_rate=44100)
        panned = apply_stereo_panning(sound, "Speaker A", ["Speaker A", "Speaker B"])
        self.assertEqual(panned.channels, 2)

    def test_detect_source_language(self):
        from dubber.utils import detect_source_language
        wav_path = os.path.join(TEMP_DIR, "test_lang_src.wav")
        sound = Sine(440).to_audio_segment(duration=100, volume=-10.0)
        sound.export(wav_path, format="wav")
        try:
            with patch('dubber.config.client_manager.execute_with_retry', return_value="Spanish"):
                lang = detect_source_language(wav_path, model="gemini")
                self.assertEqual(lang, "Spanish")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_extract_vocals_filter(self):
        from dubber.enhancer import extract_vocals_filter
        wav_path = os.path.join(TEMP_DIR, "test_vocals_src.wav")
        out_path = os.path.join(TEMP_DIR, "test_vocals_out.wav")
        sound = Sine(440).to_audio_segment(duration=100, volume=-10.0)
        sound.export(wav_path, format="wav")
        try:
            with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
                extract_vocals_filter(wav_path, out_path)
                self.assertTrue(mock_run.called)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_generate_ass_subtitles(self):
        from dubber.enhancer import generate_ass_subtitles
        segments = [{"start_time": 0.0, "end_time": 2.0, "speaker": "Speaker 1", "text": "Hello world"}]
        ass_path = os.path.join(TEMP_DIR, "test_sub.ass")
        try:
            generate_ass_subtitles(segments, ass_path)
            self.assertTrue(os.path.exists(ass_path))
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Dialogue:", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_generate_waveform_overlay(self):
        from dubber.enhancer import generate_waveform_overlay
        with patch('subprocess.run', return_value=MagicMock(returncode=0)) as mock_run:
            generate_waveform_overlay("video.mp4", "audio.wav", "out.mp4")
            self.assertTrue(mock_run.called)

    def test_analyze_eq_profile(self):
        from dubber.enhancer import analyze_eq_profile
        sound = AudioSegment.silent(duration=1000, frame_rate=44100)
        eq = analyze_eq_profile(sound)
        self.assertEqual(len(eq), 3)

    def test_benchmark_api_keys(self):
        from dubber.utils import benchmark_api_keys
        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = MagicMock(text="Hello")
            mock_client_cls.return_value = mock_client
            res = benchmark_api_keys(["key1", "key2"])
            self.assertEqual(len(res), 2)
            self.assertEqual(res[0]["status"], "valid")

    def test_align_segment_boundaries(self):
        from dubber.audio_processor import align_segment_boundaries
        segments = [{"start_time": 1.0, "end_time": 3.0, "text": "hello"}]
        with patch('dubber.utils.detect_silence_ranges', return_value=[(0.0, 0.5), (2.9, 3.2)]):
            res = align_segment_boundaries(segments, "dummy.wav")
            self.assertAlmostEqual(res[0]["end_time"], 3.05)

    def test_preset_saving_loading(self):
        from dubber.utils import save_custom_voice_preset, load_custom_voice_preset
        preset_name = "test_preset_suite"
        voice_map = {"Speaker 1": "en-US-JennyNeural"}
        try:
            saved = save_custom_voice_preset(preset_name, voice_map)
            self.assertTrue(saved)
            loaded = load_custom_voice_preset(preset_name)
            self.assertEqual(loaded["Speaker 1"], "en-US-JennyNeural")
        finally:
            preset_file = os.path.join(os.getcwd(), "presets", f"{preset_name}.json")
            if os.path.exists(preset_file):
                os.remove(preset_file)

    def test_remove_vocals_oops(self):
        from dubber.audio_processor import remove_vocals_oops
        # Stereo test
        stereo_sound = AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(2)
        removed_stereo = remove_vocals_oops(stereo_sound)
        self.assertEqual(removed_stereo.channels, 2)
        
        # Mono test
        mono_sound = AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(1)
        removed_mono = remove_vocals_oops(mono_sound)
        self.assertEqual(removed_mono.channels, 1)

if __name__ == "__main__":
    unittest.main()

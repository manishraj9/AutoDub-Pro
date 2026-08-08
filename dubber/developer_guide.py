# Idealabs Dubber PRO - System Operations and Developer Guide
# This file serves as a built-in module documentation manual.

DEVELOPER_GUIDE_TEXT = """
================================================================================
                       IDEALABS DUBBER PRO - DEVELOPER MANUAL
================================================================================

This document details the software design, logic flows, filters, API interfaces,
and unit testing procedures for the Automated Video Translation & Dubbing System.

--------------------------------------------------------------------------------
1. ARCHITECTURAL MAP & DESIGN PATTERNS
--------------------------------------------------------------------------------
The system is built on a modular pipeline designed to download raw video inputs,
split audio tracks, transcribe text blocks using the Gemini Structured JSON API,
translate sentences contextually, separate speaker IDs, adjust speech velocities
losslessly, mix audio with logarithmic cross-faded background ducking, generate
closed caption files (SRT, WebVTT), and mux output tracks using FFmpeg stream
copying.

Modules:
- config.py: Configuration variables, directories, and the retry/rotation manager.
- downloader.py: Downloads media using yt-dlp and splits visual/audible streams.
- transcriber.py: Splits audio files and queries Gemini for speech timestamps.
- translator.py: Translates paragraphs using Gemini translation structures.
- diarization.py: Analyzes speakers and maps distinct Edge-TTS vocal IDs.
- synthesizer.py: Generates vocal segments via Edge-TTS or ElevenLabs voice prints.
- audio_processor.py: Merges segments, performs ducking, and levels volume.
- enhancer.py: 3-Band Parametric Equalizer, noise gates, and subtitle burns.
- cache.py: Persistent translation caching layer with thread-safe lock files.
- profiler.py: Tracks timing splits, RAM spikes, and calculates cost stats.
- utils.py: Silence scanning, CRF compression, pinger latency checks.
- validator.py: Strict validations of inputs, structures, parameters, and directories.
- visualizer.py: zero-dependency BMP waveform visualizer and HTML timelines.

Design Patterns:
- Client-Key Rotation (Singleton/Manager): Manages list of Gemini API keys,
  intercepting HTTP 429 exceptions to cycle indices and retry transparently.
- Caching Layer: Avoids duplicate translation queries using SHA-256 string hashes.
- Zero-Dependency BMP Rendering: Generates binary image files byte-by-byte
  without PIL or Matplotlib, making it extremely lightweight and portable.

--------------------------------------------------------------------------------
2. PARASECTION: 3-BAND PARAMETRIC EQUALIZER (EQ)
--------------------------------------------------------------------------------
To ensure maximum voice clarity during translation synthesis (especially when
cloning old or noisy vocals), the system implements a 3-Band Parametric EQ.
Frequencies are divided into three discrete ranges:

- Bass (Low frequencies): Frequencies < 250 Hz.
  Extracted using a LowPassFilter at 250Hz.
  low_band = audio.low_pass_filter(250) + low_gain
  
- Presence (Mid frequencies): Frequencies between 250 Hz and 4000 Hz.
  Extracted by applying a HighPassFilter at 250Hz and a LowPassFilter at 4000Hz.
  mid_band = audio.high_pass_filter(250).low_pass_filter(4000) + mid_gain
  
- Treble (High frequencies): Frequencies > 4000 Hz.
  Extracted using a HighPassFilter at 4000Hz.
  high_band = audio.high_pass_filter(4000) + high_gain

Summation:
The three bands are combined using overlaying:
eq_audio = low_band.overlay(mid_band).overlay(high_band)

--------------------------------------------------------------------------------
3. CLIENT KEY ROTATION SCHEMAS
--------------------------------------------------------------------------------
The Gemini API client manager (`GeminiClientManager`) manages a list of keys:
`self.api_keys = [key1, key2, key3]`

When an API call fails with:
`google.genai.errors.APIError` and code == 429 (Resource Exhausted)
The decorator `execute_with_retry` catches the error, rotates the active index:
`self.current_index = (self.current_index + 1) % len(self.api_keys)`
And retries the execution with a fresh key, ensuring continuous service.

--------------------------------------------------------------------------------
4. WEBSOCKET PROGRESS PROTOCOLS
--------------------------------------------------------------------------------
FastAPI boots a WebSocket channel at `/ws/progress`.
Messages are passed in JSON format.

Stage mapping:
- `start_pipeline`: Sends URL, models, EQ bands, noise gates, and subtitle toggles.
- `EDITING_READY`: Pushes transcribed, diarized segments to index.html workspace.
- `synthesize_and_mux`: Submits refined translations back for final assembly.
- `PIPELINE_COMPLETE`: Pushes final video file URLs to output player.

--------------------------------------------------------------------------------
5. PERSISTENT CACHING ARCHITECTURE
--------------------------------------------------------------------------------
Translation segment arrays are cached in compressed gzip archives (`.json.gz`).
File locks are implemented using `threading.Lock` to support simultaneous threads:

Import/Export Sheets:
Users can export translation cache lists into standard CSV sheets:
`export_cache_to_csv(csv_path, language)`
And import them back using:
`import_cache_from_csv(csv_path, language)`

--------------------------------------------------------------------------------
6. ZERO-DEPENDENCY WAVEFORM BMP RENDERER
--------------------------------------------------------------------------------
The waveform visualizer parses the raw PCM data from WAV files.
It calculates peak amplitudes for `width` column bins and generates:
- BITMAPFILEHEADER (14 bytes)
- BITMAPINFOHEADER (40 bytes)
- BGR Pixel Matrix (width x height x 3 bytes, padded to multiples of 4 bytes)

Header packing format:
- File header: `struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)`
- Info header: `struct.pack('<IIIHHIIIIII', 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0)`

--------------------------------------------------------------------------------
7. ADVANCED SYSTEM FUNCTIONS & F5-TTS VOICE CLONING
--------------------------------------------------------------------------------
F5-TTS is a state-of-the-art non-autoregressive voice cloning framework. This
project includes programmatic clients (`f5_tts_client.py`) and deployment notebooks
(`kaggle_f5_tts.ipynb`) utilizing Pinggy reverse SSH tunnels to run inference on
free GPU platforms.

Premium Capabilities:
1. Dynamic translation length condensing.
2. 3D spatial speaker panning (automatic 2-speaker left/right panning).
3. Automatic source language detection from a 15-second audio snippet.
4. Out-of-Phase Stereo (OOPS) cancellation (`L - R`) and bandpass vocal attenuation to strip old vocals while preserving background music.
5. Styled Advanced SubStation Alpha (.ass) subtitle file creation.
6. Real-time transparent visual waveform overlay generation with high-fidelity encoding.
7. Zero Crossing Rate (ZCR) vocal analysis for automatic EQ assignment.
8. Gemini API Key RTT latency benchmarking.
9. Snapping segment boundaries to silence periods and overlap collision checks.
10. Gender-aware speaker voice mapping and preset serialization maps.

--------------------------------------------------------------------------------
8. MANUAL VERIFICATION PROCEDURES
--------------------------------------------------------------------------------
To run local tests manually:
1. Run key rotation checks:
   `python test_key_rotation.py`
2. Run functional pipeline checks:
   `python test_suite.py`
3. Run CLI integration checks:
   `python test_cli.py`
4. Launch the Web Server Dashboard:
   `python main.py --web`
   Navigate to http://localhost:8000
"""

def get_developer_guide() -> str:
    """Returns the system architecture developer guide text."""
    return DEVELOPER_GUIDE_TEXT

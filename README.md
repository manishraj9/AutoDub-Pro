# 🎙️ AutoDub Pro: Open-Source AI Video Dubbing System

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Google Gemini API](https://img.shields.io/badge/Powered%20By-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Edge-TTS](https://img.shields.io/badge/TTS-Microsoft%20Edge--TTS-green.svg)](https://github.com/rany2/edge-tts)

**Free, Open-Source AI Video Dubbing & Voice Translation Platform** — Powered by Google Gemini, Microsoft Edge-TTS, and Acoustic Vocal Isolation. Built for **Idealabs Digital**.

> ⚠️ **Project Status Note**: **Edge-TTS** is 100% fully functional, battle-tested, and recommended for production use. Secondary engines (**F5-TTS** and custom voice cloning models) are currently under active experimental development as the project is continuously updating.

---

## 🌟 Key Highlights & Major Fixes

- **100% Start-to-End Vocal Coverage**: Ensures 100% of synthesized TTS speech plays from natural start to end without cutoffs, truncations, or premature shortening.
- **100% Background Music & FX Preservation**: Uses **Side-Channel Out-of-Phase Stereo (OOPS)** phase subtraction and multi-stage bandstop filtering to cancel original vocals while keeping 100% of background music, instruments, guitars, score, and ambient sound effects completely untouched.
- **Frame-Accurate 35s Chunking**: Eliminates multi-second timestamp drift by processing audio in 35-second physical chunks with 2.0s overlap.
- **Acoustic Trailing End-Time Extension**: Uses spectral energy envelope analysis to extend segment `end_time` to natural speech release points.
- **Hybrid Acoustic Pitch & Timbre Diarization**: Combines F0 pitch autocorrelation (Male < 165Hz, Female >= 165Hz) with Gemini dialogue context analysis to cleanly separate 3+ distinct speakers into Male and Female profiles.
- **Cinematic Localization Engine**: Translates foreign dialogue into idiomatic, natural spoken language suitable for professional dubbing and voice acting.

---

## 🏗️ Architecture & Pipeline Flow

```
[YouTube Video URL]
        │
        ▼ (yt-dlp)
 ┌───────────────┐
 │ Original MP4  │
 └──────┬────────┘
        │
        ├─────────────────────────┐
        ▼ (ffmpeg)                ▼ (Video Stream Copy)
 ┌───────────────┐         ┌───────────────┐
 │ 16kHz WAV     │         │ Video (No Aud)│
 └──────┬────────┘         └───────┬───────┘
        │ (35s Precision Chunking) │
        ▼                          │
 ┌───────────────┐                 │
 │ Audio Chunks  │                 │
 └──────┬────────┘                 │
        │                          │
        ▼ (Gemini S2T + F0 Pitch)  │
 ┌───────────────┐                 │
 │ Multi-Speaker │                 │
 │ Timed JSON    │                 │
 └──────┬────────┘                 │
        │                          │
        ▼ (Gemini Cinematic Trans) │
 ┌───────────────┐                 │
 │ Translated    │                 │
 │ Timed JSON    │                 │
 └──────┬────────┘                 │
        │                          │
        ▼ (Edge-TTS / ElevenLabs)  │
 ┌───────────────┐                 │
 │ Neural Vocals │                 │
 └──────┬────────┘                 │
        │                          │
        ▼ (Acoustic End Align)     │
 ┌───────────────┐                 │
 │ Extended &    │                 │
 │ Padded Clips  │                 │
 └──────┬────────┘                 │
        │                          │
        ▼ (Side-Channel Mix)       │
 ┌───────────────┐                 │
 │ Dubbed WAV    │                 │
 └──────┬────────┘                 │
        │                          │
        └───────────┬──────────────┘
                    ▼ (FFmpeg Lossless Muxing)
            ┌───────────────┐
            │  Dubbed MP4   │
            └───────────────┘
```

---

## 🎛️ 12 Pro Enterprise Features

1. **Smart Side-Channel Vocal Isolation**: Preserves 100% of stereo background music and environmental sound effects while cancelling center-panned spoken dialogue.
2. **Hybrid Acoustic Pitch & Timbre Diarization**: Autocorrelates fundamental frequency (F0) to cluster 3+ distinct speakers into Male and Female profiles.
3. **Acoustic Trailing Release Boundary Extension**: Dynamically extends segment `end_time` into natural phrase release envelopes so speech never cuts off early.
4. **Cinematic Dubbing Translation Engine**: Translates dialogue into natural, idiomatic speech matching character intent and duration rhythm.
5. **Automatic Gender & Voice Matching**: Maps Male and Female speakers to distinct gendered neural voices (`en-US-BrianNeural`, `en-US-EmmaMultilingualNeural`, `en-US-AvaNeural`, `en-US-AndrewNeural`, etc.).
6. **Smart Dynamic Sidechain Ducking**: Gently ducks background music (-4dB default) during active speech with smooth crossfades so background music stays rich and audible.
7. **Speech Rate Auto-Pacing & Condensing**: Shortens translations exceeding WPM speed thresholds and applies pitch-preserving time-stretching if required.
8. **Multi-Format Subtitle Exporter**: Generates styled ASS, SRT, and WebVTT caption files with custom font sizes, colors, outlines, and exact frame timestamps.
9. **Voice Presets Manager & Preset Exporter**: Saves, loads, exports, and deletes custom speaker-to-voice presets to JSON/CSV for recurring video projects.
10. **Automatic API Key Rotation Manager**: Rotates through `GEMINI_API_KEY_1..10` upon rate limit (429) hits with network latency health benchmarking.
11. **Performance Profiler & HTML Diagnostics Reporter**: Generates an HTML report containing timeline allocation, character estimates, and ASCII time charts.
12. **Zero-Dependency Waveform BMP & Animated Video Overlay**: Generates visual waveform images and animated audio visualization video overlays using FFmpeg filters.

---

## ⚡ API Key Rotation

`GeminiClientManager` (`config.py`) implements automatic API key failover:
- Loads multiple keys from environment: `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, etc.
- Upon encountering rate limit (`429`) errors or quota limits, it rotates to the next available API key automatically.

---

## 🛠️ Setup & Installation

### Prerequisites
- **Python**: version `3.9` to `3.12` (tested on `3.11.4`).
- **FFmpeg**: Must be installed and available in system PATH.

### Installation
1. Clone or extract this repository.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Configuration
1. Create a `.env` file from `.env.template`:
   ```powershell
   copy .env.template .env
   ```
2. Fill in your Gemini API keys:
   ```env
   GEMINI_API_KEY_1=AIzaSy...
   GEMINI_API_KEY_2=AIzaSy...
   GEMINI_API_KEY_3=AIzaSy...
   ```

### 🔑 How to Get Free Gemini API Keys (Step-by-Step)
1. Go to **[Google AI Studio](https://aistudio.google.com/)**.
2. Sign in with your Google account.
3. Click **"Get API Key"** in the left sidebar menu.
4. Click **"Create API Key in new project"**.
5. Copy your generated API key (starts with `AIzaSy...`).
6. Paste your key into your `.env` file as `GEMINI_API_KEY_1`.
7. *(Optional for large videos)*: Create 2-3 free keys across Google projects and add them as `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` in `.env` to enable automatic API key rotation!

---

## 🖥️ Interactive Web Dashboard

Launch the FastAPI web interface at `http://localhost:8000`:
```powershell
python main.py --web
```
Features available in the dashboard:
- Live YouTube video ingestion and progress logging over WebSockets.
- Interactive segment editor table to edit translations before rendering.
- Voice Presets Manager to save, load, and delete speaker voice configurations.
- Subtitle downloader (SRT, WebVTT, ASS) and performance report viewer.

---

## 🧪 Test Suite

Run the automated unit test suite (46 tests):
```powershell
python test_suite.py
```
Run the full integration test:
```powershell
python run_integration.py
```


import os
import sys
import argparse
from dotenv import load_dotenv

# Load env variables before importing package configurations
load_dotenv()

# Fix for asyncio ConnectionResetError on Windows when websockets disconnect
if sys.platform == 'win32':
    import asyncio
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        
        def _silenced_call_connection_lost(self, exc):
            try:
                _original_call_connection_lost(self, exc)
            except ConnectionResetError:
                pass
                
        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except ImportError:
        pass

from dubber.config import (
    client_manager, 
    DEFAULT_TRANSCRIPTION_MODEL, 
    DEFAULT_TRANSLATION_MODEL, 
    DEFAULT_TTS_VOICE, 
    ELEVENLABS_API_KEY,
    F5_TTS_URL,
    DEFAULT_EQ_LOW,
    DEFAULT_EQ_MID,
    DEFAULT_EQ_HIGH,
    DEFAULT_NOISE_GATE,
    DEFAULT_CRF
)
from dubber.processor import process_dubbing
from dubber.web_app import start_web_server

def check_environment(args):
    """
    Checks that the environment is set up correctly (Gemini API keys and ElevenLabs keys configured).
    """
    if not client_manager.api_keys:
        print("="*80)
        print("                           [CRITICAL CONFIGURATION ERROR]                       ")
        print("="*80)
        print(" No Gemini API keys found in your .env file or environment!")
        print(" Please create a '.env' file in this directory and populate it with:")
        print("   GEMINI_API_KEY_1=your_first_gemini_api_key")
        print("   GEMINI_API_KEY_2=your_second_gemini_api_key (optional)")
        print("   GEMINI_API_KEY_3=your_third_gemini_api_key (optional)")
        print("\n Alternatively, set the GEMINI_API_KEY environment variable.")
        print("="*80)
        sys.exit(1)
        
    if not args.web and args.tts_engine == "elevenlabs" and not args.eleven_api_key:
        print("="*80)
        print("                           [CONFIGURATION ERROR]                       ")
        print("="*80)
        print(" ElevenLabs API Key is required when using the 'elevenlabs' speech engine!")
        print(" Please pass it via --eleven-api-key or set ELEVENLABS_API_KEY in your .env file.")
        print("="*80)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Automated Video Dubbing System - Turn any YouTube video into an English dubbed version."
    )
    # Target URL is optional if running in web mode
    parser.add_argument(
        "url",
        type=str,
        nargs="?",
        default=None,
        help="The YouTube URL of the video to dub (not required if running in --web mode)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dubbed_output.mp4",
        help="Path where the final dubbed video will be saved (default: dubbed_output.mp4)."
    )
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=["edge-tts", "elevenlabs", "f5-tts"],
        default="edge-tts",
        help="Speech synthesis engine: 'edge-tts' for free standard voices, 'elevenlabs' for automated voice cloning, or 'f5-tts' for local/Kaggle F5 voice cloning (default: edge-tts)."
    )
    parser.add_argument(
        "--eleven-api-key",
        type=str,
        default=ELEVENLABS_API_KEY,
        help="ElevenLabs API Key for voice cloning (can also be set as ELEVENLABS_API_KEY in .env file)."
    )
    parser.add_argument(
        "--f5-tts-url",
        type=str,
        default=F5_TTS_URL,
        help="F5-TTS API server URL for voice cloning (can also be set as F5_TTS_URL in .env file)."
    )
    parser.add_argument(
        "-v", "--voice",
        type=str,
        default=DEFAULT_TTS_VOICE,
        help=f"The Edge-TTS voice to use for English speech synthesis (default: {DEFAULT_TTS_VOICE})."
    )
    parser.add_argument(
        "--transcription-model",
        type=str,
        default=DEFAULT_TRANSCRIPTION_MODEL,
        help=f"The Gemini model to use for audio transcription (default: {DEFAULT_TRANSCRIPTION_MODEL})."
    )
    parser.add_argument(
        "--translation-model",
        type=str,
        default=DEFAULT_TRANSLATION_MODEL,
        help=f"The Gemini model to use for translation to English (default: {DEFAULT_TRANSLATION_MODEL})."
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="English",
        help="Target language for translation and dubbing (e.g. English, Hindi, Spanish, French, German, Japanese, Tamil, etc.)."
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="1080p",
        choices=["4k", "1080p", "720p", "480p", "360p", "audio"],
        help="Download video quality preset (default: 1080p)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["duck", "replace"],
        default="duck",
        help="Audio mixing mode: 'duck' for professional voice-over (lowers bg audio) or 'replace' (default: duck)."
    )
    parser.add_argument(
        "--duck-level",
        type=float,
        default=-20.0,
        help="Volume reduction of background track in dB when speech occurs (default: -20.0)."
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary audio chunks and downloaded files (default: False)."
    )
    parser.add_argument(
        "--eq-low",
        type=float,
        default=DEFAULT_EQ_LOW,
        help="Bass equalization gain in dB (default: 0.0)."
    )
    parser.add_argument(
        "--eq-mid",
        type=float,
        default=DEFAULT_EQ_MID,
        help="Mid-frequency presence gain in dB (default: 0.0)."
    )
    parser.add_argument(
        "--eq-high",
        type=float,
        default=DEFAULT_EQ_HIGH,
        help="Treble presence gain in dB (default: 0.0)."
    )
    parser.add_argument(
        "--noise-gate",
        type=float,
        default=DEFAULT_NOISE_GATE,
        help="Noise gate threshold in dBFS (default: -45.0)."
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Mux and burn translation subtitles directly into the video stream (default: False)."
    )
    parser.add_argument(
        "--waveform-overlay",
        action="store_true",
        help="Mux and overlay a transparent animated waveform of the dubbed audio track (default: False)."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the translation local persistence cache (forces fresh API translation queries)."
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Re-encode output video with optimal compression configurations (default: False)."
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help="Constant Rate Factor quality factor for video compression (18-28, default: 24)."
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive local Web UI editor instead of running in the CLI."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the local web server on (default: 8000)."
    )

    args = parser.parse_args()
    
    # Verify we have API keys configured
    check_environment(args)
    
    # Launch Web Server if requested
    if args.web:
        start_web_server(port=args.port)
        return

    # Otherwise run in CLI mode
    if not args.url:
        print("[Error] YouTube URL is required when running in CLI mode. Pass a URL or use the --web flag.")
        parser.print_help()
        sys.exit(1)
        
    try:
        process_dubbing(
            url=args.url,
            output_video_path=args.output,
            voice=args.voice,
            transcription_model=args.transcription_model,
            translation_model=args.translation_model,
            target_language=args.target_language,
            download_quality=args.quality,
            audio_mode=args.mode,
            duck_level_db=args.duck_level,
            tts_engine=args.tts_engine,
            eleven_api_key=args.eleven_api_key,
            f5_tts_url=args.f5_tts_url,
            eq_low=args.eq_low,
            eq_mid=args.eq_mid,
            eq_high=args.eq_high,
            noise_gate=args.noise_gate,
            burn_subtitles=args.burn_subtitles,
            use_cache=not args.no_cache,
            compress_video=args.compress,
            crf_value=args.crf,
            keep_temp=args.keep_temp,
            waveform_overlay=args.waveform_overlay
        )

    except KeyboardInterrupt:
        print("\n[Info] Process interrupted by user. Exiting.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[Error] Dubbing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

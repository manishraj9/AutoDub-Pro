import os
import subprocess
import yt_dlp
from dubber.config import TEMP_DIR

def download_youtube_video(url: str, output_dir: str = TEMP_DIR, quality: str = "1080p") -> tuple[str, str]:
    """
    Downloads a YouTube video at chosen quality (4k, 1080p, 720p, 480p, 360p, best) and extracts its audio.
    Returns:
        tuple[str, str]: (path to downloaded video file, path to extracted WAV audio file)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp format selector based on quality requested
    q = str(quality).lower().strip()
    if q in ["4k", "2160p"]:
        format_spec = 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=2160]+bestaudio/best[ext=mp4]/best'
    elif q in ["1080p", "1080"]:
        format_spec = 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[ext=mp4]/best'
    elif q in ["720p", "720"]:
        format_spec = 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[ext=mp4]/best'
    elif q in ["480p", "480"]:
        format_spec = 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[ext=mp4]/best'
    elif q in ["360p", "360"]:
        format_spec = 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[ext=mp4]/best'
    elif q == "audio":
        format_spec = 'bestaudio/best'
    else:
        format_spec = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

    ydl_opts = {
        'format': format_spec,
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    
    print(f"[Info] Downloading YouTube video from {url} (Quality Mode: {quality})...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get('id', 'video')
        video_filename = ydl.prepare_filename(info)
        
        base, ext = os.path.splitext(video_filename)
        if ext != '.mp4' and os.path.exists(base + '.mp4'):
            video_filepath = base + '.mp4'
        else:
            video_filepath = video_filename

    if not os.path.exists(video_filepath):
        possible_path = os.path.join(output_dir, f"{video_id}.mp4")
        if os.path.exists(possible_path):
            video_filepath = possible_path
        else:
            raise FileNotFoundError(f"Could not find downloaded video file: {video_filename}")

    print(f"[Info] Video downloaded successfully ({quality}): {video_filepath}")
    
    # Extract audio using FFmpeg
    audio_filepath = os.path.join(output_dir, f"{video_id}_audio.wav")
    print(f"[Info] Extracting audio to: {audio_filepath}...")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_filepath,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_filepath
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        
    print(f"[Info] Audio extracted successfully: {audio_filepath}")
    return video_filepath, audio_filepath


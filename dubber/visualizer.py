import os
import struct
from typing import List, Tuple, Dict, Any

def write_bmp_header(width: int, height: int) -> bytes:
    """
    Generates binary headers for a 24-bit uncompressed BMP image.
    Structure:
    - BITMAPFILEHEADER (14 bytes)
    - BITMAPINFOHEADER (40 bytes)
    """
    # File Header
    file_type = b'BM'
    pixel_data_offset = 54 # 14 + 40
    
    # Image row size must be a multiple of 4 bytes due to padding
    row_size = (width * 3 + 3) & ~3
    pixel_data_size = row_size * height
    file_size = pixel_data_offset + pixel_data_size
    
    file_header = struct.pack(
        '<2sIHHI',
        file_type,     # 'BM'
        file_size,     # Size of file in bytes
        0, 0,          # Reserved fields
        pixel_data_offset # Offset to pixel data
    )
    
    # Info Header
    info_header_size = 40
    planes = 1
    bits_per_pixel = 24
    compression = 0 # BI_RGB (uncompressed)
    
    info_header = struct.pack(
        '<IIIHHIIIIII',
        info_header_size,
        width,
        height,
        planes,
        bits_per_pixel,
        compression,
        pixel_data_size,
        2835, 2835,     # H & V resolutions (72 DPI)
        0,              # Number of colors
        0               # Important colors
    )
    
    return file_header + info_header

def draw_wav_waveform_bmp(wav_path: str, bmp_path: str, width: int = 800, height: int = 200, 
                          color_rgb: Tuple[int, int, int] = (143, 85, 253),
                          bg_rgb: Tuple[int, int, int] = (14, 18, 36)) -> bool:
    """
    Parses a WAV PCM audio file and draws its waveform amplitude directly into a 24-bit BMP image.
    This is written completely from scratch with ZERO external dependencies (no matplotlib or PIL)!
    """
    print(f"[Info] Drawing WAV waveform visualization: {bmp_path}...")
    if not os.path.exists(wav_path):
        print(f"[Warning] WAV file {wav_path} not found for waveform generation.")
        return False
        
    try:
        # Read WAV format parameters and samples
        with open(wav_path, 'rb') as f:
            riff = f.read(12)
            if riff[0:4] != b'RIFF' or riff[8:12] != b'WAVE':
                return False
                
            # Scan format details
            channels = 1
            bits_per_sample = 16
            data_offset = 44
            
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    break
                chunk_size = int.from_bytes(f.read(4), byteorder='little')
                if chunk_id == b'fmt ':
                    f.read(2) # AudioFormat
                    channels = int.from_bytes(f.read(2), byteorder='little')
                    f.read(4) # SampleRate
                    f.read(4) # ByteRate
                    f.read(2) # BlockAlign
                    bits_per_sample = int.from_bytes(f.read(2), byteorder='little')
                    if chunk_size > 16:
                        f.read(chunk_size - 16)
                elif chunk_id == b'data':
                    data_offset = f.tell()
                    break
                else:
                    f.seek(chunk_size, 1)
            
            # Read PCM raw samples
            f.seek(data_offset)
            raw_data = f.read()
            
        # Parse samples based on bit depth
        samples = []
        bytes_per_sample = bits_per_sample // 8
        step = channels * bytes_per_sample
        
        # Read max 200,000 points to keep visual render fast
        max_points = 200000
        length = len(raw_data)
        skip = max(1, (length // step) // max_points) * step
        
        for i in range(0, length - step, skip):
            # Parse only channel 1
            val = 0
            if bits_per_sample == 16:
                val = int.from_bytes(raw_data[i:i+2], byteorder='little', signed=True)
            elif bits_per_sample == 8:
                val = int.from_bytes(raw_data[i:i+1], byteorder='little', signed=False) - 128
            samples.append(val)
            
        if not samples:
            return False
            
        # Downsample amplitudes into 'width' bins
        max_val = max(max(abs(s) for s in samples), 1)
        bin_size = len(samples) / width
        heights = []
        
        for col in range(width):
            start = int(col * bin_size)
            end = max(start + 1, int((col + 1) * bin_size))
            sub = samples[start:end]
            if sub:
                val_max = max(abs(s) for s in sub)
                # Map to visual height boundaries
                h_val = int((val_max / max_val) * (height / 2.0) * 0.9)
                heights.append(h_val)
            else:
                heights.append(0)
                
        # Generate pixel maps (row size padded to multiples of 4 bytes)
        row_size = (width * 3 + 3) & ~3
        pixels = bytearray(row_size * height)
        
        # Draw background and vertical waveform bars
        # BMP color format is BGR
        bg_b, bg_g, bg_r = bg_rgb[2], bg_rgb[1], bg_rgb[0]
        fg_b, fg_g, fg_r = color_rgb[2], color_rgb[1], color_rgb[0]
        
        mid_y = height // 2
        
        for y in range(height):
            y_offset = y * row_size
            for x in range(width):
                x_offset = y_offset + x * 3
                
                # Check if this pixel falls inside the waveform bar height
                h_val = heights[x]
                if abs(y - mid_y) <= h_val:
                    pixels[x_offset] = fg_b
                    pixels[x_offset+1] = fg_g
                    pixels[x_offset+2] = fg_r
                else:
                    pixels[x_offset] = bg_b
                    pixels[x_offset+1] = bg_g
                    pixels[x_offset+2] = bg_r
                    
        # Write binary headers and pixels
        header = write_bmp_header(width, height)
        with open(bmp_path, 'wb') as out_f:
            out_f.write(header)
            out_f.write(pixels)
            
        print("[Info] Waveform BMP image generated successfully.")
        return True
    except Exception as e:
        print(f"[Warning] Waveform drawing failed: {e}")
        return False

def generate_speaker_timeline_html(segments: List[dict], output_html_path: str, title: str = "Video Dubbing Speaker Timeline") -> None:
    """
    Compiles speaker segment timestamps into a clean, modern interactive
    timeline visualization HTML document, depicting speaker turns and transcripts.
    """
    print(f"[Info] Compressing timeline mapping to HTML: {output_html_path}...")
    if not segments:
        return
        
    total_dur = max(seg.get("end_time", 0.0) for seg in segments)
    if total_dur <= 0:
        total_dur = 1.0
        
    # Generate unique color mappings for each speaker
    unique_speakers = sorted(list(set(seg.get("speaker", "Unknown") for seg in segments)))
    colors = ["#8f55fd", "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#ec4899", "#14b8a6"]
    speaker_colors = {spk: colors[idx % len(colors)] for idx, spk in enumerate(unique_speakers)}
    
    # Compile timeline items list
    timeline_items_html = ""
    list_items_html = ""
    
    for idx, seg in enumerate(segments):
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        
        color = speaker_colors.get(speaker, "#9aa0b9")
        pct_left = (start / total_dur) * 100.0
        pct_width = ((end - start) / total_dur) * 100.0
        
        # Timeline visual block
        timeline_items_html += f"""
        <div class="timeline-block" style="left: {pct_left:.2f}%; width: {pct_width:.2f}%; background-color: {color};" title="{speaker}: {start:.1f}s - {end:.1f}s">
        </div>
        """
        
        # Detail list row
        list_items_html += f"""
        <div class="detail-row">
            <div class="time-badge">{start:05.2f}s - {end:05.2f}s</div>
            <div class="speaker-badge" style="background-color: {color}15; color: {color}; border: 1px solid {color}30;">{speaker}</div>
            <div class="dialog-text">{text}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #070912;
            color: #f1f3f9;
            margin: 0;
            padding: 2.5rem 1.5rem;
        }}
        .timeline-container {{
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(16, 20, 38, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        }}
        h1 {{
            font-size: 1.6rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #a78bfa, #8f55fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta-p {{
            color: #9aa0b9;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}
        .timeline-canvas {{
            position: relative;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            height: 60px;
            border-radius: 8px;
            margin-bottom: 2.5rem;
            overflow: hidden;
        }}
        .timeline-block {{
            position: absolute;
            height: 100%;
            border-right: 1px solid rgba(0, 0, 0, 0.2);
            border-left: 1px solid rgba(0, 0, 0, 0.2);
            top: 0;
            opacity: 0.85;
            transition: opacity 0.2s ease;
            cursor: pointer;
        }}
        .timeline-block:hover {{
            opacity: 1.0;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }}
        .timeline-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #9aa0b9;
            margin-top: -2rem;
            margin-bottom: 2.5rem;
            padding: 0 5px;
        }}
        .detail-list {{
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 1.5rem;
        }}
        .detail-row {{
            display: grid;
            grid-template-columns: 140px 140px 1fr;
            gap: 1.2rem;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
        }}
        .time-badge {{
            font-family: monospace;
            color: #a78bfa;
            font-weight: 500;
        }}
        .speaker-badge {{
            font-size: 0.82rem;
            font-weight: 600;
            text-align: center;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            text-transform: capitalize;
        }}
        .dialog-text {{
            font-size: 0.92rem;
            color: #d1d5db;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="timeline-container">
        <h1>{title}</h1>
        <p class="meta-p">Speaker turn distributions and conversation layout</p>
        
        <div class="timeline-canvas">
            {timeline_items_html}
        </div>
        <div class="timeline-labels">
            <span>0.00s</span>
            <span>{total_dur/2.0:.1f}s</span>
            <span>{total_dur:.2f}s</span>
        </div>
        
        <div class="detail-list">
            {list_items_html}
        </div>
    </div>
</body>
</html>
"""
    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[Info] HTML Timeline generated: {output_html_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate HTML timeline: {e}")

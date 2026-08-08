import os
import time
from typing import Dict, List, Any

class TimeProfiler:
    """
    Saves timing execution benchmarks for each phase of the video dubbing pipeline.
    Also tracks CPU/Memory indicators using standard Python system methods.
    """
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
        self.memory_samples: Dict[str, float] = {}
        self.order: List[str] = []

    def start(self, stage_name: str) -> None:
        """Starts timing for a specific stage and samples memory."""
        self.timers[stage_name] = time.time()
        self.memory_samples[stage_name + "_start"] = self.sample_process_memory()
        if stage_name not in self.order:
            self.order.append(stage_name)

    def stop(self, stage_name: str) -> float:
        """Stops timing for a stage and saves its duration in seconds."""
        if stage_name not in self.timers:
            return 0.0
            
        start_t = self.timers[stage_name]
        duration = time.time() - start_t
        self.durations[stage_name] = duration
        self.memory_samples[stage_name + "_end"] = self.sample_process_memory()
        return duration

    def sample_process_memory(self) -> float:
        """
        Samples the current process memory consumption in Megabytes (MB).
        Works natively on Windows using tasklist/wmic or standard resource libraries.
        """
        try:
            import os
            # On Windows, read process memory from OS environment
            import ctypes
            from ctypes import wintypes
            
            # Simple fallback to psutil if installed, else query using process APIs
            try:
                import psutil
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)
            except ImportError:
                # Win32 API fallback
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                
                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                GetActiveProcess = ctypes.windll.kernel32.GetCurrentProcess
                
                pmc = PROCESS_MEMORY_COUNTERS()
                pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                if GetProcessMemoryInfo(GetActiveProcess(), ctypes.byref(pmc), pmc.cb):
                    return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get_stage_duration(self, stage_name: str) -> float:
        """Returns the duration of a stage in seconds."""
        return self.durations.get(stage_name, 0.0)

    def get_total_duration(self) -> float:
        """Returns total summed duration of all stages."""
        return sum(self.durations.values())

    def draw_ascii_histogram(self) -> str:
        """
        Generates a beautiful console-based ASCII histogram of stages duration split.
        """
        total = self.get_total_duration()
        if total <= 0:
            return "No profile records available."
            
        chart = "\n" + "="*50 + "\n"
        chart += "          PIPELINE TIME ALLOCATION CHART\n"
        chart += "="*50 + "\n"
        
        for stage in self.order:
            dur = self.durations.get(stage, 0.0)
            pct = (dur / total * 100.0)
            # 20 character visual bar width
            bar_width = int(round(pct / 5.0))
            bar_str = "█" * bar_width + "░" * (20 - bar_width)
            chart += f" {stage.capitalize():14} | {bar_str} | {dur:6.2f}s ({pct:5.1f}%)\n"
            
        chart += "="*50 + "\n"
        chart += f" Total Processing Duration: {total:.2f} seconds.\n"
        chart += "="*50 + "\n"
        return chart

    def generate_report_data(self) -> Dict[str, Any]:
        """
        Compiles performance benchmarks, memory spikes and percentages.
        """
        total = self.get_total_duration()
        stages_info = []
        
        for stage in self.order:
            dur = self.durations.get(stage, 0.0)
            pct = (dur / total * 100.0) if total > 0 else 0.0
            
            mem_start = self.memory_samples.get(stage + "_start", 0.0)
            mem_end = self.memory_samples.get(stage + "_end", 0.0)
            mem_spike = max(0.0, mem_end - mem_start)
            
            stages_info.append({
                "stage": stage,
                "duration_sec": round(dur, 2),
                "percentage": round(pct, 1),
                "memory_start_mb": round(mem_start, 2),
                "memory_end_mb": round(mem_end, 2),
                "memory_spike_mb": round(mem_spike, 2)
            })
            
        return {
            "total_duration_sec": round(total, 2),
            "stages": stages_info
        }

# Global time profiler instance
profiler = TimeProfiler()

def estimate_api_characters(segments: List[dict]) -> Dict[str, Any]:
    """
    Estimates translation and speech synthesis character counts.
    Calculates commercial cost value vs free tiers.
    """
    original_chars = 0
    translated_chars = 0
    
    for seg in segments:
        orig_text = seg.get("text", "")
        trans_text = seg.get("translated_text", orig_text)
        
        original_chars += len(orig_text)
        translated_chars += len(trans_text)
        
    # Cost approximations (e.g. Gemini and ElevenLabs commercial APIs)
    # ElevenLabs: $0.18 per 1000 characters (Starter level)
    # Gemini: $0.075 per 1M input tokens + $0.30 per 1M output tokens (approx $0.0001 per 1000 characters)
    commercial_tts_cost = (translated_chars / 1000.0) * 0.18
    commercial_gemini_cost = ((original_chars + translated_chars) / 1000.0) * 0.0005
    total_commercial_usd = commercial_tts_cost + commercial_gemini_cost
    
    return {
        "original_characters": original_chars,
        "translated_characters": translated_chars,
        "segments_count": len(segments),
        "commercial_cost_est_usd": round(total_commercial_usd, 4),
        "user_cost_actual_usd": 0.0 # Free tier API keys
    }

def generate_html_report(report_data: Dict[str, Any], stats: Dict[str, Any], output_html_path: str) -> None:
    """
    Generates a beautiful standalone HTML performance report page containing
    a CSS-only bar chart representing pipeline duration splits and memory footprints.
    """
    total_sec = report_data["total_duration_sec"]
    stages = report_data["stages"]
    
    chart_html = ""
    table_html = ""
    
    colors = ["#8f55fd", "#4f46e5", "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#ec4899"]
    
    for idx, s in enumerate(stages):
        color = colors[idx % len(colors)]
        pct = s["percentage"]
        
        table_html += f"""
        <tr>
            <td style="font-weight: 500;">{s['stage'].capitalize()}</td>
            <td>{s['duration_sec']} s</td>
            <td>{pct}%</td>
            <td>{s['memory_spike_mb']} MB</td>
        </tr>
        """
        
        chart_html += f"""
        <div style="margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.3rem;">
                <span style="font-weight: 500; color: #f1f3f9;">{s['stage'].capitalize()}</span>
                <span style="color: #9aa0b9;">{s['duration_sec']}s ({pct}%) | Spike: {s['memory_spike_mb']}MB</span>
            </div>
            <div style="background-color: rgba(255, 255, 255, 0.05); height: 12px; border-radius: 6px; overflow: hidden; width: 100%;">
                <div style="background-color: {color}; height: 100%; width: {pct}%; border-radius: 6px; transition: width 0.5s ease;"></div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video Dubbing Performance & Cost Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #080a13;
            color: #f1f3f9;
            margin: 0;
            padding: 2.5rem 1.5rem;
        }}
        .report-container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(16, 20, 38, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        }}
        h1 {{
            font-size: 1.8rem;
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
        .section {{
            margin-bottom: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 1.5rem;
        }}
        h2 {{
            font-size: 1.25rem;
            margin-bottom: 1.2rem;
            color: #ffffff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            margin-bottom: 1rem;
        }}
        th, td {{
            padding: 0.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        th {{
            color: #9aa0b9;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1.5rem;
        }}
        .stat-card {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
        }}
        .stat-num {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #10b981;
            margin-top: 0.4rem;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <h1>Performance & Character Usage Report</h1>
        <p class="meta-p">Automated Video Dubbing System execution metrics</p>
        
        <div class="section">
            <h2>⏱️ Pipeline Duration & Resource Splits</h2>
            {chart_html}
        </div>
        
        <div class="grid">
            <div class="section">
                <h2>📊 Execution Timing & RAM Spikes</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Stage</th>
                            <th>Time</th>
                            <th>Split</th>
                            <th>RAM Spike</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_html}
                        <tr style="font-weight: 700; border-top: 2px solid rgba(255, 255, 255, 0.1);">
                            <td>Total Duration</td>
                            <td>{total_sec} s</td>
                            <td>100%</td>
                            <td>-</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 System Stats & Costs</h2>
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div class="stat-card">
                        <div style="font-size: 0.8rem; color: #9aa0b9; text-transform: uppercase;">Translated Characters</div>
                        <div class="stat-num">{stats['translated_characters']}</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 0.8rem; color: #9aa0b9; text-transform: uppercase;">Speech Segments</div>
                        <div class="stat-num" style="color: #3b82f6;">{stats['segments_count']}</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 0.8rem; color: #9aa0b9; text-transform: uppercase;">Commercial Cost Savings</div>
                        <div class="stat-num" style="color: #eab308;">${stats['commercial_cost_est_usd']}</div>
                        <small style="color: #a78bfa; font-size: 0.72rem;">Actual Cost: $0.00 (Free APIs)</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[Info] HTML Performance Report written to: {output_html_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate HTML report: {e}")

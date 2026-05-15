"""
modules/exporter.py
Export utilities for SHM analysis results.

Author: University of Phayao | Physics Department
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def make_filename(prefix: str, ext: str = "csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"


def figure_to_html_bytes(fig) -> bytes:
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def make_metadata_json(
    video_meta: dict,
    tracking_info: dict,
    analysis_results: Dict[str, Any] = None,
    settings: Dict[str, Any] = None,
) -> bytes:
    metadata = {
        "analysis_timestamp": datetime.now().isoformat(),
        "video": video_meta,
        "tracking": tracking_info,
        "results": analysis_results or {},
        "settings": settings or {},
        "tool": "SHM Motion Tracker — University of Phayao",
    }
    return json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")


def build_summary_text(
    video_meta: dict,
    df: pd.DataFrame,
    time_result=None,
    fft_result=None,
    fit_result=None,
) -> str:
    lines = []
    lines.append(f"Video: {video_meta.get('filename', 'N/A')}")
    lines.append(f"Duration: {video_meta.get('duration_sec', 0):.2f} sec")
    lines.append(f"FPS: {video_meta.get('fps', 0)}")
    lines.append(f"Frames tracked: {len(df)}")
    if "found" in df.columns:
        n_found = df["found"].sum()
        pct = 100 * n_found / len(df) if len(df) > 0 else 0
        lines.append(f"Tracking success: {n_found}/{len(df)} ({pct:.1f}%)")
    
    if time_result is not None:
        lines.append("")
        lines.append("--- Time Domain Analysis ---")
        lines.append(f"Period T = {time_result.period_sec:.4f} sec")
        lines.append(f"Frequency f = {time_result.frequency_hz:.4f} Hz")
        lines.append(f"Angular frequency ω = {time_result.angular_freq:.4f} rad/s")
        lines.append(f"Amplitude = {time_result.amplitude:.4f}")
        lines.append(f"Peaks detected = {time_result.n_peaks}")
    
    if fft_result is not None:
        lines.append("")
        lines.append("--- Frequency Domain (FFT) ---")
        lines.append(f"Dominant frequency = {fft_result.dominant_freq:.4f} Hz")
        lines.append(f"Dominant period = {fft_result.dominant_period:.4f} sec")
        lines.append(f"Sample rate = {fft_result.sample_rate:.1f} Hz")
    
    if fit_result is not None and fit_result.success:
        lines.append("")
        lines.append("--- Damped Oscillator Fit ---")
        lines.append(f"x(t) = {fit_result.A:.3f}·exp(-{fit_result.gamma:.4f}t)·cos({fit_result.omega:.3f}t + {fit_result.phi:.3f}) + {fit_result.offset:.3f}")
        lines.append(f"Damping ratio ζ = {fit_result.damping_ratio:.4f}")
        lines.append(f"Quality factor Q = {fit_result.quality_factor:.2f}")
        lines.append(f"Natural frequency f₀ = {fit_result.natural_freq_hz:.4f} Hz")
        lines.append(f"R² = {fit_result.r_squared:.4f}")
    
    return "\n".join(lines)

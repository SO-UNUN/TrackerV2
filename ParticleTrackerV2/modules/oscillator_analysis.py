"""
modules/oscillator_analysis.py
SHM (Simple Harmonic Motion) analysis tools:
- Period/frequency from peaks (Time domain)
- FFT spectrum analysis (Frequency domain)
- Damped oscillator fitting

Author: University of Phayao | Physics Department
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# ============================================================
#  TIME DOMAIN ANALYSIS
# ============================================================
@dataclass
class TimeDomainResult:
    """ผลการวิเคราะห์ใน time domain"""
    period_sec: float          # T = ระยะเวลาต่อรอบ
    frequency_hz: float        # f = 1/T
    angular_freq: float        # ω = 2πf
    amplitude: float           # max - min (peak-to-peak / 2)
    mean_position: float       # ค่ากลางของการสั่น
    n_peaks: int               # จำนวน peak ที่ตรวจพบ
    peak_times: List[float]    # เวลาที่เกิด peak
    peak_values: List[float]   # ค่าที่ peak

    def to_dict(self) -> dict:
        return {
            "period_sec": self.period_sec,
            "frequency_hz": self.frequency_hz,
            "angular_freq_rad_s": self.angular_freq,
            "amplitude": self.amplitude,
            "mean_position": self.mean_position,
            "n_peaks": self.n_peaks,
        }


def analyze_time_domain(
    time: np.ndarray,
    position: np.ndarray,
    min_peak_prominence: float = 0.1,
    smooth_window: int = 11,
) -> TimeDomainResult:
    """หา period, frequency, amplitude จาก time series ตำแหน่ง
    
    Args:
        time: array ของเวลา (วินาที)
        position: array ของตำแหน่ง (cm หรือพิกเซล)
        min_peak_prominence: ความเด่นขั้นต่ำของ peak (filter noise)
        smooth_window: window สำหรับ Savitzky-Golay smooth
        
    Returns:
        TimeDomainResult
    """
    pos = np.asarray(position, dtype=float)
    t = np.asarray(time, dtype=float)
    
    # Remove NaN
    valid = ~np.isnan(pos)
    pos = pos[valid]
    t = t[valid]
    if len(pos) < 10:
        return TimeDomainResult(0, 0, 0, 0, 0, 0, [], [])
    
    # Smooth ก่อนหา peak (ลด noise)
    if len(pos) > smooth_window:
        win = smooth_window if smooth_window % 2 else smooth_window + 1
        poly = min(3, win - 1)
        pos_smooth = savgol_filter(pos, win, poly)
    else:
        pos_smooth = pos

    mean_pos = float(np.mean(pos_smooth))
    amplitude = float((np.max(pos_smooth) - np.min(pos_smooth)) / 2)

    # หา peak (ยอดบน)
    pos_centered = pos_smooth - mean_pos
    prominence = max(min_peak_prominence, amplitude * 0.3)
    peaks_idx, _ = find_peaks(pos_centered, prominence=prominence)
    
    if len(peaks_idx) < 2:
        return TimeDomainResult(0, 0, 0, amplitude, mean_pos, len(peaks_idx),
                                 list(t[peaks_idx]), list(pos_smooth[peaks_idx]))

    # คำนวณ period จาก peak-to-peak
    peak_times = t[peaks_idx]
    periods = np.diff(peak_times)
    period = float(np.mean(periods))
    frequency = 1.0 / period if period > 0 else 0.0
    omega = 2 * np.pi * frequency

    return TimeDomainResult(
        period_sec=period,
        frequency_hz=frequency,
        angular_freq=omega,
        amplitude=amplitude,
        mean_position=mean_pos,
        n_peaks=len(peaks_idx),
        peak_times=list(peak_times),
        peak_values=list(pos_smooth[peaks_idx]),
    )


# ============================================================
#  FREQUENCY DOMAIN ANALYSIS (FFT)
# ============================================================
@dataclass
class FFTResult:
    """ผลการวิเคราะห์ FFT"""
    frequencies: np.ndarray    # array of frequencies (Hz)
    amplitudes: np.ndarray     # FFT magnitudes (single-sided)
    dominant_freq: float       # frequency ที่มีค่ามากที่สุด
    dominant_period: float     # 1 / dominant_freq
    sample_rate: float         # fps ที่ใช้
    n_samples: int


def compute_fft(
    time: np.ndarray,
    position: np.ndarray,
    remove_mean: bool = True,
    apply_window: bool = True,
) -> FFTResult:
    """คำนวณ FFT spectrum ของตำแหน่ง
    
    Args:
        time: array ของเวลา (วินาที) — ต้อง uniform spacing
        position: array ของตำแหน่ง
        remove_mean: ลบ DC component (mean) ก่อน FFT
        apply_window: ใช้ Hann window เพื่อลด spectral leakage

    Returns:
        FFTResult
    """
    t = np.asarray(time, dtype=float)
    pos = np.asarray(position, dtype=float)
    
    # Remove NaN
    valid = ~np.isnan(pos)
    pos = pos[valid]
    t = t[valid]
    
    if len(pos) < 4:
        return FFTResult(np.array([]), np.array([]), 0, 0, 0, 0)

    # Sampling rate (เฉลี่ย)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt if dt > 0 else 30.0
    n = len(pos)

    # ลบ mean (DC component)
    if remove_mean:
        pos = pos - np.mean(pos)

    # Hann window
    if apply_window:
        window = np.hanning(n)
        pos_windowed = pos * window
        # Normalize: amplitude correction factor
        amp_correction = 2.0 / window.sum() * n
    else:
        pos_windowed = pos
        amp_correction = 2.0 / n

    # FFT
    yf = rfft(pos_windowed)
    xf = rfftfreq(n, dt)
    amplitudes = np.abs(yf) * amp_correction / 2  # single-sided amplitude

    # ตัด DC (index 0) ออกในการหา dominant
    if len(amplitudes) > 1:
        peak_idx = np.argmax(amplitudes[1:]) + 1
        dominant_freq = float(xf[peak_idx])
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
    else:
        dominant_freq = 0.0
        dominant_period = 0.0

    return FFTResult(
        frequencies=xf,
        amplitudes=amplitudes,
        dominant_freq=dominant_freq,
        dominant_period=dominant_period,
        sample_rate=fs,
        n_samples=n,
    )


# ============================================================
#  DAMPED OSCILLATOR FITTING
# ============================================================
@dataclass
class DampedOscillatorFit:
    """ผลการ fit damped harmonic oscillator
    Model: x(t) = A·exp(-γ·t)·cos(ω·t + φ) + offset
    """
    A: float          # initial amplitude
    gamma: float      # damping coefficient (1/sec)
    omega: float      # angular frequency (rad/sec)
    phi: float        # phase (rad)
    offset: float     # equilibrium position
    r_squared: float  # goodness of fit
    success: bool

    @property
    def frequency_hz(self) -> float:
        return self.omega / (2 * np.pi)

    @property
    def period_sec(self) -> float:
        return 2 * np.pi / self.omega if self.omega > 0 else 0

    @property
    def damping_ratio(self) -> float:
        """ζ = γ / ω₀ — สำหรับ underdamped: ω = ω₀·√(1-ζ²)"""
        if self.omega == 0:
            return 0
        omega_0_sq = self.omega**2 + self.gamma**2
        if omega_0_sq <= 0:
            return 0
        return self.gamma / np.sqrt(omega_0_sq)

    @property
    def quality_factor(self) -> float:
        """Q = ω / (2γ) — สูง = damping ต่ำ"""
        if self.gamma == 0:
            return float("inf")
        return self.omega / (2 * self.gamma)

    @property
    def natural_freq_hz(self) -> float:
        """f₀ = ω₀ / (2π) — frequency ที่ไม่มี damping"""
        omega_0 = np.sqrt(self.omega**2 + self.gamma**2)
        return omega_0 / (2 * np.pi)

    def predict(self, t: np.ndarray) -> np.ndarray:
        return damped_oscillator(t, self.A, self.gamma, self.omega, self.phi, self.offset)


def damped_oscillator(t, A, gamma, omega, phi, offset):
    """Model function: x(t) = A·exp(-γt)·cos(ωt + φ) + offset"""
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + offset


def fit_damped_oscillator(
    time: np.ndarray,
    position: np.ndarray,
    initial_guess: Optional[Dict] = None,
) -> DampedOscillatorFit:
    """Fit damped oscillator model ให้กับข้อมูล
    
    Args:
        time, position: time series
        initial_guess: dict with keys A, gamma, omega, phi, offset
                       ถ้าไม่ใส่ จะหา initial guess อัตโนมัติ
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(position, dtype=float)
    valid = ~np.isnan(x)
    t, x = t[valid], x[valid]
    if len(t) < 10:
        return DampedOscillatorFit(0, 0, 0, 0, 0, 0, False)

    # Initial guess
    if initial_guess is None:
        offset_g = float(np.mean(x))
        x_centered = x - offset_g
        A_g = float(np.max(np.abs(x_centered)))

        # Estimate freq จาก FFT
        fft = compute_fft(t, x_centered, remove_mean=False, apply_window=True)
        omega_g = 2 * np.pi * fft.dominant_freq if fft.dominant_freq > 0 else 2 * np.pi
        gamma_g = 0.1  # ไม่ damped มาก
        phi_g = 0.0
    else:
        A_g = initial_guess.get("A", float(np.max(np.abs(x - np.mean(x)))))
        gamma_g = initial_guess.get("gamma", 0.1)
        omega_g = initial_guess.get("omega", 2 * np.pi)
        phi_g = initial_guess.get("phi", 0.0)
        offset_g = initial_guess.get("offset", float(np.mean(x)))

    try:
        popt, _ = curve_fit(
            damped_oscillator, t, x,
            p0=[A_g, gamma_g, omega_g, phi_g, offset_g],
            maxfev=5000,
            bounds=(
                [0,        0,    0.01,    -2*np.pi, -np.inf],
                [10*A_g,   10,   100,      2*np.pi,  np.inf],
            ),
        )
        A, gamma, omega, phi, offset = popt

        # R²
        x_pred = damped_oscillator(t, *popt)
        ss_res = np.sum((x - x_pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return DampedOscillatorFit(
            A=float(A), gamma=float(gamma), omega=float(omega),
            phi=float(phi), offset=float(offset),
            r_squared=float(r_squared), success=True,
        )
    except Exception as e:
        print(f"Fit failed: {e}")
        return DampedOscillatorFit(A_g, gamma_g, omega_g, phi_g, offset_g, 0, False)


# ============================================================
#  PHYSICS CALCULATIONS
# ============================================================
def calculate_spring_constant(mass_kg: float, frequency_hz: float) -> float:
    """คำนวณ spring constant k จาก f = (1/2π)√(k/m)
    
    k = (2πf)² · m
    """
    omega = 2 * np.pi * frequency_hz
    return omega ** 2 * mass_kg


def theoretical_period(mass_kg: float, spring_constant: float) -> float:
    """T = 2π·√(m/k)"""
    if spring_constant <= 0:
        return 0
    return 2 * np.pi * np.sqrt(mass_kg / spring_constant)


def energy_analysis(
    time: np.ndarray,
    position: np.ndarray,
    mass_kg: float,
    spring_constant: float,
) -> pd.DataFrame:
    """คำนวณ KE, PE, total energy vs time
    
    Returns:
        DataFrame columns: time, position, velocity, KE, PE, total_E
    """
    t = np.asarray(time)
    x = np.asarray(position)
    
    # Velocity ด้วย central difference
    v = np.gradient(x, t)
    
    KE = 0.5 * mass_kg * v ** 2
    PE = 0.5 * spring_constant * x ** 2
    
    return pd.DataFrame({
        "time_sec": t,
        "position": x,
        "velocity": v,
        "KE_J": KE,
        "PE_J": PE,
        "total_E_J": KE + PE,
    })

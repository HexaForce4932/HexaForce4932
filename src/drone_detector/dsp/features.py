"""Spectral feature helpers used by the DSP pre-filter.

Drone rotors produce a strong "harmonic comb": a fundamental blade-passage
frequency (roughly 80-400 Hz depending on propeller size/RPM) plus energy at
integer multiples of it. We look for that pattern directly in the power
spectrum instead of training a model, so this stage is cheap enough to run
continuously on a Raspberry Pi.
"""
from __future__ import annotations

import numpy as np


def power_spectrum(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, power) for a mono signal using a Hann-windowed FFT.

    Power is normalized by the number of samples so that `band_energy_min_db`
    thresholds stay comparable across different `chunk_duration` settings
    (by Parseval's theorem this makes the total roughly equal to the mean
    squared amplitude of the chunk, independent of window length).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    windowed = signal * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    power = (np.abs(spectrum) ** 2) / n
    return freqs, power


def total_energy(power: np.ndarray) -> float:
    return float(np.sum(power)) + 1e-12


def band_energy(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(power[mask]))


def harmonic_energy_ratio(
    freqs: np.ndarray,
    power: np.ndarray,
    fundamental_hz: float,
    max_harmonics: int = 6,
    tolerance_hz: float = 15.0,
) -> float:
    """Fraction of total spectral energy found at `fundamental_hz` and its harmonics."""
    if fundamental_hz <= 0 or len(freqs) < 2:
        return 0.0

    freq_resolution = freqs[1] - freqs[0]
    tol_bins = max(1, int(round(tolerance_hz / freq_resolution)))
    nyquist = freqs[-1]

    harmonic_energy = 0.0
    for k in range(1, max_harmonics + 1):
        target = fundamental_hz * k
        if target > nyquist:
            break
        idx = int(round(target / freq_resolution))
        lo = max(0, idx - tol_bins)
        hi = min(len(power), idx + tol_bins + 1)
        harmonic_energy += np.sum(power[lo:hi])

    return float(harmonic_energy / total_energy(power))


def find_best_fundamental(
    freqs: np.ndarray,
    power: np.ndarray,
    f0_min: float,
    f0_max: float,
    max_harmonics: int = 6,
    tolerance_hz: float = 15.0,
    step_hz: float = 5.0,
) -> tuple[float, float]:
    """Search candidate fundamentals in [f0_min, f0_max] for the strongest harmonic comb.

    Returns (best_fundamental_hz, harmonic_energy_ratio).
    """
    best_f0 = float(f0_min)
    best_score = 0.0
    for f0 in np.arange(f0_min, f0_max, step_hz):
        score = harmonic_energy_ratio(freqs, power, float(f0), max_harmonics, tolerance_hz)
        if score > best_score:
            best_score = score
            best_f0 = float(f0)
    return best_f0, best_score

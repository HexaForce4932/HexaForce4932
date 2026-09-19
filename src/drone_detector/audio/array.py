"""Simple direction-of-arrival (DOA) estimation for a microphone array via GCC-PHAT.

This is a lightweight, approximate estimator meant for a rough "which way is the
drone" indication, not a rigorous beamformer. It works with any linear/circular
array of >= 2 microphones by estimating pairwise time-delays between adjacent
channels and converting them to an angle via the far-field approximation.
"""
from __future__ import annotations

import numpy as np


def gcc_phat(sig: np.ndarray, ref: np.ndarray, sample_rate: int, max_tau: float | None = None, interp: int = 1) -> float:
    """Estimate the time delay (seconds) of `sig` relative to `ref` via GCC-PHAT.

    A positive return value means `sig` lags behind `ref` by that many seconds.
    """
    n = sig.shape[0] + ref.shape[0]
    sig_fft = np.fft.rfft(sig, n=n)
    ref_fft = np.fft.rfft(ref, n=n)
    cross = sig_fft * np.conj(ref_fft)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(cross / denom, n=interp * n)

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * sample_rate * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    tau = shift / float(interp * sample_rate)
    return tau


def estimate_doa_two_mic(
    chunk_ch0: np.ndarray,
    chunk_ch1: np.ndarray,
    sample_rate: int,
    mic_spacing_m: float,
    speed_of_sound_mps: float = 343.0,
) -> float:
    """Estimate the angle of arrival (degrees, broadside-relative) for a mic pair."""
    max_tau = mic_spacing_m / speed_of_sound_mps
    tau = gcc_phat(chunk_ch0, chunk_ch1, sample_rate, max_tau=max_tau)
    ratio = np.clip(tau * speed_of_sound_mps / mic_spacing_m, -1.0, 1.0)
    return float(np.degrees(np.arcsin(ratio)))


def estimate_doa(
    chunk: np.ndarray,
    sample_rate: int,
    mic_spacing_m: float,
    speed_of_sound_mps: float = 343.0,
) -> float:
    """Estimate DOA (degrees) for a multi-channel chunk shaped (frames, channels).

    Averages the pairwise estimates between adjacent microphone channels.
    """
    if chunk.ndim != 2 or chunk.shape[1] < 2:
        raise ValueError("DOA estimation requires at least 2 microphone channels")

    n_channels = chunk.shape[1]
    angles = [
        estimate_doa_two_mic(
            chunk[:, i], chunk[:, i + 1], sample_rate, mic_spacing_m, speed_of_sound_mps
        )
        for i in range(n_channels - 1)
    ]
    return float(np.mean(angles))

"""Manual MFCC feature extraction (no librosa dependency, Pi-friendly).

Standard pipeline: pre-emphasis -> framing/windowing -> power spectrum ->
mel filterbank -> log -> DCT-II. Produces a fixed-length feature vector
(mean + std of the per-frame MFCCs) suitable as input to a small classifier.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(num_filters: int, n_fft: int, sample_rate: int, low_freq: float = 0.0, high_freq: float | None = None) -> np.ndarray:
    high_freq = high_freq or sample_rate / 2.0
    low_mel = _hz_to_mel(low_freq)
    high_mel = _hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, n_fft // 2 + 1))
    for m in range(1, num_filters + 1):
        f_prev, f_curr, f_next = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if f_curr > f_prev:
            fbank[m - 1, f_prev:f_curr] = (np.arange(f_prev, f_curr) - f_prev) / (f_curr - f_prev)
        if f_next > f_curr:
            fbank[m - 1, f_curr:f_next] = (f_next - np.arange(f_curr, f_next)) / (f_next - f_curr)
    return fbank


def frame_signal(signal: np.ndarray, sample_rate: int, frame_length_ms: float = 25.0, frame_step_ms: float = 10.0) -> np.ndarray:
    frame_len = max(1, int(round(sample_rate * frame_length_ms / 1000.0)))
    frame_step = max(1, int(round(sample_rate * frame_step_ms / 1000.0)))
    signal_len = len(signal)

    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_len - frame_len) / frame_step))

    pad_len = (num_frames - 1) * frame_step + frame_len
    padded = np.zeros(pad_len)
    padded[:signal_len] = signal

    indices = (
        np.tile(np.arange(frame_len), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step)[:num_frames], (frame_len, 1)).T
    )
    return padded[indices]


def compute_mfcc(
    signal: np.ndarray,
    sample_rate: int,
    num_cepstral: int = 13,
    num_filters: int = 26,
    frame_length_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    pre_emphasis: float = 0.97,
) -> np.ndarray:
    """Returns MFCCs shaped (num_frames, num_cepstral)."""
    signal = np.asarray(signal, dtype=np.float64)
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frames = frame_signal(emphasized, sample_rate, frame_length_ms, frame_step_ms)
    frames = frames * np.hamming(frames.shape[1])

    n_fft = 512
    while n_fft < frames.shape[1]:
        n_fft *= 2

    magnitude = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    power = (1.0 / n_fft) * (magnitude ** 2)

    fbank = mel_filterbank(num_filters, n_fft, sample_rate)
    filter_banks = power @ fbank.T
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fb = np.log(filter_banks)

    mfcc = dct(log_fb, type=2, axis=1, norm="ortho")[:, :num_cepstral]
    return mfcc


def mfcc_feature_vector(signal: np.ndarray, sample_rate: int, num_cepstral: int = 13, **kwargs) -> np.ndarray:
    """Fixed-length feature vector (2 * num_cepstral,) = mean + std of per-frame MFCCs."""
    mfcc = compute_mfcc(signal, sample_rate, num_cepstral=num_cepstral, **kwargs)
    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])

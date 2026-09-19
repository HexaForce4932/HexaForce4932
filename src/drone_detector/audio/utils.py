"""Small shared audio helpers: PCM<->float conversion, WAV loading, mono-down-mixing."""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def pcm_to_float(data: np.ndarray) -> np.ndarray:
    """Convert integer PCM samples to float32 in [-1, 1]. Float input is passed through."""
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(int(info.min)), int(info.max))
        return (data.astype(np.float32) / scale)
    return data.astype(np.float32)


def load_wav(path: str):
    """Load a WAV file. Returns (sample_rate, data) with data shape (frames, channels)."""
    sample_rate, data = wavfile.read(path)
    data = pcm_to_float(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return sample_rate, data


def to_mono(chunk: np.ndarray) -> np.ndarray:
    """Down-mix a (frames, channels) array to a mono (frames,) array by averaging channels."""
    if chunk.ndim == 1:
        return chunk
    return chunk.mean(axis=1)

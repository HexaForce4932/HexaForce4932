"""Audio source abstractions: live microphone input and offline WAV playback."""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from .utils import load_wav


class AudioSource(abc.ABC):
    """A source of audio chunks shaped (num_frames, channels), float32 in [-1, 1]."""

    sample_rate: int
    channels: int

    @abc.abstractmethod
    def read_chunk(self, num_frames: int) -> np.ndarray:
        ...

    def close(self) -> None:
        pass


class MicrophoneSource(AudioSource):
    """Live capture from a (USB) microphone or microphone array via sounddevice.

    Requires the optional `sounddevice` dependency, which in turn needs PortAudio.
    Only needed on the actual Raspberry Pi / recording machine, not for offline
    development or testing (use WavFileSource for that).
    """

    def __init__(self, sample_rate: int, channels: int, device: Optional[str] = None):
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on optional hardware dep
            raise RuntimeError(
                "sounddevice is required for live microphone capture. "
                "Install it with `pip install acoustic-drone-detector[live]`."
            ) from exc

        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = sd.InputStream(
            samplerate=sample_rate, channels=channels, device=device, dtype="float32"
        )
        self._stream.start()

    def read_chunk(self, num_frames: int) -> np.ndarray:
        data, _overflowed = self._stream.read(num_frames)
        return data

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class WavFileSource(AudioSource):
    """Reads a WAV file chunk-by-chunk, for offline testing/demoing without hardware."""

    def __init__(self, path: str):
        sample_rate, data = load_wav(path)
        self.sample_rate = sample_rate
        self.channels = data.shape[1]
        self._data = data
        self._pos = 0

    def read_chunk(self, num_frames: int) -> np.ndarray:
        end = self._pos + num_frames
        chunk = self._data[self._pos : end]
        if chunk.shape[0] < num_frames:
            pad = np.zeros((num_frames - chunk.shape[0], self.channels), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        self._pos = end
        return chunk

    def has_more(self) -> bool:
        return self._pos < len(self._data)

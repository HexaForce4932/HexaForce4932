"""Cheap DSP heuristic that flags audio chunks worth a closer (ML) look."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as F


@dataclass
class DspDetectionResult:
    is_candidate: bool
    confidence: float
    dominant_frequency_hz: float
    band_energy_db: float


class DspPrefilter:
    """Flags a chunk as a drone candidate if it has a strong harmonic comb in the
    typical rotor blade-passage frequency range, above a minimum energy floor.
    """

    def __init__(
        self,
        sample_rate: int,
        fundamental_min_hz: float,
        fundamental_max_hz: float,
        max_harmonics: int,
        harmonicity_threshold: float,
        band_energy_min_db: float,
        tolerance_hz: float = 15.0,
        step_hz: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.fundamental_min_hz = fundamental_min_hz
        self.fundamental_max_hz = fundamental_max_hz
        self.max_harmonics = max_harmonics
        self.harmonicity_threshold = harmonicity_threshold
        self.band_energy_min_db = band_energy_min_db
        self.tolerance_hz = tolerance_hz
        self.step_hz = step_hz

    def analyze(self, chunk_mono: np.ndarray) -> DspDetectionResult:
        freqs, power = F.power_spectrum(chunk_mono, self.sample_rate)

        f0, score = F.find_best_fundamental(
            freqs,
            power,
            self.fundamental_min_hz,
            self.fundamental_max_hz,
            self.max_harmonics,
            self.tolerance_hz,
            self.step_hz,
        )

        band_hi = self.fundamental_max_hz * self.max_harmonics
        band_e = F.band_energy(freqs, power, self.fundamental_min_hz, band_hi)
        band_db = 10.0 * np.log10(band_e + 1e-12)

        is_candidate = score >= self.harmonicity_threshold and band_db >= self.band_energy_min_db

        return DspDetectionResult(
            is_candidate=is_candidate,
            confidence=score,
            dominant_frequency_hz=f0,
            band_energy_db=float(band_db),
        )

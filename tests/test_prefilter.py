import numpy as np

from drone_detector.dsp.prefilter import DspPrefilter


def make_prefilter(sr=16000):
    return DspPrefilter(
        sample_rate=sr,
        fundamental_min_hz=80,
        fundamental_max_hz=400,
        max_harmonics=6,
        harmonicity_threshold=0.3,
        band_energy_min_db=-100.0,
    )


def test_prefilter_flags_harmonic_signal():
    sr = 16000
    prefilter = make_prefilter(sr)
    t = np.arange(sr) / sr
    signal = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 5))

    result = prefilter.analyze(signal)

    assert result.is_candidate
    assert abs(result.dominant_frequency_hz - 150) <= 10


def test_prefilter_ignores_noise():
    sr = 16000
    prefilter = make_prefilter(sr)
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(sr) * 0.05

    result = prefilter.analyze(signal)

    assert not result.is_candidate


def test_prefilter_ignores_near_silence_even_with_tone():
    sr = 16000
    prefilter = DspPrefilter(
        sample_rate=sr,
        fundamental_min_hz=80,
        fundamental_max_hz=400,
        max_harmonics=6,
        harmonicity_threshold=0.0,
        band_energy_min_db=0.0,  # very strict floor, nothing should pass
    )
    t = np.arange(sr) / sr
    signal = 0.001 * np.sin(2 * np.pi * 150 * t)

    result = prefilter.analyze(signal)

    assert not result.is_candidate

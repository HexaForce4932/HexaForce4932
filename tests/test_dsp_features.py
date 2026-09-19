import numpy as np

from drone_detector.dsp import features as F


def make_sine(freq, duration, sample_rate, amplitude=1.0):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_find_best_fundamental_detects_harmonic_signal():
    sr = 16000
    f0 = 150.0
    duration = 1.0
    signal = np.zeros(int(sr * duration))
    for k in range(1, 5):
        signal += make_sine(f0 * k, duration, sr, amplitude=1.0 / k)
    rng = np.random.default_rng(0)
    signal += 0.01 * rng.standard_normal(len(signal))

    freqs, power = F.power_spectrum(signal, sr)
    best_f0, score = F.find_best_fundamental(freqs, power, f0_min=80, f0_max=400, max_harmonics=6)

    assert abs(best_f0 - f0) <= 10
    assert score > 0.5


def test_find_best_fundamental_low_score_for_noise():
    sr = 16000
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(sr)

    freqs, power = F.power_spectrum(signal, sr)
    _, score = F.find_best_fundamental(freqs, power, f0_min=80, f0_max=400, max_harmonics=6)

    assert score < 0.35


def test_band_energy_is_zero_outside_range():
    sr = 16000
    signal = make_sine(1000.0, 1.0, sr)
    freqs, power = F.power_spectrum(signal, sr)
    energy = F.band_energy(freqs, power, 80.0, 400.0)
    assert energy < F.total_energy(power) * 0.01

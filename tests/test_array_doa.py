import numpy as np
import pytest

from drone_detector.audio import array as doa


def test_gcc_phat_recovers_integer_sample_delay():
    sr = 16000
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(sr)
    delay_samples = 5
    sig = np.concatenate([np.zeros(delay_samples), ref])[: len(ref)]

    tau = doa.gcc_phat(sig, ref, sr)

    expected_tau = delay_samples / sr
    assert abs(tau - expected_tau) < 2 / sr


def test_estimate_doa_two_mic_broadside_is_near_zero_for_identical_signal():
    sr = 16000
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(sr)

    angle = doa.estimate_doa_two_mic(ref, ref, sr, mic_spacing_m=0.05)

    assert abs(angle) < 1.0


def test_estimate_doa_requires_multi_channel():
    chunk = np.zeros((100, 1))
    with pytest.raises(ValueError):
        doa.estimate_doa(chunk, 16000, 0.05)


def test_estimate_doa_averages_multi_channel_pairs():
    sr = 16000
    rng = np.random.default_rng(6)
    ref = rng.standard_normal(sr)
    chunk = np.stack([ref, ref, ref], axis=1)

    angle = doa.estimate_doa(chunk, sr, mic_spacing_m=0.05)

    assert abs(angle) < 1.0

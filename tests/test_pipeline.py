import numpy as np

from drone_detector.alert.alerts import AlertBackend, AlertManager
from drone_detector.config import AppConfig
from drone_detector.pipeline import DroneDetectionPipeline


class RecordingBackend(AlertBackend):
    def __init__(self):
        self.triggers = []

    def trigger(self, message, confidence, extra=None):
        self.triggers.append((message, confidence, extra))


class FakeAudioSource:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def close(self):
        pass


def make_drone_chunk(sample_rate=16000, duration=1.0, f0=150.0):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
    return signal.reshape(-1, 1).astype(np.float32)


def make_noise_chunk(sample_rate=16000, duration=1.0):
    rng = np.random.default_rng(5)
    signal = rng.standard_normal(int(sample_rate * duration)) * 0.05
    return signal.reshape(-1, 1).astype(np.float32)


def build_pipeline(cooldown_seconds=100.0):
    config = AppConfig()
    config.dsp.harmonicity_threshold = 0.3
    config.dsp.band_energy_min_db = -100.0
    config.ml.enabled = False
    config.detection.cooldown_seconds = cooldown_seconds

    audio_source = FakeAudioSource()
    backend = RecordingBackend()
    alert_manager = AlertManager([backend])
    pipeline = DroneDetectionPipeline(config, audio_source, alert_manager, classifier=None)
    return pipeline, backend


def test_pipeline_detects_and_alerts_on_drone_chunk():
    pipeline, backend = build_pipeline()

    event = pipeline.process_chunk(make_drone_chunk())

    assert event.is_drone
    assert len(backend.triggers) == 1


def test_pipeline_ignores_noise_chunk():
    pipeline, backend = build_pipeline()

    event = pipeline.process_chunk(make_noise_chunk())

    assert not event.is_drone
    assert len(backend.triggers) == 0


def test_pipeline_respects_cooldown():
    pipeline, backend = build_pipeline(cooldown_seconds=100.0)

    pipeline.process_chunk(make_drone_chunk())
    pipeline.process_chunk(make_drone_chunk())

    assert len(backend.triggers) == 1


def test_pipeline_alerts_again_after_cooldown_expires():
    pipeline, backend = build_pipeline(cooldown_seconds=0.0)

    pipeline.process_chunk(make_drone_chunk())
    pipeline.process_chunk(make_drone_chunk())

    assert len(backend.triggers) == 2

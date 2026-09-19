"""Typed configuration loading for the drone detector (YAML -> dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class MicArrayConfig:
    enabled: bool = False
    geometry: str = "linear"
    mic_spacing_m: float = 0.05
    speed_of_sound_mps: float = 343.0


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 1.0
    device: Optional[str] = None
    mic_array: MicArrayConfig = field(default_factory=MicArrayConfig)


@dataclass
class DspConfig:
    fundamental_min_hz: float = 80.0
    fundamental_max_hz: float = 400.0
    max_harmonics: int = 6
    harmonicity_threshold: float = 0.35
    band_energy_min_db: float = -60.0


@dataclass
class MlConfig:
    enabled: bool = True
    model_path: str = "models/drone_classifier.joblib"
    confirmation_threshold: float = 0.6
    n_mfcc: int = 13


@dataclass
class DetectionConfig:
    cooldown_seconds: float = 5.0
    require_ml_confirmation: bool = False


@dataclass
class GpioLedConfig:
    enabled: bool = False
    pin: int = 17


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: Optional[str] = None
    timeout_seconds: float = 3.0


@dataclass
class AlertsConfig:
    console: bool = True
    gpio_led: GpioLedConfig = field(default_factory=GpioLedConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dsp: DspConfig = field(default_factory=DspConfig)
    ml: MlConfig = field(default_factory=MlConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AppConfig":
        data = data or {}

        audio_data = dict(data.get("audio") or {})
        mic_array_data = audio_data.pop("mic_array", {}) or {}
        audio = AudioConfig(mic_array=MicArrayConfig(**mic_array_data), **audio_data)

        dsp = DspConfig(**(data.get("dsp") or {}))
        ml = MlConfig(**(data.get("ml") or {}))
        detection = DetectionConfig(**(data.get("detection") or {}))

        alerts_data = dict(data.get("alerts") or {})
        gpio_data = alerts_data.pop("gpio_led", {}) or {}
        webhook_data = alerts_data.pop("webhook", {}) or {}
        alerts = AlertsConfig(
            gpio_led=GpioLedConfig(**gpio_data),
            webhook=WebhookConfig(**webhook_data),
            **alerts_data,
        )

        return cls(audio=audio, dsp=dsp, ml=ml, detection=detection, alerts=alerts)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig.from_dict(data)

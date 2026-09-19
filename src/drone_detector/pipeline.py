"""Hybrid detection pipeline: DSP pre-filter, optional ML confirmation, DOA, alerting."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .alert.alerts import AlertManager
from .audio import array as doa
from .audio.capture import AudioSource
from .audio.utils import to_mono
from .config import AppConfig
from .dsp.prefilter import DspPrefilter
from .ml.features import mfcc_feature_vector
from .ml.model import DroneClassifier

logger = logging.getLogger("drone_detector.pipeline")


@dataclass
class DetectionEvent:
    timestamp: float
    is_drone: bool
    confidence: float
    dominant_frequency_hz: float
    ml_confidence: Optional[float]
    direction_deg: Optional[float]


class DroneDetectionPipeline:
    def __init__(
        self,
        config: AppConfig,
        audio_source: AudioSource,
        alert_manager: AlertManager,
        classifier: Optional[DroneClassifier] = None,
    ):
        self.config = config
        self.audio_source = audio_source
        self.alert_manager = alert_manager
        self.classifier = classifier

        self.prefilter = DspPrefilter(
            sample_rate=audio_source.sample_rate,
            fundamental_min_hz=config.dsp.fundamental_min_hz,
            fundamental_max_hz=config.dsp.fundamental_max_hz,
            max_harmonics=config.dsp.max_harmonics,
            harmonicity_threshold=config.dsp.harmonicity_threshold,
            band_energy_min_db=config.dsp.band_energy_min_db,
        )

        self._last_alert_time = 0.0
        self._chunk_frames = max(1, int(audio_source.sample_rate * config.audio.chunk_duration))

    def process_chunk(self, chunk: np.ndarray) -> DetectionEvent:
        mono = to_mono(chunk)
        dsp_result = self.prefilter.analyze(mono)

        is_drone = dsp_result.is_candidate
        confidence = dsp_result.confidence
        ml_confidence: Optional[float] = None

        ml_available = self.classifier is not None and self.classifier.is_fitted and self.config.ml.enabled

        if dsp_result.is_candidate and ml_available:
            feat = mfcc_feature_vector(
                mono, self.audio_source.sample_rate, num_cepstral=self.config.ml.n_mfcc
            )
            ml_confidence = self.classifier.predict_proba_drone(feat)
            is_drone = ml_confidence >= self.config.ml.confirmation_threshold
            confidence = ml_confidence
        elif dsp_result.is_candidate and self.config.detection.require_ml_confirmation and not ml_available:
            # Policy says a DSP hit alone isn't enough, but no ML model is available to confirm.
            is_drone = False

        direction_deg = None
        if is_drone and chunk.ndim == 2 and chunk.shape[1] > 1 and self.config.audio.mic_array.enabled:
            try:
                direction_deg = doa.estimate_doa(
                    chunk,
                    self.audio_source.sample_rate,
                    self.config.audio.mic_array.mic_spacing_m,
                    self.config.audio.mic_array.speed_of_sound_mps,
                )
            except ValueError:
                direction_deg = None

        event = DetectionEvent(
            timestamp=time.time(),
            is_drone=is_drone,
            confidence=confidence,
            dominant_frequency_hz=dsp_result.dominant_frequency_hz,
            ml_confidence=ml_confidence,
            direction_deg=direction_deg,
        )

        self._handle_alerting(event)
        return event

    def _handle_alerting(self, event: DetectionEvent) -> None:
        if not event.is_drone:
            return
        if event.timestamp - self._last_alert_time < self.config.detection.cooldown_seconds:
            return

        message = f"Drone detected (~{event.dominant_frequency_hz:.0f} Hz)"
        if event.direction_deg is not None:
            message += f", direction ~{event.direction_deg:.0f} deg"

        self.alert_manager.trigger(
            message,
            event.confidence,
            extra={
                "dominant_frequency_hz": event.dominant_frequency_hz,
                "ml_confidence": event.ml_confidence,
                "direction_deg": event.direction_deg,
            },
        )
        self._last_alert_time = event.timestamp

    def run_forever(self) -> None:
        logger.info(
            "Starting drone detection pipeline (sample_rate=%d, channels=%d)",
            self.audio_source.sample_rate,
            self.audio_source.channels,
        )
        try:
            while True:
                chunk = self.audio_source.read_chunk(self._chunk_frames)
                event = self.process_chunk(chunk)
                logger.debug("event=%s", event)
        except KeyboardInterrupt:
            logger.info("Stopping drone detection pipeline.")
        finally:
            self.audio_source.close()

"""Command-line entry point: `drone-detector live|file ...`."""
from __future__ import annotations

import argparse
import logging
import os

from .alert.alerts import AlertManager, ConsoleAlert, GpioLedAlert, WebhookAlert
from .audio.capture import MicrophoneSource, WavFileSource
from .config import AppConfig, load_config
from .ml.model import DroneClassifier
from .pipeline import DroneDetectionPipeline

logger = logging.getLogger("drone_detector.cli")


def build_alert_manager(config: AppConfig) -> AlertManager:
    backends = []
    if config.alerts.console:
        backends.append(ConsoleAlert())
    if config.alerts.gpio_led.enabled:
        backends.append(GpioLedAlert(config.alerts.gpio_led.pin))
    if config.alerts.webhook.enabled and config.alerts.webhook.url:
        backends.append(WebhookAlert(config.alerts.webhook.url, config.alerts.webhook.timeout_seconds))
    return AlertManager(backends)


def load_classifier(config: AppConfig):
    if not config.ml.enabled:
        return None
    if not os.path.exists(config.ml.model_path):
        logger.warning(
            "ML confirmation enabled but model file '%s' not found; running DSP-only until trained.",
            config.ml.model_path,
        )
        return None
    return DroneClassifier.load(config.ml.model_path)


def cmd_live(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    audio_source = MicrophoneSource(config.audio.sample_rate, config.audio.channels, device=config.audio.device)
    alert_manager = build_alert_manager(config)
    classifier = load_classifier(config)
    pipeline = DroneDetectionPipeline(config, audio_source, alert_manager, classifier)
    pipeline.run_forever()


def cmd_file(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    audio_source = WavFileSource(args.wav_path)
    alert_manager = build_alert_manager(config)
    classifier = load_classifier(config)
    pipeline = DroneDetectionPipeline(config, audio_source, alert_manager, classifier)

    chunk_frames = max(1, int(audio_source.sample_rate * config.audio.chunk_duration))
    while audio_source.has_more():
        chunk = audio_source.read_chunk(chunk_frames)
        event = pipeline.process_chunk(chunk)
        status = "DRONE" if event.is_drone else "-"
        direction = f"{event.direction_deg:.0f}deg" if event.direction_deg is not None else "n/a"
        print(
            f"[{event.timestamp:.2f}] {status:5s} conf={event.confidence:.2f} "
            f"f0={event.dominant_frequency_hz:.0f}Hz dir={direction}"
        )
    audio_source.close()


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="drone-detector", description="Acoustic drone detector for Raspberry Pi.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_live = sub.add_parser("live", help="Run detection using a live microphone / mic array.")
    p_live.set_defaults(func=cmd_live)

    p_file = sub.add_parser("file", help="Run detection on a WAV file (for testing without hardware).")
    p_file.add_argument("wav_path")
    p_file.set_defaults(func=cmd_file)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

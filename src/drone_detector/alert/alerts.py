"""Alert backends: console log, GPIO LED (Raspberry Pi), and webhook."""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("drone_detector.alert")


class AlertBackend:
    def trigger(self, message: str, confidence: float, extra: Optional[dict] = None) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        pass


class ConsoleAlert(AlertBackend):
    def trigger(self, message: str, confidence: float, extra: Optional[dict] = None) -> None:
        logger.warning("DRONE ALERT: %s (confidence=%.2f) %s", message, confidence, extra or {})

    def clear(self) -> None:
        logger.info("Drone alert cleared.")


class GpioLedAlert(AlertBackend):
    """Drives an LED on a BCM GPIO pin. No-ops gracefully off a Raspberry Pi
    (e.g. during development/testing) so the pipeline still runs elsewhere.
    """

    def __init__(self, pin: int):
        self.pin = pin
        self._gpio = None
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            self._gpio = GPIO
        except (ImportError, RuntimeError):
            logger.warning("RPi.GPIO not available; GPIO LED alert disabled (not running on a Pi?).")

    def trigger(self, message: str, confidence: float, extra: Optional[dict] = None) -> None:
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.HIGH)

    def clear(self) -> None:
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.LOW)


class WebhookAlert(AlertBackend):
    def __init__(self, url: str, timeout_seconds: float = 3.0):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def trigger(self, message: str, confidence: float, extra: Optional[dict] = None) -> None:
        try:
            import requests

            payload = {"message": message, "confidence": confidence, "timestamp": time.time()}
            if extra:
                payload.update(extra)
            requests.post(self.url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to send webhook alert: %s", exc)


class AlertManager:
    def __init__(self, backends: list[AlertBackend]):
        self._backends = backends

    def trigger(self, message: str, confidence: float, extra: Optional[dict] = None) -> None:
        for backend in self._backends:
            backend.trigger(message, confidence, extra)

    def clear(self) -> None:
        for backend in self._backends:
            backend.clear()

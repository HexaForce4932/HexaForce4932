# Hardware-Aufbau

## Grundausstattung

- Raspberry Pi 3B+ / 4 / Zero 2 W (o. ä., ausreichend für FFT + kleines
  ML-Modell in Echtzeit auf 1-Sekunden-Chunks).
- Micro-SD-Karte mit Raspberry Pi OS, PortAudio installiert:
  `sudo apt install libportaudio2`.
- Stromversorgung, Gehäuse nach Bedarf.

## Option A: Einzelnes USB-Mikrofon

Einfachste Variante — erkennt *ob* eine Drohne zu hören ist, aber keine
Richtung.

- Ein beliebiges USB-Mikrofon bzw. USB-Soundkarte mit Mikrofoneingang.
- Einstecken, Gerät ggf. per `arecord -l` / `python -m sounddevice` finden
  und in `config/default.yaml` unter `audio.device` eintragen (Name oder
  Index; `null` = Systemstandard).
- Konfiguration: `audio.channels: 1`, `audio.mic_array.enabled: false`.

## Option B: Mikrofon-Array (z. B. ReSpeaker 4-Mic Array / Linear-4-Mic-Kit)

Erlaubt zusätzlich eine grobe Richtungsschätzung.

- Array laut Hersteller-Anleitung an den Pi anschließen (I2S- oder
  USB-Variante, je nach Modell).
- Den tatsächlichen Mikrofonabstand (Mitte zu Mitte, in Metern) ausmessen und
  in `audio.mic_array.mic_spacing_m` eintragen — die Richtungsschätzung
  hängt direkt von diesem Wert ab.
- Konfiguration: `audio.channels: <Anzahl Mikrofone>`,
  `audio.mic_array.enabled: true`.
- Die Schätzung mittelt paarweise GCC-PHAT-Ergebnisse benachbarter Kanäle;
  bei einem linearen Array (2–4 Mikrofone in einer Reihe) ist das
  ausreichend für eine grobe "von links/rechts/vorne"-Aussage, nicht für
  eine präzise Grad-Angabe.

## Optionale Status-LED (GPIO-Alarm)

- LED + Vorwiderstand (z. B. 220–330 Ω) in Reihe.
- LED-Anode über den Widerstand an einen freien GPIO-Pin (Standard in der
  Konfiguration: BCM 17, Pin 11 auf der Pinleiste).
- LED-Kathode an GND (z. B. Pin 6 oder 9).
- In der Konfiguration `alerts.gpio_led.enabled: true` und `pin` auf die
  BCM-Nummer des gewählten Pins setzen.
- Läuft der Code nicht auf einem Pi (z. B. beim Entwickeln am Laptop), wird
  der GPIO-Alarm automatisch übersprungen (kein Absturz) — nützlich zum
  Testen mit `file`-Modus ohne Hardware.

## Webhook-Alarm (optional)

Statt/zusätzlich zur LED kann bei Erkennung ein HTTP-POST an eine
selbstgewählte URL gesendet werden (`alerts.webhook.enabled: true`,
`alerts.webhook.url: "https://..."`) — z. B. um einen eigenen
Benachrichtigungsdienst, ein Smart-Home-System oder ein Logging-Backend
anzusteuern. Der Payload enthält Nachricht, Konfidenz, Zeitstempel sowie
erkannte Grundfrequenz und (falls verfügbar) Richtung.

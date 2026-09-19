# Akustischer Drohnendetektor

Ein Software-Projekt für Raspberry Pi, das Drohnen anhand ihres charakteristischen
Rotor-/Propellergeräuschs akustisch erkennt. Die Erkennung ist **hybrid**:

1. **DSP-Vorfilter** (immer aktiv, sehr günstig): sucht per FFT nach einem
   "harmonischen Kamm" — einer Grundfrequenz (typisch 80–400 Hz, abhängig von
   Propellergröße/Drehzahl) plus Energie bei deren Vielfachen. Das ist das
   akustische Kennzeichen rotierender Propeller.
2. **Optionales ML-Modell** (nur bei DSP-Kandidaten aktiv): bestätigt/verwirft
   den Verdacht anhand von MFCC-Audio-Features mit einem kleinen, lokal
   trainierten Klassifikator (Random Forest).

Das spart Rechenleistung auf dem Pi: das teurere ML-Modell läuft nur, wenn der
günstige DSP-Filter bereits anschlägt.

## Features

- **Konfigurierbares Mikrofon-Setup**: einzelnes USB-Mikrofon oder Mikrofon-Array
  (z. B. ReSpeaker 4-Mic), umschaltbar per Konfigurationsdatei.
- **Richtungsschätzung (DOA)** bei Array-Betrieb via GCC-PHAT (Grobschätzung,
  kein präzises Beamforming).
- **Mehrere Alarm-Kanäle**: Konsolen-Log, GPIO-LED, Webhook (HTTP POST) —
  beliebig kombinierbar.
- **Offline testbar** ganz ohne Raspberry-Pi-Hardware über WAV-Dateien.
- Reine Python-Implementierung, DSP/MFCC ohne schwere Zusatzabhängigkeiten
  (`numpy`/`scipy` statt `librosa`).

## Projektstruktur

```
src/drone_detector/
  config.py         # YAML-Konfiguration -> Dataclasses
  audio/
    capture.py      # Mikrofon- bzw. WAV-Datei-Eingabe
    array.py        # Richtungsschätzung (GCC-PHAT) für Mikrofon-Arrays
    utils.py         # WAV-Laden, PCM->float, Mono-Downmix
  dsp/
    features.py      # FFT, Bandenergie, Harmonischen-Suche
    prefilter.py      # Der eigentliche DSP-Erkennungsfilter
  ml/
    features.py       # Manuelle MFCC-Extraktion
    model.py           # Wrapper um scikit-learn-Klassifikator
    train.py            # Trainings-CLI für das ML-Modell
  alert/
    alerts.py           # Konsole / GPIO-LED / Webhook
  pipeline.py            # Orchestriert DSP -> ML -> DOA -> Alarm
  cli.py                  # `drone-detector live|file`
config/default.yaml       # Standardkonfiguration (zum Kopieren/Anpassen)
tests/                     # pytest-Suite (läuft ohne Hardware)
docs/hardware.md            # Verkabelung Pi + Mikrofon(e) + LED
```

## Installation

Auf dem Raspberry Pi (oder zum Entwickeln auf einem normalen Rechner):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # Basis + pytest, zum Entwickeln/Testen
# Auf dem Pi zusätzlich für Live-Betrieb:
pip install -e ".[live,pi]"    # sounddevice (Mikrofon) + RPi.GPIO (LED)
```

`sounddevice` benötigt PortAudio (`sudo apt install libportaudio2` auf Raspberry Pi OS).

## Verwendung

### Ohne Hardware testen (WAV-Datei)

```bash
python -m drone_detector.cli --config config/default.yaml file aufnahme.wav
```

Gibt pro Analyse-Chunk eine Zeile aus (Zeit, Status, Konfidenz, erkannte
Grundfrequenz, ggf. Richtung) und löst bei Erkennung dieselben Alarme aus wie
der Live-Betrieb.

### Live mit Mikrofon (auf dem Pi)

```bash
python -m drone_detector.cli --config config/default.yaml live
```

Läuft dauerhaft (Strg+C zum Beenden) und verarbeitet kontinuierlich
`audio.chunk_duration`-Sekunden-Blöcke.

### Konfiguration

Siehe [`config/default.yaml`](config/default.yaml) — alle Parameter sind dort
kommentiert. Wichtige Stellschrauben:

- `audio.channels`: `1` für Einzelmikrofon, `>1` für ein Array.
- `audio.mic_array.enabled` + `mic_spacing_m`: aktiviert Richtungsschätzung,
  Mikrofon-Abstand in Metern messen und eintragen.
- `dsp.fundamental_min_hz`/`fundamental_max_hz`: an die tatsächliche
  Rotorfrequenz eurer Ziel-Drohnen(klasse) anpassen (kleine Racing-Drohnen
  liegen eher am oberen Ende, größere Multicopter eher niedriger).
- `ml.enabled` + `ml.model_path`: ML-Bestätigung an/aus und Pfad zum
  trainierten Modell.
- `detection.require_ml_confirmation`: wenn `true`, löst ein reiner
  DSP-Treffer ohne geladenes ML-Modell **keinen** Alarm aus (sicherer, aber
  mehr Fehlalarm-Risiko ohne trainiertes Modell).
- `alerts.*`: Konsole, GPIO-LED (Pi-Pin per BCM-Nummerierung), Webhook-URL.

### ML-Modell trainieren (optional)

Das ML-Modell ist optional — ohne trainiertes Modell läuft die Pipeline rein
DSP-basiert weiter. Zum Trainieren werden kurze, gelabelte WAV-Clips
gebraucht:

```
data/
  drone/   *.wav   # Aufnahmen mit hörbarer Drohne
  noise/   *.wav   # Hintergrundgeräusche ohne Drohne
```

```bash
python -m drone_detector.ml.train \
  --drone-dir data/drone --noise-dir data/noise \
  --output models/drone_classifier.joblib
```

Das Skript extrahiert MFCC-Feature-Vektoren, trainiert einen Random-Forest-
Klassifikator, gibt einen Klassifikationsbericht aus und speichert das Modell.
Anschließend `ml.model_path` in der Konfiguration darauf zeigen lassen.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Die Tests laufen komplett ohne Mikrofon/Pi-Hardware (synthetische Signale +
temporäre WAV-Dateien).

## Funktionsweise / Architektur

1. Ein Audio-Chunk (Standard: 1 Sekunde) wird eingelesen (Mikrofon oder WAV).
2. **DSP-Vorfilter**: FFT über den (auf Mono heruntergemischten) Chunk, Suche
   nach der Grundfrequenz mit dem stärksten harmonischen Muster im
   konfigurierten Frequenzband, plus ein Mindest-Energiepegel gegen reine
   Stille.
3. Schlägt der Vorfilter an *und* ist ein ML-Modell geladen, wird zusätzlich
   ein MFCC-Feature-Vektor berechnet und vom Klassifikator bewertet — das
   Ergebnis entscheidet dann final über "Drohne ja/nein".
4. Bei Array-Betrieb (`channels > 1`, `mic_array.enabled`) wird bei positiver
   Erkennung zusätzlich per GCC-PHAT eine grobe Einfallsrichtung geschätzt.
5. Bei Erkennung (und außerhalb der Abklingzeit `detection.cooldown_seconds`)
   werden alle konfigurierten Alarme ausgelöst.

### Grenzen der Richtungsschätzung

Die DOA-Schätzung ist eine einfache paarweise GCC-PHAT-Näherung, kein echtes
Beamforming. Bei rein synthetischen, perfekt periodischen Testtönen ganz ohne
Rauschanteil kann PHAT-Gewichtung fehlschlagen (Mehrdeutigkeit durch die
Periodizität); reale Mikrofonaufnahmen haben durch Umgebungs- und
Eigenrauschen immer einen Breitbandanteil, mit dem die Schätzung in der
Praxis zuverlässig funktioniert (siehe Tests in `tests/test_array_doa.py`).

## Hardware

Siehe [`docs/hardware.md`](docs/hardware.md) für Bauteilliste, Verkabelung
(Mikrofon, optionale LED an GPIO) und Hinweise zur Mikrofon-Array-Geometrie.

## Hinweis zu Datenschutz/Recht

Dieses Projekt zeichnet Umgebungsgeräusche auf, um Rotorgeräusche zu
erkennen. Je nach Einsatzort können Datenschutz- bzw. Abhörgesetze gelten
(z. B. DSGVO, ggf. Vorschriften zum Mithören/Aufzeichnen nichtöffentlich
gesprochener Worte). Setzt den Detektor nur auf eigenem Grundstück bzw. mit
entsprechender Berechtigung ein, speichert keine Rohaudiodaten ohne Bedarf
und prüft die für euren Standort geltenden Vorschriften. Dies ist ein
Hobby-/Experimentierprojekt, kein zertifiziertes Sicherheitsprodukt.

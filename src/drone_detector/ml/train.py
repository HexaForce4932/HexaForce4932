"""Train the optional ML confirmation model from labeled WAV clips.

Expects two directories of short WAV clips:
    <drone-dir>/*.wav   -- recordings that contain a drone
    <noise-dir>/*.wav   -- recordings of background noise / no drone

Usage:
    python -m drone_detector.ml.train --drone-dir data/drone --noise-dir data/noise \
        --output models/drone_classifier.joblib
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..audio.utils import load_wav, to_mono
from .features import mfcc_feature_vector
from .model import DroneClassifier


def build_dataset(drone_dir: str, noise_dir: str, num_cepstral: int = 13) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for label, directory in ((1, drone_dir), (0, noise_dir)):
        for path in sorted(glob.glob(os.path.join(directory, "*.wav"))):
            sample_rate, data = load_wav(path)
            mono = to_mono(data)
            features.append(mfcc_feature_vector(mono, sample_rate, num_cepstral=num_cepstral))
            labels.append(label)
    return np.array(features), np.array(labels)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-dir", required=True, help="Directory of WAV clips containing a drone.")
    parser.add_argument("--noise-dir", required=True, help="Directory of WAV clips without a drone.")
    parser.add_argument("--output", default="models/drone_classifier.joblib")
    parser.add_argument("--num-cepstral", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)

    X, y = build_dataset(args.drone_dir, args.noise_dir, num_cepstral=args.num_cepstral)
    if len(X) < 4 or len(set(y.tolist())) < 2:
        raise SystemExit(
            "Not enough labeled data to train: need WAV clips in both --drone-dir and --noise-dir."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    classifier = DroneClassifier()
    classifier.fit(X_train, y_train)

    y_pred = [1 if classifier.predict_proba_drone(x) >= 0.5 else 0 for x in X_test]
    print(classification_report(y_test, y_pred, target_names=["noise", "drone"]))

    classifier.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

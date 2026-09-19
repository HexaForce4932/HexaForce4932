"""Thin wrapper around a scikit-learn classifier used for ML confirmation."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class DroneClassifier:
    def __init__(self, estimator=None):
        if estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self._estimator = estimator
        self._is_fitted = getattr(estimator, "classes_", None) is not None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator.fit(X, y)
        self._is_fitted = True

    def predict_proba_drone(self, feature_vector: np.ndarray) -> float:
        """Probability that the given feature vector corresponds to a drone (label 1)."""
        if not self._is_fitted:
            raise RuntimeError("Model is not trained/loaded yet.")
        proba = self._estimator.predict_proba(feature_vector.reshape(1, -1))[0]
        classes = list(self._estimator.classes_)
        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = int(np.argmax(proba))
        return float(proba[idx])

    def save(self, path: str) -> None:
        import joblib

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self._estimator, path)

    @classmethod
    def load(cls, path: str) -> "DroneClassifier":
        import joblib

        estimator = joblib.load(path)
        return cls(estimator=estimator)

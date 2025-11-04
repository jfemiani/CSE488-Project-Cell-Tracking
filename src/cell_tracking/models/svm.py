"""SVM baseline helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.svm import SVC

from ..features import sliding_window_features


def train_svm(features: np.ndarray, labels: np.ndarray, kernel: str = "rbf", probability: bool = True) -> SVC:
    clf = SVC(kernel=kernel, probability=probability)
    clf.fit(features, labels)
    return clf


def save_model(model: SVC, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as fh:
        pickle.dump(model, fh)


def load_model(source: Path) -> SVC:
    with open(source, "rb") as fh:
        return pickle.load(fh)


def predict_image(image: np.ndarray, model: SVC, window_size: int, return_probabilities: bool = False) -> np.ndarray:
    flat_features = sliding_window_features(image, window_size)
    if return_probabilities and hasattr(model, "predict_proba"):
        preds = model.predict_proba(flat_features)
    else:
        preds = model.predict(flat_features)
    return preds

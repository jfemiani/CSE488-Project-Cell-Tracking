"""Centralised configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = Path(os.environ.get("CELL_TRACKING_BASE", PROJECT_ROOT / "artifacts"))
DEFAULT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

EVALUATION_ZIP_URL = "http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip"
SEGMEASURE_FALLBACK_URL = (
    "https://gist.github.com/jfemiani/81e5f6dbed8a949306d5a84df8f4e8eb/"
    "raw/144973730afaeb28640f3b0d10f7d311df410f3c/MySEGMeasure.py"
)

TRAINING_DATASETS: Dict[str, str] = {
    "Fluo-N2DH-GOWT1": "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip",
}

TEST_DATASETS: Dict[str, str] = {
    "Fluo-N2DH-GOWT1": "http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DH-GOWT1.zip",
}


def repo_path(*segments: str) -> Path:
    """Return an absolute path rooted at the repository."""

    return PROJECT_ROOT.joinpath(*segments)


def artifacts_path(*segments: str) -> Path:
    """Return a path rooted inside the shared artifacts directory."""

    return DEFAULT_ARTIFACTS.joinpath(*segments)

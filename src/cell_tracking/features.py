"""Feature extraction utilities for classical ML baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage import io
from skimage.util import view_as_windows


@dataclass
class ImageMaskPair:
    image_path: Path
    mask_path: Path

    def exists(self) -> bool:
        return self.image_path.exists() and self.mask_path.exists()


def _sample_pixels(mask: np.ndarray, samples: int, pct_fg: float) -> Tuple[np.ndarray, np.ndarray]:
    flat_indices = np.arange(mask.size)
    fg = flat_indices[mask.reshape(-1) > 0]
    bg = flat_indices[mask.reshape(-1) == 0]
    fg_samples = int(samples * pct_fg)
    bg_samples = samples - fg_samples
    rng = np.random.default_rng()
    return rng.choice(fg, fg_samples, replace=len(fg) < fg_samples), rng.choice(
        bg, bg_samples, replace=len(bg) < bg_samples
    )


def _extract_windows(image: np.ndarray, window_size: int) -> np.ndarray:
    pad = (window_size - 1) // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    windows = view_as_windows(padded, (window_size, window_size))
    return windows.reshape(-1, window_size ** 2)


def generate_image_features(image_path: Path, mask_path: Path, window_size: int, samples_per_image: int, pct_fg: float = 0.5) -> pd.DataFrame:
    image = io.imread(str(image_path)).astype(np.float32)
    mask = io.imread(str(mask_path))
    flat_windows = _extract_windows(image, window_size)

    fg_idx, bg_idx = _sample_pixels(mask, samples_per_image, pct_fg)
    selected = np.concatenate([fg_idx, bg_idx])
    features = flat_windows[selected]
    labels = np.concatenate([np.ones_like(fg_idx), np.zeros_like(bg_idx)])

    columns = [f"f{i:03d}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels
    df["image"] = image_path.name
    return df


def process_images(image_mask_pairs: Iterable[ImageMaskPair], window_size: int, samples_per_image: int, pct_fg: float = 0.5) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for pair in image_mask_pairs:
        if pair.exists():
            frames.append(
                generate_image_features(
                    pair.image_path,
                    pair.mask_path,
                    window_size,
                    samples_per_image,
                    pct_fg,
                )
            )
        else:
            print(f"Skipping missing files: {pair.image_path} / {pair.mask_path}")
    return pd.concat(frames, ignore_index=True)


def sliding_window_features(image: np.ndarray, window_size: int) -> np.ndarray:
    return _extract_windows(image, window_size)

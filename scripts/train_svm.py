"""Train a simple SVM baseline."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from cell_tracking.features import ImageMaskPair, process_images
from cell_tracking.config import artifacts_path
from cell_tracking.models import svm as svm_module


@hydra.main(config_path="../conf", config_name="config", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    dataset_root = artifacts_path("datasets", cfg.dataset.name, "training")
    model_path = Path(cfg.model.path)

    pairs = [
        ImageMaskPair(
            dataset_root / cfg.train.track / f"t{i:03d}.tif",
            dataset_root / f"{cfg.train.track}_ST" / "SEG" / f"man_seg{i:03d}.tif",
        )
        for i in range(3)
    ]

    df = process_images(pairs, cfg.train.window, cfg.train.samples, cfg.train.pct_fg)
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].to_numpy()

    model = svm_module.train_svm(x, y, kernel=cfg.model.kernel)
    svm_module.save_model(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
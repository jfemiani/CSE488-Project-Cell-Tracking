"""Evaluate predictions with IoU + SEGMeasure."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from skimage import io

from cell_tracking.config import artifacts_path
from cell_tracking.evaluation import compute_jaccard_index_for_matches, run_segmeasure
from cell_tracking.models import svm as svm_module


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model_path = Path(cfg.model.path)
    dataset_root = artifacts_path("datasets", cfg.dataset.name, "training")

    model = svm_module.load_model(model_path)
    image = io.imread(str(dataset_root / cfg.eval.track / "t000.tif"))
    gt = io.imread(str(dataset_root / f"{cfg.eval.track}_ST" / "SEG" / "man_seg000.tif"))

    preds = svm_module.predict_image(image, model, cfg.eval.window)
    mask = (preds.reshape(image.shape) > 0.5).astype(np.uint8)

    mean_iou, per_object = compute_jaccard_index_for_matches(gt, mask)
    print(f"Mean IoU: {mean_iou:.3f}")
    for label, iou in per_object.items():
        print(f"  Label {label}: {iou:.3f}")

    pred_dir = artifacts_path("results", cfg.dataset.name, cfg.eval.track)
    pred_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(str(pred_dir / "mask000.tif"), mask * 255)

    run_segmeasure(
        dataset_root / f"{cfg.eval.track}_ST" / "SEG",
        pred_dir,
        verbose=cfg.eval.verbose,
    )


if __name__ == "__main__":
    main()
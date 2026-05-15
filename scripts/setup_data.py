"""CLI helper to download datasets and evaluation tools."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from cell_tracking.data import ensure_all


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ensure_all(cfg.dataset.name, cfg.dataset.splits)


if __name__ == "__main__":
    main()
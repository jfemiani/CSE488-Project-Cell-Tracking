# CSE488 Project – Cell Segmentation & Tracking Starter

This repository replaces the shared Colab notebook with a reproducible Python project for the Cell Tracking Challenge inspired assignment. Students fork this repo, keep it private, and share access with the instructional staff. Major goals:

- Provide scripts for downloading challenge data, evaluation tooling, and helper assets.
- Offer reusable modules for feature extraction, classical ML (SVM baseline), and evaluation (IoU + SEGMeasure wrapper).
- Encourage clean experiments tracked via version control instead of ad-hoc Colab sessions.

## Project Layout

```
CSE488-Project-Cell-Tracking/
├── README.md
├── pyproject.toml
├── environment.yml
├── scripts/
│   ├── setup_data.py
│   ├── train_svm.py
│   └── eval_seg.py
├── src/cell_tracking/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   └── models/svm.py
├── notebooks/
│   └── 00_reference_colab.ipynb
└── tests/
    └── test_features.py
```

## Quickstart

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate cse488-cell-tracking

# 2. Install repo in editable mode
pip install -e .

# 3. Download tools + datasets
python scripts/setup_data.py --dataset Fluo-N2DH-GOWT1 --splits training test

# 4. Train a toy SVM baseline
python scripts/train_svm.py --dataset Fluo-N2DH-GOWT1 --track 01 \
    --window 5 --samples 500 --model-path artifacts/models/svm_rbf.pkl

# 5. Evaluate with IoU + SEGMeasure
python scripts/eval_seg.py --gt artifacts/datasets/Fluo-N2DH-GOWT1/training/01_ST/SEG \
    --pred artifacts/results/Fluo-N2DH-GOWT1/01 --model-path artifacts/models/svm_rbf.pkl
```

Key environment variables:

- `CELL_TRACKING_BASE`: Root folder for datasets/results (defaults to `<repo>/artifacts`).
- `CELL_TRACKING_DATASETS`: Optional override for dataset cache.

## Notebooks

`notebooks/00_reference_colab.ipynb` mirrors the original Colab walkthrough but imports utilities from `src/cell_tracking`. Feel free to add exploratory notebooks; keep production code in `src/` and automation in `scripts/`.

## Next Steps for Students

1. Fork the repo (keep private) and invite Heather Shelton, Femian John Charles, and Siddhant Nene.
2. Use issues/projects to plan segmentation/tracking experiments.
3. Extend `scripts/` and `src/cell_tracking/` with custom methods (CNNs, trackers, etc.). For deep baselines, start with lightweight U-Nets from `segmentation_models_pytorch` (e.g., MobileNet or EfficientNet encoders) to keep training feasible.
4. Publish qualitative results and videos via GitHub Pages (or similar) linked from your README.

## GitHub Pages / Showcase Site

Every team must host a lightweight public page that summarizes results and embeds demo videos. This repo includes a template at `docs/index.md`; GitHub Pages can publish anything under `docs/` straight from the `main` branch.

1. Edit `docs/index.md` with your team name, updated badge link (point it at your fork), metrics tables, and embedded media (place assets under `docs/assets/`).
2. Commit/push the changes to `main`.
3. On GitHub (while viewing your fork), click **Settings** in the top nav, then **Pages** in the left sidebar.
4. Under **Build and deployment**, choose **Deploy from branch**, select branch `main` and folder `/docs`, then click **Save**. Refresh the page—GitHub will show `https://<username>.github.io/<repo>/` once the build completes (may take ~1 minute).
5. Add that URL to your README and Canvas submission (update it whenever the site changes).

Tip: run `python -m http.server 8000` from the repo root and visit `http://localhost:8000/docs/` to preview changes before pushing.

## License

MIT License – adjust to match your team preference if needed.

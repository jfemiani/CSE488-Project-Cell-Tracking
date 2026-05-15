# Copilot Instructions for CSE488 Cell Segmentation & Tracking

## Project Overview
This is a starter repository for a cell segmentation and tracking assignment based on the [Cell Tracking Challenge](http://celltrackingchallenge.net/). Students are expected to choose a data train/test split, expand the provided baseline code to train on their chosen frames, generate predictions for their evaluation frames, and produce a meaningful evaluation score across those frames.

## What Students Are Expected to Expand
The following are intentionally incomplete and must be improved by the student:
- `scripts/train_svm.py` — currently only trains on 3 frames. Students must expand this to train on their chosen training frames.
- `scripts/eval_seg.py` — currently only generates and evaluates 1 frame. Students must loop over their chosen evaluation frames.
- Students must implement at least three classical ML models (e.g. SVM, decision tree, random forest, k-NN, logistic regression). The provided SVM counts as one — students must add at least two more.
- Graduate students must additionally implement a fourth model using deep learning (e.g. a CNN or U-Net).
- Each model should have its own module in `src/cell_tracking/models/`, its own config in `conf/model/`, and its own training script in `scripts/`.

When helping with these files, suggest solutions that loop over the student's chosen set of frames rather than
hardcoding a single frame index. Never assume all frames should be used for training — students define their own split.

## Project Structure
- `conf/` — Hydra configuration files. All hyperparameters and paths live here, not hardcoded in scripts.
- `scripts/` — Entry point scripts. Each uses `@hydra.main` for configuration.
- `src/cell_tracking/` — Core library modules:
  - `config.py` — use `artifacts_path()` for all file paths, never hardcode absolute paths
  - `data.py` — download helpers for datasets and evaluation tools
  - `evaluation.py` — `JaccardEvaluator` and `compute_jaccard_index_for_matches` for IoU scoring; `run_segmeasure` for full evaluation
  - `features.py` — sliding-window feature extraction; `process_images` and `ImageMaskPair`
  - `models/svm.py` — `train_svm`, `predict_image`, `save_model`, `load_model`

## Conventions to Follow
- Always use `artifacts_path()` from `cell_tracking.config` instead of hardcoded or relative paths.
- Config values come from `cfg` (a Hydra `DictConfig`), never from `argparse` or hardcoded constants.
- New model variants belong in `conf/model/<model_name>.yaml` and a corresponding training script in `scripts/`.
- New model implementation modules belong in `src/cell_tracking/models/<model_name>.py`, following the same interface as `models/svm.py` (i.e. expose `train_<model>`, `predict_image`, `save_model`, `load_model`).
- New dataset configs belong in `conf/dataset/<dataset_name>.yaml`.
- Students are responsible for defining their own train/test split from the available frames. The starter code does not enforce a split — students must decide which frames to train on and which to hold out for evaluation, and should document their choice in their report. Do not suggest using all frames for both training and evaluation, as this would be data leakage.
- Do not modify `evaluation.py`'s public interface (`compute_jaccard_index_for_matches`, `JaccardEvaluator`, `run_segmeasure`) — other code depends on these.
- Use `tqdm` for any loop over frames so users can see progress.
- Models should be saved with `pickle` or `joblib` to `cfg.model.path`.

## Hydra Usage
This project uses [Hydra](https://hydra.cc) for configuration. Config overrides are passed on
the command line like:
```bash
python scripts/train_svm.py train.samples=1000 model=svm_linear
python scripts/train_svm.py -m model=svm_rbf,svm_linear train.samples=500,1000
```
When suggesting command line usage, always use Hydra override syntax rather than argparse flags.

## Evaluation
- Predictions must be saved as `mask{frame_number}.tif` (e.g. `mask000.tif`) in the results directory.
- Ground truth is the Silver Truth (`_ST/SEG/man_seg*.tif`), not the full ground truth.
- `run_segmeasure(gt_dir, pred_dir, verbose=True)` will score all masks found in `pred_dir` against the Silver Truth and report per-frame Jaccard scores plus a final mean.
- A submission that only evaluates one frame will not be accepted — make sure all evaluation frames have a corresponding mask.

## Dependencies
Key packages: `numpy`, `scikit-image`, `scikit-learn`, `pandas`, `hydra-core`, `omegaconf`, `tqdm`.
All dependencies are in `pyproject.toml` and `environment.yml`. Do not suggest installing packages
outside of these files without also adding them there.
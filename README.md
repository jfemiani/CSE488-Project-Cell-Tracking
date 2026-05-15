# CSE488 Project – Cell Segmentation & Tracking Starter
A reproducible Python starter repository for the Cell Tracking Challenge inspired assignment.
Students clone this repo, keep it private, and add the instructional staff as collaborators.

**Goals:**
- Provide scripts for downloading challenge data, evaluation tools, and helper assets.
- Offer reusable modules for feature extraction, classical ML baselines, and evaluation (IoU + SEGMeasure).
- Encourage clean, version-controlled experiments rather than ad-hoc notebook sessions.

## Getting Started

### 1. Use this template
On GitHub, click **Use this template → Create a new repository**.
- Keep the repo **private**.
- Name it something like `<yourname>-cell-tracking`.
- Add John Femiani and other TAs as collaborators so they can clone and grade your work.

### 2. Set up your environment
```bash
conda env create -f environment.yml
conda activate cse488-cell-tracking
pip install -e .
```

### 3. Enable GitHub Copilot (recommended)
This repo includes a `.github/copilot-instructions.md` file that gives Copilot context about the project structure, conventions, and what you are expected to implement. If you have access to GitHub Copilot, it will automatically use these instructions to give you more relevant suggestions as you work.

### 4. Download the data and evaluation tools
```bash
python scripts/setup_data.py
```
This downloads the dataset into `artifacts/`. By default it fetches both the `training` and `test` splits of `Fluo-N2DH-GOWT1`. To override:
```bash
python scripts/setup_data.py dataset.splits=[training]
```

### 5. Train the SVM baseline
To test out what you will be doing for the project, some demo code has been provided to you for a SVM model. However:
- `train_svm.py` only trains on 3 frames
- `eval_seg.py` only generates predictions for 1 frame
- The Jaccard evaluator in `evaluation.py` will silently skip any frame without a corresponding mask, so your score will look great but be meaningless if you haven't generated masks for all evalutation frames/

It is your job to expand upon this code for your project to train on all training frames, generate predictions for every evaluation frame, and produce a meaningful evaluation score across the full dataset. A submission that only evaluates one frame will not be accepted.

To run the training script (feel free to run before modifying to get a feel for what you're doing):
```bash
python scripts/train_svm.py
```
You can override any config value on the command line without editing any files:
```bash
# Change the SVM kernel
python scripts/train_svm.py model=svm_linear

# Change window size and sample count
python scripts/train_svm.py train.window=7 train.samples=1000

# Try multiple kernels in one go (multirun)
python scripts/train_svm.py -m model=svm_rbf,svm_linear
```
### 6. Evaluate
Again, it is your job to expand on `eval_seg.py` in order to evaluate across all your evaluation frames.

To run the evaluation script:
```bash
python scripts/eval_seg.py
```
Override options work the same way:
```bash
# Evaluate with a specific model
python scripts/eval_seg.py model=svm_linear

# Evaluate track 02 with verbose output
python scripts/eval_seg.py eval.track=02 eval.verbose=true
```

### 7. Understanding the config system (Hydra)
This project uses [Hydra](https://hydra.cc) for configuration management. All default settings live in the `conf/` folder:
```
conf/
  config.yaml          <- top-level defaults (which dataset/model/train/eval to use)
  dataset/
    Fluo-N2DH-GOWT1.yaml   <- dataset name and splits
  model/
    svm_rbf.yaml       <- RBF kernel SVM settings
    svm_linear.yaml    <- linear kernel SVM settings
  train/
    default.yaml       <- window size, samples, foreground ratio
  eval/
    default.yaml       <- track, window size, verbosity
```
To experiment with a new model configuration, create a new file in `conf/model/` (e.g. `conf/model/svm_poly.yaml`) and run:
```bash
python scripts/train_svm.py model=svm_poly
```
Every run automatically saves its full config and logs to `outputs/<date>/<time>/`. This means you can always reproduce exactly what settings produced a given result.

## Project Layout
```
CSE488-Project-Cell-Tracking/
├── README.md
├── pyproject.toml          <- package metadata and dependencies
├── environment.yml         <- conda environment
├── conf/                   <- Hydra configuration files
│   ├── config.yaml         <- top-level defaults
│   ├── dataset/            <- one file per dataset
│   ├── model/              <- one file per model variant
│   ├── train/              <- training hyperparameters
│   └── eval/               <- evaluation settings
├── scripts/
│   ├── setup_data.py       <- download datasets + evaluation tools
│   ├── train_svm.py        <- train the SVM baseline
│   └── eval_seg.py         <- run predictions and evaluate
├── src/cell_tracking/
│   ├── config.py           <- paths and URLs
│   ├── data.py             <- download helpers
│   ├── evaluation.py       <- IoU computation + Jaccard evaluator
│   ├── features.py         <- sliding-window feature extraction
│   ├── models/
│   │   └── svm.py          <- SVM train/predict/save/load
│   └── cli.py              <- optional Typer CLI (same commands as scripts/)
├── .github/
│   └── copilot-instructions.md  <- Copilot context for this project
└── tests/
    └── test_features.py    <- tests for the feature extraction module
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `CELL_TRACKING_BASE` | `<repo>/artifacts` | Root folder for datasets, models, and results |
| `CELL_TRACKING_DATASETS` | same as above | Optional override for the dataset cache |

## What You Need to Submit
- A private GitHub repo containing your code, a working README with setup instructions, and an environment file.
- A final PDF report (not a notebook). See the assignment instructions for required sections.
- The repo URL and PDF submitted on Canvas.

## License
MIT License
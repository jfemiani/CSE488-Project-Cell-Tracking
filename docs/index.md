# TEAM_NAME – Cell Segmentation & Tracking Showcase

[![GitHub Repo](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/jfemiani/CSE488-Project-Cell-Tracking)

> Replace `TEAM_NAME`, update the badge link to your fork, and embed your own visuals. This page is what you submit as the public-facing deliverable.

## Project Overview

- **Dataset(s):** E.g., Fluo-N2DH-GOWT1 (tracks 01 & 02)
- **Team roles:** segmentation lead, tracking lead, evaluation lead
- **Summary:** Briefly explain your approach and what makes it unique.

## Segmentation Results

| Method | Dataset | Track | Mean IoU | Notes |
| ------ | ------- | ----- | -------- | ----- |
| Classical SVM | Fluo-N2DH-GOWT1 | 01 | 0.73 | RBF kernel, window=5 |
| CNN Baseline | Fluo-N2DH-GOWT1 | 02 | 0.81 | Lightweight U-Net |

- Describe preprocessing, feature engineering, and model choices.
- Link to notebooks or scripts that reproduce each row.

## Tracking Results (Grad/Honors)

| Tracker | Metric | Score | Notes |
| ------- | ------ | ----- | ----- |
| ASM + IoU linking | ID switches | 4 | Initialized from previous frame |

Explain how you propagate identities across frames and highlight failure cases.

## Demo Videos / GIFs

Embed MP4/GIF assets committed under `docs/assets/` or hosted elsewhere.

```html
<video controls width="640">
  <source src="assets/segmentation_demo.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
```

## Reproduction Checklist

1. `conda env create -f environment.yml`
2. `python scripts/setup_data.py --dataset Fluo-N2DH-GOWT1 --splits training test`
3. `python scripts/train_svm.py ...`
4. `python scripts/eval_seg.py ...`
5. Additional steps for custom models.

## Lessons Learned / Next Steps

Share insights, what you'd try next, and open research questions.

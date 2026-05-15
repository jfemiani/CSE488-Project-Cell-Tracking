"""Evaluation helpers (IoU + SEGMeasure)."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops

from .config import DEFAULT_ARTIFACTS


ArrayLike = np.ndarray


def compute_jaccard_index_for_matches(ref_image: ArrayLike, seg_mask: ArrayLike) -> Tuple[float, Dict[int, float]]:
    """Compute mean IoU between reference labels and predicted mask components."""

    labeled_mask = label(seg_mask)
    mask_props = regionprops(labeled_mask)
    jaccard_indices: Dict[int, float] = {}

    for ref_label in np.unique(ref_image):
        if ref_label == 0:
            continue
        ref_object = ref_image == ref_label
        overlaps = []
        for prop in mask_props:
            intersection = np.sum(ref_object & (labeled_mask == prop.label))
            union = np.sum(ref_object | (labeled_mask == prop.label))
            if intersection > 0.5 * np.sum(ref_object):
                overlaps.append(intersection / union)
            else:
                overlaps.append(0.0)
        if overlaps:
            jaccard_indices[int(ref_label)] = max(overlaps)

    mean_iou = float(np.mean(list(jaccard_indices.values()))) if jaccard_indices else 0.0
    return mean_iou, jaccard_indices


class JaccardEvaluator:
    def __init__(self):
        self.jaccard_scores = []

    def add_image_pair(self, seg_image_path, gt_image_path):
        seg_image = imread(seg_image_path) > 0
        gt_image = imread(gt_image_path)
        jaccard_index, indices = compute_jaccard_index_for_matches(gt_image, seg_image)
        self.jaccard_scores += list(indices.values())
        return jaccard_index, indices

    def mean_jaccard_index(self):
        return float(np.mean(self.jaccard_scores)) if self.jaccard_scores else 0.0

    def report(self):
        return self.jaccard_scores

    @staticmethod
    def evaluate_folder(gt_folder, output_folder, image_patterns=('*.tif', '*.jpg', '*.png'), verbose=False):
        evaluator = JaccardEvaluator()
        all_files = os.listdir(gt_folder)
        gt_files = []
        for pattern in image_patterns:
            gt_files.extend(fnmatch.filter(all_files, pattern))
        for gt_file in sorted(gt_files):
            if verbose:
                print('-' * 10)
                print(gt_file)
            match = re.search(r'man_seg(\d+)', gt_file)
            if match:
                file_number = match.group(1)
                output_file = f'mask{file_number}.tif'
                gt_path = os.path.join(gt_folder, gt_file)
                output_path = os.path.join(output_folder, output_file)
                if os.path.exists(output_path):
                    jac, indices = evaluator.add_image_pair(output_path, gt_path)
                    if verbose:
                        print('JAC:', jac)
                        for idx, ijac in indices.items():
                            print('   ', idx, ':', ijac)
        return evaluator.mean_jaccard_index()


def run_segmeasure(gt_dir: Path, res_dir: Path, verbose: bool = False) -> None:
    """Evaluate segmentation results using the built-in Jaccard evaluator."""

    mean_jac = JaccardEvaluator.evaluate_folder(str(gt_dir), str(res_dir), verbose=verbose)
    print(f"Mean Jaccard Index: {mean_jac:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results using Jaccard Index")
    parser.add_argument('--verbose', '-v', action='store_true', help='Output a lot of info.')
    parser.add_argument('output_folder', type=str, help='Folder containing output segmentation images')
    parser.add_argument('gt_folder', type=str, help='Folder containing ground truth segmentation images')
    args = parser.parse_args()
    jaccard_mean = JaccardEvaluator.evaluate_folder(args.output_folder, args.gt_folder, verbose=args.verbose)
    print(f"Mean Jaccard Index: {jaccard_mean}")


if __name__ == '__main__':
    main()
"""
Tests for the feature extraction module.

Install pytest with:
    pip install pytest

Run all tests with:
    pytest

Run just this file with:
    pytest tests/test_features.py

Run with verbose output to see each test name:
    pytest tests/test_features.py -v

If you add your own feature extraction functions to src/cell_tracking/features.py, add corresponding tests here to verify they work correctly. A test should check that your function produces output of the expected shape, type, and value range.
"""

import numpy as np

from cell_tracking.features import sliding_window_features

def test_sliding_window_features_shape():
    """sliding_window_features should return one feature vector per pixel.

    For a 5x5 image (25 pixels) with a 3x3 window (9 values per window),
    the output should be shape (25, 9).
    """
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    features = sliding_window_features(image, window_size=3)
    assert features.shape == (25, 9), (
        f"Expected shape (25, 9), got {features.shape}"
    )

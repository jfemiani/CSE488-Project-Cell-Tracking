from pathlib import Path

import numpy as np

from cell_tracking.features import sliding_window_features


def test_sliding_window_features_shape(tmp_path: Path):
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    features = sliding_window_features(image, window_size=3)
    assert features.shape == (25, 9)

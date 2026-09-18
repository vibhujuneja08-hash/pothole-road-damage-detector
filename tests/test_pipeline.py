import numpy as np

from src.analysis import analyze_mask, classify_severity
from src.segmentation import segment


def test_analysis_ratio():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:2, :] = 255
    result = analyze_mask(mask)
    assert result["foreground_pixels"] == 20
    assert abs(result["damage_ratio"] - 0.2) < 1e-9


def test_severity_thresholds():
    assert classify_severity(0.01) == "Low"
    assert classify_severity(0.05) == "Medium"
    assert classify_severity(0.10) == "High"


def test_segmentation_output_shape():
    image = np.full((100, 100), 100, dtype=np.uint8)
    image[40:60, 40:60] = 20
    mask = segment(image)
    assert mask.shape == image.shape
    assert set(np.unique(mask)).issubset({0, 255})

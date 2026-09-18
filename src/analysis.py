import numpy as np

from .config import LOW_THRESHOLD, HIGH_THRESHOLD


def analyze_mask(mask):
    total = int(mask.size)
    foreground = int(np.count_nonzero(mask))
    ratio = foreground / total if total else 0.0
    return {
        "foreground_pixels": foreground,
        "total_pixels": total,
        "damage_ratio": ratio,
    }


def classify_severity(ratio):
    if ratio < LOW_THRESHOLD:
        return "Low"
    if ratio <= HIGH_THRESHOLD:
        return "Medium"
    return "High"

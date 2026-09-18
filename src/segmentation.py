import cv2
import numpy as np

from .config import MIN_COMPONENT_AREA


def _cleanup(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255
    return cleaned


def threshold_segment(image):
    mask = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7
    )
    return _cleanup(mask)


def graphcut_segment(image):
    # Graph-cut segmentation using OpenCV GrabCut with a conservative
    # initialization. It is intended as an optional comparative method.
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    margin_y, margin_x = max(5, h // 20), max(5, w // 20)
    mask[:margin_y, :] = cv2.GC_BGD
    mask[-margin_y:, :] = cv2.GC_BGD
    mask[:, :margin_x] = cv2.GC_BGD
    mask[:, -margin_x:] = cv2.GC_BGD
    mask[margin_y:-margin_y, margin_x:-margin_x] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        color, mask, None, bgd_model, fgd_model, 2,
        cv2.GC_INIT_WITH_MASK
    )
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    return _cleanup(binary)


def segment(image, method="threshold"):
    if method == "graphcut":
        return graphcut_segment(image)
    return threshold_segment(image)

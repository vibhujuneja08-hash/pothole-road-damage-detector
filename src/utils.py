import csv
from pathlib import Path
import cv2


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_mask(path, mask):
    cv2.imwrite(str(path), mask)


def save_annotated(path, image):
    cv2.imwrite(str(path), image)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

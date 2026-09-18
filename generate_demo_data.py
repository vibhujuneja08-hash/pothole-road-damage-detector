from pathlib import Path
import cv2
import numpy as np


def create_image(index: int, output_dir: Path):
    rng = np.random.default_rng(index)
    h, w = 360, 640

    # Road-like background
    base = np.full((h, w, 3), 105, dtype=np.uint8)
    noise = rng.normal(0, 12, (h, w, 1))
    image = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Perspective-like lighter road bands
    cv2.line(image, (0, 80), (w, 170), (125, 125, 125), 3)
    cv2.line(image, (0, 280), (w, 200), (80, 80, 80), 3)

    # Artificial pothole/damage regions
    for _ in range(1 + index % 3):
        cx = int(rng.integers(100, w - 100))
        cy = int(rng.integers(130, h - 60))
        ax = int(rng.integers(20, 65))
        ay = int(rng.integers(12, 40))
        color = int(rng.integers(25, 65))
        cv2.ellipse(image, (cx, cy), (ax, ay), 0, 0, 360, (color, color, color), -1)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imwrite(str(output_dir / f"demo_{index:02d}.jpg"), image)


def main():
    output_dir = Path("data/input")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        create_image(i, output_dir)
    print(f"Generated 8 demo images in {output_dir}")


if __name__ == "__main__":
    main()

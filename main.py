import argparse
import json
from pathlib import Path

import cv2

from src.preprocessing import preprocess
from src.segmentation import segment
from src.analysis import analyze_mask, classify_severity
from src.utils import ensure_dir, save_mask, save_annotated, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Pothole and road-damage detector")
    parser.add_argument("--input", default="data/input", help="Input image directory")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument(
        "--method",
        choices=["threshold", "graphcut"],
        default="threshold",
        help="Segmentation method",
    )
    parser.add_argument("--save-masks", action="store_true", help="Save binary masks")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if args.max_images:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        print(f"No images found in {input_dir}. Run generate_demo_data.py first.")
        return

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        enhanced = preprocess(image)
        mask = segment(enhanced, method=args.method)
        metrics = analyze_mask(mask)
        severity = classify_severity(metrics["damage_ratio"])

        annotated = image.copy()
        text = (
            f"{severity} | area={metrics['damage_ratio'] * 100:.2f}%"
        )
        cv2.putText(
            annotated, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 255), 2, cv2.LINE_AA
        )

        save_annotated(output_dir / f"{image_path.stem}_annotated.jpg", annotated)
        if args.save_masks:
            save_mask(output_dir / f"{image_path.stem}_mask.png", mask)

        rows.append({
            "image": image_path.name,
            "method": args.method,
            "foreground_pixels": metrics["foreground_pixels"],
            "total_pixels": metrics["total_pixels"],
            "damage_ratio": round(metrics["damage_ratio"], 6),
            "damage_percentage": round(metrics["damage_ratio"] * 100, 4),
            "severity": severity,
        })

    write_csv(output_dir / "results.csv", rows)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Processed {len(rows)} image(s). Results saved to {output_dir}")


if __name__ == "__main__":
    main()

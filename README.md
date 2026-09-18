# Pothole and Road Damage Detector

A classical computer-vision project for detecting possible pothole/road-damage regions from road images.

## Features
- Grayscale conversion and denoising
- CLAHE contrast enhancement
- Adaptive-threshold segmentation
- Morphological noise removal
- Optional OpenCV GrabCut graph-cut segmentation
- Damage-area ratio calculation
- Low/Medium/High severity classification
- Annotated output images
- CSV and JSON reports
- Demo-image generator so the project can run without an external dataset

## Requirements

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Quick start

Generate demo road images:

```bash
python generate_demo_data.py
```

Run the detector:

```bash
python main.py --input data/input --output data/output
```

Use graph-cut mode:

```bash
python main.py --input data/input --output data/output --method graphcut
```

The results will be saved in `data/output`:
- annotated images
- binary masks
- `results.csv`
- `results.json`

## Run tests

```bash
pytest -q
```

## Command-line options

```bash
python main.py --help
```

## Severity thresholds

- Low: damage ratio < 2%
- Medium: 2% to 8%
- High: > 8%

These thresholds are configurable in `src/config.py`.

## Project structure

```text
.
├── main.py
├── generate_demo_data.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── analysis.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
└── data/
    ├── input/
    └── output/
```

## Academic note

This repository is an original educational implementation of a classical computer-vision pipeline. It does not claim that the numerical results in the earlier draft report were reproduced unless the same dataset and experimental configuration are supplied.

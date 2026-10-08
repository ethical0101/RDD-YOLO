"""Project-wide constants: paths and the RDD2022 class schema."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw" / "RDD2022"
PROCESSED_DIR = DATASET_DIR / "processed" / "rdd2022_yolo"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
ARCH_DIR = MODELS_DIR / "architectures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Order defines the YOLO class index.
CLASS_CODES: list[str] = ["D00", "D10", "D20", "D40"]
CLASS_NAMES: dict[str, str] = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}
CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_CODES)}


def class_label(idx: int) -> str:
    code = CLASS_CODES[idx]
    return f"{code} {CLASS_NAMES[code]}"

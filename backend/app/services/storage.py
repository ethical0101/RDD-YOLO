"""File storage under outputs/ and conversion of paths to public URLs."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..core.config import PROJECT_ROOT, get_settings


def new_stem(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def subdir(name: str) -> Path:
    p = get_settings().outputs_dir / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def rel(path: Path) -> str:
    """Path relative to the outputs dir, stored in the DB (portable across machines)."""
    return path.resolve().relative_to(get_settings().outputs_dir.resolve()).as_posix()


def url(relpath: str | None) -> str | None:
    return f"/files/outputs/{relpath}" if relpath else None


def abs_path(relpath: str) -> Path:
    return get_settings().outputs_dir / relpath


def save_image(img: np.ndarray, folder: str, stem: str, quality: int = 90) -> str:
    p = subdir(folder) / f"{stem}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return rel(p)


def save_crop(img: np.ndarray, bbox: list[float], stem: str, pad: float = 0.15) -> str:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    xa, ya = max(0, int(x1 - px)), max(0, int(y1 - py))
    xb, yb = min(w, int(x2 + px)), min(h, int(y2 + py))
    return save_image(img[ya:yb, xa:xb], "crops", stem, quality=88)


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def experiments_url(path: Path) -> str:
    return "/files/experiments/" + path.resolve().relative_to(get_settings().experiments_dir.resolve()).as_posix()


__all__ = ["PROJECT_ROOT", "new_stem", "save_image", "save_crop", "decode_image", "url", "rel", "abs_path",
           "experiments_url", "subdir"]

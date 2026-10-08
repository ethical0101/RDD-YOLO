"""Robust image decoding for uploads: JPEG/PNG/WebP/BMP via OpenCV, AVIF and HEIC/HEIF via Pillow."""
from __future__ import annotations

import io

import cv2
import numpy as np

try:  # optional: iPhone HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:  # pragma: no cover
    HEIF_SUPPORTED = False


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode bytes to a BGR uint8 array, or None if the data is not a readable image."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

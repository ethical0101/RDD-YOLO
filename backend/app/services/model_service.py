"""Loads and serves the active detection model."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..core.config import get_settings

log = logging.getLogger(__name__)

# Preference when no checkpoint is configured: extended-data RDD-YOLO (Exp C), RDD-YOLO (Exp B), baseline.
_PREFERRED = ["rdd_yolo26n_ext_best.pt", "rdd_yolo26n_best.pt", "rdd_yolo26s_best.pt", "baseline_yolo26n_best.pt", "baseline_yolo26s_best.pt"]


class ModelService:
    def __init__(self) -> None:
        self._detector = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def available_weights(self) -> list[dict]:
        wdir = get_settings().weights_dir
        out = []
        for p in sorted(wdir.glob("*_best.pt")):
            meta_f = p.with_suffix(".json")
            meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
            out.append({"file": p.name, "size_mb": round(p.stat().st_size / 1024**2, 2), **meta})
        return out

    def _default_weights(self) -> Path | None:
        s = get_settings()
        if s.model_weights:
            p = Path(s.model_weights)
            return p if p.is_absolute() else (s.weights_dir.parent.parent / p)
        for name in _PREFERRED:
            if (s.weights_dir / name).exists():
                return s.weights_dir / name
        found = sorted(s.weights_dir.glob("*_best.pt"))
        return found[0] if found else None

    def load(self, weights: str | Path | None = None):
        from rdd_yolo.detector import Detector

        s = get_settings()
        with self._lock:
            path = Path(weights) if weights else self._default_weights()
            if path is not None and not path.is_absolute() and not path.exists():
                path = s.weights_dir / path
            if path is None or not path.exists():
                self._detector = None
                self._error = ("No trained checkpoint found in models/weights. "
                               "Train a model first (scripts/train.ps1) - inference is unavailable until then.")
                log.warning(self._error)
                return None
            try:
                self._detector = Detector(path, device=s.model_device or None, imgsz=s.model_imgsz)
                self._error = None
                log.info("Loaded model %s on %s", path.name, self._detector.device)
            except Exception as exc:  # keep API alive, report the real error
                self._detector = None
                self._error = f"Failed to load {path.name}: {exc}"
                log.exception("Model load failed")
            return self._detector

    @property
    def detector(self):
        if self._detector is None and self._error is None:
            self.load()
        return self._detector

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def version(self) -> str:
        d = self._detector
        return d.weights.stem if d else "none"


model_service = ModelService()

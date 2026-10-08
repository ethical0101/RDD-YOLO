"""Model architecture and inference tests (CPU)."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built():
    from rdd_yolo.modules import register_custom_modules

    register_custom_modules()
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    out = {}
    for name in ("baseline", "rdd"):
        cfg = yaml.safe_load((ROOT / f"models/architectures/yolo26-{name}.yaml").read_text())
        cfg["scale"] = "n"
        out[name] = DetectionModel(cfg, ch=3, nc=4, verbose=False)
    return out


def test_rdd_architecture_contains_modifications(built):
    from rdd_yolo.modules import SimAM

    rdd, base = built["rdd"], built["baseline"]
    assert sum(isinstance(m, SimAM) for m in rdd.modules()) == 1
    assert sum(type(m).__name__ == "GhostConv" for m in rdd.modules()) == 2
    ups = [m for m in rdd.modules() if isinstance(m, torch.nn.Upsample)]
    assert [(u.mode, u.align_corners) for u in ups] == [("bilinear", True)] * 2
    assert not any(isinstance(m, SimAM) for m in base.modules())
    assert all(u.mode == "nearest" for u in base.modules() if isinstance(u, torch.nn.Upsample))


def test_rdd_has_fewer_parameters(built):
    n = {k: sum(p.numel() for p in m.parameters()) for k, m in built.items()}
    assert n["rdd"] < n["baseline"]


def test_forward_shapes(built):
    for m in built.values():
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 3, 320, 320))
        assert y is not None


def test_detector_loads_and_predicts(weights, sample_image):
    from rdd_yolo.constants import CLASS_CODES
    from rdd_yolo.detector import Detector, draw_detections

    det = Detector(weights, device="cpu")
    assert det.info["classes"] == CLASS_CODES and det.info["parameters"] > 0
    img = cv2.imread(str(sample_image))
    dets, ms = det.predict(img, conf=0.01)
    assert ms > 0
    for d in dets:
        assert d.class_code in CLASS_CODES and 0 < d.confidence <= 1
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 <= x2 <= img.shape[1] + 1 and 0 <= y1 <= y2 <= img.shape[0] + 1
        assert d.severity in {"LOW", "MEDIUM", "HIGH"}
    assert draw_detections(img, dets).shape == img.shape


def test_detector_missing_weights():
    from rdd_yolo.detector import Detector

    with pytest.raises(FileNotFoundError):
        Detector(ROOT / "models/weights/does_not_exist.pt")

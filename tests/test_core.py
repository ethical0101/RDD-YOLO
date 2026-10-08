"""Unit tests for the core library: severity, geolocation, SimAM, hardware, dataset preparation."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rdd_yolo.geo import Route, extract_exif_gps, haversine_m, validate_coords
from rdd_yolo.hardware import detect_hardware, recommend_training_params
from rdd_yolo.modules import SimAM
from rdd_yolo.severity import estimate_severity, severity_rules

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- severity

def test_severity_pothole_large_is_high():
    s = estimate_severity("D40", (0, 0, 320, 320), (640, 640), 0.9)
    assert s.level == "HIGH" and s.area_score == 1.0


def test_severity_small_crack_is_low():
    s = estimate_severity("D00", (0, 0, 10, 10), (640, 640), 0.3)
    assert s.level == "LOW"


def test_severity_monotonic_in_area_and_repeat():
    small = estimate_severity("D20", (0, 0, 40, 40), (640, 640), 0.6).score
    big = estimate_severity("D20", (0, 0, 200, 200), (640, 640), 0.6).score
    rep = estimate_severity("D20", (0, 0, 40, 40), (640, 640), 0.6, nearby_count=3)
    assert big > small
    assert rep.score == pytest.approx(small + 15) and rep.repeat_bonus == 15


def test_severity_rules_documented():
    r = severity_rules()
    assert "not an engineering-certified" in r["disclaimer"]
    assert set(r["levels"]) == {"LOW", "MEDIUM", "HIGH"}


# ---------------------------------------------------------------- geo

def _jpeg_with_gps(lat: float, lon: float) -> bytes:
    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = round(((v - d) * 60 - m) * 60, 4)
        return (d, m, s)

    img = Image.new("RGB", (32, 32), "gray")
    exif = Image.Exif()
    gps = {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon)}
    exif[0x8825] = gps
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_gps_roundtrip():
    lat, lon = 12.9692, 79.1559  # VIT Vellore
    got = extract_exif_gps(_jpeg_with_gps(lat, lon))
    assert got is not None
    assert got[0] == pytest.approx(lat, abs=1e-4) and got[1] == pytest.approx(lon, abs=1e-4)


def test_exif_southern_western_hemisphere():
    got = extract_exif_gps(_jpeg_with_gps(-33.8688, -70.6693))
    assert got[0] < 0 and got[1] < 0


def test_exif_absent_returns_none():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    assert extract_exif_gps(buf.getvalue()) is None
    assert extract_exif_gps(b"not an image") is None


def test_haversine_and_validation():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert validate_coords(12.9, 79.1) and not validate_coords(91, 0) and not validate_coords(None, 1)


def test_route_csv_interpolation():
    r = Route.from_bytes(b"time_s,lat,lon\n0,12.0,79.0\n10,12.001,79.001\n", "route.csv")
    lat, lon = r.position_at(5)
    assert lat == pytest.approx(12.0005) and lon == pytest.approx(79.0005)
    assert r.position_at(11) is None  # outside the track: no fabricated position


def test_route_gpx():
    gpx = b"""<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
    <trkpt lat="12.0" lon="79.0"><time>2026-01-01T10:00:00Z</time></trkpt>
    <trkpt lat="12.002" lon="79.0"><time>2026-01-01T10:00:20Z</time></trkpt></trkseg></trk></gpx>"""
    r = Route.from_bytes(gpx, "track.gpx")
    assert r.position_at(10)[0] == pytest.approx(12.001)


def test_route_rejects_bad_coords():
    with pytest.raises(ValueError):
        Route.from_bytes(b"time_s,lat,lon\n0,120,79\n", "r.csv")


# ---------------------------------------------------------------- SimAM

def test_simam_is_parameter_free_and_shape_preserving():
    m = SimAM()
    x = torch.randn(2, 8, 10, 12)
    y = m(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in m.parameters()) == 0


def test_simam_matches_reference_formula():
    x = torch.randn(1, 3, 5, 5)
    lam = 1e-4
    n = 5 * 5 - 1
    mu = x.mean(dim=(2, 3), keepdim=True)
    var = ((x - mu) ** 2).sum(dim=(2, 3), keepdim=True) / n
    e_inv = (x - mu) ** 2 / (4 * (var + lam)) + 0.5  # = 1 / e_t*
    assert torch.allclose(SimAM(e_lambda=lam)(x), x * torch.sigmoid(e_inv), atol=1e-6)


# ---------------------------------------------------------------- hardware

def test_hardware_detection_and_recommendation():
    hw = detect_hardware()
    assert hw.cpu_cores >= 1
    rec = recommend_training_params(hw)
    assert rec.batch >= 1 and rec.imgsz > 0
    if not hw.cuda_available:
        assert rec.device == "cpu" and rec.amp is False


# ---------------------------------------------------------------- dataset preparation

VOC = """<annotation><size><width>{w}</width><height>{h}</height><depth>3</depth></size>{objs}</annotation>"""
OBJ = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


def _make_raw(root: Path) -> None:
    """Synthetic VOC fixture (test-only data) exercising every validation branch."""
    img_dir = root / "Testland" / "train" / "images"
    xml_dir = root / "Testland" / "train" / "annotations" / "xmls"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(12):
        Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)).save(img_dir / f"img_{i}.jpg")
        objs = OBJ.format(n=["D00", "D10", "D20", "D40"][i % 4], a=10, b=10, c=60, d=50)
        if i == 0:
            objs += OBJ.format(n="D44", a=1, b=1, c=20, d=20)  # non-target label -> dropped
        if i == 1:
            objs += OBJ.format(n="D40", a=150, b=40, c=400, d=90)  # out of bounds -> clipped
        if i == 2:
            objs += OBJ.format(n="D00", a=5, b=5, c=5.5, d=40)  # degenerate -> dropped
        (xml_dir / f"img_{i}.xml").write_text(VOC.format(w=200, h=100, objs=objs))
    (img_dir / "corrupt.jpg").write_bytes(b"\xff\xd8\xff garbage")  # corrupt image
    Image.new("RGB", (200, 100)).save(img_dir / "background.jpg")
    (xml_dir / "background.xml").write_text(VOC.format(w=200, h=100, objs=""))
    (xml_dir / "orphan.xml").write_text(VOC.format(w=200, h=100, objs=""))  # xml without image


def test_prepare_dataset_end_to_end(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    r = subprocess.run([sys.executable, str(ROOT / "dataset" / "prepare_dataset.py"), "--raw", str(raw),
                        "--out", str(out), "--bg-ratio", "1.0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    report = json.loads((out / "validation_report.json").read_text())
    issues = " | ".join(i["issue"] for i in report["issues"])
    assert "corrupt image" in issues and "annotation without image" in issues and "degenerate" in issues
    assert report["ignored_labels"] == {"D44": 1}
    assert report["images_valid"] == 13  # 12 + background; corrupt excluded
    stats = json.loads((out / "dataset_stats.json").read_text())
    total = sum(s["images"] for s in stats["splits"].values())
    assert total == 13
    assert sum(s["instances"] for s in stats["splits"].values()) == 13  # 12 + 1 clipped extra
    labels = [ln for p in (out / "labels").rglob("*.txt") for ln in p.read_text().splitlines()]
    for ln in labels:
        c, *xywh = ln.split()
        assert c in {"0", "1", "2", "3"}
        assert all(0.0 <= float(v) <= 1.0 for v in xywh)
    assert (out / "data.yaml").exists() and (out / "class_distribution.png").exists()

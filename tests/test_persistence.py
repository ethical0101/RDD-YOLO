"""Persistence paths with a deterministic detector *test double*.

The smoke checkpoint is too undertrained to produce boxes, so these tests patch
``Detector.predict`` to return fixed, clearly synthetic detections. This only
verifies storage / severity / map / route logic - model quality is evaluated
separately with training/evaluate.py on the real test split.
"""
from __future__ import annotations

import time

import cv2
import pytest

from rdd_yolo.detector import Detection
from rdd_yolo.severity import estimate_severity
from tests.conftest import sample_images


def _fake_predict(self, image_bgr, conf=0.25, iou=0.7, nearby_counter=None):
    h, w = image_bgr.shape[:2]
    out = []
    for k, (code, box, c) in enumerate([("D40", (0.2, 0.5, 0.6, 0.9), 0.91), ("D00", (0.05, 0.1, 0.15, 0.6), 0.55)]):
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        sev = estimate_severity(code, (x1, y1, x2, y2), (w, h), c, nearby_counter(code) if nearby_counter else 0)
        out.append(Detection(3 if code == "D40" else 0, code, "TEST", c, [x1, y1, x2, y2],
                             [box[0], box[1], box[2], box[3]], sev.level, sev.score, sev.to_dict()))
    return out, 1.0


@pytest.fixture()
def fake_model(client, monkeypatch):
    from backend.app.services.model_service import model_service

    monkeypatch.setattr(type(model_service.detector), "predict", _fake_predict)
    yield


def test_detections_persisted_with_crops_and_map(client, fake_model, sample_image):
    lat, lon = 12.9716, 79.1594
    body = client.post("/api/inference/image", files=[("file", (sample_image.name, sample_image.read_bytes(), "image/jpeg"))],
                       data={"latitude": str(lat), "longitude": str(lon), "location_source": "browser"}).json()
    assert len(body["detections"]) == 2
    inf = client.get(f"/api/inferences/{body['inference_id']}").json()
    assert inf["num_detections"] == 2
    for d in inf["detections"]:
        assert client.get(d["crop_url"]).status_code == 200
        assert d["location_source"] == "browser"
    feats = client.get("/api/map/detections", params={"classes": "D40"}).json()["features"]
    assert any(f["properties"]["inference_id"] == body["inference_id"] for f in feats)
    assert all(f["properties"]["class_code"] == "D40" for f in feats)


def test_repeat_detections_raise_severity(client, fake_model, sample_image):
    lat, lon = 13.5, 78.5  # fresh location
    scores = []
    for _ in range(3):
        b = client.post("/api/inference/image", files=[("file", (sample_image.name, sample_image.read_bytes(), "image/jpeg"))],
                        data={"latitude": str(lat), "longitude": str(lon), "location_source": "manual"}).json()
        scores.append(next(d for d in b["detections"] if d["class_code"] == "D00")["severity_score"])
    assert scores[1] == pytest.approx(scores[0] + 5) and scores[2] == pytest.approx(scores[0] + 10)


def test_video_route_mapping_and_dedup(client, fake_model, tmp_path):
    img = cv2.resize(cv2.imread(str(sample_images(1)[0])), (320, 320))
    vid = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
    for _ in range(40):  # 4 s static clip
        w.write(img)
    w.release()
    route = b"time_s,lat,lon\n0,12.0,79.0\n4,12.004,79.0\n"
    job = client.post("/api/inference/video", files=[("file", ("v.mp4", vid.read_bytes(), "video/mp4")),
                                                     ("route_file", ("r.csv", route, "text/csv"))],
                      data={"stride": "2"}).json()["job_id"]
    for _ in range(240):
        st = client.get(f"/api/inference/video/{job}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert st["status"] == "completed", st.get("error")
    assert st["extra"]["raw_frame_detections"] == 40  # 20 processed frames x 2
    # identical boxes in consecutive frames collapse into one track per class
    assert st["num_detections"] == 2
    for d in st["detections"]:
        assert d["location_source"] == "route"
        assert 12.0 <= d["latitude"] <= 12.004
        assert d["crop_url"]

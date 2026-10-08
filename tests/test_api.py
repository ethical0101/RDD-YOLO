"""API integration tests (TestClient, isolated temp DB, CPU inference)."""
from __future__ import annotations

import io
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.conftest import sample_images


def _upload(path) -> tuple:
    return ("file", (path.name, path.read_bytes(), "image/jpeg"))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"] is True


def test_model_and_meta_endpoints(client):
    m = client.get("/api/model").json()
    assert m["loaded"] and m["info"]["classes"] == ["D00", "D10", "D20", "D40"]
    assert len(client.get("/api/classes").json()) == 4
    assert "formula" in client.get("/api/severity/rules").json()
    hw = client.get("/api/system/hardware").json()
    assert hw["hardware"]["cpu_cores"] >= 1


def test_image_inference_manual_location_saved(client, sample_image):
    r = client.post("/api/inference/image", files=[_upload(sample_image)],
                    data={"conf": "0.01", "latitude": "12.9692", "longitude": "79.1559",
                          "location_source": "manual"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] and body["inference_id"]
    assert body["location"]["source"] == "manual"
    inf = client.get(f"/api/inferences/{body['inference_id']}").json()
    assert inf["num_detections"] == len(body["detections"])
    assert client.get(inf["annotated_url"]).status_code == 200
    for d in inf["detections"]:
        assert d["latitude"] == pytest.approx(12.9692) and d["location_source"] == "manual"


def test_image_inference_without_location_has_none(client, sample_image):
    r = client.post("/api/inference/image", files=[_upload(sample_image)], data={"conf": "0.01"})
    assert r.status_code == 200
    assert r.json()["location"] is None  # never fabricated


def test_exif_gps_is_used(client, sample_image):
    from tests.test_core import _jpeg_with_gps  # reuse EXIF writer

    img = Image.open(sample_image).convert("RGB")
    exif_src = Image.open(io.BytesIO(_jpeg_with_gps(13.0827, 80.2707))).getexif()
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif_src)
    r = client.post("/api/inference/image", files=[("file", ("gps.jpg", buf.getvalue(), "image/jpeg"))],
                    data={"conf": "0.01", "latitude": "1.0", "longitude": "1.0", "location_source": "browser"})
    body = r.json()
    assert body["exif_gps"]["lat"] == pytest.approx(13.0827, abs=1e-4)
    assert body["location"]["source"] == "exif"


def test_preview_not_saved(client, sample_image):
    before = client.get("/api/stats").json()["total_inferences"]
    r = client.post("/api/inference/image", files=[_upload(sample_image)], data={"save": "false"})
    assert r.status_code == 200 and r.json()["saved"] is False and r.json()["annotated_base64"]
    assert client.get("/api/stats").json()["total_inferences"] == before


def test_patch_location(client, sample_image):
    inf_id = client.post("/api/inference/image", files=[_upload(sample_image)], data={"conf": "0.01"}).json()["inference_id"]
    r = client.patch(f"/api/inferences/{inf_id}/location", json={"latitude": 12.97, "longitude": 79.16, "source": "browser",
                                                                 "accuracy_m": 20})
    assert r.status_code == 200 and r.json()["location_source"] == "browser"
    assert client.patch(f"/api/inferences/{inf_id}/location", json={"latitude": 200, "longitude": 0}).status_code == 422


def test_validation_errors(client, sample_image):
    assert client.post("/api/inference/image", files=[("file", ("x.txt", b"hello", "text/plain"))]).status_code == 415
    assert client.post("/api/inference/image", files=[("file", ("x.jpg", b"notimage", "image/jpeg"))]).status_code == 400
    bad = client.post("/api/inference/image", files=[_upload(sample_image)], data={"latitude": "95", "longitude": "10"})
    assert bad.status_code == 422
    only_lat = client.post("/api/inference/image", files=[_upload(sample_image)], data={"latitude": "10"})
    assert only_lat.status_code == 422
    assert client.get("/api/inferences/999999").status_code == 404
    assert client.get("/api/detections/999999").status_code == 404
    assert client.post("/api/model/select", json={"weights": "../evil.pt"}).status_code == 422


def test_detections_list_filters_and_stats(client):
    all_ = client.get("/api/detections", params={"limit": 1000}).json()
    assert all_["total"] == len(all_["items"])
    if all_["total"]:
        code = all_["items"][0]["class_code"]
        f = client.get("/api/detections", params={"classes": code, "limit": 1000}).json()
        assert all(i["class_code"] == code for i in f["items"])
        hc = client.get("/api/detections", params={"min_conf": 0.5}).json()
        assert all(i["confidence"] >= 0.5 for i in hc["items"])
    s = client.get("/api/stats").json()
    assert s["total_detections"] == all_["total"]
    assert sum(v["count"] for v in s["per_class"].values()) == s["total_detections"]
    assert sum(s["severity"].values()) == s["total_detections"]
    assert len(s["over_time"]) == 30


def test_map_geojson(client):
    g = client.get("/api/map/detections").json()
    assert g["type"] == "FeatureCollection"
    for f in g["features"]:
        lon, lat = f["geometry"]["coordinates"]
        assert -180 <= lon <= 180 and -90 <= lat <= 90
        assert f["properties"]["location_source"] in {"manual", "browser", "exif", "route"}


def test_webcam_frame(client, sample_image):
    r = client.post("/api/inference/frame", files=[_upload(sample_image)], data={"conf": "0.25"})
    assert r.status_code == 200 and "detections" in r.json()


def test_video_inference_with_route(client, tmp_path):
    imgs = sample_images(3)
    if not imgs:
        pytest.skip("no images")
    frames = [cv2.resize(cv2.imread(str(p)), (320, 320)) for p in imgs]
    vid = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 320))
    for k in range(30):  # 3 s synthetic clip assembled from real dataset frames
        w.write(frames[(k // 10) % len(frames)])
    w.release()
    route = b"time_s,lat,lon\n0,12.9690,79.1550\n3,12.9700,79.1560\n"
    r = client.post("/api/inference/video",
                    files=[("file", ("clip.mp4", vid.read_bytes(), "video/mp4")),
                           ("route_file", ("route.csv", route, "text/csv"))],
                    data={"conf": "0.01", "stride": "5"})
    assert r.status_code == 202, r.text
    job = r.json()["job_id"]
    for _ in range(240):
        st = client.get(f"/api/inference/video/{job}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "completed", st.get("error")
    assert st["extra"]["processed_frames"] == 6
    assert client.get(st["output_video_url"]).status_code == 200
    for d in st["detections"]:
        assert d["video_time_s"] is not None
        if d["latitude"] is not None:
            assert d["location_source"] == "route" and 12.969 <= d["latitude"] <= 12.970
    assert client.post("/api/inference/video", files=[("file", ("a.txt", b"x", "text/plain"))]).status_code == 415


def test_training_and_dataset_endpoints(client):
    exps = client.get("/api/training/experiments")
    assert exps.status_code == 200 and isinstance(exps.json(), list)
    for e in exps.json():
        d = client.get(f"/api/training/experiments/{e['name']}").json()
        assert "history" in d
    assert client.get("/api/training/experiments/..%2F..%2Fbackend").status_code == 404  # traversal blocked
    assert client.get("/api/does-not-exist").status_code == 404  # unknown API path is not served the SPA
    r = client.get("/files/outputs/..%2F..%2F.env")
    assert r.status_code == 404 and "RDD_" not in r.text  # StaticFiles refuses traversal
    assert "available" in client.get("/api/training/comparison").json()
    assert "available" in client.get("/api/dataset/stats").json()


def test_delete_detection_and_inference(client, sample_image):
    body = client.post("/api/inference/image", files=[_upload(sample_image)], data={"conf": "0.01"}).json()
    iid = body["inference_id"]
    inf = client.get(f"/api/inferences/{iid}").json()
    if inf["detections"]:
        did = inf["detections"][0]["id"]
        assert client.delete(f"/api/detections/{did}").status_code == 204
        assert client.get(f"/api/detections/{did}").status_code == 404
    assert client.delete(f"/api/inferences/{iid}").status_code == 204
    assert client.get(f"/api/inferences/{iid}").status_code == 404


@pytest.mark.parametrize("fmt,mime", [("AVIF", "image/avif"), ("HEIF", "image/heic")])
def test_modern_formats_avif_heic(client, sample_image, fmt, mime):
    import rdd_yolo.imageio  # noqa: F401  (registers HEIF encoder/decoder)

    img = Image.open(sample_image).convert("RGB")
    buf = io.BytesIO()
    kwargs = {}
    if fmt == "HEIF":  # also carry EXIF GPS through a HEIC file
        from tests.test_core import _jpeg_with_gps

        kwargs["exif"] = Image.open(io.BytesIO(_jpeg_with_gps(12.95, 79.13))).getexif()
    img.save(buf, fmt, **kwargs)
    r = client.post("/api/inference/image", files=[("file", (f"photo.{fmt.lower()}", buf.getvalue(), mime))],
                    data={"conf": "0.25"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["width"], body["height"]) == img.size
    assert client.get(body["image_url"]).status_code == 200  # stored as JPEG, viewable in any browser
    if fmt == "HEIF":
        assert body["location"]["source"] == "exif" and body["location"]["lat"] == pytest.approx(12.95, abs=1e-4)

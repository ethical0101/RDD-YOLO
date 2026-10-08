"""REST API routes."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

import rdd_yolo
from rdd_yolo.constants import CLASS_CODES, CLASS_NAMES
from rdd_yolo.geo import Route, validate_coords
from rdd_yolo.severity import severity_rules

from ..core.config import get_settings
from ..db.database import get_db
from ..db.models import Detection, Inference
from ..services import storage, training_info
from ..services.detections import detection_to_dict, filtered_query, inference_to_dict, statistics
from ..services.inference import (InferenceUnavailable, Location, detect_frame, resolve_location,
                                  run_image_inference, start_video_job)
from ..services.model_service import model_service
from .schemas import HealthResponse, LocationUpdate, ModelSelect

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
DB = Annotated[Session, Depends(get_db)]

IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp", "image/bmp", "image/avif",
               "image/heic", "image/heif", "application/octet-stream"}  # octet-stream: decoded and validated below
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


async def _read_upload(f: UploadFile, kind: str) -> bytes:
    data = await f.read()
    limit = get_settings().max_upload_mb * 1024 * 1024
    if not data:
        raise HTTPException(400, f"Empty {kind} upload")
    if len(data) > limit:
        raise HTTPException(413, f"{kind} exceeds {get_settings().max_upload_mb} MB limit")
    return data


def _conf(conf: float | None) -> float:
    c = get_settings().default_confidence if conf is None else conf
    if not 0.01 <= c <= 0.99:
        raise HTTPException(422, "conf must be between 0.01 and 0.99")
    return c


def _split(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


# ------------------------------------------------------------------ system


@router.get("/health", response_model=HealthResponse)
def health(db: DB):
    import torch

    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # pragma: no cover
        db_status = f"error: {exc}"
    return HealthResponse(status="ok", model_loaded=model_service.detector is not None,
                          model_version=model_service.version, model_error=model_service.error,
                          database=db_status, cuda_available=torch.cuda.is_available(), version=rdd_yolo.__version__)


@router.get("/system/hardware")
def hardware():
    from rdd_yolo.hardware import detect_hardware, recommend_training_params

    hw = detect_hardware()
    return {"hardware": hw.to_dict(), "training_recommendation": recommend_training_params(hw).__dict__}


@router.get("/classes")
def classes():
    return [{"id": i, "code": c, "name": CLASS_NAMES[c]} for i, c in enumerate(CLASS_CODES)]


@router.get("/severity/rules")
def severity():
    return severity_rules()


# ------------------------------------------------------------------ model


@router.get("/model")
def model_info():
    det = model_service.detector
    return {"loaded": det is not None, "error": model_service.error, "info": det.info if det else None,
            "available": model_service.available_weights()}


@router.post("/model/select")
def model_select(body: ModelSelect):
    if "/" in body.weights or "\\" in body.weights or not body.weights.endswith(".pt"):
        raise HTTPException(422, "weights must be a .pt file name inside models/weights")
    if not (get_settings().weights_dir / body.weights).exists():
        raise HTTPException(404, f"{body.weights} not found")
    det = model_service.load(body.weights)
    if det is None:
        raise HTTPException(500, model_service.error)
    return {"loaded": True, "info": det.info}


# ------------------------------------------------------------------ inference


@router.post("/inference/image")
async def infer_image(
    db: DB,
    file: UploadFile = File(...),
    conf: float | None = Form(None),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    location_source: str | None = Form(None),
    location_accuracy_m: float | None = Form(None),
    prefer_exif: bool = Form(True),
    save: bool = Form(True),
):
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported image type {file.content_type}")
    data = await _read_upload(file, "image")
    img = storage.decode_image(data)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    if (latitude is None) != (longitude is None) or (latitude is not None and not validate_coords(latitude, longitude)):
        raise HTTPException(422, "Provide both latitude and longitude within valid ranges")
    loc, exif = resolve_location(data, latitude, longitude, location_source, location_accuracy_m, prefer_exif)
    try:
        res = run_image_inference(db, img, data, file.filename, _conf(conf), loc, "image", save)
    except InferenceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    res["exif_gps"] = {"lat": exif[0], "lon": exif[1]} if exif else None
    return res


@router.post("/inference/frame")
async def infer_frame(
    db: DB,
    file: UploadFile = File(...),
    conf: float | None = Form(None),
    save: bool = Form(False),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    location_accuracy_m: float | None = Form(None),
):
    """Webcam frames. Not stored unless save=true (snapshot)."""
    data = await _read_upload(file, "frame")
    img = storage.decode_image(data)
    if img is None:
        raise HTTPException(400, "Could not decode frame")
    try:
        if save:
            loc = Location(latitude, longitude, "browser", location_accuracy_m) if validate_coords(latitude, longitude) \
                else Location()
            return run_image_inference(db, img, None, "webcam.jpg", _conf(conf), loc, "webcam", True)
        dets, ms = detect_frame(img, _conf(conf))
    except InferenceUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    h, w = img.shape[:2]
    return {"detections": [d.to_dict() for d in dets], "inference_ms": round(ms, 2), "width": w, "height": h}


@router.post("/inference/video", status_code=202)
async def infer_video(
    db: DB,
    file: UploadFile = File(...),
    route_file: UploadFile | None = File(None),
    conf: float | None = Form(None),
    stride: int | None = Form(None),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    location_source: str | None = Form(None),
):
    from pathlib import Path

    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(415, f"Unsupported video extension '{ext}'")
    stride = stride or get_settings().video_default_stride
    if not 1 <= stride <= 30:
        raise HTTPException(422, "stride must be 1..30")
    route = None
    if route_file is not None and route_file.filename:
        try:
            route = Route.from_bytes(await route_file.read(), route_file.filename)
        except Exception as exc:
            raise HTTPException(422, f"Invalid route file: {exc}") from exc
    if latitude is not None and not validate_coords(latitude, longitude):
        raise HTTPException(422, "Invalid coordinates")
    data = await _read_upload(file, "video")
    vpath = storage.subdir("videos") / f"{storage.new_stem('video')}{ext}"
    vpath.write_bytes(data)
    import cv2

    cap = cv2.VideoCapture(str(vpath))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        vpath.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode video")
    src = location_source if location_source in ("browser", "manual") else "manual"
    fixed = Location(latitude, longitude, src) if latitude is not None else Location()
    try:
        inf = start_video_job(db, vpath, file.filename or vpath.name, _conf(conf), stride, fixed, route)
    except InferenceUnavailable as exc:
        vpath.unlink(missing_ok=True)
        raise HTTPException(503, str(exc)) from exc
    return {"job_id": inf.id, "status": inf.status}


@router.get("/inference/video/{job_id}")
def video_status(job_id: int, db: DB):
    inf = db.get(Inference, job_id)
    if inf is None or inf.kind != "video":
        raise HTTPException(404, "Video job not found")
    return inference_to_dict(inf, with_detections=inf.status == "completed")


@router.get("/videos")
def list_videos(db: DB, limit: int = Query(20, ge=1, le=100)):
    rows = db.scalars(select(Inference).where(Inference.kind == "video").order_by(Inference.id.desc()).limit(limit))
    return [inference_to_dict(i, with_detections=False) for i in rows]


# ------------------------------------------------------------------ records


@router.get("/inferences/{inference_id}")
def get_inference(inference_id: int, db: DB):
    inf = db.get(Inference, inference_id)
    if inf is None:
        raise HTTPException(404, "Inference not found")
    return inference_to_dict(inf)


@router.patch("/inferences/{inference_id}/location")
def set_location(inference_id: int, body: LocationUpdate, db: DB):
    """Attach/replace coordinates of an image inference and all of its detections."""
    inf = db.get(Inference, inference_id)
    if inf is None:
        raise HTTPException(404, "Inference not found")
    if inf.kind == "video" and inf.location_source == "route":
        raise HTTPException(409, "Route-mapped video detections have per-frame coordinates")
    inf.latitude, inf.longitude = body.latitude, body.longitude
    inf.location_source, inf.location_accuracy_m = body.source, body.accuracy_m
    for d in inf.detections:
        d.latitude, d.longitude, d.location_source = body.latitude, body.longitude, body.source
    db.commit()
    return inference_to_dict(inf)


@router.delete("/inferences/{inference_id}", status_code=204)
def delete_inference(inference_id: int, db: DB):
    inf = db.get(Inference, inference_id)
    if inf is None:
        raise HTTPException(404, "Inference not found")
    db.delete(inf)
    db.commit()


@router.get("/detections")
def list_detections(
    db: DB,
    classes: str | None = None,
    min_conf: float | None = Query(None, ge=0, le=1),
    severity: str | None = None,
    source: str | None = None,
    has_location: bool | None = None,
    kind: str | None = None,
    days: int | None = Query(None, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at", pattern="^(created_at|confidence|severity_score)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    since = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    q = filtered_query(_split(classes), min_conf, _split(severity), _split(source), has_location, since, kind)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    col = getattr(Detection, sort)
    q = q.order_by(col.desc() if order == "desc" else col.asc(), Detection.id.desc()).limit(limit).offset(offset)
    return {"total": total, "limit": limit, "offset": offset, "items": [detection_to_dict(d) for d in db.scalars(q)]}


@router.get("/detections/{detection_id}")
def get_detection(detection_id: int, db: DB):
    d = db.get(Detection, detection_id)
    if d is None:
        raise HTTPException(404, "Detection not found")
    return detection_to_dict(d)


@router.delete("/detections/{detection_id}", status_code=204)
def delete_detection(detection_id: int, db: DB):
    d = db.get(Detection, detection_id)
    if d is None:
        raise HTTPException(404, "Detection not found")
    inf = d.inference
    db.delete(d)
    if inf is not None:
        inf.num_detections = max(0, inf.num_detections - 1)
    db.commit()


@router.get("/stats")
def stats(db: DB, days: int = Query(30, ge=1, le=365)):
    return statistics(db, days)


@router.get("/map/detections")
def map_detections(
    db: DB,
    classes: str | None = None,
    min_conf: float | None = Query(None, ge=0, le=1),
    severity: str | None = None,
    source: str | None = None,
    days: int | None = Query(None, ge=1, le=3650),
    limit: int = Query(5000, ge=1, le=20000),
):
    """GeoJSON FeatureCollection of geolocated detections (lon, lat order per RFC 7946)."""
    since = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    q = filtered_query(_split(classes), min_conf, _split(severity), _split(source), True, since)
    q = q.order_by(Detection.created_at.desc()).limit(limit)
    feats = []
    for d in db.scalars(q):
        props = detection_to_dict(d)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [d.longitude, d.latitude]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


# ------------------------------------------------------------------ training / dataset


@router.get("/training/experiments")
def experiments():
    return training_info.list_experiments()


@router.get("/training/experiments/{name}")
def experiment(name: str):
    d = training_info.experiment_detail(name)
    if d is None:
        raise HTTPException(404, "Experiment not found")
    return d


@router.get("/training/comparison")
def comparison():
    c = training_info.comparison()
    if c is None:
        return {"available": False, "message": "Comparison pending: train and evaluate both experiments, "
                                               "then run training/compare.py"}
    return {"available": True, **c}


@router.get("/dataset/stats")
def dataset_stats():
    s = training_info.dataset_stats()
    if s is None:
        return {"available": False, "message": "Dataset not prepared yet (run dataset/prepare_dataset.py)"}
    return {"available": True, **s}

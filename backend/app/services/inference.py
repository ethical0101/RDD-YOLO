"""Image and video inference workflows (detection + severity + location + persistence)."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from rdd_yolo.detector import Detection as Det
from rdd_yolo.detector import draw_detections, process_video
from rdd_yolo.geo import Route, extract_exif_gps, validate_coords

from ..db.database import session_factory
from ..db.models import Detection, Inference
from . import storage
from .detections import count_nearby
from .model_service import model_service

log = logging.getLogger(__name__)


class InferenceUnavailable(RuntimeError):
    pass


@dataclass
class Location:
    lat: float | None = None
    lon: float | None = None
    source: str | None = None
    accuracy_m: float | None = None

    @property
    def known(self) -> bool:
        return validate_coords(self.lat, self.lon)


def resolve_location(image_bytes: bytes | None, lat: float | None, lon: float | None, source: str | None,
                     accuracy: float | None, prefer_exif: bool = True) -> tuple[Location, tuple[float, float] | None]:
    """Choose the location for an image. EXIF GPS (if embedded) wins unless prefer_exif is False.

    Returns (chosen_location, exif_coords_found). Never invents coordinates.
    """
    exif = extract_exif_gps(image_bytes) if image_bytes else None
    if exif and (prefer_exif or not validate_coords(lat, lon)):
        return Location(exif[0], exif[1], "exif", None), exif
    if validate_coords(lat, lon):
        src = source if source in ("browser", "manual", "route") else "manual"
        return Location(lat, lon, src, accuracy), exif
    return Location(), exif


def _require_detector():
    det = model_service.detector
    if det is None:
        raise InferenceUnavailable(model_service.error or "Model not loaded")
    return det


def detect_frame(img: np.ndarray, conf: float) -> tuple[list[Det], float]:
    return _require_detector().predict(img, conf=conf)


def run_image_inference(db: Session, img: np.ndarray, image_bytes: bytes | None, filename: str | None,
                        conf: float, loc: Location, kind: str = "image", save: bool = True,
                        extra: dict | None = None) -> dict:
    det = _require_detector()
    nearby = (lambda code: count_nearby(db, loc.lat, loc.lon, code)) if loc.known else None
    dets, ms = det.predict(img, conf=conf, nearby_counter=nearby)
    annotated = draw_detections(img, dets)
    h, w = img.shape[:2]
    result = {"detections": [d.to_dict() for d in dets], "inference_ms": round(ms, 2), "width": w, "height": h,
              "model_version": model_service.version, "location": loc.__dict__ if loc.known else None,
              "saved": False, "inference_id": None}
    if not save:
        import base64
        import cv2

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["annotated_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
        return result

    stem = storage.new_stem(kind)
    inf = Inference(kind=kind, source_filename=filename, image_path=storage.save_image(img, "uploads", stem),
                    annotated_path=storage.save_image(annotated, "detections", stem), width=w, height=h,
                    latitude=loc.lat if loc.known else None, longitude=loc.lon if loc.known else None,
                    location_source=loc.source if loc.known else None, location_accuracy_m=loc.accuracy_m,
                    model_version=model_service.version, conf_threshold=conf, inference_ms=round(ms, 2),
                    num_detections=len(dets), extra=extra)
    db.add(inf)
    db.flush()
    for k, d in enumerate(dets):
        db.add(Detection(inference_id=inf.id, class_code=d.class_code, class_name=d.class_name,
                         confidence=d.confidence, x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3],
                         severity=d.severity, severity_score=d.severity_score, severity_detail=d.severity_detail,
                         latitude=inf.latitude, longitude=inf.longitude, location_source=inf.location_source,
                         crop_path=storage.save_crop(img, d.bbox, f"{stem}_{k}"), model_version=inf.model_version))
    db.commit()
    result.update(saved=True, inference_id=inf.id, annotated_url=storage.url(inf.annotated_path),
                  image_url=storage.url(inf.image_path))
    return result


# ----------------------------------------------------------------------------- video


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_track(frames, window_s: float = 1.0, iou_thr: float = 0.3):
    """Collapse consecutive near-identical detections (same class, overlapping box, within window_s)
    so one physical defect seen in many frames is stored once (keeping its most confident view)."""
    # track = [best_frame, best_det, last_seen_t, last_bbox]
    tracks: list[list] = []
    for fr in frames:
        for d in fr.detections:
            match = None
            for tr in reversed(tracks):
                if fr.timestamp_s - tr[2] > window_s:
                    continue
                if tr[1].class_code == d.class_code and _iou(tr[3], d.bbox) >= iou_thr:
                    match = tr
                    break
            if match is None:
                tracks.append([fr, d, fr.timestamp_s, d.bbox])
                continue
            match[2], match[3] = fr.timestamp_s, d.bbox  # extend the track
            if d.confidence > match[1].confidence:
                match[0], match[1] = fr, d
    return [(tr[0], tr[1]) for tr in tracks]


def start_video_job(db: Session, video_path: Path, filename: str, conf: float, stride: int,
                    fixed_loc: Location, route: Route | None) -> Inference:
    det = _require_detector()
    inf = Inference(kind="video", status="queued", progress=0.0, source_filename=filename,
                    video_path=storage.rel(video_path), model_version=model_service.version, conf_threshold=conf,
                    latitude=fixed_loc.lat if fixed_loc.known else None,
                    longitude=fixed_loc.lon if fixed_loc.known else None,
                    location_source=("route" if route else fixed_loc.source) if (route or fixed_loc.known) else None,
                    extra={"stride": stride, "route_points": len(route.points) if route else 0})
    db.add(inf)
    db.commit()
    threading.Thread(target=_video_worker, args=(inf.id, det, video_path, conf, stride, fixed_loc, route),
                     daemon=True, name=f"video-{inf.id}").start()
    return inf


def _video_worker(inf_id: int, det, video_path: Path, conf: float, stride: int, fixed_loc: Location,
                  route: Route | None) -> None:
    db = session_factory()()
    try:
        inf = db.get(Inference, inf_id)
        inf.status = "processing"
        db.commit()
        stem = Path(inf.video_path).stem
        out_path = storage.subdir("videos") / f"{stem}_detected.mp4"
        last = {"p": 0.0}

        def progress(p: float) -> None:
            if p - last["p"] >= 0.02:
                last["p"] = p
                inf.progress = round(p, 3)
                db.commit()

        frames, summary = process_video(det, video_path, out_path, conf=conf, stride=stride, progress=progress)
        timeline = [{"frame": f.frame_index, "t": f.timestamp_s, "n": len(f.detections),
                     "classes": [d.class_code for d in f.detections]} for f in frames if f.detections]
        (storage.subdir("videos") / f"{stem}_timeline.json").write_text(json.dumps(timeline))
        kept = dedupe_track(frames)
        import cv2

        cap = cv2.VideoCapture(str(video_path))  # crop from the clean source frame
        ow, oh = summary["output_size"]
        for k, (fr, d) in enumerate(kept):
            lat = lon = None
            src = None
            if route:
                pos = route.position_at(fr.timestamp_s)
                if pos:
                    lat, lon, src = pos[0], pos[1], "route"
            elif fixed_loc.known:
                lat, lon, src = fixed_loc.lat, fixed_loc.lon, fixed_loc.source
            crop = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_index)
            ok, frame = cap.read()
            if ok:
                # bboxes are in the (possibly downscaled) output-frame space
                frame = cv2.resize(frame, (ow, oh)) if frame.shape[1] != ow or frame.shape[0] != oh else frame
                crop = storage.save_crop(frame, d.bbox, f"{stem}_{k}")
            db.add(Detection(inference_id=inf.id, class_code=d.class_code, class_name=d.class_name,
                             confidence=d.confidence, x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3],
                             severity=d.severity, severity_score=d.severity_score, severity_detail=d.severity_detail,
                             latitude=lat, longitude=lon, location_source=src, frame_index=fr.frame_index,
                             video_time_s=fr.timestamp_s, crop_path=crop, model_version=inf.model_version))
        cap.release()
        inf.output_video_path = storage.rel(out_path)
        inf.width, inf.height = summary["output_size"]
        inf.num_detections = len(kept)
        inf.inference_ms = summary["mean_inference_ms"]
        inf.extra = {**(inf.extra or {}), **summary, "raw_frame_detections": sum(len(f.detections) for f in frames),
                     "timeline_url": storage.url(f"videos/{stem}_timeline.json")}
        inf.status, inf.progress = "completed", 1.0
        db.commit()
        log.info("Video job %s finished: %s unique detections", inf_id, len(kept))
    except Exception as exc:
        log.exception("Video job %s failed", inf_id)
        db.rollback()
        inf = db.get(Inference, inf_id)
        if inf:
            inf.status, inf.error = "failed", str(exc)
            db.commit()
    finally:
        db.close()

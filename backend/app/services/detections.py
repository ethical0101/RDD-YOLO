"""Persistence and querying of detections."""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rdd_yolo.constants import CLASS_CODES, CLASS_NAMES
from rdd_yolo.geo import haversine_m
from rdd_yolo.severity import REPEAT_RADIUS_M

from ..db.models import Detection, Inference
from . import storage


def count_nearby(db: Session, lat: float | None, lon: float | None, class_code: str,
                 radius_m: float = REPEAT_RADIUS_M) -> int:
    """Number of stored detections of the same class within radius_m of (lat, lon)."""
    if lat is None or lon is None:
        return 0
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    rows = db.execute(
        select(Detection.latitude, Detection.longitude).where(
            Detection.class_code == class_code,
            Detection.latitude.between(lat - dlat, lat + dlat),
            Detection.longitude.between(lon - dlon, lon + dlon),
        )
    ).all()
    return sum(1 for la, lo in rows if haversine_m(lat, lon, la, lo) <= radius_m)


def iso(dt: datetime | None) -> str | None:
    """ISO-8601 with timezone. SQLite drops tzinfo, and values are always stored in UTC."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def detection_to_dict(d: Detection) -> dict:
    inf = d.inference
    return {
        "id": d.id, "inference_id": d.inference_id, "kind": inf.kind if inf else None,
        "class_code": d.class_code, "class_name": d.class_name, "confidence": d.confidence,
        "bbox": [d.x1, d.y1, d.x2, d.y2], "severity": d.severity, "severity_score": d.severity_score,
        "severity_detail": d.severity_detail, "latitude": d.latitude, "longitude": d.longitude,
        "location_source": d.location_source, "frame_index": d.frame_index, "video_time_s": d.video_time_s,
        "crop_url": storage.url(d.crop_path),
        "image_url": storage.url(inf.annotated_path) if inf else None,
        "model_version": d.model_version, "created_at": iso(d.created_at),
    }


def inference_to_dict(i: Inference, with_detections: bool = True) -> dict:
    out = {
        "id": i.id, "kind": i.kind, "status": i.status, "progress": i.progress, "error": i.error,
        "source_filename": i.source_filename, "image_url": storage.url(i.image_path),
        "annotated_url": storage.url(i.annotated_path), "video_url": storage.url(i.video_path),
        "output_video_url": storage.url(i.output_video_path), "width": i.width, "height": i.height,
        "latitude": i.latitude, "longitude": i.longitude, "location_source": i.location_source,
        "location_accuracy_m": i.location_accuracy_m, "model_version": i.model_version,
        "conf_threshold": i.conf_threshold, "inference_ms": i.inference_ms, "num_detections": i.num_detections,
        "extra": i.extra, "created_at": iso(i.created_at),
    }
    if with_detections:
        out["detections"] = [detection_to_dict(d) for d in i.detections]
    return out


def filtered_query(classes: list[str] | None = None, min_conf: float | None = None,
                   severities: list[str] | None = None, sources: list[str] | None = None,
                   has_location: bool | None = None, since: datetime | None = None, kind: str | None = None):
    q = select(Detection).join(Inference)
    if classes:
        q = q.where(Detection.class_code.in_(classes))
    if min_conf is not None:
        q = q.where(Detection.confidence >= min_conf)
    if severities:
        q = q.where(Detection.severity.in_([s.upper() for s in severities]))
    if sources:
        q = q.where(Detection.location_source.in_(sources))
    if has_location is True:
        q = q.where(Detection.latitude.is_not(None), Detection.longitude.is_not(None))
    elif has_location is False:
        q = q.where(Detection.latitude.is_(None))
    if since is not None:
        q = q.where(Detection.created_at >= since)
    if kind:
        q = q.where(Inference.kind == kind)
    return q


def statistics(db: Session, days: int = 30) -> dict:
    rows = db.execute(select(Detection.class_code, Detection.confidence, Detection.severity,
                             Detection.created_at, Detection.latitude)).all()
    total = len(rows)
    per_class = Counter(r[0] for r in rows)
    sev = Counter(r[2] for r in rows)
    confs = [r[1] for r in rows]
    bins = [0] * 10
    for c in confs:
        bins[min(9, int(c * 10))] += 1
    since = datetime.now(timezone.utc) - timedelta(days=days - 1)
    by_day: dict[str, Counter] = {}
    for i in range(days):
        by_day[(since + timedelta(days=i)).date().isoformat()] = Counter()
    for code, _, _, created, _ in rows:
        if created is None:
            continue
        created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
        key = created.date().isoformat()
        if key in by_day:
            by_day[key][code] += 1
    class_conf: dict[str, list[float]] = {c: [] for c in CLASS_CODES}
    class_sev: dict[str, Counter] = {c: Counter() for c in CLASS_CODES}
    for code, conf, s, _, _ in rows:
        if code in class_conf:
            class_conf[code].append(conf)
            class_sev[code][s] += 1
    n_inf = db.scalar(select(func.count(Inference.id))) or 0
    kinds = Counter(k for (k,) in db.execute(select(Inference.kind)).all())
    return {
        "total_detections": total,
        "total_inferences": n_inf,
        "inferences_by_kind": dict(kinds),
        "geolocated_detections": sum(1 for r in rows if r[4] is not None),
        "per_class": {c: {"name": CLASS_NAMES[c], "count": per_class.get(c, 0),
                          "avg_confidence": (sum(class_conf[c]) / len(class_conf[c])) if class_conf[c] else None,
                          "severity": {k: class_sev[c].get(k, 0) for k in ("LOW", "MEDIUM", "HIGH")}}
                      for c in CLASS_CODES},
        "average_confidence": (sum(confs) / total) if total else None,
        "severity": {k: sev.get(k, 0) for k in ("LOW", "MEDIUM", "HIGH")},
        "confidence_histogram": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": b} for i, b in enumerate(bins)],
        "over_time": [{"date": d, **{c: cnt.get(c, 0) for c in CLASS_CODES}, "total": sum(cnt.values())}
                      for d, cnt in by_day.items()],
    }

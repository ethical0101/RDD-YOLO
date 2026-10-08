# REST API

Base URL: `http://127.0.0.1:8000`. Interactive OpenAPI docs: **`/docs`** (Swagger UI) and `/redoc`.
All endpoints are under `/api`. Errors return `{"detail": "..."}` with a meaningful status code
(400 bad input, 404 not found, 409 conflict, 413 too large, 415 unsupported type, 422 validation, 503 model unavailable).

Static files: `/files/outputs/...` (uploads, annotated images, crops, videos) and `/files/experiments/...` (training plots).

## System

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Status, model loaded/version/error, DB status, CUDA availability |
| GET | `/api/system/hardware` | Detected hardware + training recommendation |
| GET | `/api/classes` | The 4 classes (id, code, name) |
| GET | `/api/severity/rules` | Machine-readable severity formula, weights, thresholds, disclaimer |

## Model

| Method | Path | Description |
|---|---|---|
| GET | `/api/model` | Active model info (weights, params, device, training metadata) + available checkpoints |
| POST | `/api/model/select` | `{"weights": "rdd_yolo26n_best.pt"}` – hot-swap the checkpoint (file name inside `models/weights`) |

## Inference

### `POST /api/inference/image` (multipart)

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | image | required | JPEG/PNG/WebP/BMP, ≤ `RDD_MAX_UPLOAD_MB` |
| `conf` | float | 0.25 | 0.01–0.99 |
| `latitude`, `longitude` | float | – | both or neither |
| `location_source` | `browser`\|`manual` | `manual` | label for client-provided coordinates |
| `location_accuracy_m` | float | – | browser GPS accuracy |
| `prefer_exif` | bool | true | EXIF GPS in the file overrides client coordinates |
| `save` | bool | true | false = preview only (returns `annotated_base64`, nothing stored) |

Response (abridged):

```json
{
  "detections": [{"class_code": "D40", "class_name": "Pothole", "confidence": 0.81,
                  "bbox": [312.0, 401.5, 455.2, 498.0], "bbox_norm": [...],
                  "severity": "HIGH", "severity_score": 71.3,
                  "severity_detail": {"class_weight": 1.0, "area_ratio": 0.0377, "area_score": 0.388, "confidence": 0.81, "repeat_count": 0, "repeat_bonus": 0}}],
  "inference_ms": 21.4, "width": 640, "height": 640, "model_version": "rdd_yolo26n_best",
  "location": {"lat": 12.9692, "lon": 79.1559, "source": "exif", "accuracy_m": null},
  "exif_gps": {"lat": 12.9692, "lon": 79.1559},
  "saved": true, "inference_id": 17, "annotated_url": "/files/outputs/detections/image_....jpg"
}
```
(The values above illustrate the schema only.)

### `POST /api/inference/frame` (multipart)
Webcam frame: `file`, `conf`. Not stored unless `save=true` (snapshot; optional `latitude`/`longitude`/`location_accuracy_m` from browser GPS).

### `POST /api/inference/video` (multipart) → `202 {"job_id": N}`
`file` (mp4/mov/avi/mkv/webm/m4v), `conf`, `stride` (1–30), optional `route_file` (CSV `time_s,lat,lon` or
`timestamp,lat,lon`, or GPX with `<time>`), or a single `latitude`/`longitude` for the whole clip.

### `GET /api/inference/video/{job_id}`
`status` (`queued|processing|completed|failed`), `progress` (0–1), output video URL, summary (`extra`:
frames, processed frames, mean inference ms, inference FPS, timeline URL), and the merged detections once completed.

`GET /api/videos?limit=20` lists video jobs.

## Records

| Method | Path | Description |
|---|---|---|
| GET | `/api/inferences/{id}` | One inference with all detections |
| PATCH | `/api/inferences/{id}/location` | `{"latitude", "longitude", "source": "browser|manual|exif|route", "accuracy_m"}` – attach/replace location (also updates its detections) |
| DELETE | `/api/inferences/{id}` | Delete with cascade |
| GET | `/api/detections` | Filters: `classes=D00,D40`, `min_conf`, `severity=HIGH,MEDIUM`, `source=browser,exif`, `has_location`, `kind=image|video|webcam`, `days`; paging `limit` (≤1000) / `offset`; `sort=created_at|confidence|severity_score`, `order=asc|desc` |
| GET | `/api/detections/{id}` | One detection |
| DELETE | `/api/detections/{id}` | Delete one detection |

## Statistics & map

| Method | Path | Description |
|---|---|---|
| GET | `/api/stats?days=30` | Totals, per-class counts / avg confidence / severity, severity totals, confidence histogram, per-day counts (UTC days) |
| GET | `/api/map/detections` | **GeoJSON** `FeatureCollection` (RFC 7946, `[lon, lat]`) of geolocated detections; same filters as `/api/detections`; `limit` ≤ 20000 |

## Training & dataset (read-only, from real artefacts)

| Method | Path | Description |
|---|---|---|
| GET | `/api/training/experiments` | Runs in `experiments/` with status, epochs, best val metrics, test metrics |
| GET | `/api/training/experiments/{name}` | `run_info.json`, per-epoch history (`results.csv`), plot URLs, evaluation metrics/plots |
| GET | `/api/training/comparison` | Baseline vs RDD-YOLO comparison (`available: false` until both are evaluated) |
| GET | `/api/dataset/stats` | Split/class statistics and validation summary of the prepared dataset |

## Examples (PowerShell)

```powershell
curl.exe -F "file=@road.jpg" -F "conf=0.3" -F "latitude=12.9692" -F "longitude=79.1559" -F "location_source=manual" http://127.0.0.1:8000/api/inference/image
curl.exe "http://127.0.0.1:8000/api/map/detections?classes=D40&severity=HIGH"
curl.exe -F "file=@dashcam.mp4" -F "route_file=@route.csv" -F "stride=3" http://127.0.0.1:8000/api/inference/video
```

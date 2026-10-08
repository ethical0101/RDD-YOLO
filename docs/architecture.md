# System architecture

## Overview

```mermaid
flowchart LR
    subgraph Data["Dataset pipeline (offline)"]
        A[RDD2022 on figshare<br/>13.3 GB archive] -->|HTTP range requests<br/>download_rdd2022.py| B[raw/RDD2022/&lt;country&gt;<br/>Pascal VOC]
        B -->|prepare_dataset.py<br/>validate · filter · split| C[processed/rdd2022_yolo<br/>YOLO format + stats]
    end
    subgraph Train["Training pipeline (offline, GPU)"]
        C --> D[train.py<br/>Exp A baseline / Exp B RDD-YOLO]
        D --> E[experiments/&lt;run&gt;<br/>results.csv · plots · weights]
        E --> F[evaluate.py<br/>test split · FPS]
        F --> G[compare.py]
        E -->|best.pt| W[models/weights]
    end
    subgraph Serve["Online system"]
        W --> M[rdd_yolo.detector<br/>Detector]
        M --> API[FastAPI backend]
        API <--> DB[(SQLite<br/>PostgreSQL-ready)]
        API --> FS[outputs/<br/>uploads · crops · videos]
        UI[React dashboard<br/>Leaflet + OSM · Recharts] <-->|REST /api| API
        BR[Browser GPS · webcam] --> UI
    end
```

## Components

| Layer | Location | Responsibility |
|---|---|---|
| Core library | `rdd_yolo/` | Class schema, hardware detection, SimAM module, inference engine, severity heuristic, geolocation (EXIF, routes). Shared by training, CLI and backend. |
| Dataset | `dataset/` | Selective download of the official RDD2022 archive; validation and VOC→YOLO conversion; statistics. |
| Architectures | `models/architectures/` | `yolo26-baseline.yaml` (unmodified YOLO26) and `yolo26-rdd.yaml` (SimAM + bilinear + GhostConv). |
| Training | `training/` | `train.py`, `evaluate.py`, `compare.py`; orchestration in `scripts/run_experiments.py`. |
| CLI inference | `inference/predict.py` | Images / folders / videos without the database. |
| Backend | `backend/app/` | FastAPI app: `api/routes.py` (REST), `services/` (model, inference, storage, queries, training artefacts), `db/` (SQLAlchemy models). |
| Frontend | `frontend/` | React 19 + Vite + TypeScript + Tailwind 4, React-Leaflet (OpenStreetMap), Leaflet.markercluster, Leaflet.heat, Recharts. |
| Tests | `tests/` | pytest: core logic, dataset prep, architectures, inference, DB, API end-to-end. |

## Request flow: image detection

1. The dashboard posts `multipart/form-data` to `POST /api/inference/image` (image, confidence, optional lat/lon + source).
2. `resolve_location` chooses the location: **EXIF GPS** embedded in the photo wins (if present), else the
   browser/manual coordinates sent by the client, else **no location** (nothing is guessed).
3. `Detector.predict` runs the checkpoint (CUDA if available, CPU otherwise), producing boxes, classes and confidences.
4. For each box the severity heuristic (`rdd_yolo/severity.py`) is computed; the repeat-report term queries the
   database for same-class detections within 15 m.
5. The original image, annotated image and per-detection crops are written to `outputs/`; one `inferences` row
   and N `detections` rows are committed.
6. The JSON response drives the UI (SVG box overlay, table, map link).

## Video flow

`POST /api/inference/video` stores the upload, creates an `inferences` row (`status=queued`) and starts a
background thread. The worker decodes frames with OpenCV, runs the detector on every *stride*-th frame,
encodes an annotated H.264 MP4 through the ffmpeg binary bundled with `imageio-ffmpeg` (playable in browsers),
merges repeated sightings of the same defect (same class, IoU ≥ 0.3, gap ≤ 1 s) into one record, and — if a
GPS route (CSV/GPX) was uploaded — interpolates a position for each detection's timestamp. The UI polls
`GET /api/inference/video/{id}` for progress.

## Data model

```mermaid
erDiagram
    inferences ||--o{ detections : contains
    inferences {
        int id PK
        string kind "image | video | webcam"
        string status
        string image_path
        string annotated_path
        string video_path
        string output_video_path
        float latitude
        float longitude
        string location_source "browser | exif | manual | route"
        string model_version
        float conf_threshold
        float inference_ms
        json extra
        datetime created_at
    }
    detections {
        int id PK
        int inference_id FK
        string class_code "D00 D10 D20 D40"
        float confidence
        float x1
        float y1
        float x2
        float y2
        string severity
        float severity_score
        json severity_detail
        float latitude
        float longitude
        string location_source
        int frame_index
        float video_time_s
        string crop_path
        string model_version
        datetime created_at
    }
```

Only portable SQLAlchemy types (`Integer`, `Float`, `String`, `Text`, `DateTime(timezone=True)`, `JSON`) are used,
so switching to PostgreSQL is a matter of setting `RDD_DATABASE_URL=postgresql+psycopg://…` and installing
`psycopg`. File paths are stored relative to `outputs/`. Timestamps are stored in UTC.

## Why these choices

* **YOLO26n (Ultralytics 8.4.174)** — the current Ultralytics detector family at development time (Oct 2026).
  The nano scale trains at batch 16 within ~2.5 GB VRAM on the RTX 3050 4 GB at ~210 s/epoch on 16k images.
* **SQLite** — zero-config, file-based, free; adequate for a single-node demo.
* **OpenStreetMap + Leaflet** — no API key, no billing.
* **Single process deployment** — the backend also serves the built dashboard (`frontend/dist`), so the demo
  needs one command and one port.

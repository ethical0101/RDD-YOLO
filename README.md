# RDD-YOLO: AI-Based Road Damage Detection, Geolocation and Mapping System

A complete, working deep-learning system that **trains** road-damage detectors on the public RDD2022 dataset,
**evaluates** them on a held-out test split, and serves them through a **FastAPI** backend and a **React**
dashboard with image, video and live-camera detection, **GPS/EXIF/route geolocation**, an
**OpenStreetMap** damage map, analytics and a detection database. Zero paid APIs, zero API keys.

Detected classes (RDD2022 codes): **D00** longitudinal crack · **D10** transverse crack · **D20** alligator crack · **D40** pothole.

> **What model is this?** The academic report describes RDD-YOLO on **YOLOv8**. This implementation uses
> **Ultralytics 8.4.174 with YOLO26** (the current Ultralytics detector at development time, Oct 2026), nano
> scale for a 4 GB GPU. The three RDD-YOLO modifications — **SimAM attention, GhostConv neck, bilinear
> upsampling** — are implemented on YOLO26 and compared against an unmodified YOLO26 baseline trained identically.
> It is therefore *not* called "YOLOv8" anywhere in the code.

---

## Contents
[Features](#features) · [Architecture](#architecture) · [Tech stack](#technology-stack) · [Installation](#installation) ·
[Dataset](#dataset-setup) · [Training](#training) · [Evaluation](#evaluation) · [Results](#results) · [Inference](#inference) ·
[Backend](#backend) · [Frontend](#frontend) · [Map & GPS](#map-and-geolocation) · [Severity](#severity-estimate) ·
[Database](#database) · [Tests](#tests) · [Demo workflow](#professor-demonstration-workflow) · [Troubleshooting](#troubleshooting) ·
[Limitations](#limitations-and-honesty-notes)

## Features

| Area | What is implemented |
|---|---|
| Dataset | Selective download of the official RDD2022 archive (HTTP range requests), full image/annotation validation, VOC→YOLO conversion, stratified split, statistics + class-distribution chart |
| Training | Reproducible training of **Exp A: baseline YOLO26n** and **Exp B: RDD-YOLO26n** with automatic hardware-aware settings, identical initialisation, checkpointing, resume |
| Evaluation | Test-split P / R / F1 / mAP@50 / mAP@50-95 (overall + per class), confusion matrix, PR/F1 curves, loss/mAP curves, parameters, GFLOPs, model size, measured FPS; automatic A-vs-B comparison |
| Inference | Image (bounding boxes, class, confidence, severity), video (frame stride, annotated H.264 output, timestamps, de-duplicated detections), live webcam, CLI |
| Geolocation | Browser GPS (with permission), manual point (typed or clicked on the map), **EXIF GPS** auto-extraction, **video GPS route** (CSV/GPX) interpolation — every record is labelled with its source; nothing is ever invented |
| Map | OpenStreetMap + Leaflet, marker clustering, heatmap, filters (class, severity, confidence, source, time), marker popups with crop image, link to full annotated image |
| Severity | Transparent, documented heuristic (class, relative box area, confidence, repeat reports within 15 m) → LOW / MEDIUM / HIGH |
| Storage | SQLite via SQLAlchemy 2 (PostgreSQL-ready), images/crops/videos on disk |
| Dashboard | Dashboard, Image Detection, Video, Live Camera, Map, Analytics, History (filter/sort/paginate/CSV export), Model, Training & Experiments |
| Ops | PowerShell automation for setup → data → training → evaluation → startup; tests; Docker files |

## Architecture

```
RDD-YOLO/
├── rdd_yolo/            core library: constants, hardware, SimAM, detector, severity, geo
├── dataset/             download_rdd2022.py, prepare_dataset.py  (raw/ and processed/ are generated)
├── models/
│   ├── architectures/   yolo26-baseline.yaml, yolo26-rdd.yaml
│   └── weights/         trained checkpoints served by the API (+ COCO yolo26n.pt for init)
├── training/            train.py, evaluate.py, compare.py
├── inference/           predict.py (CLI)
├── backend/app/         FastAPI: api/, services/, db/, core/
├── frontend/            React + Vite + TypeScript + Tailwind + React-Leaflet + Recharts
├── experiments/         training runs, evaluation outputs, comparison (generated)
├── outputs/             uploads, annotated images, crops, videos, SQLite DB (generated)
├── scripts/             PowerShell automation + run_experiments.py
├── tests/               pytest suite
├── docs/                architecture.md, training.md, api.md, experiments.md
├── docker/ + docker-compose.yml
└── .env.example
```

Details and diagrams: [docs/architecture.md](docs/architecture.md).

## Technology stack

| | Version (verified Oct 2026) |
|---|---|
| Python | 3.13.4 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 (CUDA 13.0 wheels) |
| Ultralytics | 8.4.174 (YOLO26) |
| OpenCV | 5.0.0 |
| FastAPI / Uvicorn / SQLAlchemy | 0.142.2 / 0.54.0 / 2.1.3 |
| Node / npm | 20.20 / 10.8 |
| React / Vite / TypeScript / Tailwind | 19 / 8.3 / 6.0 / 4.3 |
| React-Leaflet / Leaflet / Recharts | 5.0 / 1.9.4 / 3.10 |
| Maps | OpenStreetMap tiles (free, attribution shown) |

Development machine: Windows 11, NVIDIA RTX 3050 Laptop 4 GB, 16 CPU threads, 16 GB RAM.

## Installation

Prerequisites: Windows 10/11, Python 3.11+ (3.13 tested), Node.js 20.19+, Git; NVIDIA driver for GPU training (CPU works but is slow).

```powershell
git clone <repo> RDD-YOLO; cd RDD-YOLO
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # venv, CUDA PyTorch, deps, npm install, .env
```

Use `-Cpu` to force the CPU build of PyTorch. If PowerShell blocks scripts, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Dataset setup

```powershell
scripts\prepare_data.ps1           # downloads ~2.4 GB (5 countries) and builds dataset\processed\rdd2022_yolo
```

RDD2022 (CRDDC'2022) by Arya et al., CC BY 4.0, figshare article 21431547. Countries used: Japan, India,
Czech Republic, United States, China (motorbike). See [docs/training.md](docs/training.md#2-dataset) for the
validation rules and the actual split statistics (16,156 / 2,017 / 2,023 images).

## Training

```powershell
scripts\train.ps1                  # Exp A baseline + Exp B RDD-YOLO, 40 epochs each, then evaluation + comparison
scripts\train.ps1 -Experiment rdd -Epochs 60
scripts\train.ps1 -Smoke           # quick sanity check
```

Hardware is detected automatically (CUDA, GPU name, VRAM, CPU threads, RAM) and batch / workers / AMP / model
scale are chosen accordingly (RTX 3050 4 GB → YOLO26n, batch 16 at ≈2.5 GB, AMP). Training is resumable.
Full description: [docs/training.md](docs/training.md).

## Evaluation

```powershell
scripts\evaluate.ps1               # test split metrics, plots, FPS, comparison table
```

## Results

Measured on the held-out test split (2,023 images), 40 epochs each, RTX 3050 Laptop 4 GB
(source: `experiments/comparison.json`, generated 8 Oct 2026):

| Model | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Params | GFLOPs | FPS (batch 1) |
|---|---|---|---|---|---|---|---|---|
| A · Baseline YOLO26n | 63.09 % | 55.05 % | 58.80 % | 59.66 % | 30.24 % | 2.51 M | 5.90 | 87.2 |
| B · RDD-YOLO26n | 63.08 % | 55.28 % | 58.92 % | 59.46 % | 30.30 % | 2.42 M | 5.81 | 80.1 |

**Reading:** accuracy is statistically indistinguishable (all differences < 0.3 pp, single run each);
RDD-YOLO is 3.6 % smaller but 8 % slower. The paper's gains (YOLOv8x, 180 epochs) were not reproduced at
nano scale / 40 epochs — reported as measured.

**Experiment C — extended data (the model served by the app).** RDD-YOLO26n fine-tuned for 25 more epochs
(1.9 h on the RTX 3050) on 21,477 images: the original set + RDD2022 Norway & China_Drone + two public
ground-level pothole datasets (pothole class only). Measured before → after:

| Test set | mAP@50 | Recall |
|---|---|---|
| Original RDD2022 test (unchanged, 2,023 imgs) | 59.46 % → **60.50 %** | 55.28 % → **57.19 %** |
| Norway + China_Drone held-out (532 imgs) | 12.92 % → **33.37 %** | 20.67 % → **34.68 %** |
| Ground-level potholes, external (221 imgs) | 18.34 % → **44.61 %** | 19.99 % → **45.81 %** |

Reproduce with `scripts	rain_extended.ps1`.

See **[docs/experiments.md](docs/experiments.md)** for per-class results and details — it contains the measured numbers of both experiments
(written from `experiments/comparison.json` after the runs finished) and how to interpret them. The same
results, curves, confusion matrices and PR curves are shown live on the dashboard's *Model* and
*Training & Experiments* pages, read directly from the files in `experiments/`.

## Inference

* **Dashboard:** Image Detection / Video Detection / Live Camera pages.
* **CLI:** `scripts\infer.ps1 -Source road.jpg` (also folders and videos) → `outputs\cli\<timestamp>\`.
* **API:** `POST /api/inference/image`, `/api/inference/video`, `/api/inference/frame` — see [docs/api.md](docs/api.md).

## Backend

```powershell
scripts\start_backend.ps1          # http://127.0.0.1:8000  (Swagger UI: /docs)
```

Configuration via `.env` (prefix `RDD_`, see `.env.example`): database URL, checkpoint, device, image size,
default confidence, upload limit, video stride, CORS origins. No secrets are required.

## Frontend

```powershell
scripts\start_frontend.ps1         # http://localhost:5173 (dev server, proxies /api to :8000)
scripts\start_all.ps1              # one command: build dashboard + start backend, open http://localhost:8000
scripts\start_all.ps1 -Dev         # backend + hot-reload dev server in two windows
```

## Map and geolocation

A normal photo does **not** contain live GPS. The system therefore supports, and labels, four honest sources:

| Source | How |
|---|---|
| **Browser GPS** | "Use my location" asks for permission (`navigator.geolocation`), accuracy is stored |
| **Manual** | type coordinates or click the OpenStreetMap |
| **EXIF GPS** | extracted automatically from the uploaded photo (takes priority when present) |
| **Route** | upload a CSV (`time_s,lat,lon` or `timestamp,lat,lon`) or GPX track with a video; each detection gets the position interpolated at its timestamp, frames outside the track get none |

If no source is available the detection is stored **without** coordinates and is not shown on the map.
Location can be attached later (`PATCH /api/inferences/{id}/location`, "Attach selected location" button).
Browser geolocation and the webcam require `localhost` or HTTPS.

## Severity estimate

```
score = 100 × (0.45·class_weight + 0.40·min(1, √(box_area/image_area / 0.25)) + 0.15·confidence)
        + min(15, 5 × same-class detections stored within 15 m)
class_weight: D00 0.45 · D10 0.50 · D20 0.80 · D40 1.00
LOW < 40 ≤ MEDIUM < 65 ≤ HIGH
```

This is a **heuristic for prioritising inspection**, not an engineering-certified pavement condition rating
(e.g. not PCI/ASTM D6433). Every detection stores the components of its score (`severity_detail`).

## Database

SQLite file `outputs/rdd_yolo.db` (tables `inferences`, `detections`; schema in
[docs/architecture.md](docs/architecture.md#data-model)). Stored per detection: ID, class, confidence, bounding
box, severity (+ components), latitude, longitude, location source, timestamp (UTC), image/crop reference,
video reference + frame time, model version. Switch to PostgreSQL by setting
`RDD_DATABASE_URL=postgresql+psycopg://user:pass@host/db` and `pip install psycopg[binary]`.

## Tests

```powershell
scripts\run_tests.ps1              # pytest (core, dataset prep, architectures, inference, DB, API) + frontend build
```

Tests use an isolated temporary database and CPU inference, so they can run while the GPU is training.

## Professor demonstration workflow

1. `scripts\start_all.ps1` → browser opens the **Dashboard** (status pill shows the loaded checkpoint).
2. **Model** page: trained RDD-YOLO checkpoint, dataset, classes, image size, epochs, test metrics.
3. **Image Detection**: drop a road image (e.g. from `demo_images/`, held-out test photos) → *Detect road damage*.
4. Bounding boxes, class, confidence and severity appear (hover a row to highlight its box).
5. Location: *Use my location* (browser GPS) or click the map (manual); an EXIF-tagged photo is located automatically.
6. Detections are saved (checkbox on) → *View on map*.
7. **Damage Map**: the marker appears; click it → popup with type, confidence, severity, coordinates, timestamp, crop image and full-image link. Try filters, clustering and heatmap.
8. **Analytics** and **Detection History** show the new records.
9. **Training & Experiments**: loss/mAP curves, confusion matrix, PR curves, and the **baseline vs RDD-YOLO** table with the architectural explanation.
10. Optional: **Video Detection** with a dash-cam clip (+ GPS route CSV), **Live Camera**.

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Install the CUDA wheel: `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130` (needs a recent NVIDIA driver). |
| PyTorch download stalls | Re-run `setup.ps1`; pip resumes. On flaky networks, download the wheel with `curl -C -` and `pip install` the file. |
| CUDA out of memory | `scripts\train.ps1 -Batch 8` (or `training\train.py --batch 8`). |
| Training is slow / RAM pressure on Windows | Lower `-Workers` (4). |
| Dataset download 403 from S3 links | The script uses figshare instead; just re-run (it resumes). |
| "Inference unavailable" banner | No checkpoint in `models\weights` yet — finish training, or set `RDD_MODEL_WEIGHTS`. |
| Map tiles don't load | Needs internet access to `tile.openstreetmap.org`. |
| Webcam / browser GPS denied | Use `http://localhost` (not an IP) or HTTPS and allow the permission. |
| Output video doesn't play | Requires the ffmpeg binary from `imageio-ffmpeg` (installed by `requirements.txt`). |

## Limitations and honesty notes

* Model, data and hardware differ from the RDD-YOLO paper (YOLOv8x, all RDD2022 countries, 180 epochs, RTX 4090).
  Our numbers are **not** comparable to the paper's and are reported only for our own setup.
* The official RDD2022 test set has no public labels; our "test" split is a held-out 10 % of the labelled data.
* **Domain gap** (reduced but not eliminated by Experiment C). RDD2022 images come from vehicle-mounted cameras looking down the road; the median pothole
  covers only 0.6 % of the image (only 1.6 % of training potholes cover ≥ 15 %). Ground-level, wide-angle or
  stock photos where one pothole fills the frame are largely outside the training distribution and are often
  missed. Use road photos taken from a vehicle (or the held-out samples in `demo_images/`) for demonstrations.
* Severity is a heuristic, not a certified assessment.
* Docker files are provided but were not exercised in the development environment (Docker daemon not running);
  the native PowerShell workflow is the tested path.
* Webcam throughput depends on network round-trips to the local API (one frame in flight).

## Citation

* Y. Li, C. Yin, Y. Lei, J. Zhang, Y. Yan, "RDD-YOLO: Road Damage Detection Algorithm Based on Improved You Only Look Once Version 8," *Applied Sciences* 14(8):3360, 2024.
* D. Arya et al., "RDD2022: A multi-national image dataset for automatic road damage detection," *Geoscience Data Journal*, 2024.
* L. Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for CNNs," ICML 2021.
* K. Han et al., "GhostNet: More Features from Cheap Operations," CVPR 2020.
* Ultralytics YOLO, https://github.com/ultralytics/ultralytics (AGPL-3.0).

# Training pipeline

Everything below runs on Windows in PowerShell from the project root. Each step writes its artefacts to disk,
and all reported numbers come from those artefacts.

## 1. Environment

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Verified stack (October 2026): Python 3.13.4, PyTorch 2.14.1 + CUDA 13.0 wheels (driver 610.74 / CUDA 13.3),
torchvision 0.29.1, **Ultralytics 8.4.174**, OpenCV 5.0. `python -m rdd_yolo.hardware` prints detected hardware and
the automatic recommendation (device, batch, workers, AMP, model scale).

## 2. Dataset

**Source:** RDD2022, released through CRDDC'2022 (Arya et al., *Geoscience Data Journal*, 2024), CC BY 4.0,
figshare article 21431547. The legacy S3 links in the RDD GitHub README return HTTP 403 now, so
`dataset/download_rdd2022.py` reads the figshare archive (13.3 GB, one stored ZIP per country) with HTTP
Range requests and copies only the requested countries' ZIPs (resumable; CRC verified on extraction).

Default subset: **Japan, India, Czech, United_States, China_MotorBike** (≈2.4 GB). Norway (9.9 GB of
very high-resolution images) and China_Drone (aerial view) are excluded to fit the hardware/time budget.

```powershell
scripts\prepare_data.ps1
```

`dataset/prepare_dataset.py`:

* decodes every image fully (catches corrupt/truncated JPEGs), parses every VOC XML;
* reports images without XML, XML without images, size mismatches, degenerate/out-of-bounds boxes (clipped);
* keeps D00, D10, D20, D40; counts and drops other labels (D43, D44, D50, Repair, D01, D11, D0w0);
* keeps background (no target damage) images up to 10 % of the positive images per country;
* stratified per-country split 80/10/10 with seed 42 (the official RDD2022 test split has **no public labels**);
* writes YOLO labels (hard-linked images, no extra disk), `data.yaml`, `validation_report.json`,
  `dataset_stats.json` and `class_distribution.png`.

Result of the actual run:

| | images | boxes | D00 | D10 | D20 | D40 |
|---|---|---|---|---|---|---|
| train | 16,156 | 32,481 | 12,838 | 7,054 | 7,887 | 4,702 |
| val | 2,017 | 4,096 | 1,632 | 877 | 957 | 630 |
| test | 2,023 | 4,132 | 1,550 | 906 | 1,011 | 665 |

Validation: 27,823 / 27,823 images decodable; 18,934 contain at least one target box; 1 degenerate box dropped;
non-target labels dropped: D44 5,057 · D50 3,581 · D43 793 · Repair 277 · D01 179 · D11 45 · D0w0 1.

## 3. Training

```powershell
scripts\train.ps1                       # Exp A + Exp B + evaluation + comparison (resumable)
scripts\train.ps1 -Experiment rdd -Epochs 60
scripts\train.ps1 -Smoke                # 1 epoch on 2 % of the data
```

`training/train.py` steps:

1. Detect hardware → choose batch/workers/AMP (overridable).
2. Build the model from the YAML (`yolo26-baseline.yaml` or `yolo26-rdd.yaml`), scale `n`.
3. **Initialisation (identical for both experiments):** copy the COCO-pretrained weights of backbone layers
   0–10 from the official `yolo26n.pt` (240 tensors; identical indices and shapes in both YAMLs). Everything
   after the backbone — SimAM, neck, detection head — is randomly initialised with a fixed seed and learned on
   RDD2022 only. `--no-pretrained` trains every layer from scratch.
4. Save that as `experiments/_init/<run>_init.pt` and train with Ultralytics.
5. Write `run_info.partial.json` at start and `run_info.json` at the end (hardware, versions, args,
   initialisation report, wall time, best/last epoch metrics read from `results.csv`); copy `best.pt` to
   `models/weights/<run>_best.pt` with a metadata JSON.

Settings used for both experiments:

| Setting | Value |
|---|---|
| Model scale | n |
| Image size | 640 |
| Epochs | 40 (early stopping patience 15) |
| Batch | 16 (≈2.5 GB VRAM with AMP) |
| Workers | 6 |
| Optimizer | Ultralytics `auto` → resolved to **MuSGD** (lr 0.01, momentum 0.9, weight decay 5e-4), linear LR decay to lrf=0.01, 3 warm-up epochs |
| Mosaic | on, disabled for the last 10 epochs |
| AMP | on |
| Seed / deterministic | 0 / true |

Measured throughput on the RTX 3050 Laptop 4 GB: ~5 it/s at batch 16, ≈210 s per epoch including validation.

Resume after an interruption: re-run `scripts\train.ps1` — finished steps are skipped and an unfinished run
resumes from `weights/last.pt`.

## 4. Evaluation

```powershell
scripts\evaluate.ps1
```

`training/evaluate.py` runs `model.val(split="test")` on the 2,023 held-out images and saves
`experiments/<run>/eval_test/metrics.json` plus Ultralytics plots (confusion matrix, PR/P/R/F1 curves,
prediction mosaics). It reports overall and per-class precision, recall, F1 (= 2PR/(P+R) at the confidence that
Ultralytics selects for max mean F1), mAP@50, mAP@50-95, parameters, GFLOPs, checkpoint size, and a speed benchmark
(200 test images, batch 1, 20 warm-up runs; end-to-end latency incl. pre/post-processing and inference-only latency).

`training/compare.py` produces `experiments/comparison.{json,md}` and comparison plots.

## 5. Outputs

```
experiments/
  baseline_yolo26n/        results.csv, results.png, args.yaml, run_info.json,
                           confusion_matrix*.png, Box*_curve.png, weights/{best,last}.pt,
                           eval_test/metrics.json + plots
  rdd_yolo26n/             (same)
  comparison.json / comparison.md / comparison_*.png
  logs/                    stdout of each pipeline step
models/weights/            <run>_best.pt + <run>_best.json (served by the backend)
```

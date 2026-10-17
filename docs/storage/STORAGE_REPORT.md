# Storage report

Generated 2026-10-17T20:14:33+00:00 by `scripts/storage_report.py` (live measurement).

Drive of `E:\RDD-YOLO`: **16.8 GB free** of 100.5 GB. Reserve that downloads must keep: 8 GB (`RDD_STORAGE_RESERVE_GB`). Dataset budget: 30 GB (`RDD_DATASET_BUDGET_GB`), currently 17.85 GB in `dataset/raw`.

| Path | Size (GB) | Class | Auto-removable | Why |
|---|---|---|---|---|
| `dataset/raw/RDD2022/Norway` | 8.21 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/processed/rdd2022_extended_v3` | 5.62 | derived dataset (regenerable from sources; mostly hard links) | no | rebuild with dataset/*.py; keep while experiments reference it |
| `dataset/raw/external/deeksha` | 5.46 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/processed/rdd2022_extended_v2` | 2.60 | derived dataset (regenerable from sources; mostly hard links) | no | rebuild with dataset/*.py; keep while experiments reference it |
| `dataset/processed/rdd2022_extended` | 2.32 | derived dataset (regenerable from sources; mostly hard links) | no | rebuild with dataset/*.py; keep while experiments reference it |
| `.cache` | 1.85 | regenerable cache | yes (quarantine) | downloaded wheels (re-downloadable) |
| `dataset/processed/rdd2022_yolo` | 1.43 | derived dataset (regenerable from sources; mostly hard links) | no | rebuild with dataset/*.py; keep while experiments reference it |
| `dataset/raw/RDD2022/Japan` | 0.80 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/raw/external/rupesh_pothole2` | 0.65 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/raw/external/mwpd` | 0.64 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/raw/RDD2022/India` | 0.49 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/raw/external/egypt_rdd` | 0.38 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/raw/RDD2022/United_States` | 0.34 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `deploy` | 0.27 | regenerable cache | yes (quarantine) | deployment bundles rebuilt by scripts/build_hf_space.py / export_static_site.py |
| `dataset/raw/RDD2022/Czech` | 0.19 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/raw/RDD2022/China_Drone` | 0.16 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/raw/RDD2022/China_MotorBike` | 0.14 | source dataset (never auto-removed) | no | original RDD2022 country data used by experiments |
| `dataset/raw/external/manot_pothole2` | 0.13 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/raw/external/rupesh_pothole` | 0.11 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/raw/external/manot_road_damage` | 0.10 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `models/weights` | 0.08 | final artefact (never auto-removed) | no | published checkpoints |
| `experiments/_init` | 0.07 | regenerable cache | yes (quarantine) | initial checkpoints rebuilt by train.py |
| `experiments/bench_res/weights` | 0.04 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `experiments/yolo26s_full/weights` | 0.04 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `dataset/raw/external/potholes_rf` | 0.02 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `models/backup_v1.0-expD` | 0.02 | final artefact (never auto-removed) | no | user-requested safety copy |
| `dataset/raw/external/keremberke_pothole` | 0.01 | source dataset (never auto-removed) | no | original external dataset (license recorded in docs/datasets) |
| `dataset/processed/smoke` | 0.01 | derived dataset (regenerable from sources; mostly hard links) | no | rebuild with dataset/*.py; keep while experiments reference it |
| `experiments/baseline_yolo26n/weights` | 0.01 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `experiments/rdd_yolo26n/weights` | 0.01 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `experiments/rdd_yolo26n_ext/weights` | 0.01 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `experiments/rdd_yolo26n_ext2/weights` | 0.01 | regenerable cache | yes (quarantine) | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `experiments/smoke_rdd/weights` | 0.01 | final artefact (never auto-removed) | no | best.pt duplicated in models/weights; last.pt only needed to resume a finished run |
| `outputs` | 0.01 | final artefact (never auto-removed) | no | user detections database and uploads |
| `dataset/raw/_zips` | 0.00 | regenerable cache | yes (quarantine) | partial/complete download archives; data already extracted |

Regenerable items total **2.31 GB**. They can be moved (not deleted) to `_quarantine/` with `scripts/storage_report.py --quarantine <path> --apply`; source datasets, checkpoints without a copy, results, manifests, licenses, logs, documents and code are never eligible. Deleting a source dataset requires the owner's approval.

Every check and move is appended to `docs/storage/storage_log.jsonl`.

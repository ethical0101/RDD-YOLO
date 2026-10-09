# Datasets, licenses and attribution

All data used by RDD-YOLO is publicly available. Every source below was downloaded by a script in
`dataset/`; nothing was hand-labelled or synthesised. Sources without a stated license were **not** used.

| # | Dataset | Used for | License | Source | Download script |
|---|---|---|---|---|---|
| 1 | **RDD2022** (CRDDC'2022) — Japan, India, Czech Republic, United States, China MotorBike | Experiments A–E (train / val / **original test split**) | CC BY 4.0 | [figshare 21431547](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) | `download_rdd2022.py` |
| 2 | **RDD2022** — Norway, China Drone | C–E (train / val / `test_rdd_new`) | CC BY 4.0 | same archive | `download_rdd2022.py` |
| 3 | Egypt RDD (Roboflow `rdd-6706t` v8) — pothole class only | C–E | CC BY 4.0 | [HF: Hanno100/RoadDamageDetection-Egypt](https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt) | `download_external.py` |
| 4 | Potholes Detection (Roboflow `potholes-detection-d4rma`) | C–E | CC BY 4.0 | [HF: Ryukijano/Pothole-detection-Yolov8](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8) | `download_external.py` |
| 5 | road damage (Roboflow `road-damage-xvt2d`) | D–E | CC BY 4.0 | [HF: manot/pothole-segmentation](https://huggingface.co/datasets/manot/pothole-segmentation) | `download_external.py` |
| 6 | pothole-detection (Roboflow `pothole-detection-gilij`) | D–E | CC BY 4.0 | [HF: manot/pothole-segmentation2](https://huggingface.co/datasets/manot/pothole-segmentation2) | `download_external.py` |
| 7 | Pothole Detection (Roboflow `pothole-detection-irkz9`) | D–E | CC BY 4.0 | [HF: keremberke/pothole-segmentation](https://huggingface.co/datasets/keremberke/pothole-segmentation) | `download_external.py` |
| 8 | pothole YOLO dataset | D–E | MIT | [HF: rupesh002/pothole_dataset_2](https://huggingface.co/datasets/rupesh002/pothole_dataset_2) | `download_external.py` |
| 9 | pothole detection dataset (Roboflow `pothole` v1) | D–E | MIT | [HF: rupesh002/pothole-detection-dataset](https://huggingface.co/datasets/rupesh002/pothole-detection-dataset) | `download_external.py` |
| 10 | MWPD — multi-weather pothole dataset | D–E | Apache-2.0 | [HF: sumadixSk/pothole-dataset](https://huggingface.co/datasets/sumadixSk/pothole-dataset) | `download_external.py` |
| 11 | Pavement Distress Detection aggregate — SVRDD, HighRPD, RD0, "archive", Potholes, Attain sources only | E | MIT | [HF: Deeksha9/pavement-distress-detection](https://huggingface.co/datasets/Deeksha9/pavement-distress-detection) | `download_deeksha.py` |

## Class mapping and filtering

| Our class | RDD2022 code | External labels mapped to it |
|---|---|---|
| D00 Longitudinal crack | D00 | "Longitudinal Crack" (11) |
| D10 Transverse crack | D10 | "Transverse Crack" (11) |
| D20 Alligator crack | D20 | "Alligator Crack" (11) |
| D40 Pothole | D40 | "pothole", "potholes", "Pothole", "Potholes" (3–11) |

* Dropped labels: RDD2022 D43, D44, D50, Repair, D01, D11, D0w0; "Patch" (11).
* Images skipped entirely when they carry a label that cannot be mapped but is visually similar to a target
  class (it would become a false negative): generic "Crack" (3), "Block Crack" (11).
* From source 11, its embedded copy of RDD2022 and the older Japanese RDD release were not used because they
  overlap our held-out test split.

## Leakage control

* The original RDD2022 test split (2,023 images) is fixed since Experiment A and never altered.
* Every added image is compared with every held-out image of all five test sets using a 64-bit difference hash
  (dHash) of the image and of its mirror image (Roboflow exports contain flipped augmentations); matches within
  Hamming distance 6 are removed from training/validation. Near-duplicates across sources are kept only once.
* Counts of removed images per source are written to `dataset/processed/*/dataset_stats_v*.json` and summarised
  in [experiments.md](experiments.md).

## Final splits (extended set v3, used by Experiment E)

| Split | Images | Purpose |
|---|---|---|
| train | 36,223 | training |
| val | 4,501 | model selection |
| test | 2,023 | **original RDD2022 held-out test** (comparable across all experiments) |
| test_rdd_new | 532 | Norway + China Drone held-out |
| test_external | 221 | ground-level potholes held-out |
| test_closeup | 235 | close-up potholes held-out |
| test_pavement | 1,803 | pavement-distress sources held-out |

## Citation

D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, Y. Sekimoto, "RDD2022: A multi-national image dataset for automatic
road damage detection," *Geoscience Data Journal*, 2024. Roboflow-hosted datasets are credited to their
publishers at the Roboflow Universe URLs given in each dataset's README.

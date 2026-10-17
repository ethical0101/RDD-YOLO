# Dataset catalog

Generated 2026-10-17T20:52:09+00:00 by `scripts/build_dataset_docs.py` from `dataset/registry.py` and a scan of the local files. Image counts are measured on disk.

Canonical classes: D00 Longitudinal Crack, D10 Transverse Crack, D20 Alligator Crack, D40 Pothole.

## Datasets on disk and their use in training

Counts are images in `dataset/processed/rdd2022_extended_v3` (the set used by Experiments E and F) after class mapping, de-duplication and leakage removal.

| Dataset | Status | Local images | Local size | train | val | test | test_clean | test_rdd_new | test_external | test_closeup | test_pavement | test_rome | Used in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [RDD2022 (Crowdsensing-based Road Damage Detection Challenge 2022)](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) | used | 38,385 | 10.33 GB | 20,408 | 2,548 | 2,023 | 1,804 | 532 | – | – | – | – | A-F (5 countries); C-F (+Norway, China_Drone) |
| [RDD (Egypt) - Roboflow rdd-6706t v8](https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt) | used | 3,563 | 0.38 GB | 962 | 127 | – | – | – | 121 | – | – | – | C-F |
| [Potholes Detection - Roboflow potholes-detection-d4rma v1](https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8) | used | 300 | 0.02 GB | 100 | 98 | – | – | – | 100 | – | – | – | C-F |
| [road damage - Roboflow road-damage-xvt2d](https://huggingface.co/datasets/manot/pothole-segmentation) | used | 819 | 0.10 GB | 192 | 8 | – | – | – | – | 73 | – | – | D-F |
| [pothole-detection - Roboflow pothole-detection-gilij](https://huggingface.co/datasets/manot/pothole-segmentation2) | used | 864 | 0.13 GB | 426 | 121 | – | – | – | – | 64 | – | – | D-F |
| [Pothole Detection - Roboflow pothole-detection-irkz9](https://huggingface.co/datasets/keremberke/pothole-segmentation) | used | 90 | 0.01 GB | – | – | – | – | – | – | 5 | – | – | D-F |
| [pothole_dataset_2 (YOLO)](https://huggingface.co/datasets/rupesh002/pothole_dataset_2) | used | 665 | 0.65 GB | 5 | – | – | – | – | – | – | – | – | D-F |
| [pothole-detection-dataset (Roboflow pothole v1 export)](https://huggingface.co/datasets/rupesh002/pothole-detection-dataset) | used | 715 | 0.11 GB | 425 | 118 | – | – | – | – | – | – | – | D-F |
| [MWPD - Multi-Weather Pothole Dataset](https://huggingface.co/datasets/sumadixSk/pothole-dataset) | used | 3,139 | 0.64 GB | 2,048 | 180 | – | – | – | – | 93 | – | – | D-F |
| [Pavement Distress Detection aggregate (SVRDD, HighRPD, RD0, 'archive', Potholes, Attain sources)](https://huggingface.co/datasets/Deeksha9/pavement-distress-detection) | used | 35,223 | 5.46 GB | 11,657 | 1,301 | – | – | – | – | – | 1,803 | – | E-F |
| [N-RDD2024: Road damage and defects (refined RDD2022 annotations)](https://data.mendeley.com/datasets/27c8pwsd6v/3) | downloaded (annotations only) | 0 | 0.02 GB | – | – | – | – | – | – | – | – | – | label-agreement analysis only (not training) |
| [Road Damage Dataset: Potholes, Cracks and Manholes (Rome, Italy)](https://zenodo.org/records/18528034) | used (test only) | 2,009 | 0.18 GB | – | – | – | – | – | – | – | – | 634 | held-out test only (test_rome, 634 images) |
| **Total** | | | | **36,223** | **4,501** | **2,023** | **1,804** | **532** | **221** | **235** | **1,803** | **634** | |

## Details

### RDD2022 (Crowdsensing-based Road Damage Detection Challenge 2022)

* Source: https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547
* License: CC BY 4.0
* Attribution: Arya D., Maeda H., Ghosh S.K., Toshniwal D., Sekimoto Y.
* Publication: RDD2022: A multi-national image dataset for automatic road damage detection, Geoscience Data Journal, 2024
* Original format / classes: Pascal VOC XML / D00, D10, D20, D40 (+ D01, D11, D43, D44, D50, Repair, D0w0)
* Mapping to canonical classes: D00/D10/D20/D40 kept; all other labels dropped (image kept)
* Limitations: Official test split has no public labels; consecutive dash-cam frames cause near-duplicates; Norway images 3650 px wide (downscaled to 1280)
* Local copies: `dataset/raw/RDD2022/Japan` (10,506 images, 10,506 annotation files); `dataset/raw/RDD2022/India` (7,706 images, 7,706 annotation files); `dataset/raw/RDD2022/Czech` (2,829 images, 2,829 annotation files); `dataset/raw/RDD2022/United_States` (4,805 images, 4,805 annotation files); `dataset/raw/RDD2022/China_MotorBike` (1,977 images, 1,977 annotation files); `dataset/raw/RDD2022/Norway` (8,161 images, 8,161 annotation files); `dataset/raw/RDD2022/China_Drone` (2,401 images, 2,401 annotation files)

### RDD (Egypt) - Roboflow rdd-6706t v8

* Source: https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt
* License: CC BY 4.0 (Roboflow export) / Apache-2.0 (HF card)
* Attribution: Roboflow Universe workspace road-damage-detection-wgjj4
* Publication: Roboflow Universe dataset, exported 2024-04-16
* Original format / classes: YOLOv8 TXT / Crack, Pothole
* Mapping to canonical classes: Pothole -> D40; images with generic 'Crack' skipped (cannot map to D00/D10/D20)
* Limitations: Generic crack label; 70 % of images skipped
* Local copies: `dataset/raw/external/egypt_rdd` (3,563 images, 3,565 annotation files)

### Potholes Detection - Roboflow potholes-detection-d4rma v1

* Source: https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8
* License: CC BY 4.0
* Attribution: Roboflow Universe workspace project-ssayl
* Publication: Roboflow Universe, 2023
* Original format / classes: YOLOv8 TXT / pothole
* Mapping to canonical classes: pothole -> D40
* Limitations: Only 300 images mirrored on Hugging Face; 640x640 stretched
* Local copies: `dataset/raw/external/potholes_rf` (300 images, 301 annotation files)

### road damage - Roboflow road-damage-xvt2d

* Source: https://huggingface.co/datasets/manot/pothole-segmentation
* License: CC BY 4.0
* Attribution: Roboflow Universe workspace abdulmohsen-fahad-f7pdw
* Publication: Roboflow Universe, 2023
* Original format / classes: COCO JSON (boxes + polygons) / potholes, pothole
* Mapping to canonical classes: pothole(s) -> D40
* Limitations: Many near-duplicates of other pothole sets (removed by dHash)
* Local copies: `dataset/raw/external/manot_road_damage` (819 images, 4 annotation files)

### pothole-detection - Roboflow pothole-detection-gilij

* Source: https://huggingface.co/datasets/manot/pothole-segmentation2
* License: CC BY 4.0
* Attribution: Roboflow Universe workspace gurgen-hovsepyan-mbrnv
* Publication: Roboflow Universe, 2023
* Original format / classes: COCO JSON / pothole
* Mapping to canonical classes: pothole -> D40
* Limitations: Close-up photos
* Local copies: `dataset/raw/external/manot_pothole2` (864 images, 4 annotation files)

### Pothole Detection - Roboflow pothole-detection-irkz9

* Source: https://huggingface.co/datasets/keremberke/pothole-segmentation
* License: CC BY 4.0
* Attribution: Roboflow Universe workspace imacs-pothole-detection-wo8mu
* Publication: Roboflow Universe, 2023
* Original format / classes: COCO JSON / pothole
* Mapping to canonical classes: pothole -> D40
* Limitations: 90 images; mostly duplicates of other sets
* Local copies: `dataset/raw/external/keremberke_pothole` (90 images, 4 annotation files)

### pothole_dataset_2 (YOLO)

* Source: https://huggingface.co/datasets/rupesh002/pothole_dataset_2
* License: MIT
* Attribution: rupesh002 (Hugging Face)
* Publication: Hugging Face dataset
* Original format / classes: YOLO TXT / pothole
* Mapping to canonical classes: pothole -> D40
* Limitations: Almost entirely duplicates of other pothole sets
* Local copies: `dataset/raw/external/rupesh_pothole2` (665 images, 665 annotation files)

### pothole-detection-dataset (Roboflow pothole v1 export)

* Source: https://huggingface.co/datasets/rupesh002/pothole-detection-dataset
* License: MIT
* Attribution: rupesh002 (Hugging Face)
* Publication: Hugging Face dataset
* Original format / classes: YOLOv4-PyTorch TXT / pothole
* Mapping to canonical classes: pothole -> D40
* Limitations: Partial duplicates
* Local copies: `dataset/raw/external/rupesh_pothole` (715 images, 715 annotation files)

### MWPD - Multi-Weather Pothole Dataset

* Source: https://huggingface.co/datasets/sumadixSk/pothole-dataset
* License: Apache-2.0
* Attribution: sumadixSk (Hugging Face)
* Publication: Hugging Face dataset
* Original format / classes: YOLO TXT / Potholes
* Mapping to canonical classes: Potholes -> D40
* Limitations: Weather augmentations create near-duplicates (removed by dHash)
* Local copies: `dataset/raw/external/mwpd` (3,139 images, 3,139 annotation files)

### Pavement Distress Detection aggregate (SVRDD, HighRPD, RD0, 'archive', Potholes, Attain sources)

* Source: https://huggingface.co/datasets/Deeksha9/pavement-distress-detection
* License: MIT
* Attribution: Deeksha9 / DeekshithaKalluri; original sources include SVRDD and HighRPD (Mendeley, CC BY 4.0, doi:10.17632/sywswj7djj.1)
* Publication: Hugging Face dataset (aggregate of public datasets)
* Original format / classes: YOLO TXT / Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole, Patch, Block Crack
* Mapping to canonical classes: first four -> D00/D10/D20/D40; Patch boxes dropped; images with Block Crack skipped; embedded RDD2022 and older Japanese RDD not used (test overlap)
* Limitations: Class mapping done by the aggregator; 'archive' source undocumented
* Local copies: `dataset/raw/external/deeksha` (35,223 images, 35,223 annotation files)

### N-RDD2024: Road damage and defects (refined RDD2022 annotations)

* Source: https://data.mendeley.com/datasets/27c8pwsd6v/3
* License: CC BY 4.0
* Attribution: Kaya O., Codur M.Y.
* Publication: Mendeley Data, v3, 2024, doi:10.17632/27c8pwsd6v.3
* Original format / classes: Pascal VOC / COCO / YOLO (Roboflow export, 640x640) / D00, D10, D20, D30, D40, D50, D60, D70, D80, D90
* Mapping to canonical classes: annotations only (images are RDD2022); compared on D00/D10/D20/D40: box F1 98.2 %, class agreement 99.9 % (docs/diagnostics/LABEL_AGREEMENT.md), so no corrected labels to train with
* Limitations: Own train/valid split overlaps our test images; resized 640x640
* Local copies: `dataset/raw/external/nrdd2024` (0 images, 18,996 annotation files)

### Road Damage Dataset: Potholes, Cracks and Manholes (Rome, Italy)

* Source: https://zenodo.org/records/18528034
* License: CC BY 4.0
* Attribution: Giordani E., Arcioni L., Gil-Martin M., Marini M.R.
* Publication: Zenodo 2026, doi:10.5281/zenodo.18528034; Scientific Reports, doi:10.1038/s41598-026-46679-4
* Original format / classes: YOLO TXT / Pothole, Crack, Manhole
* Mapping to canonical classes: Pothole -> D40; images with generic Crack excluded; Manhole boxes dropped (manhole-only images kept as negatives)
* Limitations: Generic crack class (1,375 of 2,009 images excluded); 640x360; one city, Feb-Mar 2025
* Local copies: `dataset/raw/external/rome_rdd` (2,009 images, 4,020 annotation files)

## Reviewed and not used

| Dataset | Reason |
|---|---|
| [IRRDD (Iran Road Damage Dataset)](https://github.com/IranRoadDamageDataset/IRRDD) | No license stated (25,000 images, YOLO, D00/D10/D20/D40) |
| [UAPD (UAV Asphalt Pavement Distress)](https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset) | No license stated; UAV top-down view |
| [PaveTrack](https://www.scidb.cn/en/detail?dataSetId=e9ffbca43e6e4a5ebdd25663d0de9ad1) | CC BY-NC terms, generic 'crack' class incompatible with D00/D10, 46k images exceed the storage budget |
| [SUT-Crack](https://data.mendeley.com/datasets/gsbmknrhkv/6) | 130 images, generic crack class only |
| [Outerview Global Potholes](https://huggingface.co/datasets/Outerview/global-potholes-dataset) | Image-level labels only, no boxes |
| [miia-pothole](https://huggingface.co/datasets/mabughali/miia-pothole-train) | Classification labels only |
| [RDD2020 / GRDDC'2020 copies](https://github.com/sekilab/RoadDamageDetector) | Contained in RDD2022 (duplicates, test leakage) |
| [Crack500, CFD, GAPs, Pothole-600, CrackTree260 and other segmentation sets](https://github.com/large-road-damage-datalake) | Pixel masks without crack type; no boxes would be honest for D00/D10 (GAPs also requires registration) |
| [Unlicensed Hugging Face uploads (RyzenPuppy, VarunVV, robiulhasanjisan88, hf-vision, adhisetiawan)](https://huggingface.co/datasets) | No license stated |

"""Single source of truth for dataset provenance (static facts). Local counts are measured by
scripts/build_dataset_docs.py, which merges this registry with a scan of the files on disk.

Fields: name, source_url, license, attribution, publication, original_format, original_classes,
mapping (their class -> canonical class or rule), local_dirs (relative to project), v3_prefixes (file-name prefixes
in dataset/processed/rdd2022_extended_v3), used_in (experiments), limitations, status.
"""

CANONICAL = {"D00": "Longitudinal Crack", "D10": "Transverse Crack", "D20": "Alligator Crack", "D40": "Pothole"}

DATASETS = [
    {
        "id": "rdd2022", "name": "RDD2022 (Crowdsensing-based Road Damage Detection Challenge 2022)",
        "source_url": "https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547",
        "license": "CC BY 4.0", "attribution": "Arya D., Maeda H., Ghosh S.K., Toshniwal D., Sekimoto Y.",
        "publication": "RDD2022: A multi-national image dataset for automatic road damage detection, Geoscience Data Journal, 2024",
        "original_format": "Pascal VOC XML", "original_classes": "D00, D10, D20, D40 (+ D01, D11, D43, D44, D50, Repair, D0w0)",
        "mapping": "D00/D10/D20/D40 kept; all other labels dropped (image kept)",
        "local_dirs": ["dataset/raw/RDD2022/Japan", "dataset/raw/RDD2022/India", "dataset/raw/RDD2022/Czech",
                       "dataset/raw/RDD2022/United_States", "dataset/raw/RDD2022/China_MotorBike",
                       "dataset/raw/RDD2022/Norway", "dataset/raw/RDD2022/China_Drone"],
        "v3_prefixes": ["Japan_", "India_", "Czech_", "United_States_", "China_MotorBike_", "Norway_", "China_Drone_"],
        "used_in": "A-F (5 countries); C-F (+Norway, China_Drone)",
        "limitations": "Official test split has no public labels; consecutive dash-cam frames cause near-duplicates; Norway images 3650 px wide (downscaled to 1280)",
        "status": "used",
    },
    {
        "id": "egypt_rdd", "name": "RDD (Egypt) - Roboflow rdd-6706t v8", "source_url": "https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt",
        "license": "CC BY 4.0 (Roboflow export) / Apache-2.0 (HF card)", "attribution": "Roboflow Universe workspace road-damage-detection-wgjj4",
        "publication": "Roboflow Universe dataset, exported 2024-04-16", "original_format": "YOLOv8 TXT", "original_classes": "Crack, Pothole",
        "mapping": "Pothole -> D40; images with generic 'Crack' skipped (cannot map to D00/D10/D20)",
        "local_dirs": ["dataset/raw/external/egypt_rdd"], "v3_prefixes": ["egypt_rdd_"], "used_in": "C-F",
        "limitations": "Generic crack label; 70 % of images skipped", "status": "used",
    },
    {
        "id": "potholes_rf", "name": "Potholes Detection - Roboflow potholes-detection-d4rma v1", "source_url": "https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8",
        "license": "CC BY 4.0", "attribution": "Roboflow Universe workspace project-ssayl", "publication": "Roboflow Universe, 2023",
        "original_format": "YOLOv8 TXT", "original_classes": "pothole", "mapping": "pothole -> D40",
        "local_dirs": ["dataset/raw/external/potholes_rf"], "v3_prefixes": ["potholes_rf_"], "used_in": "C-F",
        "limitations": "Only 300 images mirrored on Hugging Face; 640x640 stretched", "status": "used",
    },
    {
        "id": "manot_road_damage", "name": "road damage - Roboflow road-damage-xvt2d", "source_url": "https://huggingface.co/datasets/manot/pothole-segmentation",
        "license": "CC BY 4.0", "attribution": "Roboflow Universe workspace abdulmohsen-fahad-f7pdw", "publication": "Roboflow Universe, 2023",
        "original_format": "COCO JSON (boxes + polygons)", "original_classes": "potholes, pothole", "mapping": "pothole(s) -> D40",
        "local_dirs": ["dataset/raw/external/manot_road_damage"], "v3_prefixes": ["manot_road_damage_"], "used_in": "D-F",
        "limitations": "Many near-duplicates of other pothole sets (removed by dHash)", "status": "used",
    },
    {
        "id": "manot_pothole2", "name": "pothole-detection - Roboflow pothole-detection-gilij", "source_url": "https://huggingface.co/datasets/manot/pothole-segmentation2",
        "license": "CC BY 4.0", "attribution": "Roboflow Universe workspace gurgen-hovsepyan-mbrnv", "publication": "Roboflow Universe, 2023",
        "original_format": "COCO JSON", "original_classes": "pothole", "mapping": "pothole -> D40",
        "local_dirs": ["dataset/raw/external/manot_pothole2"], "v3_prefixes": ["manot_pothole2_"], "used_in": "D-F", "limitations": "Close-up photos", "status": "used",
    },
    {
        "id": "keremberke_pothole", "name": "Pothole Detection - Roboflow pothole-detection-irkz9", "source_url": "https://huggingface.co/datasets/keremberke/pothole-segmentation",
        "license": "CC BY 4.0", "attribution": "Roboflow Universe workspace imacs-pothole-detection-wo8mu", "publication": "Roboflow Universe, 2023",
        "original_format": "COCO JSON", "original_classes": "pothole", "mapping": "pothole -> D40",
        "local_dirs": ["dataset/raw/external/keremberke_pothole"], "v3_prefixes": ["keremberke_pothole_"], "used_in": "D-F",
        "limitations": "90 images; mostly duplicates of other sets", "status": "used",
    },
    {
        "id": "rupesh_pothole2", "name": "pothole_dataset_2 (YOLO)", "source_url": "https://huggingface.co/datasets/rupesh002/pothole_dataset_2",
        "license": "MIT", "attribution": "rupesh002 (Hugging Face)", "publication": "Hugging Face dataset", "original_format": "YOLO TXT",
        "original_classes": "pothole", "mapping": "pothole -> D40", "local_dirs": ["dataset/raw/external/rupesh_pothole2"],
        "v3_prefixes": ["rupesh_pothole2_"], "used_in": "D-F", "limitations": "Almost entirely duplicates of other pothole sets", "status": "used",
    },
    {
        "id": "rupesh_pothole", "name": "pothole-detection-dataset (Roboflow pothole v1 export)", "source_url": "https://huggingface.co/datasets/rupesh002/pothole-detection-dataset",
        "license": "MIT", "attribution": "rupesh002 (Hugging Face)", "publication": "Hugging Face dataset", "original_format": "YOLOv4-PyTorch TXT",
        "original_classes": "pothole", "mapping": "pothole -> D40", "local_dirs": ["dataset/raw/external/rupesh_pothole"],
        "v3_prefixes": ["rupesh_pothole_"], "used_in": "D-F", "limitations": "Partial duplicates", "status": "used",
    },
    {
        "id": "mwpd", "name": "MWPD - Multi-Weather Pothole Dataset", "source_url": "https://huggingface.co/datasets/sumadixSk/pothole-dataset",
        "license": "Apache-2.0", "attribution": "sumadixSk (Hugging Face)", "publication": "Hugging Face dataset", "original_format": "YOLO TXT",
        "original_classes": "Potholes", "mapping": "Potholes -> D40", "local_dirs": ["dataset/raw/external/mwpd"], "v3_prefixes": ["mwpd_"],
        "used_in": "D-F", "limitations": "Weather augmentations create near-duplicates (removed by dHash)", "status": "used",
    },
    {
        "id": "deeksha_pavement", "name": "Pavement Distress Detection aggregate (SVRDD, HighRPD, RD0, 'archive', Potholes, Attain sources)",
        "source_url": "https://huggingface.co/datasets/Deeksha9/pavement-distress-detection", "license": "MIT",
        "attribution": "Deeksha9 / DeekshithaKalluri; original sources include SVRDD and HighRPD (Mendeley, CC BY 4.0, doi:10.17632/sywswj7djj.1)",
        "publication": "Hugging Face dataset (aggregate of public datasets)", "original_format": "YOLO TXT",
        "original_classes": "Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole, Patch, Block Crack",
        "mapping": "first four -> D00/D10/D20/D40; Patch boxes dropped; images with Block Crack skipped; embedded RDD2022 and older Japanese RDD not used (test overlap)",
        "local_dirs": ["dataset/raw/external/deeksha"], "v3_prefixes": ["deeksha_"], "used_in": "E-F",
        "limitations": "Class mapping done by the aggregator; 'archive' source undocumented", "status": "used",
    },
    {
        "id": "nrdd2024", "name": "N-RDD2024: Road damage and defects (refined RDD2022 annotations)", "source_url": "https://data.mendeley.com/datasets/27c8pwsd6v/3",
        "license": "CC BY 4.0", "attribution": "Kaya O., Codur M.Y.", "publication": "Mendeley Data, v3, 2024, doi:10.17632/27c8pwsd6v.3",
        "original_format": "Pascal VOC / COCO / YOLO (Roboflow export, 640x640)", "original_classes": "D00, D10, D20, D30, D40, D50, D60, D70, D80, D90",
        "mapping": "annotations only (images are RDD2022); compared on D00/D10/D20/D40: box F1 98.2 %, class agreement 99.9 % (docs/diagnostics/LABEL_AGREEMENT.md), so no corrected labels to train with",
        "local_dirs": ["dataset/raw/external/nrdd2024"], "v3_prefixes": [], "used_in": "label-agreement analysis only (not training)",
        "limitations": "Own train/valid split overlaps our test images; resized 640x640", "status": "downloaded (annotations only)",
    },
    {
        "id": "rome_rdd", "name": "Road Damage Dataset: Potholes, Cracks and Manholes (Rome, Italy)", "source_url": "https://zenodo.org/records/18528034",
        "license": "CC BY 4.0", "attribution": "Giordani E., Arcioni L., Gil-Martin M., Marini M.R.",
        "publication": "Zenodo 2026, doi:10.5281/zenodo.18528034; Scientific Reports, doi:10.1038/s41598-026-46679-4",
        "original_format": "YOLO TXT", "original_classes": "Pothole, Crack, Manhole",
        "mapping": "Pothole -> D40; images with generic Crack excluded; Manhole boxes dropped (manhole-only images kept as negatives)",
        "local_dirs": ["dataset/raw/external/rome_rdd"], "v3_prefixes": ["rome_rdd_"], "used_in": "held-out test only (test_rome, 634 images)",
        "limitations": "Generic crack class (1,375 of 2,009 images excluded); 640x360; one city, Feb-Mar 2025", "status": "used (test only)",
    },
]

EXCLUDED = [
    {"name": "IRRDD (Iran Road Damage Dataset)", "source_url": "https://github.com/IranRoadDamageDataset/IRRDD", "reason": "No license stated (25,000 images, YOLO, D00/D10/D20/D40)"},
    {"name": "UAPD (UAV Asphalt Pavement Distress)", "source_url": "https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset", "reason": "No license stated; UAV top-down view"},
    {"name": "PaveTrack", "source_url": "https://www.scidb.cn/en/detail?dataSetId=e9ffbca43e6e4a5ebdd25663d0de9ad1", "reason": "CC BY-NC terms, generic 'crack' class incompatible with D00/D10, 46k images exceed the storage budget"},
    {"name": "SUT-Crack", "source_url": "https://data.mendeley.com/datasets/gsbmknrhkv/6", "reason": "130 images, generic crack class only"},
    {"name": "Outerview Global Potholes", "source_url": "https://huggingface.co/datasets/Outerview/global-potholes-dataset", "reason": "Image-level labels only, no boxes"},
    {"name": "miia-pothole", "source_url": "https://huggingface.co/datasets/mabughali/miia-pothole-train", "reason": "Classification labels only"},
    {"name": "RDD2020 / GRDDC'2020 copies", "source_url": "https://github.com/sekilab/RoadDamageDetector", "reason": "Contained in RDD2022 (duplicates, test leakage)"},
    {"name": "Crack500, CFD, GAPs, Pothole-600, CrackTree260 and other segmentation sets", "source_url": "https://github.com/large-road-damage-datalake", "reason": "Pixel masks without crack type; no boxes would be honest for D00/D10 (GAPs also requires registration)"},
    {"name": "Unlicensed Hugging Face uploads (RyzenPuppy, VarunVV, robiulhasanjisan88, hf-vision, adhisetiawan)", "source_url": "https://huggingface.co/datasets", "reason": "No license stated"},
]

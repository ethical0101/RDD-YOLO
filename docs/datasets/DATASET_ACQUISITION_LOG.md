# Dataset acquisition log

Dates come from the completion markers written by the download scripts (`.extracted`, `.complete`,
`download_manifest.json`). From 2026-10-18 on, every download also appends a machine-readable event to
`docs/storage/storage_log.jsonl` (free-space check before the download, then completion).

| Date | Dataset | Script | Source | What was stored | Notes / errors |
|---|---|---|---|---|---|
| 2026-10-07 | RDD2022 Czech, India, Japan, United_States, China_MotorBike | `dataset/download_rdd2022.py` | figshare 21431547 (`ndownloader.figshare.com/files/38030910`) | `dataset/raw/RDD2022/<country>/train` (images + VOC XML) | Only the five country zips were read from the 13.3 GB archive by HTTP range requests; CRC verified by `zipfile`; country zips deleted after a verified extraction (re-downloadable). The official, unlabelled test images were not extracted |
| 2026-10-08 | RDD2022 China_Drone, Norway | same | same | same | Network switch caused SSL/DNS errors; retried automatically by exponential back-off. Norway images (3650 px) are downscaled to 1280 px only in the processed copy |
| 2026-10-08 | egypt_rdd, potholes_rf, manot_road_damage, manot_pothole2, keremberke_pothole, rupesh_pothole, rupesh_pothole2, mwpd | `dataset/download_external.py` | Hugging Face dataset repositories (see catalog) | `dataset/raw/external/<name>` | Only repositories with an explicit license were downloaded; unlicensed uploads were skipped |
| 2026-10-09 | Deeksha pavement-distress aggregate | `dataset/download_deeksha.py` | `huggingface.co/datasets/Deeksha9/pavement-distress-detection` | `dataset/raw/external/deeksha` | Its embedded RDD2022 and older Japanese RDD copies are not used (would leak into our test split) |
| 2026-10-18 | N-RDD2024 annotations (VOC XML only) | `dataset/download_nrdd2024.py` | Mendeley Data 27c8pwsd6v v3 public API | `dataset/raw/external/nrdd2024/<country>_voc/` | Mendeley returns HTTP 403 to Python's default User-Agent; a descriptive User-Agent header is now sent (public files, no authentication involved). Images were streamed but not written: they are the RDD2022 images already on disk |
| 2026-10-18 | Rome road-damage dataset (potholes, cracks, manholes) | `dataset/download_rome.py` | Zenodo record 18528034 (`data.zip`, 194 MB) | `dataset/raw/external/rome_rdd/data` | MD5 verified against the Zenodo record; zip removed after extraction. The two conversion scripts inside the archive were not executed. Used only as the held-out `test_rome` split (`dataset/build_test_rome.py`) |

## Datasets that were found but not acquired

See "Reviewed and not used" in `DATASET_CATALOG.md`: no dataset without a clear license was downloaded, nothing
behind a registration wall was accessed, and no access restriction was bypassed.

## Cleanup history

* RDD2022 country zips (`dataset/raw/_zips`) were removed by the downloader only after a verified extraction.
* No source dataset, checkpoint or evaluation result has been deleted. Regenerable items (caches, build outputs)
  are listed in `docs/storage/STORAGE_REPORT.md` and can only be moved to `_quarantine/`, never deleted
  automatically.

# Accuracy diagnosis

Produced from real files by `scripts/diagnose_accuracy.py` (raw numbers: `diagnosis.json`; example images:
`examples/`). Model analysed: the served YOLO26s (`models/weights/yolo26s_full_best.pt`, Experiment E) on the
original RDD2022 test split (2,023 images, 4,132 boxes), confidence 0.25, IoU match 0.5.

## Summary — likely causes, ranked by evidence

| Rank | Cause | Evidence | Planned fix |
|---|---|---|---|
| 1 | **Missed small objects** | Recall 47.7 % for boxes < 1 % of the image vs 73.6 % for boxes ≥ 5 %; 65.7 % of training potholes and 45.8 % of transverse cracks are < 1 % of the image | Higher input resolution (Experiment F2, 800 px) |
| 2 | **Detection, not classification, errors** | Of 4,132 test boxes only 75 were found with the wrong class; 1,726 were missed entirely and 1,167 predictions matched no box | Resolution, longer training, label quality; class re-weighting is *not* the priority |
| 3 | **Under-training** | Experiments D and E stopped at their last epoch with validation mAP still rising (+2.0 pp over the last 5 epochs for E) and validation loss at its minimum in the final epoch | Continue training (Experiment F1 control, 10 more epochs) |
| 4 | Annotation inconsistency (suspected, **not confirmed**) | Crack boxes in RDD2022 are drawn at varying granularity. The independent N-RDD2024 annotation (CC BY 4.0) of 18,095 of the same images agrees with RDD2022 on 98.2 % of D00/D10/D20/D40 boxes (IoU ≥ 0.5) and on 99.9 % of their classes (`LABEL_AGREEMENT.md`): it added new classes but did not redraw ours | No label-refinement experiment: N-RDD2024 offers no corrected labels for these four classes |
| 5 | **Test-set leakage (minor, inflates scores)** | 219 of 2,023 test images (10.8 %) have a near-duplicate training image (dHash Hamming ≤ 6; visual check: 6 of 8 sampled pairs were consecutive frames of the same scene); most in China_MotorBike (61/199) and India (102/355). Recall on those images 63.2 % vs 57.6 % on the rest | Report a **clean subset** (1,804 images); group-aware splits for any new split |
| 6 | Class imbalance (moderate) | Training boxes D00 12,838 · D20 7,887 · D10 7,054 · D40 4,702 (2.7 : 1); D40 has the lowest recall (43.3 %) but also the smallest objects (median 0.59 % of image area) | Size, not count, is the dominant factor for D40; no re-sampling yet |

## 1. Dataset and labels (original split, 16,156 / 2,017 / 2,023 images)

* Annotation format verified: every label line has 5 fields, class ids 0–3, coordinates in [0, 1].
* Label defects are rare: 0 duplicate boxes (IoU > 0.9, same class) in train, 1 in test; 20 boxes in train are
  narrower than 8 px at 640 px input.
* Median box area (share of image): D00 1.80 %, D10 1.10 %, D20 8.94 %, D40 0.59 %.
* Background (no target damage) images: 6.2 % of train.

## 2. Leakage between train and test

| Country | Test images | Near-duplicate in train (≤ 6) |
|---|---|---|
| China_MotorBike | 199 | 61 |
| India | 355 | 102 |
| Japan | 869 | 27 |
| United_States | 481 | 24 |
| Czech | 119 | 5 |

The original split was random per image within each country; RDD2022 contains consecutive dash-cam frames,
so a random split places neighbouring frames on both sides. Effect on reported numbers (mAP@50, full test →
clean subset of 1,804 images): A 59.66 → 58.87, B 59.46 → 58.53, C 60.50 → 59.74, D 61.02 → 60.17,
E 62.45 → 61.78. Inflation is 0.7–0.9 pp and the ranking of models does not change.

## 3. Training curves

| Run | Epochs | Best epoch | Val mAP@50 gain, last 5 epochs | Val-loss minimum at epoch |
|---|---|---|---|---|
| A baseline_yolo26n | 40 | 37 | +0.08 pp | 37 |
| B rdd_yolo26n | 40 | 40 | +0.57 pp | 39 |
| C rdd_yolo26n_ext | 25 | 24 | +0.91 pp | 24 |
| D rdd_yolo26n_ext2 | 15 | 15 | +2.25 pp | 15 |
| E yolo26s_full | 20 | 20 | +1.99 pp | 20 |

A and B had converged (no over-fitting: validation loss still at its minimum near the end). D and E were still
improving when they stopped.

## 4. Error analysis of Experiment E (test split)

| Class | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| D00 Longitudinal crack | 958 | 520 | 592 | 64.8 % | 61.8 % |
| D10 Transverse crack | 512 | 299 | 394 | 63.1 % | 56.5 % |
| D20 Alligator crack | 573 | 257 | 438 | 69.0 % | 56.7 % |
| D40 Pothole | 288 | 166 | 377 | 63.4 % | 43.3 % |

Confusion matrix (rows = predicted, columns = ground truth, "bg" = background):

| | D00 | D10 | D20 | D40 | bg |
|---|---|---|---|---|---|
| D00 | 958 | 8 | 33 | 1 | 478 |
| D10 | 5 | 512 | 3 | 0 | 291 |
| D20 | 18 | 2 | 573 | 0 | 237 |
| D40 | 1 | 0 | 4 | 288 | 161 |
| bg | 568 | 384 | 398 | 376 | — |

Recall by object size: small (< 1 %) 47.7 % (654/1,371) · medium (1–5 %) 56.2 % (907/1,613) · large (≥ 5 %)
73.6 % (845/1,148). Example misses and false alarms: `examples/fn_*.jpg`, `examples/fp_*.jpg`.

## 5. What this rules out

* Not a broken pipeline: labels, class ids and checkpoint loading are correct (inter-class confusion is < 2 %).
* Not over-fitting: validation loss tracks training loss and keeps decreasing.
* More data of the same kind helped (Experiments C–E) but cannot fix the small-object recall gap by itself.

## 6. Label agreement with N-RDD2024

See `LABEL_AGREEMENT.md` (`scripts/label_agreement.py`). N-RDD2024 re-exports RDD2022 with ten classes; for the four
classes we model its boxes are essentially the RDD2022 boxes (box F1 98.2 % train, 98.5 % test; class agreement
99.9 %). It therefore cannot be used to measure or reduce label noise in D00–D40, and 1,520 of our test images are in
its own training split. It is kept as an analysed but unused source.

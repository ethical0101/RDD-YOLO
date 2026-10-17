# Generalization to a new country: Rome (test_rome)

Held-out split built by `dataset/build_test_rome.py` from the Rome road-damage dataset (Zenodo 18528034, CC BY 4.0):
634 images at 640x360 (282 with potholes, 352 manhole-only negatives), 396 pothole boxes. No image of this source is
used for training; a dHash check against all non-RDD training images found no duplicates (17 candidate matches, all
visually different streets). Only the pothole class (D40) has ground truth; crack predictions on these images count
as false positives in the error analysis below but do not affect D40 AP.

| Exp | Run | D40 AP@50 test_rome | D40 AP@50-95 | P | R | D40 AP@50 on RDD2022 test (for reference) |
|---|---|---|---|---|---|---|
| A | baseline_yolo26n | 17.9% | 5.9% | 30.3% | 23.0% | 45.9% |
| B | rdd_yolo26n | 18.6% | 6.1% | 29.7% | 26.8% | 45.3% |
| C | rdd_yolo26n_ext | 19.4% | 6.7% | 29.1% | 30.6% | 46.8% |
| D | rdd_yolo26n_ext2 | 22.5% | 8.2% | 32.3% | 31.1% | 48.5% |
| E | yolo26s_full | 25.7% | 8.8% | 32.9% | 33.2% | 51.8% |

## Error analysis (Experiment E, confidence 0.25)

* Potholes found: 86 of 396; missed: 310. Missed potholes are small: median 0.36 % of the image area vs 0.66 % for found ones (at 640x360 that is roughly a 29x29 px box).
* Pothole false alarms: 83 on manhole covers, 46 badly localised boxes on real potholes, 14 elsewhere. Manhole covers are not labelled as negatives in any training source.
* Crack false alarms: 56 D00, 15 D10, 37 D20 on images that the Rome annotators marked crack-free.

## Interpretation

The drop from in-domain scores to about 26 % is a genuine domain-shift result: new city and camera, low resolution,
and an object (manhole cover) that resembles a pothole but never appears as a labelled negative in training. More
diverse data and a larger model did transfer: Rome AP rose steadily from 17.9 % (A) to 25.7 % (E), larger relative
gains than on the RDD2022 test (45.9 % to 51.8 %), but the gap remains wide. Likely remedies, not yet tested: training
images containing manhole covers as negatives, and higher input resolution for small potholes (Experiment F2 will
be evaluated on this split).

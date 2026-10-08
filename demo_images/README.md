# Demo images

12 road photos from the **held-out test split** of RDD2022 (never used for training or model selection),
3 per class, from different countries. Source: RDD2022, Arya et al., CC BY 4.0.

**These are selected best cases for a live demonstration**: they were chosen automatically
(see `manifest.json`) as test images where the trained RDD-YOLO model detects the class that is present in the
ground truth with confidence ≥ 0.4. They show what the model looks like when it works; they are **not**
representative of overall accuracy. The representative numbers are the test-split metrics in
`docs/experiments.md` (mAP@50 59.5 %, recall 55.3 %).

Use them in the dashboard's *Image Detection* page (drag & drop), optionally with "Use my location".

"""Build the held-out `test_rome` split from the Rome road-damage dataset (Zenodo 18528034, CC BY 4.0).

The Rome set labels potholes, cracks (one generic class) and manholes. A generic crack cannot be mapped to
D00/D10/D20, so images containing any crack are excluded. In the remaining images every crack would have been
labelled, so they are fully annotated for our four classes: potholes -> D40, manholes dropped (not damage),
manhole-only images kept as true negatives. The split is used for evaluation only (never for training).

Output: dataset/processed/rdd2022_extended_v3/{images,labels}/test_rome + data_test_rome.yaml
Usage: python dataset/build_test_rome.py
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "raw" / "external" / "rome_rdd" / "data"
V3 = ROOT / "processed" / "rdd2022_extended_v3"


def main() -> None:
    img_out, lab_out = V3 / "images" / "test_rome", V3 / "labels" / "test_rome"
    for d in (img_out, lab_out):
        if d.exists():
            shutil.rmtree(d)  # derived split, rebuilt from the raw source every time
        d.mkdir(parents=True)
    stats = Counter()
    for img in sorted((SRC / "images").glob("*.jpg")):
        lab = SRC / "labels-YOLO" / f"{img.stem}.txt"
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()] if lab.exists() else []
        if any(r[0] == "1" for r in rows):
            stats["skipped: contains generic crack"] += 1
            continue
        out = [f"3 {' '.join(r[1:5])}" for r in rows if r[0] == "0"]
        stats["images"] += 1
        stats["D40 boxes"] += len(out)
        stats["negative images (manhole only)"] += not out
        name = f"rome_rdd_{img.stem}"
        shutil.copy2(img, img_out / f"{name}.jpg")
        (lab_out / f"{name}.txt").write_text("\n".join(out) + ("\n" if out else ""))
    (V3 / "data_test_rome.yaml").write_text(
        f"path: {V3.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test_rome\nnc: 4\n"
        "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    (V3 / "test_rome_stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(dict(stats))


if __name__ == "__main__":
    main()

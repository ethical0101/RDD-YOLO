"""Build the extended training set (Experiment C) on top of the original prepared dataset.

Sources
  1. Original 5-country RDD2022 set (dataset/processed/rdd2022_yolo) - copied split-for-split, unchanged,
     so the original test split stays identical and results remain comparable with Experiments A/B.
  2. RDD2022 China_Drone and Norway (official, all 4 classes) - validated with the same rules as
     prepare_dataset.py, own seeded 80/10/10 split per country, images downscaled to <= MAX_SIDE px
     (labels are normalised, so resizing does not change them).
  3. External pothole datasets (Roboflow exports, CC BY 4.0, see download_external.py) - only the pothole
     class is used and mapped to D40. Images that also contain a generic "Crack" label are skipped: those
     cracks cannot be mapped to D00/D10/D20 and would otherwise become false negatives. The publishers'
     own train/valid/test splits are kept.

Output: dataset/processed/rdd2022_extended/ with
  images|labels/{train,val,test,test_rdd_new,test_external}
  data.yaml                (train/val = everything, test = ORIGINAL test split)
  data_test_rdd_new.yaml   (test = Norway + China_Drone held-out images)
  data_test_external.yaml  (test = external pothole test images)
  dataset_stats.json, class_distribution.png
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import scan, validate  # noqa: E402

from rdd_yolo.constants import CLASS_CODES, CLASS_INDEX, DATASET_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

OUT = DATASET_DIR / "processed" / "rdd2022_extended"
EXT = DATASET_DIR / "raw" / "external"
MAX_SIDE = 1280
SEED = 42
NEW_COUNTRIES = ["China_Drone", "Norway"]
EXTERNAL = {
    # name: (folder with train/valid/test, class names in their data.yaml -> our class index or None to skip image)
    "egypt_rdd": (EXT / "egypt_rdd" / "RDD.v8i.yolov8", {"0": "skip_image", "1": CLASS_INDEX["D40"]}),
    "potholes_rf": (EXT / "potholes_rf", {"0": CLASS_INDEX["D40"]}),
}


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_image(src: Path, dst: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        link(src, dst)
    return w, h


def main() -> None:
    if not (PROCESSED_DIR / "data.yaml").exists():
        sys.exit("Run prepare_dataset.py first (original 5-country set).")
    if OUT.exists():
        shutil.rmtree(OUT)
    splits = ["train", "val", "test", "test_rdd_new", "test_external"]
    for s in splits:
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
    stats: dict = {"sources": {}, "splits": {s: Counter() for s in splits}, "images": Counter(), "skipped": Counter()}

    def add_labels(split: str, lines: list[str], source: str) -> None:
        stats["images"][split] += 1
        for ln in lines:
            stats["splits"][split][CLASS_CODES[int(ln.split()[0])]] += 1
        stats["sources"].setdefault(source, Counter())[split] += 1

    # 1. original set, unchanged
    for s in ("train", "val", "test"):
        for img in (PROCESSED_DIR / "images" / s).iterdir():
            lbl = PROCESSED_DIR / "labels" / s / f"{img.stem}.txt"
            link(img, OUT / "images" / s / img.name)
            link(lbl, OUT / "labels" / s / lbl.name)
            add_labels(s, [x for x in lbl.read_text().splitlines() if x.strip()], "rdd2022_original5")
    print("original set linked", dict(stats["images"]))

    # 2. new RDD2022 countries
    report: dict = {"countries": {}, "issues": []}
    for country in NEW_COUNTRIES:
        if not (RAW_DIR / country / ".extracted").exists():
            print(f"[skip] {country} not downloaded")
            continue
        samples = validate(scan(RAW_DIR, [country], report), report)
        rng = random.Random(f"{SEED}:{country}")
        pos = [x for x in samples if x.boxes]
        neg = [x for x in samples if not x.boxes]
        rng.shuffle(pos)
        rng.shuffle(neg)
        chosen = pos + neg[: int(len(pos) * 0.1)]
        rng.shuffle(chosen)
        n = len(chosen)
        parts = {"train": chosen[: int(n * 0.8)], "val": chosen[int(n * 0.8): int(n * 0.9)],
                 "test_rdd_new": chosen[int(n * 0.9):]}
        for split, items in parts.items():
            for x in items:
                stem = f"{country}_{x.image.stem}"
                if write_image(x.image, OUT / "images" / split / f"{stem}{x.image.suffix.lower()}") is None:
                    stats["skipped"][f"{country}: unreadable"] += 1
                    continue
                lines = [f"{CLASS_INDEX[b.cls]} {(b.xmin + b.xmax) / 2 / x.width:.6f} {(b.ymin + b.ymax) / 2 / x.height:.6f} "
                         f"{(b.xmax - b.xmin) / x.width:.6f} {(b.ymax - b.ymin) / x.height:.6f}" for b in x.boxes]
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                add_labels(split, lines, f"rdd2022_{country}")
        print(f"{country}: {n} images", {k: len(v) for k, v in parts.items()})
    stats["new_rdd_validation"] = {"issues": len(report["issues"]), "ignored_labels": report.get("ignored_labels")}

    # 3. external pothole datasets
    for name, (root, mapping) in EXTERNAL.items():
        if not root.exists():
            print(f"[skip] {name} not downloaded")
            continue
        for their, ours in (("train", "train"), ("valid", "val"), ("test", "test_external")):
            img_dir = root / their / "images"
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.iterdir()):
                lbl = root / their / "labels" / f"{img.stem}.txt"
                if not lbl.exists():
                    stats["skipped"][f"{name}: no label file"] += 1
                    continue
                out_lines, skip = [], False
                for ln in lbl.read_text().splitlines():
                    p = ln.split()
                    if len(p) != 5:  # polygons / malformed -> skip line
                        stats["skipped"][f"{name}: non-box line"] += 1
                        continue
                    m = mapping.get(p[0])
                    if m == "skip_image":
                        skip = True
                        break
                    vals = [float(v) for v in p[1:]]
                    if m is None or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                        stats["skipped"][f"{name}: invalid box"] += 1
                        continue
                    out_lines.append(f"{m} {' '.join(f'{v:.6f}' for v in vals)}")
                if skip:
                    stats["skipped"][f"{name}: image has generic Crack label"] += 1
                    continue
                if not out_lines:
                    stats["skipped"][f"{name}: no pothole box"] += 1
                    continue
                if cv2.imread(str(img)) is None:
                    stats["skipped"][f"{name}: unreadable image"] += 1
                    continue
                stem = f"{name}_{img.stem}"[:150]
                link(img, OUT / "images" / ours / f"{stem}{img.suffix.lower()}")
                (OUT / "labels" / ours / f"{stem}.txt").write_text("\n".join(out_lines) + "\n")
                add_labels(ours, out_lines, name)
        print(f"{name} added", dict(stats["sources"].get(name, {})))

    base = f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    names = f"nc: {len(CLASS_CODES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_CODES))
    (OUT / "data.yaml").write_text("# Extended set (Experiment C). test = ORIGINAL RDD2022 5-country test split\n"
                                   + base + "test: images/test\n" + names)
    (OUT / "data_test_rdd_new.yaml").write_text(base + "test: images/test_rdd_new\n" + names)
    (OUT / "data_test_external.yaml").write_text(base + "test: images/test_external\n" + names)

    out_stats = {
        "images": dict(stats["images"]),
        "instances_per_class": {s: {c: stats["splits"][s].get(c, 0) for c in CLASS_CODES} for s in splits},
        "images_per_source": {k: dict(v) for k, v in stats["sources"].items()},
        "skipped": dict(stats["skipped"]),
        "new_rdd_validation": stats["new_rdd_validation"],
        "config": {"max_side": MAX_SIDE, "seed": SEED, "new_countries": NEW_COUNTRIES, "external": list(EXTERNAL)},
    }
    (OUT / "dataset_stats.json").write_text(json.dumps(out_stats, indent=2))
    print(json.dumps(out_stats, indent=2))


if __name__ == "__main__":
    main()

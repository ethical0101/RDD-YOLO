"""Round 2 of the extended dataset: add close-up / ground-level pothole datasets with de-duplication.

Starts from dataset/processed/rdd2022_extended (round 1, built by build_extended.py) and writes
dataset/processed/rdd2022_extended_v2:

  * New sources (licenses stated in each dataset README; see download_external.py):
      manot_road_damage (CC BY 4.0, COCO), manot_pothole2 (CC BY 4.0, COCO), keremberke_pothole (CC BY 4.0, COCO),
      rupesh_pothole2 (MIT, YOLO), rupesh_pothole (MIT, YOLO), mwpd (Apache-2.0, YOLO)
    All of them annotate only potholes -> mapped to D40.
  * Their own test splits form a NEW held-out set: images/test_closeup.
  * Leakage control: a 64-bit difference hash (dHash) of every image (and of its mirror image, because Roboflow
    exports contain flipped augmentations) is compared with every held-out test image of ALL test sets.
    Training/validation images within Hamming distance <= HAMMING of any test image are dropped - this also
    removes round-1 external training images that duplicate a new test image. Near-duplicates among the new
    training images themselves are kept only once.
  * The original RDD2022 test split, test_rdd_new and test_external are carried over unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdd_yolo.constants import CLASS_CODES, CLASS_INDEX, DATASET_DIR  # noqa: E402

V1 = DATASET_DIR / "processed" / "rdd2022_extended"
OUT = DATASET_DIR / "processed" / "rdd2022_extended_v2"
EXT = DATASET_DIR / "raw" / "external"
HAMMING = 6
D40 = CLASS_INDEX["D40"]
POTHOLE_NAMES = {"pothole", "potholes"}
IGNORE_NAMES = {"object"}  # Roboflow's root super-category, carries no boxes of its own

SOURCES = {
    "manot_road_damage": ("coco", EXT / "manot_road_damage"),
    "manot_pothole2": ("coco", EXT / "manot_pothole2"),
    "keremberke_pothole": ("coco", EXT / "keremberke_pothole"),
    "rupesh_pothole2": ("yolo", EXT / "rupesh_pothole2" / "pothole_yolo_dataset"),
    "rupesh_pothole": ("yolo", EXT / "rupesh_pothole" / "pothole.v1i.yolov4pytorch"),
    "mwpd": ("yolo", EXT / "mwpd" / "MWPD"),
}
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test_closeup"}


def dhash(img: np.ndarray) -> tuple[int, int]:
    """(hash, hash of the horizontally flipped image)."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    gf = g[:, ::-1]
    bits_f = (gf[:, 1:] > gf[:, :-1]).flatten()
    to_int = lambda b: int("".join("1" if x else "0" for x in b), 2)  # noqa: E731
    return to_int(bits), to_int(bits_f)


class HashIndex:
    def __init__(self) -> None:
        self.hashes: list[int] = []

    def add(self, h: int) -> None:
        self.hashes.append(h)

    def near(self, hs: tuple[int, int]) -> bool:
        if not self.hashes:
            return False
        arr = np.array(self.hashes, dtype=np.uint64)
        for h in hs:
            x = np.bitwise_xor(arr, np.uint64(h))
            # popcount via byte view
            pc = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
            if (pc <= HAMMING).any():
                return True
        return False


def iter_coco(root: Path):
    for split in ("train", "valid", "test"):
        jf = root / split / "_annotations.coco.json"
        if not jf.exists():
            continue
        j = json.loads(jf.read_text())
        cats = {c["id"]: c["name"].strip().lower() for c in j["categories"]}
        anns: dict[int, list] = {}
        for a in j["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in j["images"]:
            w, h = im["width"], im["height"]
            boxes, bad = [], False
            for a in anns.get(im["id"], []):
                name = cats.get(a["category_id"], "")
                if name in IGNORE_NAMES:
                    continue
                if name not in POTHOLE_NAMES:
                    bad = True  # unknown damage label -> skip image rather than create false negatives
                    break
                x, y, bw, bh = a.get("bbox") or [0, 0, 0, 0]
                if (bw <= 1 or bh <= 1) and a.get("segmentation"):
                    pts = np.array(a["segmentation"][0], dtype=float).reshape(-1, 2)
                    x, y = pts.min(0)
                    bw, bh = pts.max(0) - pts.min(0)
                if bw <= 1 or bh <= 1:
                    continue
                boxes.append((D40, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
            yield split, root / split / im["file_name"], boxes, bad


def iter_yolo(root: Path):
    for split in ("train", "valid", "test"):
        idir = root / split / "images"
        if not idir.exists():
            continue
        for img in sorted(idir.iterdir()):
            lbl = root / split / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                yield split, img, None, False
                continue
            boxes = []
            for ln in lbl.read_text().split("\n"):
                p = ln.split()
                if not p:
                    continue
                if p[0] != "0":
                    boxes = None
                    break
                v = [float(t) for t in p[1:]]
                if len(v) > 4:  # polygon -> bbox
                    pts = np.array(v[: len(v) // 2 * 2]).reshape(-1, 2)
                    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
                    v = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                boxes.append((D40, *v))
            yield split, img, boxes, boxes is None


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    if not (V1 / "data.yaml").exists():
        sys.exit("Build round 1 first: python dataset/build_extended.py")
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(V1, OUT, copy_function=link)
    (OUT / "images" / "test_closeup").mkdir()
    (OUT / "labels" / "test_closeup").mkdir()
    stats = {"added": Counter(), "skipped": Counter(), "removed_round1_external": 0}

    # 1. hashes of every held-out image already present
    test_idx = HashIndex()
    for t in ("test", "test_rdd_new", "test_external"):
        for p in (OUT / "images" / t).iterdir():
            img = cv2.imread(str(p))
            if img is not None:
                test_idx.add(dhash(img)[0])
    print("indexed test images:", len(test_idx.hashes))

    # 2. collect new samples
    samples = []
    for name, (kind, root) in SOURCES.items():
        if not root.exists():
            print(f"[skip] {name}: not downloaded")
            continue
        it = iter_coco(root) if kind == "coco" else iter_yolo(root)
        for split, img_path, boxes, bad in it:
            samples.append((name, SPLIT_MAP[split], img_path, boxes, bad))

    def write(name, split, img_path, img, boxes):
        stem = f"{name}_{img_path.stem}"[:150]
        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]) \
            if img_path.suffix.lower() not in (".jpg", ".jpeg") else link(img_path, OUT / "images" / split / f"{stem}.jpg")
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n")
        stats["added"][f"{name}/{split}"] += 1

    def usable(name, img_path, boxes, bad):
        if bad:
            stats["skipped"][f"{name}: non-pothole label"] += 1
            return None
        if not boxes:
            stats["skipped"][f"{name}: no boxes"] += 1
            return None
        boxes = [(c, min(max(x, 0), 1), min(max(y, 0), 1), min(w, 1), min(h, 1)) for c, x, y, w, h in boxes if w > 0 and h > 0]
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"][f"{name}: unreadable"] += 1
            return None
        return img, boxes

    # 3. new test split first (deduplicated against existing tests and within itself)
    for name, split, img_path, boxes, bad in samples:
        if split != "test_closeup":
            continue
        r = usable(name, img_path, boxes, bad)
        if r is None:
            continue
        hs = dhash(r[0])
        if test_idx.near(hs):
            stats["skipped"][f"{name}: test image duplicates another test image"] += 1
            continue
        test_idx.add(hs[0])
        write(name, split, img_path, r[0], r[1])

    # 4. drop round-1 external train/val images that duplicate any test image
    for split in ("train", "val"):
        for p in list((OUT / "images" / split).iterdir()):
            if not p.name.startswith(("egypt_rdd_", "potholes_rf_")):
                continue
            img = cv2.imread(str(p))
            if img is not None and test_idx.near(dhash(img)):
                p.unlink()
                (OUT / "labels" / split / f"{p.stem}.txt").unlink(missing_ok=True)
                stats["removed_round1_external"] += 1

    # 5. new train/val, deduplicated against tests and against each other
    seen = HashIndex()
    for name, split, img_path, boxes, bad in samples:
        if split == "test_closeup":
            continue
        r = usable(name, img_path, boxes, bad)
        if r is None:
            continue
        hs = dhash(r[0])
        if test_idx.near(hs):
            stats["skipped"][f"{name}: duplicates a test image"] += 1
            continue
        if seen.near(hs):
            stats["skipped"][f"{name}: duplicate of an already added training image"] += 1
            continue
        seen.add(hs[0])
        write(name, split, img_path, r[0], r[1])

    base = f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    names = f"nc: {len(CLASS_CODES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_CODES))
    (OUT / "data.yaml").write_text("# Extended set v2. test = ORIGINAL RDD2022 5-country test split\n" + base
                                   + "test: images/test\n" + names)
    for t in ("test_rdd_new", "test_external", "test_closeup"):
        (OUT / f"data_{t}.yaml").write_text(base + f"test: images/{t}\n" + names)

    counts = {s: len(list((OUT / "images" / s).iterdir())) for s in ("train", "val", "test", "test_rdd_new", "test_external", "test_closeup")}
    inst = {}
    for s in counts:
        c = Counter(ln.split()[0] for f in (OUT / "labels" / s).glob("*.txt") for ln in f.read_text().splitlines() if ln.strip())
        inst[s] = {CLASS_CODES[int(k)]: v for k, v in sorted(c.items())}
    out = {"images": counts, "instances_per_class": inst, "added": dict(stats["added"]), "skipped": dict(stats["skipped"]),
           "removed_round1_external_duplicates": stats["removed_round1_external"], "hamming_threshold": HAMMING}
    (OUT / "dataset_stats_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

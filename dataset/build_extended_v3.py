"""Round 3 of the extended dataset: add the non-RDD sources of the Deeksha9 "Pavement Distress Detection"
aggregate (MIT license; see download_deeksha.py) on top of rdd2022_extended_v2.

Output: dataset/processed/rdd2022_extended_v3
  * All earlier splits are carried over unchanged (hard links).
  * Class mapping (their -> ours): Longitudinal Crack->D00, Transverse Crack->D10, Alligator Crack->D20,
    Pothole->D40; "Patch" boxes are dropped (repairs, like RDD2022's "Repair"); images containing
    "Block Crack" are skipped (no RDD class; it would become a false negative).
  * Their test split becomes a new held-out set images/test_pavement.
  * De-duplication (64-bit dHash incl. mirrored image, Hamming <= 6): new images that match any held-out
    test image (all test sets) are dropped from train/val; new images that duplicate an already present
    external training image or each other are kept once.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_extended_v2 import HAMMING, dhash  # noqa: E402

from rdd_yolo.constants import CLASS_CODES, DATASET_DIR  # noqa: E402

V2 = DATASET_DIR / "processed" / "rdd2022_extended_v2"
OUT = DATASET_DIR / "processed" / "rdd2022_extended_v3"
SRC = DATASET_DIR / "raw" / "external" / "deeksha" / "split"
CLASS_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": "drop", "5": "skip_image"}
SPLIT_MAP = {"train": "train", "val": "val", "test": "test_pavement"}
EXTERNAL_PREFIXES = ("egypt_rdd_", "potholes_rf_", "manot_", "keremberke_", "rupesh_", "mwpd_")


class FastHashIndex:
    """Growable uint64 array with vectorised Hamming search."""

    def __init__(self, cap: int = 1024) -> None:
        self.arr = np.zeros(cap, dtype=np.uint64)
        self.n = 0

    def add(self, h: int) -> None:
        if self.n == len(self.arr):
            self.arr = np.concatenate([self.arr, np.zeros(len(self.arr), dtype=np.uint64)])
        self.arr[self.n] = h
        self.n += 1

    def near(self, hs: tuple[int, int]) -> bool:
        if self.n == 0:
            return False
        a = self.arr[: self.n]
        for h in hs:
            x = np.bitwise_xor(a, np.uint64(h))
            if (np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1) <= HAMMING).any():
                return True
        return False


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    if not (V2 / "data.yaml").exists():
        sys.exit("Build round 2 first: python dataset/build_extended_v2.py")
    if not (SRC.parent / ".complete").exists():
        sys.exit("Deeksha download incomplete: python dataset/download_deeksha.py")
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(V2, OUT, copy_function=link)
    for d in ("images", "labels"):
        (OUT / d / "test_pavement").mkdir()
    stats = {"added": Counter(), "skipped": Counter()}

    test_idx, seen = FastHashIndex(), FastHashIndex()
    for t in ("test", "test_rdd_new", "test_external", "test_closeup"):
        for p in (OUT / "images" / t).iterdir():
            im = cv2.imread(str(p))
            if im is not None:
                test_idx.add(dhash(im)[0])
    for s in ("train", "val"):
        for p in (OUT / "images" / s).iterdir():
            if p.name.startswith(EXTERNAL_PREFIXES):
                im = cv2.imread(str(p))
                if im is not None:
                    seen.add(dhash(im)[0])
    print(f"indexed {test_idx.n} test and {seen.n} existing external training images", flush=True)

    def load(split: str):
        idir, ldir = SRC / split / "images", SRC / split / "labels"
        for img in sorted(idir.iterdir()) if idir.exists() else []:
            src_name = img.name.split("_")[0]
            lbl = ldir / f"{img.stem}.txt"
            if not lbl.exists():
                stats["skipped"][f"{src_name}: no label"] += 1
                continue
            boxes, skip = [], False
            for ln in lbl.read_text().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                m = CLASS_MAP.get(p[0])
                if m == "skip_image":
                    skip = True
                    break
                if m in (None, "drop"):
                    continue
                v = [min(max(float(x), 0.0), 1.0) for x in p[1:]]
                if v[2] > 0 and v[3] > 0:
                    boxes.append((m, *v))
            if skip:
                stats["skipped"][f"{src_name}: block crack"] += 1
                continue
            if not boxes:
                stats["skipped"][f"{src_name}: no target boxes"] += 1
                continue
            yield src_name, img, boxes

    def put(split: str, src_name: str, img: Path, boxes) -> None:
        stem = f"deeksha_{img.stem}"[:150]
        link(img, OUT / "images" / split / f"{stem}{img.suffix.lower()}")
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n")
        stats["added"][f"{src_name}/{split}"] += 1

    # new held-out set first
    for src_name, img, boxes in load("test"):
        im = cv2.imread(str(img))
        if im is None:
            continue
        hs = dhash(im)
        if test_idx.near(hs):
            stats["skipped"][f"{src_name}: test duplicates another test image"] += 1
            continue
        test_idx.add(hs[0])
        put("test_pavement", src_name, img, boxes)
    print("test_pavement:", sum(v for k, v in stats["added"].items() if k.endswith("test_pavement")), flush=True)

    for their in ("train", "val"):
        for k, (src_name, img, boxes) in enumerate(load(their), 1):
            im = cv2.imread(str(img))
            if im is None:
                stats["skipped"][f"{src_name}: unreadable"] += 1
                continue
            hs = dhash(im)
            if test_idx.near(hs):
                stats["skipped"][f"{src_name}: duplicates a test image"] += 1
                continue
            if seen.near(hs):
                stats["skipped"][f"{src_name}: duplicate training image"] += 1
                continue
            seen.add(hs[0])
            put(SPLIT_MAP[their], src_name, img, boxes)
            if k % 5000 == 0:
                print(f"  {their}: {k} processed", flush=True)

    base = f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    names = f"nc: {len(CLASS_CODES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_CODES))
    (OUT / "data.yaml").write_text("# Extended set v3. test = ORIGINAL RDD2022 5-country test split\n" + base
                                   + "test: images/test\n" + names)
    for t in ("test_rdd_new", "test_external", "test_closeup", "test_pavement"):
        (OUT / f"data_{t}.yaml").write_text(base + f"test: images/{t}\n" + names)
    splits = ("train", "val", "test", "test_rdd_new", "test_external", "test_closeup", "test_pavement")
    counts = {s: len(list((OUT / "images" / s).iterdir())) for s in splits}
    inst = {}
    for s in splits:
        c = Counter(ln.split()[0] for f in (OUT / "labels" / s).glob("*.txt") for ln in f.read_text().splitlines() if ln.strip())
        inst[s] = {CLASS_CODES[int(k)]: v for k, v in sorted(c.items())}
    out = {"images": counts, "instances_per_class": inst, "added": dict(stats["added"]),
           "skipped": dict(stats["skipped"]), "hamming_threshold": HAMMING}
    (OUT / "dataset_stats_v3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

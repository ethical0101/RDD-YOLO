"""Evidence-based accuracy diagnosis for the RDD-YOLO models.

Reads the real datasets, training logs and the served checkpoint and writes
  docs/diagnostics/diagnosis.json      (all numbers)
  docs/diagnostics/examples/*.jpg      (false-negative / false-positive examples)
Nothing here is estimated by hand: every figure comes from files or model runs.

Usage: .venv\\Scripts\\python scripts\\diagnose_accuracy.py [--weights models/weights/yolo26s_full_best.pt]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdd_yolo.constants import CLASS_CODES  # noqa: E402

DS = ROOT / "dataset" / "processed" / "rdd2022_yolo"   # original 5-country split (Exp A/B, primary test)
OUT = ROOT / "docs" / "diagnostics"
RUNS = ["baseline_yolo26n", "rdd_yolo26n", "rdd_yolo26n_ext", "rdd_yolo26n_ext2", "yolo26s_full"]


def read_labels(split: str) -> dict[str, list[tuple[int, float, float, float, float]]]:
    out = {}
    for f in sorted((DS / "labels" / split).glob("*.txt")):
        out[f.stem] = [(int(p[0]), *map(float, p[1:5])) for p in (ln.split() for ln in f.read_text().splitlines()) if len(p) == 5]
    return out


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix, iy = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = a[3] * a[4] + b[3] * b[4] - inter
    return inter / u if u > 0 else 0.0


def dataset_stats() -> dict:
    res = {}
    for split in ("train", "val", "test"):
        lab = read_labels(split)
        cls = Counter(b[0] for v in lab.values() for b in v)
        areas = defaultdict(list)
        dup = tiny = 0
        for v in lab.values():
            for i, b in enumerate(v):
                areas[b[0]].append(b[3] * b[4])
                if b[3] * 640 < 8 or b[4] * 640 < 8:
                    tiny += 1
                for c in v[i + 1:]:
                    if c[0] == b[0] and iou_xywh(b, c) > 0.9:
                        dup += 1
        res[split] = {
            "images": len(lab), "background_images": sum(1 for v in lab.values() if not v),
            "boxes": sum(cls.values()), "boxes_per_class": {CLASS_CODES[k]: cls.get(k, 0) for k in range(4)},
            "near_duplicate_boxes_iou_gt_0.9": dup, "tiny_boxes_lt_8px_at_640": tiny,
            "median_rel_area_per_class": {CLASS_CODES[k]: float(np.median(areas[k])) if areas[k] else None for k in range(4)},
            "share_small_lt_1pct_area": {CLASS_CODES[k]: float(np.mean(np.array(areas[k]) < 0.01)) if areas[k] else None for k in range(4)},
        }
    return res


def dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)


def leakage() -> dict:
    """Near-duplicate test images whose closest training image is within a small Hamming distance."""
    hs = {}
    for split in ("train", "test"):
        files = sorted((DS / "images" / split).iterdir())
        hs[split] = (files, np.array([dhash(f) for f in files], dtype=np.uint64))
    tr_files, tr = hs["train"]
    te_files, te = hs["test"]
    mins = np.empty(len(te), dtype=np.int64)
    for i, h in enumerate(te):
        mins[i] = popcount64(np.bitwise_xor(tr, h)).min()
    by_country = defaultdict(lambda: [0, 0])
    for f, m in zip(te_files, mins):
        c = f.name.split("_")[0] if not f.name.startswith(("China", "United")) else "_".join(f.name.split("_")[:2])
        by_country[c][1] += 1
        if m <= 6:
            by_country[c][0] += 1
    hist = Counter(int(min(m, 20)) for m in mins)
    return {
        "test_images": int(len(te)), "near_dup_hamming_le_6": int((mins <= 6).sum()),
        "near_dup_hamming_le_10": int((mins <= 10).sum()), "exact_hash_match": int((mins == 0).sum()),
        "per_country_le_6": {k: {"near_dup": v[0], "test_images": v[1]} for k, v in sorted(by_country.items())},
        "min_hamming_histogram_capped_20": {str(k): v for k, v in sorted(hist.items())},
        "_mins": mins.tolist(), "_test_files": [f.name for f in te_files],
    }


def curves() -> dict:
    out = {}
    for r in RUNS:
        p = ROOT / "experiments" / r / "results.csv"
        if not p.exists():
            continue
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(p.open())]
        m = [x["metrics/mAP50(B)"] for x in rows]
        tl = [x["train/box_loss"] + x["train/cls_loss"] for x in rows]
        vl = [x["val/box_loss"] + x["val/cls_loss"] for x in rows]
        best = int(np.argmax(m))
        out[r] = {
            "epochs": len(rows), "best_epoch": best + 1, "best_val_mAP50": m[best], "last_val_mAP50": m[-1],
            "val_mAP50_gain_last_5_epochs": m[-1] - m[max(0, len(m) - 6)],
            "train_loss_first_last": [tl[0], tl[-1]], "val_loss_first_last": [vl[0], vl[-1]],
            "val_loss_min_epoch": int(np.argmin(vl)) + 1,
            "val_minus_train_loss_last": vl[-1] - tl[-1],
        }
    return out


def error_analysis(weights: Path, conf: float, leak_mins: dict) -> dict:
    from rdd_yolo.modules import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    lab = read_labels("test")
    files = sorted((DS / "images" / "test").iterdir())
    K = 4
    cm = np.zeros((K + 1, K + 1), dtype=int)  # rows = predicted (last = background), cols = true (last = background)
    tp = Counter(); fp = Counter(); fn = Counter()
    fn_size = defaultdict(lambda: [0, 0])  # size bin -> [found, total]
    examples_fn, examples_fp = [], []
    near_dup_tp = [0, 0]; clean_tp = [0, 0]  # recall on near-duplicate vs clean test images
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        for f, r in zip(batch, model.predict([str(x) for x in batch], conf=conf, iou=0.7, verbose=False, device=0)):
            gt = lab.get(f.stem, [])
            h, w = r.orig_shape
            preds = [(int(c), *((b[0] / w + b[2] / w) / 2, (b[1] / h + b[3] / h) / 2, (b[2] - b[0]) / w, (b[3] - b[1]) / h), float(s))
                     for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            preds.sort(key=lambda x: -x[5])
            used = set()
            for p in preds:
                best, bj = 0.0, -1
                for j, g in enumerate(gt):
                    if j in used:
                        continue
                    o = iou_xywh(p, g)
                    if o > best:
                        best, bj = o, j
                if best >= 0.5:
                    used.add(bj)
                    cm[p[0], gt[bj][0]] += 1
                    if p[0] == gt[bj][0]:
                        tp[p[0]] += 1
                    else:
                        fp[p[0]] += 1; fn[gt[bj][0]] += 1
                else:
                    cm[p[0], K] += 1
                    fp[p[0]] += 1
                    if len(examples_fp) < 8 and p[5] > 0.5:
                        examples_fp.append((f, p))
            nd = leak_mins.get(f.name, 99) <= 6
            for j, g in enumerate(gt):
                a = g[3] * g[4]
                bin_ = "small <1%" if a < 0.01 else "medium 1-5%" if a < 0.05 else "large >=5%"
                found = j in used and any(cm is not None for _ in [0])
                matched_same = j in used
                fn_size[bin_][1] += 1
                (near_dup_tp if nd else clean_tp)[1] += 1
                if matched_same:
                    fn_size[bin_][0] += 1
                    (near_dup_tp if nd else clean_tp)[0] += 1
                else:
                    cm[K, g[0]] += 1
                    fn[g[0]] += 1
                    if len(examples_fn) < 8 and a > 0.002:
                        examples_fn.append((f, g))
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    for kind, ex in (("fn", examples_fn), ("fp", examples_fp)):
        for n, (f, b) in enumerate(ex):
            img = cv2.imread(str(f))
            H, W = img.shape[:2]
            x1, y1, x2, y2 = int((b[1] - b[3] / 2) * W), int((b[2] - b[4] / 2) * H), int((b[1] + b[3] / 2) * W), int((b[2] + b[4] / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if kind == "fn" else (255, 0, 0), 3)
            cv2.putText(img, f"{kind.upper()} {CLASS_CODES[b[0]]}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(OUT / "examples" / f"{kind}_{n}_{f.stem}.jpg"), img)
    names = CLASS_CODES + ["background"]
    return {
        "weights": weights.name, "conf": conf, "iou_match": 0.5,
        "per_class": {CLASS_CODES[k]: {"tp": tp[k], "fp": fp[k], "fn": fn[k],
                                       "precision": tp[k] / max(1, tp[k] + fp[k]), "recall": tp[k] / max(1, tp[k] + fn[k])}
                      for k in range(K)},
        "confusion_matrix": {"rows_predicted": names, "cols_true": names, "matrix": cm.tolist()},
        "recall_by_object_size": {k: {"found": v[0], "total": v[1], "recall": v[0] / max(1, v[1])} for k, v in fn_size.items()},
        "recall_near_duplicate_vs_clean_test_images": {
            "near_duplicate": {"found": near_dup_tp[0], "total": near_dup_tp[1], "recall": near_dup_tp[0] / max(1, near_dup_tp[1])},
            "clean": {"found": clean_tp[0], "total": clean_tp[1], "recall": clean_tp[0] / max(1, clean_tp[1])}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "weights" / "yolo26s_full_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("dataset stats ..."); ds = dataset_stats()
    print("leakage (dHash) ..."); lk = leakage()
    leak_map = dict(zip(lk.pop("_test_files"), lk.pop("_mins")))
    print("curves ..."); cv = curves()
    print("error analysis ..."); ea = error_analysis(args.weights, args.conf, leak_map)
    res = {"dataset": ds, "leakage_train_vs_test": lk, "training_curves": cv, "error_analysis": ea}
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("leakage_train_vs_test", "training_curves")}, indent=1)[:6000])
    print(json.dumps(ea, indent=1)[:4000])


if __name__ == "__main__":
    main()

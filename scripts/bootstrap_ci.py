"""Bootstrap 95 % confidence intervals for mAP@50 and paired differences between models.

Each model predicts the split once (conf 0.001, NMS IoU 0.7, max 300 boxes, as in validation). Predictions are
matched to ground truth per image and class (greedy by confidence, IoU >= 0.5), and AP uses 101-point interpolation.
Images are then resampled with replacement (the same resamples for every model), so the interval of a difference
between two models is paired. The full-sample mAP@50 can differ from Ultralytics' value by a few tenths of a point
because Ultralytics matches boxes slightly differently; the intervals are meant for comparing models with each other.

Usage:
  .venv\\Scripts\\python scripts\\bootstrap_ci.py --data dataset/processed/rdd2022_extended_v3/data.yaml --split test \\
      --models E=models/weights/yolo26s_full_best.pt F1=models/weights/yolo26s_ft640_best.pt --tag test
Output: docs/diagnostics/bootstrap/<tag>.json (+ cached predictions under .cache/bootstrap/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdd_yolo.modules import register_custom_modules  # noqa: E402  (RDD-YOLO checkpoints reference SimAM etc.)

CACHE = ROOT / ".cache" / "bootstrap"
OUT = ROOT / "docs" / "diagnostics" / "bootstrap"
NC = 4


def split_images(data: Path, split: str) -> tuple[list[Path], Path]:
    cfg = yaml.safe_load(data.read_text())
    img_dir = Path(cfg["path"]) / cfg[split]
    lbl_dir = Path(cfg["path"]) / cfg[split].replace("images", "labels", 1)
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), lbl_dir


def read_gt(lbl: Path, w: int, h: int) -> np.ndarray:
    rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
    out = np.zeros((len(rows), 5))
    for i, (c, cx, cy, bw, bh) in enumerate(rows):
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out[i] = [int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def predict(weights: Path, images: list[Path], lbl_dir: Path, imgsz: int, device: str) -> dict:
    """Per prediction: image index, class, confidence, TP flag. Per image: GT count per class. Cached."""
    key = CACHE / f"{weights.stem}_{lbl_dir.name}_{imgsz}_{len(images)}.npz"
    if key.exists():
        z = np.load(key)
        return {k: z[k] for k in z.files}
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(str(weights))
    img_idx, cls, conf, tp, ngt = [], [], [], [], np.zeros((len(images), NC), dtype=np.int32)
    for i, p in enumerate(images):
        r = model.predict(str(p), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device=device, verbose=False)[0]
        h, w = r.orig_shape
        gt = read_gt(lbl_dir / f"{p.stem}.txt", w, h)
        for c in range(NC):
            ngt[i, c] = int((gt[:, 0] == c).sum()) if len(gt) else 0
        b = r.boxes
        if b is None or not len(b):
            continue
        pb, pc, ps = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        flags = np.zeros(len(pb), dtype=bool)
        for c in range(NC):
            pi = np.where(pc == c)[0]
            g = gt[gt[:, 0] == c, 1:] if len(gt) else np.zeros((0, 4))
            if not len(pi) or not len(g):
                continue
            pi = pi[np.argsort(-ps[pi])]
            m = iou_matrix(pb[pi], g)
            used = np.zeros(len(g), dtype=bool)
            for k, row in enumerate(m):
                row = np.where(used, -1, row)
                j = int(row.argmax())
                if row[j] >= 0.5:
                    used[j] = True
                    flags[pi[k]] = True
        img_idx += [i] * len(pb)
        cls += pc.tolist()
        conf += ps.tolist()
        tp += flags.tolist()
        if (i + 1) % 200 == 0:
            print(f"  {weights.stem}: {i + 1}/{len(images)}", flush=True)
    d = {"img": np.array(img_idx, dtype=np.int32), "cls": np.array(cls, dtype=np.int8), "conf": np.array(conf, dtype=np.float32),
         "tp": np.array(tp, dtype=bool), "ngt": ngt}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(key, **d)
    return d


def ap101(tp_w: np.ndarray, fp_w: np.ndarray, n_gt: float) -> float:
    if n_gt <= 0:
        return np.nan
    ctp, cfp = np.cumsum(tp_w), np.cumsum(fp_w)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], prec, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def map50(d: dict, weights: np.ndarray) -> tuple[float, list[float]]:
    """mAP@50 with per-image weights (bootstrap counts)."""
    aps = []
    for c in range(NC):
        sel = d["order"][c]
        w = weights[d["img"][sel]]
        aps.append(ap101(w * d["tp"][sel], w * ~d["tp"][sel], float((weights[:, None] * d["ngt"])[:, c].sum())))
    return float(np.nanmean(aps)), aps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True, help="NAME=weights.pt")
    ap.add_argument("--imgsz", nargs="*", type=int, default=[], help="per-model image size (default 640)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    args = ap.parse_args()

    images, lbl_dir = split_images(args.data, args.split)
    if args.limit:
        images = images[: args.limit]
    names, preds = [], {}
    for k, spec in enumerate(args.models):
        name, w = spec.split("=", 1)
        size = args.imgsz[k] if k < len(args.imgsz) else 640
        d = predict(Path(w), images, lbl_dir, size, args.device)
        d["order"] = [np.where(d["cls"] == c)[0][np.argsort(-d["conf"][d["cls"] == c])] for c in range(NC)]
        names.append(name)
        preds[name] = d
    rng = np.random.default_rng(0)
    n_img = len(images)
    full = {m: map50(preds[m], np.ones(n_img)) for m in names}
    boots = {m: [] for m in names}
    for _ in range(args.n):
        wts = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        for m in names:
            boots[m].append(map50(preds[m], wts)[0])
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]  # noqa: E731
    res = {"data": str(args.data), "split": args.split, "images": n_img, "bootstrap_resamples": args.n, "iou": 0.5,
           "models": {m: {"mAP50": round(full[m][0], 4), "ci95": ci(boots[m]),
                          "per_class_AP50": dict(zip(["D00", "D10", "D20", "D40"], [round(a, 4) for a in full[m][1]]))}
                      for m in names},
           "paired_differences": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array(boots[b]) - np.array(boots[a])
            res["paired_differences"][f"{b} - {a}"] = {"delta_mAP50": round(full[b][0] - full[a][0], 4), "ci95": ci(diff),
                                                       "p_b_better": round(float((diff > 0).mean()), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    json.dump(res, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

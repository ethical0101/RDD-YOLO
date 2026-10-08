"""Evaluate a trained checkpoint on the held-out split and benchmark its speed.

Outputs (in experiments/<run>/eval_<split>/):
    metrics.json            overall + per-class P / R / F1 / mAP50 / mAP50-95,
                            params, GFLOPs, file size, measured latency/FPS
    confusion_matrix*.png, *PR_curve.png, *F1_curve.png ... (Ultralytics plots)

Every number is produced by actually running the model; nothing is typed in.

Usage:
    .venv\\Scripts\\python training\\evaluate.py --weights experiments/rdd_yolo26n/weights/best.pt
    .venv\\Scripts\\python training\\evaluate.py --run rdd_yolo26n --split test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdd_yolo.constants import CLASS_CODES, CLASS_NAMES, EXPERIMENTS_DIR, PROCESSED_DIR  # noqa: E402
from rdd_yolo.modules import register_custom_modules  # noqa: E402


def benchmark_speed(model, images: list[Path], imgsz: int, device: str, n: int = 200, warmup: int = 20) -> dict:
    """Measure end-to-end single-image latency (pre + inference + post), batch=1."""
    import cv2

    frames = [cv2.imread(str(p)) for p in images[: max(n, 1)]]
    frames = [f for f in frames if f is not None]
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device=device, verbose=False)
    lat, inf = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, device=device, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        inf.append(r.speed["inference"])
    return {"images": len(frames), "batch": 1, "imgsz": imgsz, "device": device,
            "end_to_end_ms_mean": round(statistics.mean(lat), 2), "end_to_end_ms_median": round(statistics.median(lat), 2),
            "inference_ms_mean": round(statistics.mean(inf), 2),
            "fps_end_to_end": round(1000 / statistics.mean(lat), 1), "fps_inference_only": round(1000 / statistics.mean(inf), 1)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", type=Path)
    g.add_argument("--run", help="Experiment run name under experiments/")
    ap.add_argument("--data", type=Path, default=PROCESSED_DIR / "data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--speed-images", type=int, default=200)
    args = ap.parse_args()

    register_custom_modules()
    import torch
    from ultralytics import YOLO, __version__ as ul_version

    weights = args.weights or EXPERIMENTS_DIR / args.run / "weights" / "best.pt"
    if not weights.exists():
        sys.exit(f"Checkpoint not found: {weights}")
    run_dir = weights.parent.parent if weights.parent.name == "weights" else EXPERIMENTS_DIR / weights.stem
    out_dir = run_dir / f"eval_{args.split}"
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(str(weights), task="detect")
    print(f"Evaluating {weights} on '{args.split}' split ({device})")
    m = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch, device=device,
                  project=str(run_dir), name=f"eval_{args.split}", exist_ok=True, plots=True, verbose=True)

    box = m.box
    p, r = float(box.mp), float(box.mr)
    per_class = {}
    for i, k in enumerate(box.ap_class_index):
        code = CLASS_CODES[int(k)]
        pc, rc = float(box.p[i]), float(box.r[i])
        per_class[code] = {"name": CLASS_NAMES[code], "precision": pc, "recall": rc,
                           "f1": 2 * pc * rc / (pc + rc) if pc + rc else 0.0,
                           "mAP50": float(box.ap50[i]), "mAP50_95": float(box.ap[i])}
    nt = getattr(m, "nt_per_class", None)
    if nt is not None:
        for i, code in enumerate(CLASS_CODES):
            if code in per_class:
                per_class[code]["instances"] = int(nt[i])

    n_params = sum(x.numel() for x in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(float(get_flops(model.model, args.imgsz)), 2)
    except Exception:
        gflops = None
    img_dir = Path(args.data).parent / "images" / args.split
    imgs = sorted(img_dir.glob("*.*"))
    speed = benchmark_speed(model, imgs, args.imgsz, device, n=args.speed_images)

    result = {
        "weights": str(weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights),
        "split": args.split, "dataset": str(args.data), "images": len(imgs), "imgsz": args.imgsz,
        "ultralytics_version": ul_version, "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else "CPU",
        "overall": {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "mAP50": float(box.map50), "mAP50_95": float(box.map)},
        "per_class": per_class,
        "parameters": n_params, "gflops": gflops,
        "model_size_mb": round(weights.stat().st_size / 1024**2, 2),
        "ultralytics_val_speed_ms": {k: round(v, 3) for k, v in m.speed.items()},
        "speed_benchmark": speed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print(f"FPS (end-to-end, batch 1): {speed['fps_end_to_end']}")
    print(f"Saved -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

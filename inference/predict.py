"""Command-line inference for images, image folders and videos.

Writes annotated outputs and a detections.json to outputs/cli/<timestamp>/.
Does not touch the database (use the API/dashboard for stored, geolocated detections).

Usage:
    .venv\\Scripts\\python inference\\predict.py --source road.jpg
    .venv\\Scripts\\python inference\\predict.py --source images_dir --conf 0.3
    .venv\\Scripts\\python inference\\predict.py --source dashcam.mp4 --stride 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from rdd_yolo.constants import OUTPUTS_DIR, WEIGHTS_DIR  # noqa: E402
from rdd_yolo.detector import Detector, draw_detections, process_video  # noqa: E402
from rdd_yolo.geo import extract_exif_gps  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def default_weights() -> Path:
    for n in ("yolo26s_full_best.pt", "rdd_yolo26n_ext2_best.pt", "rdd_yolo26n_ext_best.pt", "rdd_yolo26n_best.pt", "baseline_yolo26n_best.pt"):
        if (WEIGHTS_DIR / n).exists():
            return WEIGHTS_DIR / n
    sys.exit("No trained checkpoint in models/weights. Train first (scripts/train.ps1) or pass --weights.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1, help="Video: analyse every N-th frame")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    det = Detector(args.weights or default_weights(), device=args.device)
    out = args.out or OUTPUTS_DIR / "cli" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Model: {det.weights.name} on {det.device} -> {out}")
    report: dict = {"weights": det.weights.name, "conf": args.conf, "results": []}

    src = args.source
    if src.is_file() and src.suffix.lower() in VID:
        frames, summary = process_video(det, src, out / f"{src.stem}_detected.mp4", conf=args.conf, stride=args.stride)
        report["video"] = summary
        report["results"] = [{"frame": f.frame_index, "time_s": f.timestamp_s, "detections": [d.to_dict() for d in f.detections]}
                             for f in frames if f.detections]
        print(f"{summary['processed_frames']} frames analysed, {sum(len(f.detections) for f in frames)} detections, "
              f"{summary['inference_fps']} inference FPS")
    else:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG) if src.is_dir() else [src]
        if not files:
            sys.exit(f"No images found at {src}")
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  skip (unreadable): {p}")
                continue
            dets, ms = det.predict(img, conf=args.conf)
            cv2.imwrite(str(out / f"{p.stem}_detected.jpg"), draw_detections(img, dets))
            gps = extract_exif_gps(p.read_bytes())
            report["results"].append({"image": str(p), "inference_ms": round(ms, 1), "exif_gps": gps,
                                      "detections": [d.to_dict() for d in dets]})
            summary = ", ".join(f"{d.class_code} {d.confidence:.2f} {d.severity}" for d in dets) or "no damage"
            print(f"  {p.name}: {summary} ({ms:.0f} ms)")
    (out / "detections.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {out / 'detections.json'}")


if __name__ == "__main__":
    main()

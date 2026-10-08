"""Train a baseline YOLO26 or RDD-YOLO model on the prepared RDD2022 dataset.

Experiments (see docs/experiments.md):
    baseline  -> models/architectures/yolo26-baseline.yaml   (Experiment A)
    rdd       -> models/architectures/yolo26-rdd.yaml        (Experiment B)

Initialisation (identical for both experiments, for a fair comparison):
    Backbone layers 0-10 are initialised from the official COCO-pretrained
    yolo26{scale}.pt (same layer indices/shapes in both YAMLs). Everything
    after the backbone (SimAM, neck, detection head) starts from random
    initialisation with a fixed seed and is learned on RDD2022 only.
    Use --no-pretrained to train everything from scratch.

Examples (PowerShell, from the project root):
    .venv\\Scripts\\python training\\train.py --experiment baseline --epochs 60
    .venv\\Scripts\\python training\\train.py --experiment rdd --epochs 60
    .venv\\Scripts\\python training\\train.py --experiment rdd --epochs 1 --fraction 0.02 --name smoke   # smoke test
    .venv\\Scripts\\python training\\train.py --resume experiments/rdd_yolo26n/weights/last.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdd_yolo.constants import ARCH_DIR, EXPERIMENTS_DIR, PROCESSED_DIR, WEIGHTS_DIR  # noqa: E402
from rdd_yolo.hardware import detect_hardware, recommend_training_params  # noqa: E402
from rdd_yolo.modules import register_custom_modules  # noqa: E402

EXPERIMENTS = {
    "baseline": {"yaml": "yolo26-baseline.yaml", "title": "Experiment A - Baseline YOLO26"},
    "rdd": {"yaml": "yolo26-rdd.yaml", "title": "Experiment B - RDD-YOLO (SimAM + GhostConv + Bilinear)"},
}
BACKBONE_LAST_LAYER = 10  # index of C2PSA, the last backbone layer in both YAMLs


def build_initial_checkpoint(yaml_path: Path, scale: str, out_path: Path, pretrained: bool, seed: int,
                             all_layers: bool = False) -> dict:
    """Build the model from YAML, transfer COCO weights (backbone only, or every shape-compatible layer)."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(seed, deterministic=True)
    import yaml as _yaml

    cfg = _yaml.safe_load(yaml_path.read_text())
    cfg["scale"] = scale
    cfg["yaml_file"] = str(yaml_path.with_name(yaml_path.stem + scale + yaml_path.suffix))
    model = DetectionModel(cfg, ch=3, nc=cfg["nc"], verbose=False)
    report = {"pretrained_source": None, "transferred_tensors": 0, "backbone_tensors": 0}
    if pretrained:
        from ultralytics import YOLO

        src_name = f"yolo26{scale}.pt"
        src = YOLO(str(WEIGHTS_DIR / src_name) if (WEIGHTS_DIR / src_name).exists() else src_name)
        src_sd = src.model.float().state_dict()
        dst_sd = model.state_dict()
        transfer = {}
        for k, v in src_sd.items():
            parts = k.split(".")
            if parts[0] != "model" or not parts[1].isdigit() or (not all_layers and int(parts[1]) > BACKBONE_LAST_LAYER):
                continue
            report["backbone_tensors"] += 1
            if k in dst_sd and dst_sd[k].shape == v.shape:
                transfer[k] = v
        missing = model.load_state_dict(transfer, strict=False)
        report.update(pretrained_source=src_name, transferred_tensors=len(transfer), scope="all layers" if all_layers else "backbone",
                      model_tensors=len(dst_sd), not_initialised_from_pretrained=len(missing.missing_keys))
        # keep a local copy of the COCO weights for reproducibility/offline use
        src_path = Path(src.ckpt_path) if getattr(src, "ckpt_path", None) else None
        if src_path and src_path.exists() and not (WEIGHTS_DIR / src_name).exists():
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, WEIGHTS_DIR / src_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "train_args": {}, "date": datetime.now(timezone.utc).isoformat(),
                "init_report": report}, out_path)
    return report


def summarise_results(run_dir: Path) -> dict:
    """Read the metrics Ultralytics actually logged (results.csv) - nothing is computed by hand."""
    import csv

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(csv_path.open())]
    if not rows:
        return {}
    best = max(rows, key=lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]))
    last = rows[-1]

    def pick(r):
        p, rc = float(r["metrics/precision(B)"]), float(r["metrics/recall(B)"])
        return {"epoch": int(float(r["epoch"])), "precision": p, "recall": rc,
                "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                "mAP50": float(r["metrics/mAP50(B)"]), "mAP50_95": float(r["metrics/mAP50-95(B)"])}

    return {"epochs_completed": len(rows), "best_epoch_val": pick(best), "last_epoch_val": pick(last)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=EXPERIMENTS, default="rdd")
    ap.add_argument("--scale", default=None, help="Model scale n/s/m (default: hardware recommendation)")
    ap.add_argument("--data", type=Path, default=PROCESSED_DIR / "data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None, help="Default: hardware recommendation")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of the train split to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="Run name (default: <experiment>_yolo26<scale>)")
    ap.add_argument("--no-pretrained", action="store_true", help="Train every layer from scratch")
    ap.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt checkpoint")
    ap.add_argument("--cache", default=False, help="Ultralytics cache option: False/ram/disk")
    ap.add_argument("--non-deterministic", action="store_true",
                    help="Allow non-deterministic CUDA kernels (faster, esp. bilinear upsampling backward)")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Fine-tune: start from this trained checkpoint instead of the COCO-backbone init")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--title", default=None, help="Human-readable experiment title stored in run_info.json")
    ap.add_argument("--init-all-layers", action="store_true",
                    help="Transfer every shape-compatible COCO layer (backbone+neck+head), not only the backbone")
    args = ap.parse_args()

    register_custom_modules()
    from ultralytics import YOLO, __version__ as ul_version

    if args.resume:
        run_dir = args.resume.resolve().parent.parent
        print(f"Resuming {args.resume}")
        t0 = time.time()
        YOLO(str(args.resume)).train(resume=True)
        finalize(run_dir, time.time() - t0, resumed=True)
        return

    if not args.data.exists():
        sys.exit(f"Dataset config not found: {args.data}. Run: python dataset/prepare_dataset.py")

    hw = detect_hardware()
    rec = recommend_training_params(hw, args.imgsz)
    scale = args.scale or rec.model_scale
    batch = args.batch or rec.batch
    workers = rec.workers if args.workers is None else args.workers
    device = args.device or rec.device
    exp = dict(EXPERIMENTS[args.experiment])
    if args.title:
        exp["title"] = args.title
    name = args.name or f"{args.experiment}_yolo26{scale}"
    run_dir = EXPERIMENTS_DIR / name
    print(f"== {exp['title']} ==")
    print(f"Hardware: {json.dumps(hw.to_dict())}")
    print(f"Recommendation: {rec.reason}")
    print(f"Using scale={scale} batch={batch} imgsz={args.imgsz} workers={workers} device={device} amp={rec.amp}")

    yaml_path = ARCH_DIR / exp["yaml"]
    if args.init_weights:  # fine-tuning from an already trained model
        if not args.init_weights.exists():
            sys.exit(f"--init-weights not found: {args.init_weights}")
        init_ckpt = args.init_weights
        init_report = {"fine_tuned_from": str(args.init_weights)}
    else:
        init_ckpt = EXPERIMENTS_DIR / "_init" / f"{name}_init.pt"
        init_report = build_initial_checkpoint(yaml_path, scale, init_ckpt, not args.no_pretrained, args.seed,
                                               all_layers=args.init_all_layers)
    print(f"Initialisation: {init_report}")

    model = YOLO(str(init_ckpt), task="detect")
    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.partial.json").write_text(json.dumps({
        "experiment": args.experiment, "title": exp["title"], "run_name": name,
        "architecture_yaml": f"models/architectures/{exp['yaml']}", "scale": scale,
        "ultralytics_version": ul_version, "started_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hw.to_dict(), "hardware_recommendation": asdict(rec),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective": {"batch": batch, "workers": workers, "device": device, "amp": rec.amp},
        "initialisation": init_report,
    }, indent=2))
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=batch, workers=workers, device=device,
        amp=rec.amp, patience=args.patience, fraction=args.fraction, seed=args.seed, deterministic=not args.non_deterministic,
        project=str(EXPERIMENTS_DIR), name=name, exist_ok=True, cache=args.cache, plots=True, val=True,
        pretrained=True,  # use the weights of the init checkpoint (COCO backbone + seeded random rest)
        optimizer="auto", cos_lr=False, close_mosaic=args.close_mosaic,
    )
    finalize(run_dir, time.time() - t0)


def finalize(run_dir: Path, elapsed: float, resumed: bool = False) -> None:
    """Write run_info.json (from the partial written at start) and publish best.pt to models/weights."""
    partial = run_dir / "run_info.partial.json"
    info = json.loads(partial.read_text()) if partial.exists() else {"run_name": run_dir.name}
    info["train_time_hours"] = round(info.get("train_time_hours", 0) + elapsed / 3600, 3)
    info["finished_utc"] = datetime.now(timezone.utc).isoformat()
    info["resumed"] = info.get("resumed", False) or resumed
    info["results"] = summarise_results(run_dir)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dst = WEIGHTS_DIR / f"{run_dir.name}_best.pt"
        shutil.copy2(best, dst)
        dst.with_suffix(".json").write_text(json.dumps({
            "model_name": f"{info.get('title', run_dir.name)}", "experiment": info.get("experiment"),
            "run_name": run_dir.name, "architecture": info.get("architecture_yaml"), "scale": info.get("scale"),
            "ultralytics_version": info.get("ultralytics_version"), "dataset": (info.get("args") or {}).get("data"),
            "imgsz": (info.get("args") or {}).get("imgsz"), "epochs_requested": (info.get("args") or {}).get("epochs"),
            "results": info["results"], "trained_on": (info.get("hardware") or {}).get("gpu_name") or "CPU",
            "finished_utc": info["finished_utc"],
        }, indent=2))
        print(f"Best checkpoint copied to {dst}")
    print(json.dumps(info["results"], indent=2))


if __name__ == "__main__":
    main()

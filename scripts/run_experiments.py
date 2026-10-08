"""Run the full experiment pipeline sequentially (one GPU):

    Experiment A (baseline) train -> evaluate(test)
    Experiment B (RDD-YOLO)  train -> evaluate(test)
    compare

Each step is skipped if its output already exists, so the script can be
re-run after an interruption (an interrupted training run is resumed from
weights/last.pt). Logs: experiments/logs/<step>.log

Usage:
    .venv\\Scripts\\python scripts\\run_experiments.py --epochs 40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "experiments" / "logs"


def run(step: str, args: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{step}.log"
    print(f"[{datetime.now():%H:%M:%S}] {step}: {' '.join(args)}  (log: {log})", flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {datetime.now().isoformat()} {' '.join(args)}\n")
        fh.flush()
        rc = subprocess.call([PY, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit(f"{step} failed with exit code {rc}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--scale", default="n")
    ap.add_argument("--only", choices=["baseline", "rdd"], default=None)
    args = ap.parse_args()

    for exp in ([args.only] if args.only else ["baseline", "rdd"]):
        name = f"{exp}_yolo26{args.scale}"
        run_dir = ROOT / "experiments" / name
        if (run_dir / "run_info.json").exists():
            print(f"{name}: training already finished, skipping")
        elif (run_dir / "weights" / "last.pt").exists():
            run(f"{name}_train", ["training/train.py", "--resume", str(run_dir / "weights" / "last.pt")])
        else:
            run(f"{name}_train", ["training/train.py", "--experiment", exp, "--scale", args.scale,
                                  "--epochs", str(args.epochs), "--batch", str(args.batch),
                                  "--workers", str(args.workers), "--patience", str(args.patience)])
        if not (run_dir / "eval_test" / "metrics.json").exists():
            run(f"{name}_eval", ["training/evaluate.py", "--run", name, "--split", "test"])
    if not args.only:
        run("compare", ["training/compare.py", "--baseline", f"baseline_yolo26{args.scale}",
                        "--rdd", f"rdd_yolo26{args.scale}"])
        run("docs", ["scripts/write_experiments_doc.py"])
    print("All experiments finished.")


if __name__ == "__main__":
    main()

"""Experiment C: fine-tune RDD-YOLO on the extended dataset and evaluate before/after.

Steps (each skipped if its output exists, so the script is resumable):
  1. dataset/build_extended.py                        -> dataset/processed/rdd2022_extended
  2. evaluate Experiment B (rdd_yolo26n) on the two NEW test sets      ("before")
  3. train rdd_yolo26n_ext: fine-tune from models/weights/rdd_yolo26n_best.pt
  4. evaluate rdd_yolo26n_ext on the original test split + the two new test sets ("after")
Logs: experiments/logs/<step>.log

Usage:  .venv\\Scripts\\python scripts\\run_extended.py --epochs 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiments import run  # noqa: E402

EXT = ROOT / "dataset" / "processed" / "rdd2022_extended"
EXP = ROOT / "experiments"
TESTS = {"test": EXT / "data.yaml", "test_rdd_new": EXT / "data_test_rdd_new.yaml",
         "test_external": EXT / "data_test_external.yaml"}


def evaluate(run_name: str, tag: str) -> None:
    if (EXP / run_name / f"eval_{tag}" / "metrics.json").exists():
        return
    split = "test"  # each yaml maps its own folder to the 'test' key
    run(f"{run_name}_eval_{tag}", ["training/evaluate.py", "--run", run_name, "--data", str(TESTS[tag]),
                                   "--split", split, "--tag", tag])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if not (EXT / "data.yaml").exists():
        run("build_extended", ["dataset/build_extended.py"])
    for tag in ("test_rdd_new", "test_external"):
        evaluate("rdd_yolo26n", tag)  # "before"

    name = "rdd_yolo26n_ext"
    if not (EXP / name / "run_info.json").exists():
        last = EXP / name / "weights" / "last.pt"
        if last.exists():
            run(f"{name}_train", ["training/train.py", "--resume", str(last)])
        else:
            run(f"{name}_train", ["training/train.py", "--experiment", "rdd", "--name", name, "--data", str(EXT / "data.yaml"),
                                  "--init-weights", str(ROOT / "models" / "weights" / "rdd_yolo26n_best.pt"),
                                  "--epochs", str(args.epochs), "--batch", str(args.batch), "--workers", str(args.workers),
                                  "--patience", "10", "--close-mosaic", "5", "--non-deterministic"])
    for tag in TESTS:
        evaluate(name, tag)  # "after"
    print("Experiment C finished.")


if __name__ == "__main__":
    main()

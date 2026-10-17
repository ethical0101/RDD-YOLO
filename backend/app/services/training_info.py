"""Reads real training/evaluation artefacts from experiments/ (never synthesises metrics)."""
from __future__ import annotations

import csv
import re
import time
import json
from pathlib import Path

from ..core.config import get_settings
from .storage import experiments_url

PLOT_FILES = ["results.png", "confusion_matrix_normalized.png", "confusion_matrix.png", "BoxPR_curve.png",
              "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "labels.jpg", "val_batch0_pred.jpg",
              "val_batch0_labels.jpg", "train_batch0.jpg"]


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_results_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                try:
                    row[k.strip()] = float(v)
                except (TypeError, ValueError):
                    row[k.strip()] = v
            out.append(row)
    return out


def list_experiments() -> list[dict]:
    root = get_settings().experiments_dir
    exps = []
    if not root.exists():
        return exps
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (d / "results.csv").exists() and not (d / "run_info.json").exists():
            continue
        info = _read_json(d / "run_info.json") or {}
        rows = _read_results_csv(d / "results.csv")
        test = _read_json(d / "eval_test" / "metrics.json")
        status = ("aborted" if (d / "ABORTED.txt").exists() else "completed" if info.get("finished_utc")
                  else "running_or_interrupted" if rows else "pending")
        exps.append({
            "name": d.name, "title": info.get("title"), "experiment": info.get("experiment"), "status": status,
            "epochs_completed": len(rows), "epochs_requested": (info.get("args") or {}).get("epochs"),
            "train_time_hours": info.get("train_time_hours"), "best_val": (info.get("results") or {}).get("best_epoch_val"),
            "test_metrics": test.get("overall") if test else None,
            "has_weights": (d / "weights" / "best.pt").exists(),
        })
    return exps


def experiment_detail(name: str) -> dict | None:
    root = get_settings().experiments_dir
    d = (root / name).resolve()
    if not d.is_relative_to(root.resolve()) or not d.is_dir():
        return None
    plots = [{"name": f, "url": experiments_url(d / f)} for f in PLOT_FILES if (d / f).exists()]
    evals = {}
    for split in ("val", "test"):
        e = d / f"eval_{split}"
        if e.exists():
            evals[split] = {
                "metrics": _read_json(e / "metrics.json"),
                "plots": [{"name": f, "url": experiments_url(e / f)} for f in PLOT_FILES if (e / f).exists()],
            }
    return {"name": name, "run_info": _read_json(d / "run_info.json"), "history": _read_results_csv(d / "results.csv"),
            "plots": plots, "evaluations": evals}


def comparison() -> dict | None:
    root = get_settings().experiments_dir
    data = _read_json(root / "comparison.json")
    if data is None:
        return None
    data["plots"] = [{"name": f, "url": experiments_url(root / f)}
                     for f in ("comparison_curves.png", "comparison_metrics.png") if (root / f).exists()]
    return data


def dataset_stats() -> dict | None:
    d = get_settings().dataset_dir
    stats = _read_json(d / "dataset_stats.json")
    if stats is None:
        return None
    report = _read_json(d / "validation_report.json") or {}
    stats["validation"] = {k: report.get(k) for k in ("images_total", "images_valid", "images_with_target_damage",
                                                      "issue_count", "issue_summary", "ignored_labels", "countries")}
    return stats


# Ultralytics progress line, e.g. "  3/10  2.43G  1.71  1.83  0.013  17  640: 52% ... 2370/4528 5.2it/s 7:23<6:51"
_PROGRESS = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s.*?(\d+)%.*?(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)\s+([\d:]+)<([\d:?]+)")


def _secs(t: str) -> float | None:
    try:
        parts = [float(x) for x in t.split(":")]
    except ValueError:
        return None
    s = 0.0
    for x in parts:
        s = s * 60 + x
    return s


def live_progress(max_age_s: float = 300) -> list[dict]:
    """Progress of training runs whose log was written in the last few minutes, parsed from the real
    experiments/logs/<run>_train.log (Ultralytics console output). Returns [] when nothing is training."""
    logs = get_settings().experiments_dir / "logs"
    out = []
    if not logs.exists():
        return out
    now = time.time()
    for log in logs.glob("*_train.log"):
        age = now - log.stat().st_mtime
        if age > max_age_s:
            continue
        with log.open("rb") as fh:
            fh.seek(max(0, log.stat().st_size - 20000))
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", fh.read().decode("utf-8", "replace"))
            tail = text.replace("\r", "\n").splitlines()
        m = next((_PROGRESS.search(ln) for ln in reversed(tail) if _PROGRESS.search(ln)), None)
        if not m:
            out.append({"run": log.name[:-10], "phase": "starting / validating", "log_age_s": round(age)})
            continue
        ep, eps, _, it, its, rate, unit, el, rem = m.groups()
        ep, eps, it, its = int(ep), int(eps), int(it), int(its)
        speed = float(rate) if unit == "it/s" else 1 / float(rate)
        epoch_s = its / speed if speed else None
        rem_s = _secs(rem)
        run = log.name[:-10]
        rows = _read_results_csv(get_settings().experiments_dir / run / "results.csv")
        val_last = rows[-1].get("metrics/mAP50(B)") if rows else None
        out.append({"run": run, "phase": "training", "epoch": ep, "epochs": eps, "batch": it, "batches": its,
                    "iter_per_s": round(speed, 2), "epoch_eta_s": rem_s,
                    "run_eta_s": round((rem_s or 0) + (eps - ep) * (epoch_s or 0) * 1.08) if epoch_s else None,  # +8 % validation
                    "progress": round(((ep - 1) + it / its) / eps, 4), "log_age_s": round(age),
                    "epochs_logged": len(rows), "last_val_mAP50": val_last,
                    "history": [{"epoch": int(r.get("epoch", i + 1)), "mAP50": r.get("metrics/mAP50(B)"),
                                 "mAP50_95": r.get("metrics/mAP50-95(B)")} for i, r in enumerate(rows)]})
    return out


def dataset_catalog() -> dict | None:
    return _read_json(get_settings().experiments_dir.parent / "docs" / "datasets" / "dataset_manifest.json")

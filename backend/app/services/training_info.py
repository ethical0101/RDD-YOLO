"""Reads real training/evaluation artefacts from experiments/ (never synthesises metrics)."""
from __future__ import annotations

import csv
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
        status = "completed" if info.get("finished_utc") else ("running_or_interrupted" if rows else "pending")
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

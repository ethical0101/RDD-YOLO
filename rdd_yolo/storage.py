"""Storage manager: free-space guard, inventory, logging and reversible quarantine.

Configuration (environment or .env):
  RDD_STORAGE_RESERVE_GB  minimum free space that must remain on the project drive (default 8)
  RDD_DATASET_BUDGET_GB   maximum total size of dataset/raw (default 30)

Safety rules implemented here:
  * downloads/extractions call `guard.require(bytes)` first and stop safely when the reserve would be crossed;
  * only items classified as REGENERABLE may be quarantined automatically; source datasets, final checkpoints,
    evaluation results, manifests, licenses, logs, docs and source code are never touched;
  * "removal" = a move into <project>/_quarantine/<timestamp>/ (reversible), never a permanent delete.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .constants import PROJECT_ROOT

LOG = PROJECT_ROOT / "docs" / "storage" / "storage_log.jsonl"
QUARANTINE = PROJECT_ROOT / "_quarantine"
GB = 1024 ** 3


class InsufficientSpace(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def free_bytes(path: Path = PROJECT_ROOT) -> int:
    return shutil.disk_usage(path).free


def log_event(event: str, **fields) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event,
           "free_gb": round(free_bytes() / GB, 2), **fields}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


class StorageGuard:
    def __init__(self, reserve_gb: float | None = None, budget_gb: float | None = None):
        self.reserve = (reserve_gb if reserve_gb is not None else _env_float("RDD_STORAGE_RESERVE_GB", 8)) * GB
        self.budget = (budget_gb if budget_gb is not None else _env_float("RDD_DATASET_BUDGET_GB", 30)) * GB

    def _raw_size(self) -> int:
        """Size of dataset/raw, rescanned at most once a minute (downloaders call require() per file)."""
        now = time.monotonic()
        if getattr(self, "_raw", None) is None or now - self._raw[0] > 60:
            self._raw = (now, dir_size(PROJECT_ROOT / "dataset" / "raw"))
        return self._raw[1]

    def require(self, nbytes: int, what: str, dataset_bytes: bool = True) -> None:
        """Raise InsufficientSpace if writing nbytes would cross the reserve or the dataset budget."""
        free = free_bytes()
        raw = self._raw_size() if dataset_bytes else 0
        ok_reserve = free - nbytes >= self.reserve
        ok_budget = (not dataset_bytes) or raw + nbytes <= self.budget
        log_event("check", what=what, need_gb=round(nbytes / GB, 3), ok=ok_reserve and ok_budget,
                  reserve_gb=round(self.reserve / GB, 1), dataset_raw_gb=round(raw / GB, 2))
        if not ok_reserve:
            raise InsufficientSpace(f"{what}: needs {nbytes / GB:.2f} GB but only {free / GB:.2f} GB free "
                                    f"(reserve {self.reserve / GB:.0f} GB). Free space or lower RDD_STORAGE_RESERVE_GB.")
        if not ok_budget:
            raise InsufficientSpace(f"{what}: dataset budget {self.budget / GB:.0f} GB would be exceeded "
                                    f"({raw / GB:.2f} GB used). Raise RDD_DATASET_BUDGET_GB to continue.")


guard = StorageGuard()

# ------------------------------------------------------------------ inventory

SOURCE = "source dataset (never auto-removed)"
FINAL = "final artefact (never auto-removed)"
DERIVED = "derived dataset (regenerable from sources; mostly hard links)"
REGEN = "regenerable cache"


@dataclass
class Item:
    path: str
    size_gb: float
    kind: str
    removable: bool
    reason: str


def _rel(p: Path) -> str:
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def inventory() -> list[Item]:
    R = PROJECT_ROOT
    items: list[tuple[Path, str, bool, str]] = []
    for p in sorted((R / "dataset" / "raw" / "RDD2022").glob("*")):
        items.append((p, SOURCE, False, "original RDD2022 country data used by experiments"))
    for p in sorted((R / "dataset" / "raw" / "external").glob("*")):
        items.append((p, SOURCE, False, "original external dataset (license recorded in docs/datasets)"))
    zips = R / "dataset" / "raw" / "_zips"
    if zips.exists():
        items.append((zips, REGEN, True, "partial/complete download archives; data already extracted"))
    for p in sorted((R / "dataset" / "processed").glob("*")):
        if p.is_dir():
            items.append((p, DERIVED, False, "rebuild with dataset/*.py; keep while experiments reference it"))
    items.append((R / "models" / "weights", FINAL, False, "published checkpoints"))
    for p in sorted((R / "experiments").glob("*/weights")):
        items.append((p, REGEN if (R / "models" / "weights" / f"{p.parent.name}_best.pt").exists() else FINAL,
                      (R / "models" / "weights" / f"{p.parent.name}_best.pt").exists(),
                      "best.pt duplicated in models/weights; last.pt only needed to resume a finished run"))
    items.append((R / "experiments" / "_init", REGEN, True, "initial checkpoints rebuilt by train.py"))
    items.append((R / "models" / "backup_v1.0-expD", FINAL, False, "user-requested safety copy"))
    items.append((R / "deploy", REGEN, True, "deployment bundles rebuilt by scripts/build_hf_space.py / export_static_site.py"))
    items.append((R / ".cache", REGEN, True, "downloaded wheels (re-downloadable)"))
    items.append((R / "outputs", FINAL, False, "user detections database and uploads"))
    out = []
    for p, kind, removable, reason in items:
        if p.exists():
            out.append(Item(_rel(p), round(dir_size(p) / GB, 3), kind, removable, reason))
    return out


def quarantine(rel_path: str, dry_run: bool = True) -> str:
    """Move a REGENERABLE item into _quarantine/<timestamp>/ (reversible). Refuses anything else."""
    allowed = {i.path for i in inventory() if i.removable}
    if rel_path not in allowed:
        raise PermissionError(f"{rel_path} is not classified as regenerable; refusing to move it")
    src = PROJECT_ROOT / rel_path
    dst = QUARANTINE / time.strftime("%Y%m%d_%H%M%S") / rel_path
    if dry_run:
        return f"[dry-run] would move {rel_path} -> {_rel(dst)}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = dir_size(src)
    shutil.move(str(src), str(dst))
    log_event("quarantine", path=rel_path, to=_rel(dst), size_gb=round(size / GB, 3))
    return f"moved {rel_path} -> {_rel(dst)}"


def report() -> dict:
    du = shutil.disk_usage(PROJECT_ROOT)
    inv = inventory()
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": str(PROJECT_ROOT),
        "drive": {"total_gb": round(du.total / GB, 1), "used_gb": round(du.used / GB, 1), "free_gb": round(du.free / GB, 1)},
        "reserve_gb": guard.reserve / GB, "dataset_budget_gb": guard.budget / GB,
        "dataset_raw_gb": round(dir_size(PROJECT_ROOT / "dataset" / "raw") / GB, 2),
        "items": [asdict(i) for i in inv],
        "removable_total_gb": round(sum(i.size_gb for i in inv if i.removable), 2),
    }

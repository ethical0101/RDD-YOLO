"""Write docs/storage/STORAGE_REPORT.md and storage_inventory.json; optionally quarantine regenerable items.

Usage:
  .venv\\Scripts\\python scripts\\storage_report.py                       # report only
  .venv\\Scripts\\python scripts\\storage_report.py --quarantine deploy     # dry run
  .venv\\Scripts\\python scripts\\storage_report.py --quarantine deploy --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdd_yolo import storage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarantine", nargs="*", default=[])
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for q in args.quarantine:
        print(storage.quarantine(q, dry_run=not args.apply))
    r = storage.report()
    out = ROOT / "docs" / "storage"
    out.mkdir(parents=True, exist_ok=True)
    (out / "storage_inventory.json").write_text(json.dumps(r, indent=2))
    d = r["drive"]
    lines = ["# Storage report", "", f"Generated {r['generated_utc']} by `scripts/storage_report.py` (live measurement).", "",
             f"Drive of `{r['project']}`: **{d['free_gb']} GB free** of {d['total_gb']} GB. Reserve that downloads must keep: "
             f"{r['reserve_gb']:.0f} GB (`RDD_STORAGE_RESERVE_GB`). Dataset budget: {r['dataset_budget_gb']:.0f} GB "
             f"(`RDD_DATASET_BUDGET_GB`), currently {r['dataset_raw_gb']} GB in `dataset/raw`.", "",
             "| Path | Size (GB) | Class | Auto-removable | Why |", "|---|---|---|---|---|"]
    for i in sorted(r["items"], key=lambda x: -x["size_gb"]):
        lines.append(f"| `{i['path']}` | {i['size_gb']:.2f} | {i['kind']} | {'yes (quarantine)' if i['removable'] else 'no'} | {i['reason']} |")
    lines += ["", f"Regenerable items total **{r['removable_total_gb']} GB**. They can be moved (not deleted) to `_quarantine/` with "
              "`scripts/storage_report.py --quarantine <path> --apply`; source datasets, checkpoints without a copy, results, "
              "manifests, licenses, logs, documents and code are never eligible. Deleting a source dataset requires the owner's approval.",
              "", "Every check and move is appended to `docs/storage/storage_log.jsonl`."]
    (out / "STORAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

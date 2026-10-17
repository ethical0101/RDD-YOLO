"""Generate docs/datasets/{dataset_manifest.json, DATASET_CATALOG.md, DATASET_LICENSES.md} from
dataset/registry.py plus a live scan of the files on disk (counts and sizes are measured, never typed).

Usage: .venv\\Scripts\\python scripts\\build_dataset_docs.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset"))
from registry import CANONICAL, DATASETS, EXCLUDED  # noqa: E402

V3 = ROOT / "dataset" / "processed" / "rdd2022_extended_v3"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = ROOT / "docs" / "datasets"


def scan_dir(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {"path": rel, "exists": False}
    files = [f for f in p.rglob("*") if f.is_file()]
    return {"path": rel, "exists": True, "files": len(files), "images": sum(f.suffix.lower() in IMG_EXT for f in files),
            "annotation_files": sum(f.suffix.lower() in {".xml", ".txt", ".json"} for f in files),
            "bytes": sum(f.stat().st_size for f in files)}


def v3_usage() -> dict[str, Counter]:
    """Images per (registry prefix, split) actually present in the training set used by Experiments E/F."""
    prefixes = sorted({p: d["id"] for d in DATASETS for p in d["v3_prefixes"]}.items(), key=lambda x: -len(x[0]))
    use: dict[str, Counter] = {d["id"]: Counter() for d in DATASETS}
    for split_dir in sorted((V3 / "images").iterdir()):
        for f in split_dir.iterdir():
            for p, ds in prefixes:
                if f.name.startswith(p):
                    use[ds][split_dir.name] += 1
                    break
    return use


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    use = v3_usage()
    entries = []
    for d in DATASETS:
        scans = [scan_dir(r) for r in d["local_dirs"]]
        e = {k: d[k] for k in ("id", "name", "source_url", "license", "attribution", "publication", "original_format",
                               "original_classes", "mapping", "used_in", "limitations", "status")}
        e["local"] = scans
        e["local_images"] = sum(s.get("images", 0) for s in scans)
        e["local_bytes"] = sum(s.get("bytes", 0) for s in scans)
        e["downloaded"] = any(s.get("exists") and s.get("files") for s in scans)
        e["images_in_training_set_v3"] = dict(sorted(use[d["id"]].items()))
        entries.append(e)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator": "scripts/build_dataset_docs.py", "canonical_classes": CANONICAL,
                "training_set": str(V3.relative_to(ROOT)).replace("\\", "/"), "datasets": entries, "excluded": EXCLUDED}
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    splits = ["train", "val", "test", "test_clean", "test_rdd_new", "test_external", "test_closeup", "test_pavement", "test_rome"]
    L = ["# Dataset catalog", "", f"Generated {manifest['generated_utc']} by `scripts/build_dataset_docs.py` from "
         "`dataset/registry.py` and a scan of the local files. Image counts are measured on disk.", "",
         "Canonical classes: " + ", ".join(f"{k} {v}" for k, v in CANONICAL.items()) + ".", "",
         "## Datasets on disk and their use in training", "",
         "Counts are images in `dataset/processed/rdd2022_extended_v3` (the set used by Experiments E and F) after class "
         "mapping, de-duplication and leakage removal.", "",
         "| Dataset | Status | Local images | Local size | " + " | ".join(splits) + " | Used in |",
         "|---|---|---|---|" + "---|" * len(splits) + "---|"]
    for e in entries:
        u = e["images_in_training_set_v3"]
        L.append(f"| [{e['name']}]({e['source_url']}) | {e['status']} | {e['local_images']:,} | {e['local_bytes'] / 2**30:.2f} GB | "
                 + " | ".join(f"{u.get(s, 0):,}" if u.get(s) else "–" for s in splits) + f" | {e['used_in']} |")
    tot = Counter()
    for e in entries:
        tot.update(e["images_in_training_set_v3"])
    L.append("| **Total** | | | | " + " | ".join(f"**{tot.get(s, 0):,}**" for s in splits) + " | |")
    L += ["", "## Details", ""]
    for e in entries:
        L += [f"### {e['name']}", "", f"* Source: {e['source_url']}", f"* License: {e['license']}",
              f"* Attribution: {e['attribution']}", f"* Publication: {e['publication']}",
              f"* Original format / classes: {e['original_format']} / {e['original_classes']}",
              f"* Mapping to canonical classes: {e['mapping']}", f"* Limitations: {e['limitations']}",
              "* Local copies: " + "; ".join(f"`{s['path']}` ({s.get('images', 0):,} images, {s.get('annotation_files', 0):,} "
                                              f"annotation files)" if s.get("exists") else f"`{s['path']}` (not downloaded)"
                                              for s in e["local"]), ""]
    L += ["## Reviewed and not used", "", "| Dataset | Reason |", "|---|---|"]
    L += [f"| [{x['name']}]({x['source_url']}) | {x['reason']} |" for x in EXCLUDED]
    (OUT / "DATASET_CATALOG.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    Lic = ["# Dataset licenses and attribution", "", "Generated from `dataset/registry.py`. Only datasets with an explicit "
           "open license are used; the trained weights are released for research and education with these attributions.", "",
           "| Dataset | License | Attribution | Citation |", "|---|---|---|---|"]
    Lic += [f"| {e['name']} | {e['license']} | {e['attribution']} | {e['publication']} |" for e in entries if e["downloaded"]]
    Lic += ["", "Datasets that were not used because of their license are listed in `DATASET_CATALOG.md`."]
    (OUT / "DATASET_LICENSES.md").write_text("\n".join(Lic) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()

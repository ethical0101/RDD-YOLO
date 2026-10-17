"""Download the Rome road-damage dataset (Zenodo 18528034, CC BY 4.0) and verify its MD5.

Giordani E., Arcioni L., Gil-Martin M., Marini M.R., "Road Damage Dataset: Potholes, Cracks and Manholes",
Zenodo 2026, doi:10.5281/zenodo.18528034. Output: dataset/raw/external/rome_rdd/ (zip extracted, zip removed
after a verified extraction; it can be downloaded again).

Usage: python dataset/download_rome.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from rdd_yolo.storage import guard, log_event  # noqa: E402

RECORD = "https://zenodo.org/api/records/18528034"
OUT = ROOT / "raw" / "external" / "rome_rdd"
UA = {"User-Agent": "Mozilla/5.0 (RDD-YOLO dataset tools)"}


def main() -> None:
    if (OUT / ".complete").exists():
        print("already downloaded")
        return
    rec = json.load(urllib.request.urlopen(urllib.request.Request(RECORD, headers=UA), timeout=60))
    assert rec["metadata"]["license"]["id"] == "cc-by-4.0", rec["metadata"]["license"]
    f = next(x for x in rec["files"] if x["key"] == "data.zip")
    guard.require(2 * f["size"], "Rome road-damage dataset (zip + extraction)")
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "data.zip"
    md5 = hashlib.md5()
    with urllib.request.urlopen(urllib.request.Request(f["links"]["self"], headers=UA), timeout=120) as r, open(zpath, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            md5.update(chunk)
    want = f["checksum"].split(":", 1)[1]
    if md5.hexdigest() != want:
        zpath.unlink()
        sys.exit(f"MD5 mismatch: got {md5.hexdigest()}, expected {want}")
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        zf.extractall(OUT)
    zpath.unlink()
    (OUT / "download_manifest.json").write_text(json.dumps(
        {"source": "https://zenodo.org/records/18528034", "doi": "10.5281/zenodo.18528034", "license": "CC BY 4.0",
         "file": "data.zip", "bytes": f["size"], "md5": want, "members": len(names),
         "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    (OUT / ".complete").write_text("ok\n")
    log_event("download", dataset="rome_rdd", files=len(names), md5_ok=True)
    print(f"{len(names)} files extracted to {OUT}")


if __name__ == "__main__":
    main()

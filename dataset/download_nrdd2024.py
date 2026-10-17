"""Download only the refined annotations of N-RDD2024 (Mendeley Data, CC BY 4.0).

N-RDD2024 (Kaya and Codur, 2024; doi:10.17632/27c8pwsd6v.3) re-annotates RDD2022 images with 10 classes.
Its images are the RDD2022 images we already hold, so only the PASCAL VOC XML files are extracted from the
per-country `<country>_voc.zip` files by streaming the remote zips with HTTP range requests; image bytes are
read but never written to disk. Output: dataset/raw/external/nrdd2024/<zip stem>/<split>/<file>.xml

Usage: python dataset/download_nrdd2024.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from download_rdd2022 import HttpRangeFile  # noqa: E402

from rdd_yolo.storage import guard, log_event  # noqa: E402

API = "https://data.mendeley.com/public-api/datasets/27c8pwsd6v"
UA = {"User-Agent": "Mozilla/5.0 (RDD-YOLO dataset tools)"}
OUT = ROOT / "raw" / "external" / "nrdd2024"


def get_json(url: str):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def main() -> None:
    folders = get_json(API + "/folders/3")
    tv = next(f for f in folders if f["name"] == "Training and Validation Dataset")
    files = [f for f in get_json(f"{API}/files?folder_id={tv['id']}&version=3") if f["filename"].endswith("_voc.zip")]
    guard.require(50 * 2**20, "N-RDD2024 annotations (XML only, est. <50 MB)")
    manifest = []
    for f in files:
        stem = f["filename"][:-4]
        marker = OUT / stem / ".complete"
        if marker.exists():
            print(f"[skip] {stem}")
            continue
        url = f["content_details"]["download_url"]
        remote = HttpRangeFile(url, f["size"], block=8 << 20)
        zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
        xmls = sorted((i for i in zf.infolist() if i.filename.lower().endswith(".xml")), key=lambda i: i.header_offset)
        for i in xmls:
            dst = OUT / stem / i.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(zf.read(i))
        marker.write_text(f"{len(xmls)} xml files from {f['filename']} (sha256 {f['content_details'].get('sha256_hash')})\n")
        manifest.append({"zip": f["filename"], "xml_files": len(xmls), "zip_bytes": f["size"],
                         "bytes_streamed": remote.bytes_fetched, "sha256": f["content_details"].get("sha256_hash")})
        log_event("download", dataset="N-RDD2024", file=f["filename"], xml_files=len(xmls))
        print(f"{stem}: {len(xmls)} XML files", flush=True)
    (OUT / "download_manifest.json").write_text(json.dumps(
        {"source": "https://data.mendeley.com/datasets/27c8pwsd6v/3", "license": "CC BY 4.0",
         "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": manifest}, indent=2))


if __name__ == "__main__":
    main()

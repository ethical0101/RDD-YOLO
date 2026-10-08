"""Selective download of the "Pavement Distress Detection" aggregate (Deeksha9, Hugging Face, MIT license).

The archive (split.zip, 20.4 GB, YOLO format, classes: Longitudinal/Transverse/Alligator Crack, Pothole,
Patch, Block Crack) bundles several public datasets, identified by a filename prefix. Only sources we do
not already have are extracted, via HTTP range requests (no full download):

  kept    1.SVRDD, 2.archive, 4.HighRPD, 5.Potholes, 7.RD0, 8.Attain
  skipped 3.RDD2022            - already used; contains our held-out test images (leakage)
          9.RoadDamageDataset  - older Japanese RDD release that overlaps RDD2022 (leakage risk)
          6.Uncracked          - damage-free images, no boxes to learn from

Output: dataset/raw/external/deeksha/split/{train,val,test}/{images,labels}/ (as published, CRC verified).
Usage:  python dataset/download_deeksha.py
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_rdd2022 import HttpRangeFile  # noqa: E402

URL = "https://huggingface.co/datasets/Deeksha9/pavement-distress-detection/resolve/main/split.zip"
SIZE = 20382190784
KEEP = ("1.SVRDD", "2.archive", "4.HighRPD", "5.Potholes", "7.RD0", "8.Attain")
OUT = Path(__file__).resolve().parent / "raw" / "external" / "deeksha"


def main() -> None:
    remote = HttpRangeFile(URL, SIZE, block=16 << 20)
    zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    sel = [i for i in zf.infolist()
           if not i.is_dir() and ("/images/" in i.filename or "/labels/" in i.filename)
           and i.filename.split("/")[-1].startswith(KEEP)]
    sel.append(zf.getinfo("split/data.yaml"))
    sel.sort(key=lambda i: i.header_offset)
    total = sum(i.compress_size for i in sel)
    print(f"selected {len(sel)} files, {total / 1e9:.2f} GB", flush=True)
    done, t0 = 0, time.time()
    for k, info in enumerate(sel, 1):
        dest = OUT / info.filename
        done += info.compress_size
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src:  # CRC32 checked by zipfile
            data = src.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        if k % 2000 == 0:
            rate = remote.bytes_fetched / max(time.time() - t0, 1e-6) / 1e6
            print(f"  {k}/{len(sel)} files, {done / 1e9:.2f}/{total / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
    (OUT / ".complete").write_text("ok")
    print("done", flush=True)


if __name__ == "__main__":
    main()

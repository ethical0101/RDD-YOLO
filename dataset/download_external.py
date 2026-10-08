"""Download public, ungated pothole datasets (YOLO format) from the Hugging Face hub.

No account or token is needed. Both are Roboflow exports published under CC BY 4.0
(license stated in each dataset's README.roboflow.txt / data.yaml):

  egypt_rdd   Hanno100/RoadDamageDetection-Egypt   classes: Crack, Pothole   (zip, ~204 MB)
              https://universe.roboflow.com/road-damage-detection-wgjj4/rdd-6706t/dataset/8
  potholes_rf Ryukijano/Pothole-detection-Yolov8   classes: pothole           (~300 images)
              https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma/dataset/1

Files land in dataset/raw/external/<name>/ exactly as published (nothing is altered here).
Usage:  python dataset/download_external.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw" / "external"
HF = "https://huggingface.co"

DATASETS = {
    "egypt_rdd": {"repo": "Hanno100/RoadDamageDetection-Egypt", "files": ["RDD.v8i.yolov8.zip"]},
    "potholes_rf": {"repo": "Ryukijano/Pothole-detection-Yolov8", "dirs": ["train", "valid", "test"],
                    "files": ["data.yaml", "README.roboflow.txt"]},
}


def get(url: str, dest: Path, retries: int = 6) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:
            print(f"  retry {k + 1}: {exc}", file=sys.stderr)
            time.sleep(2 ** k)
    raise IOError(f"failed: {url}")


def tree(repo: str, path: str) -> list[dict]:
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main() -> None:
    for name, spec in DATASETS.items():
        dst = OUT / name
        print(f"[{name}] {spec['repo']}")
        files = list(spec.get("files", []))
        for d in spec.get("dirs", []):
            files += [e["path"] for e in tree(spec["repo"], d) if e["type"] == "file"]
        for i, f in enumerate(files, 1):
            get(f"{HF}/datasets/{spec['repo']}/resolve/main/{urllib.parse.quote(f)}", dst / f)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}")
        for z in dst.glob("*.zip"):
            marker = dst / f".{z.stem}.extracted"
            if not marker.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dst)
                marker.write_text("ok")
        print(f"[{name}] {len(files)} files -> {dst}")


if __name__ == "__main__":
    main()

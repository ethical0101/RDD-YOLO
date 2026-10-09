"""Build the free static deployment (GitHub Pages): the dashboard + the YOLO26s model running in the browser.

Steps
  1. Export the served checkpoint to ONNX (if not already exported).
  2. Build the dashboard with VITE_STATIC=1 and the Pages base path.
  3. Snapshot the read-only API responses (model, training runs, comparison, dataset, classes, severity rules)
     by calling the real FastAPI app, and copy every experiment plot they reference.
  4. Add the held-out demo images and the SPA fallback (404.html) for GitHub Pages.

Usage:
    .venv\\Scripts\\python scripts\\export_static_site.py [--base /RDD-YOLO/]
Output: deploy/static_site/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "deploy" / "static_site"
WEIGHTS = ROOT / "models" / "weights" / "yolo26s_full_best.pt"
ONNX = ROOT / "deploy" / "onnx" / "yolo26s_full_best.onnx"


def export_onnx() -> None:
    if ONNX.exists() and ONNX.stat().st_mtime > WEIGHTS.stat().st_mtime:
        return
    from rdd_yolo.modules import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    ONNX.parent.mkdir(parents=True, exist_ok=True)
    tmp = ONNX.parent / WEIGHTS.name
    shutil.copy2(WEIGHTS, tmp)  # export next to a copy so nothing is written into models/weights
    out = Path(YOLO(str(tmp)).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False))
    out.replace(ONNX)
    tmp.unlink()


def build_frontend(base: str) -> None:
    env = {**os.environ, "VITE_STATIC": "1", "VITE_BASE": base}
    npm = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run([npm, "run", "build", "--", "--outDir", str(OUT), "--emptyOutDir"], cwd=ROOT / "frontend", env=env,
                   check=True)


def snapshot_api() -> None:
    os.environ.setdefault("RDD_MODEL_DEVICE", "cpu")
    from fastapi.testclient import TestClient

    from backend.app.core.config import get_settings
    from backend.app.main import app

    get_settings.cache_clear()
    data = OUT / "data" / "api"
    data.mkdir(parents=True, exist_ok=True)
    urls: set[str] = set()

    def grab(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                grab(v)
        elif isinstance(obj, list):
            for v in obj:
                grab(v)
        elif isinstance(obj, str) and obj.startswith("/files/experiments/"):
            urls.add(obj)

    def put(name: str, obj) -> None:
        grab(obj)
        (data / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")

    with TestClient(app) as c:
        model = c.get("/api/model").json()
        served = model["info"]["weights"]
        model["available"] = [w for w in model["available"] if w["file"] == served]
        model["info"]["device"] = "browser"
        put("model", model)
        for path, name in (("/api/classes", "classes"), ("/api/severity/rules", "severity_rules"),
                           ("/api/dataset/stats", "dataset_stats"), ("/api/training/comparison", "training_comparison")):
            put(name, c.get(path).json())
        exps = [e for e in c.get("/api/training/experiments").json() if not e["name"].startswith("smoke")]
        put("training_experiments", exps)
        for e in exps:
            put(f"training_experiment_{e['name']}", c.get(f"/api/training/experiments/{e['name']}").json())
    for u in sorted(urls):
        src = ROOT / "experiments" / u[len("/files/experiments/"):]
        if src.exists():
            dst = OUT / u.lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"API snapshot: {len(list(data.glob('*.json')))} files, {len(urls)} plots")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/RDD-YOLO/")
    args = ap.parse_args()
    export_onnx()
    build_frontend(args.base)
    snapshot_api()
    (OUT / "data" / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ONNX, OUT / "data" / "model" / ONNX.name)
    demo = OUT / "data" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted((ROOT / "demo_images").glob("*.jpg")):
        shutil.copy2(p, demo / p.name)
        cls = p.name.split("_")[0]
        items.append({"file": p.name, "label": f"{cls} sample ({p.name.split('_', 3)[-1].rsplit('.', 1)[0]})"})
    (demo / "index.json").write_text(json.dumps(items), encoding="utf-8")
    shutil.copy2(OUT / "index.html", OUT / "404.html")  # SPA deep links on GitHub Pages
    (OUT / ".nojekyll").write_text("")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Static site ready: {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

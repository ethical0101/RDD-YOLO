"""Assemble a self-contained Hugging Face Spaces (Docker) bundle in deploy/hf_space/.

The bundle contains only what the running app needs: the backend + core library, the built dashboard,
the trained checkpoints, the real experiment results/plots (for the Model and Training pages), the
dataset statistics and the held-out demo images. No dataset images, no secrets.

Usage:
    cd frontend; npm run build; cd ..
    .venv\\Scripts\\python scripts\\build_hf_space.py
    .venv\\Scripts\\python scripts\\deploy_hf_space.py --space <your-hf-username>/rdd-yolo
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy" / "hf_space"
RUNS = ["baseline_yolo26n", "rdd_yolo26n", "rdd_yolo26n_ext", "rdd_yolo26n_ext2", "yolo26s_full"]
RUN_FILES = ["results.csv", "results.png", "run_info.json", "args.yaml", "confusion_matrix_normalized.png",
             "confusion_matrix.png", "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
             "labels.jpg", "val_batch0_pred.jpg", "val_batch0_labels.jpg", "train_batch0.jpg"]
EVAL_FILES = ["metrics.json", "confusion_matrix_normalized.png", "confusion_matrix.png", "BoxPR_curve.png",
              "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "val_batch0_pred.jpg", "val_batch0_labels.jpg"]

DOCKERFILE = """# Hugging Face Spaces (Docker SDK). CPU inference; the app serves API + dashboard on port 7860.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user \\
    YOLO_CONFIG_DIR=/tmp/ultralytics \\
    RDD_OUTPUTS_DIR=/tmp/rdd_outputs \\
    RDD_DATABASE_URL=sqlite:////tmp/rdd_outputs/rdd_yolo.db \\
    RDD_MODEL_DEVICE=cpu \\
    RDD_CORS_ORIGINS=* \\
    RDD_MAX_UPLOAD_MB=50 \\
    RDD_VIDEO_DEFAULT_STRIDE=5
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

SPACE_README = """---
title: RDD-YOLO Road Damage Detection
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Road damage detection, geolocation and OpenStreetMap mapping
---

# RDD-YOLO — Road Damage Detection, Geolocation and Mapping

Detects **D00 longitudinal cracks, D10 transverse cracks, D20 alligator cracks and D40 potholes** with a
YOLO26s model trained on RDD2022 and other public datasets (see the *Training & Experiments* page for the
measured results). Upload an image or video, attach browser GPS / a map point / EXIF GPS, and view the
detections on an OpenStreetMap map.

* Runs on the free CPU hardware: images take ~0.3–1 s, videos are processed slowly (use a high frame stride).
* Storage is ephemeral: saved detections are cleared when the Space restarts.
* Sample photos to try are in `demo_images/` (held-out RDD2022 test images, CC BY 4.0).
* Severity is a transparent heuristic, not an engineering-certified road condition rating.

Source code: https://github.com/ethical0101/RDD-YOLO
"""

REQS = """ultralytics==8.4.174
fastapi==0.142.2
uvicorn[standard]==0.54.0
python-multipart==0.0.32
sqlalchemy==2.1.3
pydantic-settings==2.15.0
python-dotenv==1.2.4
imageio-ffmpeg==0.6.0
pillow-heif==1.8.0
psutil==7.2.2
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").exists():
        sys.exit("Build the dashboard first: cd frontend; npm run build")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "rdd_yolo", OUT / "rdd_yolo", ignore=ignore)
    shutil.copytree(ROOT / "backend", OUT / "backend", ignore=ignore)
    shutil.copytree(ROOT / "models" / "architectures", OUT / "models" / "architectures")
    shutil.copytree(dist, OUT / "frontend" / "dist")
    shutil.copytree(ROOT / "demo_images", OUT / "demo_images")
    for w in (ROOT / "models" / "weights").glob("*_best.*"):
        copy(w, OUT / "models" / "weights" / w.name)
    exp = ROOT / "experiments"
    for f in ("comparison.json", "comparison.md", "comparison_curves.png", "comparison_metrics.png"):
        if (exp / f).exists():
            copy(exp / f, OUT / "experiments" / f)
    for run in RUNS:
        for f in RUN_FILES:
            if (exp / run / f).exists():
                copy(exp / run / f, OUT / "experiments" / run / f)
        for ev in (exp / run).glob("eval_*"):
            for f in EVAL_FILES:
                if (ev / f).exists():
                    copy(ev / f, OUT / "experiments" / run / ev.name / f)
    ds = ROOT / "dataset" / "processed" / "rdd2022_yolo"
    for f in ("dataset_stats.json", "validation_report.json"):
        if (ds / f).exists():
            copy(ds / f, OUT / "dataset" / "processed" / "rdd2022_yolo" / f)
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQS, encoding="utf-8")
    (OUT / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.jpg filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"Bundle ready: {OUT} ({n} files, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

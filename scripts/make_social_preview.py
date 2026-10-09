"""Render the GitHub social preview card (1280x640) -> docs/social-preview.png.

The four tiles are real detections of the served YOLO26s model on held-out demo images (demo_images/);
the metric shown is read from experiments/yolo26s_full/eval_test/metrics.json.
Usage: .venv\\Scripts\\python scripts\\make_social_preview.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdd_yolo.detector import Detector  # noqa: E402

W, H = 1280, 640
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
COLORS = {"D00": (42, 120, 214), "D10": (27, 175, 122), "D20": (74, 58, 167), "D40": (235, 104, 52)}
TILES = [("D40_pothole_1_Japan_Japan_001247.jpg", "Pothole"),
         ("D20_alligator_3_China_MotorBike_China_MotorBike_000219.jpg", "Alligator crack"),
         ("D00_longitudinal_1_China_MotorBike_China_MotorBike_000941.jpg", "Longitudinal crack"),
         ("D10_transverse_1_China_MotorBike_China_MotorBike_001324.jpg", "Transverse crack")]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def tile(det: Detector, name: str, size: int) -> Image.Image:
    img = cv2.imread(str(ROOT / "demo_images" / name))
    dets, _ = det.predict(img, conf=0.4)
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(pil)
    sx, sy = size / w, size / h
    f = font(BOLD, 15)
    for x in dets:
        c = COLORS[x.class_code]
        x1, y1, x2, y2 = x.bbox[0] * sx, x.bbox[1] * sy, x.bbox[2] * sx, x.bbox[3] * sy
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = f"{x.class_code} {x.confidence:.2f}"
        tw = d.textlength(label, font=f)
        ty = max(0, y1 - 22)
        lx = min(max(0, x1), size - tw - 10)  # keep the label inside the tile
        d.rectangle([lx, ty, lx + tw + 10, ty + 21], fill=c)
        d.text((lx + 5, ty + 1), label, font=f, fill="white")
    return pil


def main() -> None:
    m = json.loads((ROOT / "experiments" / "yolo26s_full" / "eval_test" / "metrics.json").read_text())
    map50 = 100 * m["overall"]["mAP50"]

    card = Image.new("RGB", (W, H), (11, 15, 25))
    # subtle diagonal glow
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-300, -360, 700, 520], fill=(40, 52, 88))
    gd.ellipse([760, 260, 1600, 1000], fill=(70, 38, 20))
    card = Image.blend(card, glow.filter(ImageFilter.GaussianBlur(160)), 0.55)
    d = ImageDraw.Draw(card)

    # --- right: 2x2 real detections (inside the 80 px safe border)
    size, gap, x0, y0 = 226, 14, 714, 94
    det = Detector(ROOT / "models" / "weights" / "yolo26s_full_best.pt")
    for i, (name, caption) in enumerate(TILES):
        t = tile(det, name, size)
        x, y = x0 + (i % 2) * (size + gap), y0 + (i // 2) * (size + gap)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], 14, fill=255)
        card.paste(t, (x, y), mask)
        d.rounded_rectangle([x, y, x + size - 1, y + size - 1], 14, outline=(255, 255, 255), width=2)

    # --- left: text
    lx = 96
    d.text((lx, 100), "DEEP LEARNING  ·  COMPUTER VISION  ·  GIS", font=font(BOLD, 17), fill=(235, 104, 52))
    d.text((lx - 4, 128), "RDD-YOLO", font=font(BOLD, 92), fill="white")
    d.text((lx, 248), "AI road damage detection,", font=font(FONT, 32), fill=(226, 232, 240))
    d.text((lx, 290), "geolocation & mapping", font=font(FONT, 32), fill=(226, 232, 240))

    chips = [f"{map50:.2f}% mAP@50", "YOLO26s", "4 damage classes", "10 public datasets"]
    cx, cy = lx, 362
    fc = font(BOLD, 19)
    for c in chips:
        tw = d.textlength(c, font=fc)
        if cx + tw + 30 > 680:
            cx, cy = lx, cy + 48
        d.rounded_rectangle([cx, cy, cx + tw + 28, cy + 38], 19, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
        d.text((cx + 14, cy + 6), c, font=fc, fill="white")
        cx += tw + 40

    d.text((lx, 478), "FastAPI · React · OpenStreetMap · ONNX Runtime WebGPU", font=font(FONT, 19), fill=(148, 163, 184))
    d.rounded_rectangle([lx, 512, lx + 8, 540], 3, fill=(235, 104, 52))
    d.text((lx + 20, 510), "Live demo  ethical0101.github.io/RDD-YOLO", font=font(BOLD, 22), fill="white")

    out = ROOT / "docs" / "social-preview.png"
    card.save(out, optimize=True)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

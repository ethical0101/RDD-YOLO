"""Inference engine: image, video and frame detection with a trained checkpoint."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .constants import CLASS_CODES, CLASS_NAMES
from .modules import register_custom_modules
from .severity import estimate_severity

# BGR colours per class for drawing.
CLASS_COLORS = {"D00": (214, 120, 42), "D10": (122, 175, 27), "D20": (167, 58, 74), "D40": (52, 104, 235)}  # matches the dashboard palette


@dataclass
class Detection:
    class_id: int
    class_code: str
    class_name: str
    confidence: float
    bbox: list[float]  # x1, y1, x2, y2 in pixels
    bbox_norm: list[float]  # x1, y1, x2, y2 in [0, 1]
    severity: str
    severity_score: float
    severity_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    """Thread-safe wrapper around an Ultralytics checkpoint."""

    def __init__(self, weights: str | Path, device: str | None = None, imgsz: int = 640):
        register_custom_modules()  # required before unpickling RDD-YOLO checkpoints
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.weights}")
        self.model = YOLO(str(self.weights), task="detect")
        if device is None:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        self._lock = threading.Lock()
        names = self.model.names
        if [names[i] for i in sorted(names)] != CLASS_CODES:
            raise ValueError(f"Checkpoint classes {names} do not match expected {CLASS_CODES}")
        meta_file = self.weights.with_suffix(".json")
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    @property
    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.model.parameters())
        return {"weights": self.weights.name, "device": self.device, "imgsz": self.imgsz,
                "parameters": n_params, "classes": CLASS_CODES, **self.meta}

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7,
                nearby_counter: Callable[[str], int] | None = None) -> tuple[list[Detection], float]:
        """Run detection on one BGR image. Returns (detections, inference_ms)."""
        h, w = image_bgr.shape[:2]
        with self._lock:
            t0 = time.perf_counter()
            res = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, iou=iou, device=self.device,
                                     verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
        dets: list[Detection] = []
        boxes = res.boxes
        for xyxy, c, k in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            k = int(k)
            code = CLASS_CODES[k]
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            nearby = nearby_counter(code) if nearby_counter else 0
            sev = estimate_severity(code, (x1, y1, x2, y2), (w, h), float(c), nearby)
            dets.append(Detection(k, code, CLASS_NAMES[code], round(float(c), 4),
                                  [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                  [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
                                  sev.level, sev.score, sev.to_dict()))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets, ms


def draw_detections(image_bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    th = max(1, round(min(out.shape[:2]) / 300))
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = CLASS_COLORS.get(d.class_code, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th + 1)
        label = f"{d.class_code} {d.confidence:.2f} {d.severity}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * th, th)
        yt = max(y1, tht + 6)
        cv2.rectangle(out, (x1, yt - tht - 6), (x1 + tw + 6, yt), color, -1)
        cv2.putText(out, label, (x1 + 3, yt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * th, (255, 255, 255), th,
                    cv2.LINE_AA)
    return out


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    detections: list[Detection]
    inference_ms: float


def iter_video(path: str | Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def video_properties(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    props = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
             "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    props["duration_s"] = props["frames"] / props["fps"] if props["fps"] else 0
    return props


def process_video(detector: Detector, src: str | Path, dst: str | Path, conf: float = 0.25, stride: int = 1,
                  max_side: int = 1280, progress: Callable[[float], None] | None = None) -> tuple[list[FrameResult], dict]:
    """Detect on every ``stride``-th frame and write an H.264 MP4 with boxes drawn.

    Frames between processed frames reuse the latest detections so the output
    video keeps the original frame rate. H.264 encoding uses the ffmpeg binary
    bundled with ``imageio-ffmpeg`` so the result plays in browsers.
    """
    import imageio_ffmpeg

    props = video_properties(src)
    scale = min(1.0, max_side / max(props["width"], props["height"], 1))
    ow, oh = int(props["width"] * scale) // 2 * 2, int(props["height"] * scale) // 2 * 2
    writer = imageio_ffmpeg.write_frames(str(dst), (ow, oh), fps=props["fps"], codec="libx264",
                                         pix_fmt_in="rgb24", output_params=["-pix_fmt", "yuv420p"],
                                         macro_block_size=2, quality=7)
    writer.send(None)
    results: list[FrameResult] = []
    last: list[Detection] = []
    t_start = time.perf_counter()
    infer_ms_total = 0.0
    try:
        for idx, ts, frame in iter_video(src, stride=1):
            if scale < 1.0:
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] != ow or frame.shape[0] != oh:
                frame = cv2.resize(frame, (ow, oh))
            if idx % stride == 0:
                last, ms = detector.predict(frame, conf=conf)
                infer_ms_total += ms
                results.append(FrameResult(idx, round(ts, 3), last, round(ms, 2)))
                if progress and props["frames"]:
                    progress(min(1.0, idx / props["frames"]))
            writer.send(np.ascontiguousarray(cv2.cvtColor(draw_detections(frame, last), cv2.COLOR_BGR2RGB)))
    finally:
        writer.close()
    wall = time.perf_counter() - t_start
    n = len(results)
    summary = {**props, "processed_frames": n, "stride": stride, "output_size": [ow, oh],
               "wall_time_s": round(wall, 2),
               "mean_inference_ms": round(infer_ms_total / n, 2) if n else None,
               "inference_fps": round(1000 * n / infer_ms_total, 2) if infer_ms_total else None}
    return results, summary

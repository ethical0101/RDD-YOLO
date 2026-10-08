"""Transparent, rule-based severity estimate for a single detection.

IMPORTANT: this is an explainable *heuristic* for prioritising review. It is
NOT an engineering-certified pavement condition assessment (e.g. it is not
PCI / ASTM D6433) and must not be presented as one.

Score (0-100) = 100 * (W_CLASS * class_weight
                       + W_AREA  * area_score
                       + W_CONF  * confidence)
                + repeat_bonus

  class_weight  : prior hazard of the damage type (pothole > alligator > cracks)
  area_score    : min(1, sqrt(relative_box_area / AREA_FULL)) — the box area as
                  a fraction of the image; sqrt so small-but-real damage still
                  scores, saturating at AREA_FULL (25 % of the frame)
  confidence    : model confidence (low-confidence detections are less certain)
  repeat_bonus  : +REPEAT_STEP per previously stored detection within
                  REPEAT_RADIUS_M metres (same location reported repeatedly),
                  capped at REPEAT_MAX. Only applies when GPS is known.

Levels: LOW < 40 <= MEDIUM < 65 <= HIGH
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

CLASS_WEIGHT = {"D00": 0.45, "D10": 0.50, "D20": 0.80, "D40": 1.00}
W_CLASS, W_AREA, W_CONF = 0.45, 0.40, 0.15
AREA_FULL = 0.25
REPEAT_RADIUS_M = 15.0
REPEAT_STEP = 5.0
REPEAT_MAX = 15.0
MEDIUM_THRESHOLD = 40.0
HIGH_THRESHOLD = 65.0


@dataclass
class Severity:
    level: str
    score: float
    class_weight: float
    area_ratio: float
    area_score: float
    confidence: float
    repeat_count: int
    repeat_bonus: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_severity(class_code: str, bbox_xyxy: tuple[float, float, float, float], image_wh: tuple[int, int],
                      confidence: float, nearby_count: int = 0) -> Severity:
    x1, y1, x2, y2 = bbox_xyxy
    w, h = image_wh
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(w * h, 1)
    area_score = min(1.0, math.sqrt(area_ratio / AREA_FULL))
    cw = CLASS_WEIGHT.get(class_code, 0.5)
    bonus = min(REPEAT_MAX, REPEAT_STEP * max(0, nearby_count))
    score = 100 * (W_CLASS * cw + W_AREA * area_score + W_CONF * float(confidence)) + bonus
    score = round(min(100.0, score), 1)
    level = "HIGH" if score >= HIGH_THRESHOLD else "MEDIUM" if score >= MEDIUM_THRESHOLD else "LOW"
    return Severity(level, score, cw, round(area_ratio, 5), round(area_score, 4), round(float(confidence), 4),
                    nearby_count, bonus)


def severity_rules() -> dict:
    """Machine-readable description of the rules (served by the API / shown in the UI)."""
    return {
        "disclaimer": "Heuristic prioritisation aid only - not an engineering-certified road condition assessment.",
        "formula": "score = 100*(0.45*class_weight + 0.40*min(1, sqrt(area_ratio/0.25)) + 0.15*confidence)"
                   " + min(15, 5*nearby_detections_within_15m)",
        "class_weight": CLASS_WEIGHT,
        "weights": {"class": W_CLASS, "area": W_AREA, "confidence": W_CONF},
        "area_saturation_ratio": AREA_FULL,
        "repeat": {"radius_m": REPEAT_RADIUS_M, "step": REPEAT_STEP, "max": REPEAT_MAX},
        "levels": {"LOW": f"< {MEDIUM_THRESHOLD}", "MEDIUM": f"{MEDIUM_THRESHOLD} - {HIGH_THRESHOLD}",
                   "HIGH": f">= {HIGH_THRESHOLD}"},
    }

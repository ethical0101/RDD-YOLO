"""Agreement between the RDD2022 labels we train on and the N-RDD2024 re-annotation of the same images.

N-RDD2024 (Kaya and Codur, Mendeley Data, CC BY 4.0) is a Roboflow export: every image was stretched to 640x640,
so normalised box coordinates are directly comparable with our YOLO labels. Only the four classes we model are
compared (D00, D10, D20, D40); N-RDD2024's extra classes (D30, D50-D90) are ignored.

For each image present in both: greedy one-to-one matching at IoU >= 0.5, ignoring class first ("is the same damage
boxed?"), then the class agreement of the matched pairs. Writes docs/diagnostics/label_agreement.json and .md.

Usage: .venv\\Scripts\\python scripts\\label_agreement.py
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NRDD = ROOT / "dataset" / "raw" / "external" / "nrdd2024"
V3 = ROOT / "dataset" / "processed" / "rdd2022_extended_v3" / "labels"
OUT = ROOT / "docs" / "diagnostics"
CLS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
NAMES = list(CLS)
SPLITS = ["train", "val", "test", "test_rdd_new"]


def read_nrdd() -> dict[str, list[tuple[int, list[float]]]]:
    out = {}
    for x in NRDD.glob("*/*/*.xml"):
        stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", x.stem)
        r = ET.parse(x).getroot()
        w, h = float(r.findtext("size/width")), float(r.findtext("size/height"))
        boxes = []
        for o in r.iter("object"):
            c = CLS.get(o.findtext("name"))
            if c is None:
                continue
            b = o.find("bndbox")
            x1, x2, y1, y2 = (float(b.findtext(k)) for k in ("xmin", "xmax", "ymin", "ymax"))
            boxes.append((c, [x1 / w, y1 / h, x2 / w, y2 / h]))
        country = stem.rsplit("_", 1)[0]
        out[f"{country}_{stem}"] = boxes  # our processed name: <Country>_<original stem>
    return out


def read_yolo(p: Path) -> list[tuple[int, list[float]]]:
    out = []
    for ln in p.read_text().split("\n"):
        f = ln.split()
        if len(f) == 5:
            c, cx, cy, w, h = int(f[0]), *map(float, f[1:])
            out.append((c, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return out


def iou(a, b) -> float:
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(ours, theirs, thr=0.5):
    pairs = sorted(((iou(a[1], b[1]), i, j) for i, a in enumerate(ours) for j, b in enumerate(theirs)), reverse=True)
    ui, uj, m = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in ui and j not in uj:
            ui.add(i), uj.add(j), m.append((i, j))
    return m


def main() -> None:
    nrdd = read_nrdd()
    res, conf = {}, defaultdict(Counter)
    for split in SPLITS:
        d = V3 / split
        if not d.exists():
            continue
        s = Counter()
        per_country = defaultdict(Counter)
        for p in d.glob("*.txt"):
            if p.stem not in nrdd:
                continue
            ours, theirs = read_yolo(p), nrdd[p.stem]
            m = match(ours, theirs)
            same = sum(ours[i][0] == theirs[j][0] for i, j in m)
            prefix = p.stem.rsplit("_", 1)[0]  # "<Country>_<Country>"
            country = prefix[: len(prefix) // 2]
            for c in (s, per_country[country]):
                c.update(images=1, ours=len(ours), theirs=len(theirs), matched=len(m), same_class=same,
                         ours_empty=not ours, theirs_empty=not theirs)
            for i, j in m:
                conf[split][(NAMES[ours[i][0]], NAMES[theirs[j][0]])] += 1
        if not s["images"]:
            continue

        def summ(c):
            return {"images_in_both": c["images"], "boxes_rdd2022": c["ours"], "boxes_nrdd2024": c["theirs"],
                    "matched_iou50": c["matched"],
                    "recall_of_rdd2022_boxes": round(c["matched"] / max(c["ours"], 1), 4),
                    "precision_vs_rdd2022": round(c["matched"] / max(c["theirs"], 1), 4),
                    "box_f1": round(2 * c["matched"] / max(c["ours"] + c["theirs"], 1), 4),
                    "class_agreement_of_matched": round(c["same_class"] / max(c["matched"], 1), 4)}
        res[split] = {**summ(s), "per_country": {k: summ(v) for k, v in sorted(per_country.items())},
                      "matched_class_pairs": {f"{a}->{b}": n for (a, b), n in sorted(conf[split].items())}}
    test_in_nrdd = res.get("test", {}).get("images_in_both", 0)
    out = {"source": "N-RDD2024, Mendeley Data doi:10.17632/27c8pwsd6v.3 (CC BY 4.0)", "nrdd_images_with_xml": len(nrdd),
           "iou_threshold": 0.5, "classes_compared": NAMES, "splits": res,
           "note": f"{test_in_nrdd} of our RDD2022 test images are in N-RDD2024's own train/valid split; "
                   "N-RDD2024 labels must not be used to train a model evaluated on our test split without excluding them."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_agreement.json").write_text(json.dumps(out, indent=2))

    L = ["# Label agreement: RDD2022 vs N-RDD2024", "", "Generated by `scripts/label_agreement.py`. Same images, two independent "
         "annotations. Boxes are matched one-to-one at IoU ≥ 0.5 ignoring class; class agreement is measured on the matched "
         "pairs. Only D00, D10, D20 and D40 are compared.", "",
         "| Split | Images in both | RDD2022 boxes | N-RDD2024 boxes | Matched | Share of RDD2022 boxes matched | Box F1 | Class agreement |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        L.append(f"| {k} | {v['images_in_both']:,} | {v['boxes_rdd2022']:,} | {v['boxes_nrdd2024']:,} | {v['matched_iou50']:,} | "
                 f"{v['recall_of_rdd2022_boxes']:.1%} | {v['box_f1']:.1%} | {v['class_agreement_of_matched']:.1%} |")
    if "train" in res:
        L += ["", "Per country (train split):", "", "| Country | Images | Box F1 | Class agreement |", "|---|---|---|---|"]
        L += [f"| {k} | {v['images_in_both']:,} | {v['box_f1']:.1%} | {v['class_agreement_of_matched']:.1%} |"
              for k, v in res["train"]["per_country"].items()]
    L += ["", out["note"]]
    (OUT / "LABEL_AGREEMENT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

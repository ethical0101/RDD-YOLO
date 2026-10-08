"""Validate RDD2022 (Pascal VOC) and convert it into a YOLO detection dataset.

Pipeline:
  1. Scan  raw/RDD2022/<Country>/train/{images,annotations/xmls}
  2. Validate every image (decodable, size matches XML) and annotation
     (parseable XML, known class, valid box). Problems are logged, never hidden.
  3. Keep only the 4 target classes D00/D10/D20/D40 (other RDD2022 labels such
     as D43/D44/D50/Repair are counted and dropped).
  4. Stratified (per-country) train/val/test split with a fixed seed.
     The official RDD2022 test split has no public labels, so the labelled
     train split is re-split for honest evaluation.
  5. Write YOLO labels, data.yaml, a validation report and dataset statistics
     (JSON + class-distribution chart).

Usage:
    python dataset/prepare_dataset.py                       # full prepare
    python dataset/prepare_dataset.py --validate-only        # report only
    python dataset/prepare_dataset.py --max-per-country 50 --out dataset/processed/smoke   # tiny smoke set
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdd_yolo.constants import CLASS_CODES, CLASS_INDEX, CLASS_NAMES, PROCESSED_DIR, RAW_DIR  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Box:
    cls: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Sample:
    country: str
    image: Path
    xml: Path | None
    width: int = 0
    height: int = 0
    boxes: list[Box] = field(default_factory=list)
    dropped_labels: list[str] = field(default_factory=list)


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            objs.append((name, float("nan"), 0, 0, 0))
            continue
        objs.append((name, *(float(bb.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return w, h, objs


def validate_image(path: Path) -> tuple[int, int] | str:
    """Return (w, h) for a valid image or an error string."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:
            im.load()  # full decode catches truncated files
            return im.size
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def scan(raw_dir: Path, countries: list[str] | None, report: dict) -> list[Sample]:
    samples: list[Sample] = []
    for cdir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if countries and cdir.name not in countries:
            continue
        img_dir, xml_dir = cdir / "train" / "images", cdir / "train" / "annotations" / "xmls"
        if not img_dir.exists():
            report["issues"].append({"country": cdir.name, "issue": "missing train/images directory"})
            continue
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        xmls = {p.stem: p for p in xml_dir.glob("*.xml")} if xml_dir.exists() else {}
        for stem in sorted(set(xmls) - set(images)):
            report["issues"].append({"file": str(xmls[stem]), "issue": "annotation without image"})
        for stem, img in sorted(images.items()):
            samples.append(Sample(cdir.name, img, xmls.get(stem)))
        report["countries"][cdir.name] = {"images_found": len(images), "xmls_found": len(xmls)}
    return samples


def validate(samples: list[Sample], report: dict) -> list[Sample]:
    valid: list[Sample] = []
    ignored = Counter()
    for i, s in enumerate(samples, 1):
        if i % 2000 == 0:
            print(f"  validated {i}/{len(samples)}", flush=True)
        res = validate_image(s.image)
        if isinstance(res, str):
            report["issues"].append({"file": str(s.image), "issue": f"corrupt image ({res})"})
            continue
        s.width, s.height = res
        if s.xml is None:
            # RDD2022 ships an XML for every train image; a missing one is reported and the image
            # is treated as background-only (no objects), which is the only honest interpretation.
            report["issues"].append({"file": str(s.image), "issue": "missing annotation (kept as background)"})
            valid.append(s)
            continue
        try:
            xw, xh, objs = parse_voc(s.xml)
        except Exception as exc:
            report["issues"].append({"file": str(s.xml), "issue": f"unparseable XML ({exc})"})
            continue
        if (xw, xh) != (0, 0) and (xw, xh) != (s.width, s.height):
            report["issues"].append({"file": str(s.xml),
                                     "issue": f"XML size {xw}x{xh} != image {s.width}x{s.height} (image size used)"})
        for name, x1, y1, x2, y2 in objs:
            if name not in CLASS_INDEX:
                ignored[name] += 1
                s.dropped_labels.append(name)
                continue
            if any(v != v for v in (x1, y1, x2, y2)):  # NaN
                report["issues"].append({"file": str(s.xml), "issue": f"{name}: missing bndbox values"})
                continue
            x1, x2 = sorted((max(0.0, min(x1, s.width)), max(0.0, min(x2, s.width))))
            y1, y2 = sorted((max(0.0, min(y1, s.height)), max(0.0, min(y2, s.height))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                report["issues"].append({"file": str(s.xml), "issue": f"{name}: degenerate box dropped"})
                continue
            s.boxes.append(Box(name, x1, y1, x2, y2))
        valid.append(s)
    report["ignored_labels"] = dict(ignored.most_common())
    return valid


def split(samples: list[Sample], ratios: tuple[float, float, float], seed: int,
          bg_ratio: float, max_per_country: int | None) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    by_country: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_country[s.country].append(s)
    out = {"train": [], "val": [], "test": []}
    for country, items in sorted(by_country.items()):
        items.sort(key=lambda s: s.image.name)
        pos = [s for s in items if s.boxes]
        neg = [s for s in items if not s.boxes]
        rng.shuffle(pos)
        rng.shuffle(neg)
        # Cap background (no-damage) images to a fraction of positive images.
        neg = neg[: int(len(pos) * bg_ratio)]
        chosen = pos + neg
        rng.shuffle(chosen)
        if max_per_country:
            chosen = chosen[:max_per_country]
        n = len(chosen)
        n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
        out["train"] += chosen[:n_tr]
        out["val"] += chosen[n_tr:n_tr + n_va]
        out["test"] += chosen[n_tr + n_va:]
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # hard link: no extra disk usage on the same volume
    except OSError:
        shutil.copy2(src, dst)


def write_yolo(splits: dict[str, list[Sample]], out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for name, items in splits.items():
        (out / "images" / name).mkdir(parents=True)
        (out / "labels" / name).mkdir(parents=True)
        for s in items:
            stem = f"{s.country}_{s.image.stem}"
            link_or_copy(s.image, out / "images" / name / f"{stem}{s.image.suffix.lower()}")
            lines = []
            for b in s.boxes:
                cx = (b.xmin + b.xmax) / 2 / s.width
                cy = (b.ymin + b.ymax) / 2 / s.height
                bw = (b.xmax - b.xmin) / s.width
                bh = (b.ymax - b.ymin) / s.height
                lines.append(f"{CLASS_INDEX[b.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / name / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    data_yaml = (
        "# Auto-generated by dataset/prepare_dataset.py from the official RDD2022 release.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_CODES)}\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_CODES))
    )
    (out / "data.yaml").write_text(data_yaml)


def statistics(splits: dict[str, list[Sample]]) -> dict:
    stats: dict = {"classes": {c: CLASS_NAMES[c] for c in CLASS_CODES}, "splits": {}}
    for name, items in splits.items():
        cls = Counter(b.cls for s in items for b in s.boxes)
        per_country = Counter(s.country for s in items)
        rel_areas = [((b.xmax - b.xmin) * (b.ymax - b.ymin)) / (s.width * s.height) for s in items for b in s.boxes]
        rel_areas.sort()
        stats["splits"][name] = {
            "images": len(items),
            "background_images": sum(1 for s in items if not s.boxes),
            "instances": sum(cls.values()),
            "instances_per_class": {c: cls.get(c, 0) for c in CLASS_CODES},
            "images_per_country": dict(sorted(per_country.items())),
            "box_relative_area": {
                "median": rel_areas[len(rel_areas) // 2] if rel_areas else None,
                "p10": rel_areas[len(rel_areas) // 10] if rel_areas else None,
                "p90": rel_areas[9 * len(rel_areas) // 10] if rel_areas else None,
            },
        }
    return stats


def plot_distribution(stats: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.27
    for k, split_name in enumerate(("train", "val", "test")):
        counts = [stats["splits"][split_name]["instances_per_class"][c] for c in CLASS_CODES]
        xs = [i + (k - 1) * width for i in range(len(CLASS_CODES))]
        bars = ax.bar(xs, counts, width, label=split_name)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(range(len(CLASS_CODES)), [f"{c}\n{CLASS_NAMES[c]}" for c in CLASS_CODES])
    ax.set_ylabel("Instances")
    ax.set_title("RDD2022 subset: class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--countries", nargs="*", default=None, help="Default: all extracted countries")
    ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--bg-ratio", type=float, default=0.1,
                    help="Max background (no target damage) images as a fraction of positive images")
    ap.add_argument("--max-per-country", type=int, default=None, help="Subsample for quick experiments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if not args.raw.exists():
        sys.exit(f"Raw dataset not found at {args.raw}. Run: python dataset/download_rdd2022.py")

    report: dict = {"raw_dir": str(args.raw), "countries": {}, "issues": []}
    print("Scanning ...")
    samples = scan(args.raw, args.countries, report)
    print(f"Found {len(samples)} images. Validating ...")
    valid = validate(samples, report)
    report["images_total"] = len(samples)
    report["images_valid"] = len(valid)
    report["images_with_target_damage"] = sum(1 for s in valid if s.boxes)
    report["issue_count"] = len(report["issues"])
    issue_types = Counter(i["issue"].split(" (")[0].split(":")[-1].strip() for i in report["issues"])
    report["issue_summary"] = dict(issue_types.most_common())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"Valid images: {len(valid)}/{len(samples)}; with target damage: {report['images_with_target_damage']}")
    print(f"Issues: {report['issue_summary']}")
    print(f"Ignored (non-target) labels: {report['ignored_labels']}")
    if args.validate_only:
        return

    splits = split(valid, tuple(args.split), args.seed, args.bg_ratio, args.max_per_country)
    print("Writing YOLO dataset ...")
    write_yolo(splits, args.out)
    stats = statistics(splits)
    stats["config"] = {"countries": sorted({s.country for s in valid}), "split": args.split,
                       "bg_ratio": args.bg_ratio, "seed": args.seed, "max_per_country": args.max_per_country}
    # Validation report goes next to the dataset as well so each processed set is self-describing.
    (args.out / "validation_report.json").write_text(json.dumps(report, indent=2))
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    plot_distribution(stats, args.out / "class_distribution.png")
    for name, st in stats["splits"].items():
        print(f"  {name:5s}: {st['images']:6d} images, {st['instances']:6d} boxes {st['instances_per_class']}")
    print(f"Done -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()

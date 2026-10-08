"""Geolocation helpers: EXIF GPS extraction, route interpolation, distances.

No coordinates are ever invented. Every function returns ``None`` when the
information is genuinely unavailable.
"""
from __future__ import annotations

import bisect
import csv
import io
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

LOCATION_SOURCES = ("browser", "exif", "manual", "route")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def validate_coords(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def _dms_to_deg(dms, ref: str) -> float:
    d, m, s = (float(x) for x in dms)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def extract_exif_gps(image_bytes: bytes) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF GPS tags, or None if absent/invalid."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            gps = im.getexif().get_ifd(0x8825)  # GPSInfo IFD
        if not gps or 2 not in gps or 4 not in gps:
            return None
        lat = _dms_to_deg(gps[2], gps.get(1, "N"))
        lon = _dms_to_deg(gps[4], gps.get(3, "E"))
        if not validate_coords(lat, lon) or (lat == 0 and lon == 0):
            return None
        return round(lat, 7), round(lon, 7)
    except Exception:
        return None


@dataclass
class RoutePoint:
    t: float  # seconds from the start of the video
    lat: float
    lon: float


class Route:
    """A GPS track aligned to video time; positions are linearly interpolated.

    Accepted inputs:
      * CSV with header ``time_s,lat,lon`` (seconds relative to video start), or
        ``timestamp,lat,lon`` with ISO-8601 timestamps (first row = video start)
      * GPX 1.1 track (``<trkpt lat lon><time>``), first point = video start
    """

    def __init__(self, points: list[RoutePoint]):
        if len(points) < 1:
            raise ValueError("Route needs at least one point")
        for p in points:
            if not validate_coords(p.lat, p.lon):
                raise ValueError(f"Invalid coordinate in route: {p}")
        self.points = sorted(points, key=lambda p: p.t)
        self._ts = [p.t for p in self.points]

    @classmethod
    def from_bytes(cls, data: bytes, filename: str) -> "Route":
        text = data.decode("utf-8-sig")
        if filename.lower().endswith(".gpx") or text.lstrip().startswith("<"):
            return cls._from_gpx(text)
        return cls._from_csv(text)

    @classmethod
    def _from_csv(cls, text: str) -> "Route":
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise ValueError("Empty route CSV")
        keys = {k.strip().lower(): k for k in rows[0]}
        lat_k, lon_k = keys.get("lat") or keys.get("latitude"), keys.get("lon") or keys.get("longitude")
        if not lat_k or not lon_k:
            raise ValueError("Route CSV needs lat/lon columns")
        if "time_s" in keys:
            pts = [RoutePoint(float(r[keys["time_s"]]), float(r[lat_k]), float(r[lon_k])) for r in rows]
        elif "timestamp" in keys:
            ts = [datetime.fromisoformat(r[keys["timestamp"]].replace("Z", "+00:00")) for r in rows]
            pts = [RoutePoint((t - ts[0]).total_seconds(), float(r[lat_k]), float(r[lon_k])) for t, r in zip(ts, rows)]
        else:
            raise ValueError("Route CSV needs a time_s or timestamp column")
        return cls(pts)

    @classmethod
    def _from_gpx(cls, text: str) -> "Route":
        root = ET.fromstring(text)
        pts, t0 = [], None
        for el in root.iter():
            if el.tag.endswith("trkpt") or el.tag.endswith("rtept"):
                tnode = next((c for c in el if c.tag.endswith("time")), None)
                if tnode is None or not tnode.text:
                    raise ValueError("GPX points need <time> to align with video")
                t = datetime.fromisoformat(tnode.text.strip().replace("Z", "+00:00"))
                t0 = t0 or t
                pts.append(RoutePoint((t - t0).total_seconds(), float(el.get("lat")), float(el.get("lon"))))
        return cls(pts)

    def position_at(self, t: float) -> tuple[float, float] | None:
        """Interpolated (lat, lon) at video time t; None outside the track's time span."""
        if t < self._ts[0] or t > self._ts[-1]:
            return None
        i = bisect.bisect_right(self._ts, t)
        if i >= len(self.points):
            p = self.points[-1]
            return p.lat, p.lon
        a, b = self.points[i - 1], self.points[i]
        f = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
        return round(a.lat + f * (b.lat - a.lat), 7), round(a.lon + f * (b.lon - a.lon), 7)

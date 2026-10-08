"""Selective downloader for the official RDD2022 dataset (figshare, CC BY 4.0).

The full archive is ~13.3 GB (Norway alone is ~9.9 GB of very high-resolution
images). Instead of downloading everything, this script reads the ZIP central
directory through HTTP Range requests and extracts only the members belonging
to the requested countries. Nothing is fabricated: every file written comes
byte-for-byte from the official archive (CRC32 is verified by `zipfile`).

Source: https://figshare.com/articles/dataset/21431547
Citation: Arya et al., "RDD2022: A multi-national image dataset for automatic
road damage detection", Geoscience Data Journal, 2024.

Usage:
    python dataset/download_rdd2022.py --countries Japan India Czech United_States China_MotorBike
    python dataset/download_rdd2022.py --list            # print member counts per country
"""
from __future__ import annotations

import argparse
import io
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_FILE_URL = "https://ndownloader.figshare.com/files/38030910"
ARCHIVE_SIZE = 13264172619  # bytes, from the figshare API (article 21431547)
DEFAULT_COUNTRIES = ["Japan", "India", "Czech", "United_States", "China_MotorBike"]
ROOT = Path(__file__).resolve().parent


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a block cache."""

    def __init__(self, url: str, size: int, block: int = 4 * 1024 * 1024):
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        # The figshare redirect yields a short-lived signed S3 URL, so the
        # ndownloader URL is requested every time (urllib keeps the Range header).
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"Server did not honour Range request (HTTP {r.status})")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError("short read")
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # network hiccup -> back off and retry
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}] {exc}; waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise IOError(f"Failed to fetch bytes {start}-{end}")

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            if len(self._cache) > 16:
                self._cache.pop(next(iter(self._cache)))
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            self._cache[idx] = self._fetch(start, end)
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._block(idx)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def inner_zip_data_offset(remote: HttpRangeFile, info: zipfile.ZipInfo) -> int:
    """Return the absolute offset of a stored member's data (after its local header)."""
    remote.seek(info.header_offset)
    hdr = remote.read(30)
    if hdr[:4] != b"PK":
        raise IOError("Bad local file header")
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    return info.header_offset + 30 + name_len + extra_len


def download_member(remote: HttpRangeFile, info: zipfile.ZipInfo, dest: Path, chunk: int = 32 << 20) -> None:
    """Stream a stored (uncompressed) member to disk with resume support."""
    if info.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"{info.filename} is compressed; direct range copy not possible")
    data_start = inner_zip_data_offset(remote, info)
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    t0, got = time.time(), 0
    with open(dest, "ab") as fh:
        while have < info.file_size:
            n = min(chunk, info.file_size - have)
            fh.write(remote._fetch(data_start + have, data_start + have + n - 1))
            have += n
            got += n
            print(f"  {dest.name}: {have / 1e6:.0f}/{info.file_size / 1e6:.0f} MB "
                  f"({got / 1e6 / max(time.time() - t0, 1e-6):.1f} MB/s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    ap.add_argument("--out", type=Path, default=ROOT / "raw")
    ap.add_argument("--include-test", action="store_true",
                    help="Also extract the official (unlabeled) test images")
    ap.add_argument("--list", action="store_true", help="Only list the per-country archives")
    ap.add_argument("--keep-zips", action="store_true", help="Keep the downloaded country zips")
    args = ap.parse_args()

    remote = HttpRangeFile(FIGSHARE_FILE_URL, ARCHIVE_SIZE)
    outer = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    members = {Path(i.filename).stem: i for i in outer.infolist() if i.filename.endswith(".zip")}
    if args.list:
        for name, info in members.items():
            print(f"{name:16s} {info.file_size / 1e6:9.1f} MB")
        return

    for country in args.countries:
        if country not in members:
            sys.exit(f"Unknown country {country!r}; available: {sorted(members)}")
        info = members[country]
        zpath = args.out / "_zips" / f"{country}.zip"
        marker = args.out / "RDD2022" / country / ".extracted"
        if marker.exists():
            print(f"[{country}] already extracted, skipping")
            continue
        print(f"[{country}] downloading {info.file_size / 1e6:.1f} MB")
        download_member(remote, info, zpath)
        print(f"[{country}] extracting")
        with zipfile.ZipFile(zpath) as inner:
            bad = inner.testzip()
            if bad:
                sys.exit(f"[{country}] CRC failure in {bad}; delete {zpath} and retry")
            names = [n for n in inner.namelist() if args.include_test or "/test/" not in n]
            dest_root = args.out / "RDD2022"
            # Inner zip paths may or may not start with the country folder.
            prefix = "" if names and names[0].split("/")[0] == country else country + "/"
            for n in names:
                if n.endswith("/"):
                    continue
                target = dest_root / (prefix + n)
                target.parent.mkdir(parents=True, exist_ok=True)
                with inner.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{len(names)} files extracted from {zpath.name}\n")
        if not args.keep_zips:
            zpath.unlink()
        print(f"[{country}] done ({len(names)} files)")
    print(f"All done. Data in {args.out / 'RDD2022'}")


if __name__ == "__main__":
    main()

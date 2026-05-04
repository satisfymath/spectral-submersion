"""Download Rongorongo Barthel transcript images from Archive.org.

This downloads the PNG transcript images from the archive.org item
'rongorongotexts' for local reference and potential future OCR processing.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import requests

ARCHIVE_ID = "rongorongotexts"
BASE_URL = f"https://archive.org/download/{ARCHIVE_ID}"
META_URL = f"https://archive.org/metadata/{ARCHIVE_ID}"


def download_rongorongo_images(output_dir: str, max_files: int | None = None):
    """Download transcript PNG images from archive.org."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata from {META_URL}...")
    resp = requests.get(META_URL, timeout=60)
    resp.raise_for_status()
    meta = resp.json()

    files = meta.get("files", [])
    png_files = [
        f
        for f in files
        if f.get("format") == "PNG" and "transcript" in f.get("name", "").lower()
    ]

    print(f"Found {len(png_files)} transcript PNG files")

    downloaded = 0
    for f in png_files[:max_files] if max_files else png_files:
        name = f["name"]
        url = f"{BASE_URL}/{quote(name)}"
        dest = out_path / name.replace(" ", "_")
        if dest.exists():
            print(f"  SKIP (exists): {name}")
            continue
        print(f"  DOWNLOAD: {name} ({f.get('size', '?')} bytes)")
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            downloaded += 1
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDownloaded {downloaded} new files to {out_path}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download Rongorongo transcript images"
    )
    parser.add_argument(
        "--output", default="data/raw/lost_language/rongorongo_archive_images"
    )
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    download_rongorongo_images(args.output, args.max_files)


if __name__ == "__main__":
    main()

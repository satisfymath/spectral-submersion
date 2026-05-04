"""Download real referent images from Wikimedia Commons.

The script records full provenance for every downloaded file. It is intended
as a lightweight bootstrap dataset for the real iconic-grounding pipeline, not
as a final curated paleoecological image corpus.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, "src")

from spectral_submersion.iconic_grounding import RapaNuiWorld1500  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "spectral-submersion-iconic-grounding/0.3 (research data bootstrap)"


def _safe_filename(text: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    if Path(stem).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        stem = str(Path(stem).with_suffix(""))
    return f"{stem[:120]}{suffix}"


def _image_suffix(url: str, mime: str | None = None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".jpg"


def _metadata_value(imageinfo: dict, key: str) -> str:
    ext = imageinfo.get("extmetadata") or {}
    value = ext.get(key, {}).get("value", "")
    return re.sub(r"<[^>]+>", "", value)


def commons_search_images(query: str, limit: int = 5) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": max(limit * 3, limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 512,
    }
    response = requests.get(
        COMMONS_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    results = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 999999)):
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = info.get("mime", "")
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        results.append(
            {
                "title": page.get("title", ""),
                "download_url": url,
                "source_url": info.get("url", ""),
                "description_url": info.get("descriptionurl", ""),
                "mime": mime,
                "license": _metadata_value(info, "LicenseShortName"),
                "artist": _metadata_value(info, "Artist"),
                "credit": _metadata_value(info, "Credit"),
            }
        )
        if len(results) >= limit:
            break
    return results


def _referent_query(referent) -> str:
    if referent.scientific_name and "sp." not in referent.scientific_name:
        return referent.scientific_name
    return referent.label


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real referent images")
    parser.add_argument(
        "--output-root",
        default="data/external/iconic_referents/rapa_nui_1500",
    )
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Pause between Wikimedia requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--referents",
        default="",
        help="Comma-separated referent IDs. Empty means all Rapa Nui 1500 referents.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    existing_rows = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    world = RapaNuiWorld1500()
    referents_by_id = world.by_id()
    if args.referents:
        referent_ids = [r.strip() for r in args.referents.split(",") if r.strip()]
    else:
        referent_ids = world.get_referent_set()

    rows = []
    for referent_id in referent_ids:
        if referent_id not in referents_by_id:
            print(f"Skipping unknown referent: {referent_id}")
            continue
        referent = referents_by_id[referent_id]
        query = _referent_query(referent)
        print(f"Searching {referent_id}: {query}")
        try:
            results = commons_search_images(query, limit=args.max_images)
            time.sleep(args.sleep_seconds)
        except Exception as exc:
            print(f"  FAILED search: {exc}")
            continue

        ref_dir = output_root / referent_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            suffix = _image_suffix(result["download_url"], result["mime"])
            local_path = ref_dir / _safe_filename(
                result["title"].replace("File:", ""), suffix
            )
            if not local_path.exists():
                try:
                    image_response = requests.get(
                        result["download_url"],
                        headers={"User-Agent": USER_AGENT},
                        timeout=60,
                    )
                    image_response.raise_for_status()
                    local_path.write_bytes(image_response.content)
                    time.sleep(args.sleep_seconds)
                except Exception as exc:
                    print(f"  FAILED download {result['title']}: {exc}")
                    continue
            rows.append(
                {
                    "referent_id": referent_id,
                    "query": query,
                    "title": result["title"],
                    "local_path": str(local_path),
                    "download_url": result["download_url"],
                    "source_url": result["source_url"],
                    "description_url": result["description_url"],
                    "mime": result["mime"],
                    "license": result["license"],
                    "artist": result["artist"],
                    "credit": result["credit"],
                }
            )
            print(f"  saved {local_path}")

    fieldnames = [
        "referent_id",
        "query",
        "title",
        "local_path",
        "download_url",
        "source_url",
        "description_url",
        "mime",
        "license",
        "artist",
        "credit",
    ]
    by_path = {
        row.get("local_path", ""): {key: row.get(key, "") for key in fieldnames}
        for row in existing_rows
        if row.get("local_path")
    }
    for row in rows:
        by_path[row["local_path"]] = row

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(by_path.values(), key=lambda row: row["local_path"]))

    print(f"\nDownloaded/verified {len(rows)} images")
    print(f"Manifest rows: {len(by_path)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

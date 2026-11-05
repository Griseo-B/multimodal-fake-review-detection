"""Turn the browser-collected Dianping reviews into a raw dataset folder.

`zh_collector.py` only records what the browser read: text, author, date, star
and the photo URLs. This script downloads those photos and writes the same
`reviews.json` envelope the 2026-10-29 collection uses, so the rest of the
pipeline (`build_supplement.py` -> `build_three_class.py`) can pick it up
without special cases.

Everything is de-duplicated on two keys: the review id and the normalised text,
because the canteen crawler revisits shops.

    python code/data/ingest_zh_collection.py
    python code/data/ingest_zh_collection.py --max-photos 6 --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "supplement" / "collected_dianping_chinese" / "dianping"
REVIEWS_JSONL = SRC / "reviews.jsonl"
IMAGES = SRC / "images"
OUT = SRC / "reviews.json"
MANIFEST = SRC / "manifest.json"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://www.dianping.com/"}
MIN_CHARS = 20
MAX_SIDE = 1024


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def key_of(text: str) -> str:
    return hashlib.md5(norm(text).encode("utf-8")).hexdigest()


def download(url: str, target: Path) -> bool:
    if target.is_file() and target.stat().st_size > 0:
        return True
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=8)
            if response.status_code != 200 or not response.content:
                return False
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=88)
            return True
        except requests.exceptions.Timeout:
            if attempt == 1:
                return False
        except Exception:  # noqa: BLE001 - a dead photo must not kill the run
            return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-photos", type=int, default=6)
    parser.add_argument("--refresh", action="store_true", help="re-download existing photos")
    parser.add_argument("--skip-download", action="store_true",
                        help="only record photos already on disk (fast pass)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not REVIEWS_JSONL.is_file():
        raise SystemExit(f"nothing collected yet: {REVIEWS_JSONL}")

    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    shops: dict[str, dict] = {}
    stats = Counter()
    failures = []

    for line in REVIEWS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row.get("review_id") or "").strip()
        text = norm(row.get("content"))
        if not rid or rid in seen_ids:
            stats["duplicate_id"] += 1
            continue
        if len(text) < MIN_CHARS:
            stats["too_short"] += 1
            continue
        text_key = key_of(text)
        if text_key in seen_text:
            stats["duplicate_text"] += 1
            continue
        seen_ids.add(rid)
        seen_text.add(text_key)

        shop_id = row.get("shop_id") or ""
        shop = shops.setdefault(shop_id, {"shop_id": shop_id, "shop": "", "city": "", "reviews": []})
        shop["shop"] = row.get("shop") or shop["shop"]
        shop["city"] = row.get("city") or shop["city"]

        photo_files = []
        urls = [url for url in (row.get("pictures") or []) if url][: args.max_photos]
        for index, url in enumerate(urls, start=1):
            name = f"{rid}_{index:02d}.jpg"
            target = IMAGES / name
            if args.skip_download:
                if target.is_file() and target.stat().st_size > 0:
                    photo_files.append(name)
                    stats["photos"] += 1
                else:
                    stats["photo_missing"] += 1
                continue
            if download(url, target):
                photo_files.append(name)
                stats["photos"] += 1
            else:
                failures.append({"review_id": rid, "url": url})
                stats["photo_failed"] += 1

        shop["reviews"].append(
            {
                "id": rid,
                "user": row.get("user") or "",
                "date": row.get("time") or "",
                "text": text,
                "images": urls,
                "photo_files": photo_files,
                "star": row.get("star") or "",
            }
        )
        stats["reviews"] += 1
        if args.limit and stats["reviews"] >= args.limit:
            break

    payload = {
        "platform": "dianping",
        "scene": "food",
        "basis": "human",
        "label": "real",
        "judged": True,
        "note": (
            "2026-11-01 补采的中文带图真评。浏览器登录态下逐店抓取店铺评价预览，"
            "每条保留最多 {max} 张原图。真评判定依据为平台真实消费评价（人工标注）。"
        ).format(max=args.max_photos),
        "collected_at": time.strftime("%Y-%m-%d"),
        "shop_count": len(shops),
        "review_count": stats["reviews"],
        "image_file_count": stats["photos"],
        "reviews_with_photos": sum(1 for s in shops.values() for r in s["reviews"] if r["photo_files"]),
        "download_failures": failures[:200],
        "shops": [shops[key] for key in sorted(shops)],
    }
    SRC.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST.write_text(
        json.dumps(
            {
                "stats": dict(stats),
                "shops": len(shops),
                "reviews": stats["reviews"],
                "failed_downloads": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"stats": dict(stats), "shops": len(shops)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

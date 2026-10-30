"""Fill reviews that still have no photo.

Missing hotels and phone brands are searched on Bing. Each saved file is a
real image, and the source address is kept. The same photo pool for a hotel
or phone is then shared by real reviews, human-written fake reviews, and AI
reviews of that same place, so the picture is not itself the label.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_human_fake_images as base

ROOT = base.ROOT
OUT = base.OUT
IMAGE_DIR = base.IMAGE_DIR
CATALOG = base.CATALOG
ASSIGNMENT = base.ASSIGNMENT
SUMMARY = base.SUMMARY
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
BLOCKED = (
    "alamy", "shutterstock", "gettyimages", "istock", "dreamstime",
    "depositphotos", "123rf", "pinterest.", "facebook.", "instagram.",
    "tiktok.",
)
QUERIES = {
    "ott-affinia": "Affinia Chicago hotel exterior",
    "ott-allegro": "Hotel Allegro Chicago exterior",
    "ott-amalfi": "Amalfi Hotel Chicago exterior",
    "ott-hardrock": "Hard Rock Hotel Chicago building",
    "ott-homewood": "Homewood Suites Chicago Downtown hotel exterior",
    "ott-omni": "Omni Chicago Hotel exterior",
    "ott-swissotel": "Swissotel Chicago exterior",
    "ott-talbott": "The Talbott Hotel Chicago exterior",
    "ott-james": "The James Chicago hotel exterior",
    "ott-conrad": "Conrad Chicago hotel exterior",
    "ott-sofitel": "Sofitel Chicago Magnificent Mile hotel",
    "ott-monaco": "Hotel Monaco Chicago exterior",
    "ott-fairmont": "Fairmont Chicago Millennium Park hotel",
    "ott-knickerbocker": "Millennium Knickerbocker Hotel Chicago",
    "fc-palomar": "Hotel Palomar Chicago exterior",
    "fc-sheraton-lax": "Sheraton Gateway Los Angeles hotel exterior",
    "fc-wchicago": "W Chicago City Center hotel exterior",
    "fc-crowne": "Crowne Plaza Houston River Oaks hotel",
    "fc-westin-lax": "The Westin Los Angeles Airport hotel exterior",
    "fc-magnolia": "Magnolia Hotel Houston exterior",
    "phone-lg": "LG G7 ThinQ smartphone",
}


def bing_images(query: str) -> list[str]:
    url = "https://www.bing.com/images/async?" + urllib.parse.urlencode({
        "q": query,
        "first": "1",
        "count": "35",
        "adlt": "off",
    })
    request = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(request, timeout=25).read().decode("utf-8", "replace")
    found = re.findall(r"murl&quot;:&quot;(https?://[^&]+)&quot;", html)
    if not found:
        found = re.findall(r'"murl":"(https?://[^"]+)"', html)
    unique = []
    for item in found:
        item = item.replace("\\u0026", "&")
        host = urllib.parse.urlparse(item).netloc.lower()
        if any(part in host for part in BLOCKED):
            continue
        if item not in unique:
            unique.append(item)
    return unique


def save_image(url: str, path: Path) -> tuple[int, int]:
    request = urllib.request.Request(url, headers={**UA, "Referer": "https://www.bing.com/"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(8_000_000)
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    if min(image.size) < 200:
        raise ValueError("too small")
    image.thumbnail((1024, 1024))
    image.save(path, "JPEG", quality=90)
    return image.size


def existing_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["entity"]] += 1
    return counts


def download_missing(rows: list[dict]) -> list[dict]:
    counts = existing_counts(rows)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for entity, query in QUERIES.items():
        have = counts[entity]
        need = 6 - have
        if need <= 0:
            continue
        try:
            urls = bing_images(query)
        except Exception as exc:
            print(f"{entity}: search failed {exc}", flush=True)
            continue
        added = 0
        for url in urls:
            if added >= need:
                break
            filename = f"{entity}-web-{have + added + 1:02d}.jpg"
            path = IMAGE_DIR / filename
            try:
                width, height = save_image(url, path)
            except Exception as exc:
                print(f"  skip {entity}: {exc}", flush=True)
                continue
            rows.append({
                "entity": entity,
                "file": filename,
                "commons_title": "",
                "license": "web",
                "license_url": url,
                "page_url": url,
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
            })
            added += 1
            time.sleep(0.15)
        print(f"{entity}: added {added}, urls {len(urls)}", flush=True)
    return rows


def assign_all(catalog: list[dict]) -> tuple[list[dict], dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in catalog:
        pools[row["entity"]].append(row)
    for items in pools.values():
        items.sort(key=lambda row: row["file"])
    text_to_hotel = base.four_city_hotels()
    counts: dict[str, int] = defaultdict(int)
    assigned: list[dict] = []
    by_label: dict[str, int] = defaultdict(int)

    def add(review_id: str, source: str, item: str, label: str, entity: str) -> None:
        pool = pools.get(entity) or []
        if not pool:
            return
        image = pool[counts[entity] % len(pool)]
        counts[entity] += 1
        by_label[label] += 1
        assigned.append({
            "review_id": review_id,
            "source": source,
            "item": item,
            "label": label,
            "entity": entity,
            "image_file": image["file"],
            "commons_title": image["commons_title"],
            "license": image["license"],
            "page_url": image["page_url"],
        })

    with base.TWO_CLASS.open(encoding="utf-8", newline="") as handle:
        for review in csv.DictReader(handle):
            entity = ""
            if review["source"] == "ott":
                entity = "ott-" + review["item"]
            elif review["source"] == "four_city" and review["label"] == "human_fake":
                entity = text_to_hotel.get(base.normalize(review["text"]), "")
            elif review["source"] == "soldner" and review["item"] in base.PHONE_QUERIES:
                entity = "phone-" + base.slug(review["item"])
            if entity:
                add(review["id"], review["source"], review["item"], review["label"], entity)

    ai_path = ROOT / "data" / "raw" / "hotel_ai" / "reviews_v2.csv"
    with ai_path.open(encoding="utf-8", newline="") as handle:
        for index, review in enumerate(csv.DictReader(handle), start=1):
            add(f"hotel-ai-{index:04d}", "hotel_ai", review["hotel_key"], "ai_fake", "ott-" + review["hotel_key"])

    reuse: dict[str, int] = defaultdict(int)
    for row in assigned:
        reuse[row["image_file"]] += 1
    summary = {
        "images": len(catalog),
        "reviews_assigned": len(assigned),
        "by_label": dict(by_label),
        "max_reviews_per_image": max(reuse.values()) if reuse else 0,
        "entities": {key: len(items) for key, items in sorted(pools.items())},
    }
    return assigned, summary


def main() -> None:
    with CATALOG.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = download_missing(rows)
    assigned, summary = assign_all(rows)
    fields = [
        "entity", "file", "commons_title", "license", "license_url", "page_url",
        "width", "height", "bytes",
    ]
    assign_fields = [
        "review_id", "source", "item", "label", "entity", "image_file",
        "commons_title", "license", "page_url",
    ]
    base.write_csv(CATALOG, fields, rows)
    base.write_csv(ASSIGNMENT, assign_fields, assigned)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "reviews_assigned", "by_label", "max_reviews_per_image")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

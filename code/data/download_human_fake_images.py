"""Attach freely licensed photos to human-written fake reviews.

Ott and Four-city reviews name a hotel. Soldner reviews name a phone.
Images come from Wikimedia Commons. A file is kept only when the page
states a free or Creative Commons license and the title or description
mentions that hotel or phone. Reviews of other sources have no stable
visual object in the frozen table, so they are left unmatched.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TWO_CLASS = ROOT / "data" / "frozen" / "two_class.csv"
OUT = RAW / "human_fake_images"
IMAGE_DIR = OUT / "images"
CATALOG = OUT / "catalog.csv"
ASSIGNMENT = OUT / "assignment.csv"
SUMMARY = OUT / "summary.json"
PROGRESS = OUT / "progress.json"
BUDGET_SECONDS = 480
API = "https://commons.wikimedia.org/w/api.php"
UA = "fake-review-research/1.0 (undergraduate dataset; local educational use)"
MAX_SIDE = 1024
MIN_SIDE = 480
CAP = 8
GENERIC = {"chicago", "hotel", "angeles", "york", "houston", "smartphone"}
LICENSE_MARKERS = (
    ("cc-by-sa-4.0", "CC BY-SA 4.0"),
    ("cc-by-4.0", "CC BY 4.0"),
    ("cc-by-sa-3.0", "CC BY-SA 3.0"),
    ("cc-by-3.0", "CC BY 3.0"),
    ("cc-by-sa-2.5", "CC BY-SA 2.5"),
    ("cc-by-2.5", "CC BY 2.5"),
    ("cc-by-sa-2.0", "CC BY-SA 2.0"),
    ("cc-by-2.0", "CC BY 2.0"),
    ("cc-by-sa", "CC BY-SA"),
    ("cc-by", "CC BY"),
    ("cc-zero", "CC0"),
    ("cc0", "CC0"),
    ("pd-self", "Public domain"),
    ("pd-us", "Public domain"),
    ("pd-old", "Public domain"),
)

# key, queries, tokens that must all appear, tokens of which none may appear
ENTITIES: list[tuple[str, list[str], list[str], list[str]]] = [
    ("ott-affinia", ["Affinia Chicago hotel"], ["affinia"], ["logo", "map"]),
    ("ott-allegro", ["Hotel Allegro Chicago"], ["allegro", "chicago"], ["logo", "map"]),
    ("ott-amalfi", ["Amalfi Hotel Chicago"], ["amalfi", "chicago"], ["logo", "map", "coast"]),
    ("ott-ambassador", ["Ambassador East Hotel Chicago", "Ambassador Chicago hotel"], ["ambassador", "chicago"], ["logo", "map"]),
    ("ott-conrad", ["Conrad Chicago hotel"], ["conrad", "chicago"], ["logo", "map"]),
    ("ott-fairmont", ["Fairmont Chicago hotel"], ["fairmont", "chicago"], ["logo", "map"]),
    ("ott-hardrock", ["Hard Rock Hotel Chicago"], ["hard rock", "chicago"], ["logo", "map"]),
    ("ott-hilton", ["Hilton Chicago hotel"], ["hilton", "chicago"], ["logo", "map", "midway", "ohare", "o'hare", "palmer"]),
    ("ott-homewood", ["Homewood Suites Chicago Downtown hotel"], ["homewood", "chicago"], ["logo", "map"]),
    ("ott-hyatt", ["Hyatt Regency Chicago hotel"], ["hyatt", "chicago"], ["logo", "map"]),
    ("ott-intercontinental", ["InterContinental Chicago hotel"], ["intercontinental", "chicago"], ["logo", "map"]),
    ("ott-james", ["The James Chicago hotel"], ["james", "chicago", "hotel"], ["logo", "map", "lebron"]),
    ("ott-knickerbocker", ["Knickerbocker Hotel Chicago"], ["knickerbocker", "chicago"], ["logo", "map"]),
    ("ott-monaco", ["Hotel Monaco Chicago"], ["monaco", "chicago"], ["logo", "map"]),
    ("ott-omni", ["Omni Chicago Hotel"], ["omni", "chicago"], ["logo", "map"]),
    ("ott-palmer", ["Palmer House Hilton Chicago"], ["palmer", "chicago"], ["logo", "map"]),
    ("ott-sheraton", ["Sheraton Grand Chicago hotel", "Sheraton Chicago Hotel"], ["sheraton", "chicago"], ["logo", "map", "gateway", "los angeles"]),
    ("ott-sofitel", ["Sofitel Chicago hotel"], ["sofitel", "chicago"], ["logo", "map"]),
    ("ott-swissotel", ["Swissotel Chicago"], ["swissotel"], ["logo", "map"]),
    ("ott-talbott", ["Talbott Hotel Chicago"], ["talbott"], ["logo", "map"]),
    ("fc-palomar", ["Hotel Palomar Chicago"], ["palomar", "chicago"], ["logo", "map"]),
    ("fc-wchicago", ["W Chicago City Center hotel"], ["city center", "chicago"], ["logo", "map"]),
    ("fc-crowne", ["Crowne Plaza Houston River Oaks"], ["crowne", "houston"], ["logo", "map"]),
    ("fc-magnolia", ["Magnolia Hotel Houston"], ["magnolia", "houston"], ["logo", "map"]),
    ("fc-sheraton-lax", ["Sheraton Gateway Los Angeles hotel"], ["sheraton", "gateway"], ["logo", "map", "chicago"]),
    ("fc-westin-lax", ["Westin Los Angeles Airport hotel"], ["westin", "angeles"], ["logo", "map"]),
    ("fc-pennsylvania", ["Hotel Pennsylvania New York"], ["pennsylvania", "hotel"], ["logo", "map"]),
    ("fc-waldorf", ["Waldorf Astoria New York hotel"], ["waldorf"], ["logo", "map"]),
]

PHONE_QUERIES = {
    "Apple": (["iPhone smartphone"], ["iphone"], ["mlb", "wafer", "chip", "motherboard", "logic board", "logo", "map"]),
    "Google": (["Google Pixel smartphone"], ["pixel"], ["mlb", "wafer", "chip", "motherboard", "logo", "map"]),
    "Huawei": (["Huawei smartphone"], ["huawei"], ["mlb", "chip", "motherboard", "logo", "map", "router"]),
    "LG": (["LG smartphone"], ["lg"], ["mlb", "chip", "motherboard", "logo", "map", "washer", "refrigerator", "tv"]),
    "Motorola": (["Motorola smartphone"], ["motorola"], ["mlb", "chip", "motherboard", "logo", "map", "radio"]),
    "OnePlus": (["OnePlus smartphone"], ["oneplus"], ["mlb", "chip", "motherboard", "logo", "map"]),
    "Oppo": (["Oppo smartphone"], ["oppo"], ["mlb", "chip", "motherboard", "logo", "map"]),
    "Samsung": (["Samsung Galaxy smartphone"], ["samsung", "galaxy"], ["mlb", "chip", "motherboard", "logo", "map", "tv"]),
    "Xiaomi": (["Xiaomi smartphone"], ["xiaomi"], ["mlb", "chip", "motherboard", "logo", "map", "scooter"]),
}

CATALOG_FIELDS = [
    "entity", "file", "commons_title", "license", "license_url", "page_url",
    "width", "height", "bytes",
]
ASSIGN_FIELDS = [
    "review_id", "source", "item", "entity", "image_file", "commons_title",
    "license", "page_url",
]


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def fetch(url: str, timeout: int = 25) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(last_error)


def read_license(file_title: str) -> str:
    name = file_title.removeprefix("File:")
    url = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(name.replace(" ", "_")) + "?action=raw"
    text = fetch(url, timeout=20).decode("utf-8", "replace").lower()
    if "non-free" in text or "fair use" in text or "all rights reserved" in text:
        return ""
    for marker, label in LICENSE_MARKERS:
        if marker in text:
            return label
    if "public domain" in text:
        return "Public domain"
    return ""


def search_files(query: str) -> list[str]:
    url = "https://commons.wikimedia.org/w/index.php?" + urllib.parse.urlencode({
        "title": "Special:MediaSearch",
        "type": "image",
        "search": query,
    })
    html_text = fetch(url, timeout=30).decode("utf-8", "replace")
    names = re.findall(r"File:[A-Za-z0-9_(),.'% -]{8,110}\.(?:jpg|jpeg|png|JPG|JPEG)", html_text)
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.replace("_", " ").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name.replace("_", " "))
    return unique


def search_entity(queries: list[str], required: list[str], forbidden: list[str]) -> list[dict]:
    distinctive = [token for token in required if token not in GENERIC]
    found: dict[str, dict] = {}
    for query in queries:
        try:
            titles = search_files(query)
        except Exception as exc:
            print(f"  search failed for {query}: {exc}", flush=True)
            continue
        for title in titles:
            filename = title.removeprefix("File:").lower()
            if any(token in filename for token in forbidden):
                continue
            if distinctive and not any(token in filename for token in distinctive):
                continue
            if "%" in title or len(title) > 140 or title in found:
                continue
            try:
                license_name = read_license(title)
            except Exception as exc:
                print(f"  license skip {title[:60]}: {exc}", flush=True)
                continue
            if not license_name:
                continue
            name = title.removeprefix("File:")
            found[title] = {
                "commons_title": title,
                "license": license_name,
                "license_url": "",
                "page_url": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name) + "?width=500",
            }
            if len(found) >= CAP:
                return list(found.values())
    return list(found.values())


def save_jpeg(payload: bytes, path: Path) -> tuple[int, int]:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    if min(image.size) < 320:
        raise ValueError("too small")
    image.save(path, "JPEG", quality=90)
    return image.size


def load_progress() -> tuple[list[str], list[dict]]:
    if not PROGRESS.exists():
        return [], []
    payload = json.loads(PROGRESS.read_text(encoding="utf-8"))
    return list(payload.get("done") or []), list(payload.get("catalog") or [])


def save_progress(done: list[str], rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"done": done, "catalog": rows}, ensure_ascii=False), encoding="utf-8")


def download_entities(entities: list[tuple[str, list[str], list[str], list[str]]]) -> list[dict]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    done, rows = load_progress()
    started = time.time()
    for key, queries, required, forbidden in entities:
        if key in done:
            continue
        if time.time() - started > BUDGET_SECONDS:
            print("time budget reached", flush=True)
            break
        matches = search_entity(queries, required, forbidden)
        print(f"{key}: {len(matches)} licensed matches", flush=True)
        for index, match in enumerate(matches, start=1):
            filename = f"{key}-{index:02d}.jpg"
            path = IMAGE_DIR / filename
            if not path.exists():
                req = urllib.request.Request(match["download_url"], headers={"User-Agent": UA})
                try:
                    with urllib.request.urlopen(req, timeout=40) as response:
                        payload = response.read()
                    width, height = save_jpeg(payload, path)
                except Exception as exc:
                    print(f"  skip {filename}: {exc}", flush=True)
                    continue
                time.sleep(0.05)
            else:
                with Image.open(path) as image:
                    width, height = image.size
            rows.append({
                "entity": key,
                "file": filename,
                "commons_title": match["commons_title"],
                "license": match["license"],
                "license_url": match["license_url"],
                "page_url": match["page_url"],
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
            })
        done.append(key)
        save_progress(done, rows)
    return rows


def four_city_hotels() -> dict[str, str]:
    mapping = {
        "Palomar Chicago, a Kimpton Hotel": "fc-palomar",
        "W Chicago - City Center": "fc-wchicago",
        "Crowne Plaza Houston River Oaks": "fc-crowne",
        "Magnolia Hotel Houston": "fc-magnolia",
        "Sheraton Gateway Los Angeles": "fc-sheraton-lax",
        "The Westin Los Angeles Airport": "fc-westin-lax",
        "Hotel Pennsylvania New York": "fc-pennsylvania",
        "Waldorf Astoria New York": "fc-waldorf",
    }
    found: dict[str, str] = {}
    with zipfile.ZipFile(RAW / "human_written_fake" / "four_city" / "fake.zip") as archive:
        for name in archive.namelist():
            if not name.endswith(".txt") or name.startswith("__MACOSX") or "/." in name:
                continue
            for line in archive.read(name).decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                payload = json.loads(line)
                hotel = payload.get("Input", {}).get("hotel_name") or ""
                text = normalize(payload.get("Answer", {}).get("review_text") or "")
                if hotel in mapping and text:
                    found[text] = mapping[hotel]
    return found


def assign(catalog: list[dict]) -> tuple[list[dict], dict]:
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for row in catalog:
        by_entity[row["entity"]].append(row)
    for rows in by_entity.values():
        rows.sort(key=lambda row: row["file"])
    text_to_hotel = four_city_hotels()
    counts: dict[str, int] = defaultdict(int)
    assigned: list[dict] = []
    skipped = defaultdict(int)
    with TWO_CLASS.open(encoding="utf-8", newline="") as handle:
        for review in csv.DictReader(handle):
            if review["label"] != "human_fake":
                continue
            entity = ""
            if review["source"] == "ott":
                entity = "ott-" + review["item"]
            elif review["source"] == "four_city":
                entity = text_to_hotel.get(normalize(review["text"]), "")
            elif review["source"] == "soldner" and review["item"] in PHONE_QUERIES:
                entity = "phone-" + slug(review["item"])
            else:
                skipped[review["source"]] += 1
                continue
            pool = by_entity.get(entity) or []
            if not pool:
                skipped["no_image_" + entity] += 1
                continue
            image = pool[counts[entity] % len(pool)]
            counts[entity] += 1
            assigned.append({
                "review_id": review["id"],
                "source": review["source"],
                "item": review["item"],
                "entity": entity,
                "image_file": image["file"],
                "commons_title": image["commons_title"],
                "license": image["license"],
                "page_url": image["page_url"],
            })
    summary = {
        "images": len(catalog),
        "reviews_assigned": len(assigned),
        "by_source": dict(defaultdict(int, **{row["source"]: 0 for row in assigned})),
        "skipped": dict(skipped),
        "entities_without_image": sorted(key for key in counts if False),
    }
    source_counts: dict[str, int] = defaultdict(int)
    reuse: dict[str, int] = defaultdict(int)
    for row in assigned:
        source_counts[row["source"]] += 1
        reuse[row["image_file"]] += 1
    summary["by_source"] = dict(source_counts)
    summary["max_reviews_per_image"] = max(reuse.values()) if reuse else 0
    summary["images_used"] = len(reuse)
    summary["entities"] = {
        key: {"images": len(rows), "reviews": counts.get(key, 0)}
        for key, rows in sorted(by_entity.items())
    }
    return assigned, summary


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    entities = list(ENTITIES)
    for brand, (queries, required, forbidden) in PHONE_QUERIES.items():
        entities.append(("phone-" + slug(brand), queries, required, forbidden))
    if "--count" in sys.argv:
        for key, queries, required, forbidden in entities:
            try:
                matches = search_entity(queries, required, forbidden)
            except Exception as exc:
                print(f"{key}: ERROR {exc}", flush=True)
                continue
            print(f"{key}: {len(matches)}", flush=True)
        return
    catalog = download_entities(entities)
    done, _ = load_progress()
    if len(done) < len(entities):
        print(f"saved {len(done)}/{len(entities)} entities, {len(catalog)} images", flush=True)
        return
    assigned, summary = assign(catalog)
    write_csv(CATALOG, CATALOG_FIELDS, catalog)
    write_csv(ASSIGNMENT, ASSIGN_FIELDS, assigned)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "reviews_assigned", "by_source", "max_reviews_per_image", "skipped")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

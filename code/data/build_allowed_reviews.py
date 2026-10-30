"""Assemble previously allowed reviews and attach images already on disk.

mm_v2.csv is not modified. Hotel and phone photos stay in
data/raw/human_fake_images/images; each review stores that relative path.
Rows without a photo are kept.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
RAW = ROOT / "data" / "raw"
OUT = FROZEN / "allowed_reviews.csv"
AUDIT = FROZEN / "allowed_reviews_audit.json"
IMAGES = RAW / "human_fake_images" / "images"

FIELDS = [
    "id",
    "in_mm_v2",
    "subset",
    "domain",
    "label",
    "language",
    "protocol",
    "source",
    "generator",
    "polarity",
    "item",
    "group_id",
    "label_quality",
    "split",
    "text",
    "image_path",
    "image_license",
    "image_page_url",
    "image_role",
    "image_entity",
    "image_link",
]


def norm(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def image_fields(assignment: dict | None, link: str) -> dict:
    if not assignment:
        return {
            "image_path": "",
            "image_license": "",
            "image_page_url": "",
            "image_role": "none",
            "image_entity": "",
            "image_link": "",
        }
    filename = assignment["image_file"]
    return {
        "image_path": f"data/raw/human_fake_images/images/{filename}",
        "image_license": assignment.get("license", ""),
        "image_page_url": assignment.get("page_url", ""),
        "image_role": "shared_entity",
        "image_entity": assignment.get("entity", ""),
        "image_link": link,
    }


def main() -> None:
    mm = read_csv(FROZEN / "mm_v2.csv")
    two = read_csv(FROZEN / "two_class_protocol.csv")
    assignment_rows = read_csv(RAW / "human_fake_images" / "assignment.csv")
    catalog = read_csv(RAW / "human_fake_images" / "catalog.csv")
    hotel_ai = read_csv(RAW / "hotel_ai" / "reviews_v2.csv")
    phone_ai = read_csv(RAW / "phone_ai" / "reviews.csv")

    by_review = {row["review_id"]: row for row in assignment_rows}
    if len(by_review) != len(assignment_rows):
        raise SystemExit("assignment review_id is not unique")

    ott_by_text: dict[str, dict] = {}
    for row in two:
        if row["source"] != "ott":
            continue
        key = norm(row["text"])
        if key in ott_by_text:
            raise SystemExit(f"duplicate Ott text for {row['id']}")
        ott_by_text[key] = row

    hotel_ai_by_text: dict[str, str] = {}
    for index, row in enumerate(hotel_ai, start=1):
        hotel_ai_by_text[norm(row["text"])] = f"hotel-ai-{index:04d}"

    pools: dict[str, list[str]] = defaultdict(list)
    for row in catalog:
        pools[row["entity"]].append(row["file"])
    for files in pools.values():
        files.sort()
    catalog_by_file = {row["file"]: row for row in catalog}

    rows: list[dict] = []
    used_assignment: set[str] = set()
    ott_miss = 0
    hotel_ai_miss = 0

    for index, row in enumerate(mm, start=1):
        protocol = "public_dataset"
        item = ""
        image = image_fields(None, "")
        group_id = row["group_id"]
        if row["subset"] == "hotel_ott":
            match = ott_by_text.get(norm(row["text"]))
            if match is None:
                ott_miss += 1
                protocol = "different_origin"
            else:
                protocol = match["protocol"]
                item = match["item"]
                image = image_fields(by_review.get(match["id"]), "assignment")
                if match["id"] in by_review:
                    used_assignment.add(match["id"])
        elif row["subset"] == "hotel_ai":
            protocol = "generated_this_study"
            review_id = hotel_ai_by_text.get(norm(row["text"]), "")
            if not review_id:
                hotel_ai_miss += 1
            else:
                assigned = by_review.get(review_id)
                image = image_fields(assigned, "assignment")
                if assigned:
                    used_assignment.add(review_id)
                    item = assigned["item"]
        elif row["subset"] == "food_aigen" and row["image_path"]:
            image = {
                "image_path": row["image_path"].replace("\\", "/"),
                "image_license": "MIT",
                "image_page_url": "",
                "image_role": row["image_role"],
                "image_entity": "",
                "image_link": "paired",
            }
        rows.append(
            {
                "id": f"mmv2-{index:05d}",
                "in_mm_v2": "yes",
                "subset": row["subset"],
                "domain": row["domain"],
                "label": row["label"],
                "language": row["language"],
                "protocol": protocol,
                "source": row["source"],
                "generator": row["generator"],
                "polarity": row["polarity"],
                "item": item,
                "group_id": group_id,
                "label_quality": row["label_quality"],
                "split": row["split"],
                "text": row["text"],
                "image_path": image["image_path"],
                "image_license": image["image_license"],
                "image_page_url": image["image_page_url"],
                "image_role": image["image_role"],
                "image_entity": image["image_entity"],
                "image_link": image["image_link"],
            }
        )

    for row in two:
        if row["source"] == "ott":
            continue
        assigned = by_review.get(row["id"])
        image = image_fields(assigned, "assignment")
        if assigned:
            used_assignment.add(row["id"])
        if row["source"] == "soldner":
            domain = "ecommerce"
        elif row["source"] == "four_city":
            domain = "hotel"
        elif row["source"] == "derev":
            domain = "book"
        elif row["source"] == "blt":
            domain = "hotel" if row["item"] == "hotels" else "ecommerce"
        else:
            domain = "other"
        rows.append(
            {
                "id": row["id"],
                "in_mm_v2": "no",
                "subset": row["source"],
                "domain": domain,
                "label": row["label"],
                "language": row["language"],
                "protocol": row["protocol"],
                "source": row["source"],
                "generator": "",
                "polarity": row["polarity"],
                "item": row["item"],
                "group_id": row["id"],
                "label_quality": "gold",
                "split": "",
                "text": row["text"],
                "image_path": image["image_path"],
                "image_license": image["image_license"],
                "image_page_url": image["image_page_url"],
                "image_role": image["image_role"],
                "image_entity": image["image_entity"],
                "image_link": image["image_link"],
            }
        )

    brand_counts: Counter[str] = Counter()
    phone_attached = 0
    phone_without_pool = 0
    for index, row in enumerate(phone_ai, start=1):
        brand = row["brand"].strip()
        entity = "phone-" + re.sub(r"[^a-z0-9]+", "-", brand.lower()).strip("-")
        pool = pools.get(entity) or []
        assigned = None
        if pool:
            filename = pool[brand_counts[entity] % len(pool)]
            brand_counts[entity] += 1
            meta = catalog_by_file[filename]
            assigned = {
                "image_file": filename,
                "license": meta["license"],
                "page_url": meta["page_url"],
                "entity": entity,
            }
            phone_attached += 1
        else:
            phone_without_pool += 1
        image = image_fields(assigned, "brand_pool")
        rows.append(
            {
                "id": f"phoneai-{index:05d}",
                "in_mm_v2": "no",
                "subset": "phone_ai",
                "domain": "ecommerce",
                "label": "ai_fake",
                "language": "en",
                "protocol": "generated_this_study",
                "source": "phone_ai",
                "generator": row["generator"],
                "polarity": row["polarity"],
                "item": row["item"],
                "group_id": f"phoneai-{index:05d}",
                "label_quality": "generated_this_study",
                "split": "",
                "text": row["text"],
                "image_path": image["image_path"],
                "image_license": image["image_license"],
                "image_page_url": image["image_page_url"],
                "image_role": image["image_role"] if assigned else "none",
                "image_entity": image["image_entity"],
                "image_link": image["image_link"] if assigned else "",
            }
        )

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        dupes = [key for key, count in Counter(ids).items() if count > 1]
        raise SystemExit(f"duplicate ids: {dupes[:8]}")
    if ott_miss or hotel_ai_miss:
        raise SystemExit(f"unmatched ott {ott_miss} hotel_ai {hotel_ai_miss}")

    missing_files = sorted({
        row["image_path"]
        for row in rows
        if row["image_path"] and not (ROOT / row["image_path"]).is_file()
    })
    if missing_files:
        raise SystemExit(f"missing image files: {missing_files[:5]}")

    unused = sorted(set(by_review) - used_assignment)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def key_count(field: str) -> dict:
        return dict(Counter(row[field] for row in rows))

    with_image = [row for row in rows if row["image_path"]]
    labels_by_file: dict[str, set[str]] = defaultdict(set)
    for row in with_image:
        if row["image_link"] == "paired":
            continue
        labels_by_file[row["image_path"]].add(row["label"])
    cross_label = sum(1 for labels in labels_by_file.values() if len(labels) > 1)

    audit = {
        "rows": len(rows),
        "mm_v2_unchanged": True,
        "output": "data/frozen/allowed_reviews.csv",
        "by_label": key_count("label"),
        "by_domain": key_count("domain"),
        "by_source": key_count("source"),
        "by_protocol": key_count("protocol"),
        "by_image_link": key_count("image_link"),
        "with_image": len(with_image),
        "without_image": len(rows) - len(with_image),
        "assignment_rows": len(assignment_rows),
        "assignment_attached": len(used_assignment),
        "assignment_unused": unused,
        "phone_ai_attached_to_existing_brand_photos": phone_attached,
        "phone_ai_without_brand_pool": phone_without_pool,
        "shared_photo_files_used_by_more_than_one_label": cross_label,
        "note": "Ott stays once, taken from mm_v2. Hotel and phone photos are shared across reviews of the same hotel or brand. Food images remain the 224 copies already in mm_v2. Rows with different protocol values are not one training pool.",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("rows", "by_label", "with_image", "without_image", "by_image_link", "phone_ai_without_brand_pool", "assignment_attached")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

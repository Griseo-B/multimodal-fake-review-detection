"""Attach the existing Chicago-hotel AI reviews to the Ott rows for those hotels.

The AI text was already generated from the hotel name, polarity, and topic
words, with 10-word collisions removed. This file does not rewrite mm_v2.csv.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TWO = ROOT / "data" / "frozen" / "two_class.csv"
AI = ROOT / "data" / "raw" / "hotel_ai" / "reviews_v2.csv"
OUT = ROOT / "data" / "frozen" / "hotel_three_class.csv"
AUDIT = ROOT / "data" / "frozen" / "hotel_three_class_audit.json"
FIELDS = ["id", "text", "label", "language", "source", "subtype", "polarity", "item", "protocol"]


def generator_name(value: str) -> str:
    if "Qwen" in value:
        return "qwen2.5-0.5b"
    if "TinyLlama" in value:
        return "tinyllama-1.1b"
    return value


def main() -> None:
    with TWO.open(encoding="utf-8", newline="") as handle:
        ott = [row for row in csv.DictReader(handle) if row["source"] == "ott"]
    rows = []
    for index, row in enumerate(ott, start=1):
        rows.append(
            {
                "id": f"ott-{index:05d}",
                "text": row["text"],
                "label": row["label"],
                "language": "en",
                "source": "ott",
                "subtype": row["subtype"],
                "polarity": row["polarity"],
                "item": row["item"],
                "protocol": "different_origin",
            }
        )
    with AI.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            polarity = "pos" if row["polarity"] == "positive" else "neg"
            rows.append(
                {
                    "id": f"hotelai-{index:05d}",
                    "text": row["text"],
                    "label": "ai_fake",
                    "language": "en",
                    "source": "hotel_ai",
                    "subtype": generator_name(row["generator"]),
                    "polarity": polarity,
                    "item": row["hotel_key"],
                    "protocol": "generated_this_study",
                }
            )
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_item = {}
    for row in rows:
        slot = by_item.setdefault(row["item"], Counter())
        slot[row["label"] if row["label"] != "ai_fake" else row["subtype"]] += 1
    missing = [
        item
        for item, counts in sorted(by_item.items())
        if counts["real"] == 0 or counts["human_fake"] == 0 or counts["qwen2.5-0.5b"] == 0 or counts["tinyllama-1.1b"] == 0
    ]
    audit = {
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "generators": dict(Counter(row["subtype"] for row in rows if row["label"] == "ai_fake")),
        "hotels": len(by_item),
        "hotels_missing_a_class": missing,
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

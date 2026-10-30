"""Pair the accepted phone AI reviews with the same phones' human reviews."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "data" / "raw" / "phone_ai" / "reviews.csv"
HUMAN = ROOT / "data" / "raw" / "human_written_fake" / "soldner2022" / "review_data.csv"
OUT = ROOT / "data" / "frozen" / "phone_paired.csv"
FIELDS = ["id", "text", "label", "language", "source", "subtype", "polarity", "item", "protocol"]


def polarity_from_rating(value: str) -> str:
    rating = int(float(value))
    if rating >= 4:
        return "pos"
    if rating <= 2:
        return "neg"
    return "neu"


def main() -> None:
    with AI.open(encoding="utf-8", newline="") as handle:
        generated = [row for row in csv.DictReader(handle) if row["text"].strip()]
    names = {row["item"] for row in generated}
    rows = []
    with HUMAN.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record["data_source_label"] != "prolific":
                continue
            item = f"{record['brand']} {record['phone_type']}"
            if item not in names:
                continue
            if record["veracity_label"] == "truthful" and record["ownership_label"] == "owner":
                label, subtype = "real", "owner_truthful"
            elif record["veracity_label"] == "deceptive" and record["ownership_label"] == "owner":
                label, subtype = "human_fake", "owner_opposite"
            elif record["veracity_label"] == "deceptive" and record["ownership_label"] == "non_owner":
                label, subtype = "human_fake", "non_owner"
            else:
                continue
            rows.append(
                {
                    "text": record["review_text"],
                    "label": label,
                    "language": "en",
                    "source": "soldner",
                    "subtype": subtype,
                    "polarity": polarity_from_rating(record["review_rating"]),
                    "item": item,
                    "protocol": "same_experiment",
                }
            )
    for record in generated:
        rows.append(
            {
                "text": record["text"],
                "label": "ai_fake",
                "language": "en",
                "source": "phone_ai",
                "subtype": record["generator"],
                "polarity": record["polarity"],
                "item": record["item"],
                "protocol": "generated_this_study",
            }
        )
    for index, row in enumerate(rows, start=1):
        row["id"] = f"phone-{index:05d}"
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "phones", len(names))


if __name__ == "__main__":
    main()

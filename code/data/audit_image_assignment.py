"""Audit how the shared photos were attached to reviews.

Two batches exist and they are not the same kind of match:

`shared_entity`  one pool per hotel or phone brand, 3 to 8 web photos each, a
                 review draws one photo from the pool of the place it reviews.
                 Matched at entity level, so the picture is of the right hotel.
`shared_web`     one pool per scene (food / hotel / ecommerce), filled with
                 photos taken from real reviews' own posts, a human_fake row
                 draws one by round-robin. Matched at scene level only, so the
                 picture may be a different restaurant or product.

The frozen table keeps neither the entity nor the pool, only the 224 path, so
this audit rebuilds the mapping and writes it beside the run. It does not
touch `three_class.csv`.

    python code/data/audit_image_assignment.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
ASSIGNMENT = ROOT / "data" / "raw" / "human_fake_images" / "assignment.csv"
OUT = ROOT / "experiments" / "20261031_image_assignment"


def key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FROZEN / "three_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    frame["key"] = frame["text"].map(key)
    assignment = pd.read_csv(ASSIGNMENT, dtype=str, keep_default_na=False)
    two_class = pd.read_csv(FROZEN / "two_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    two_class["key"] = two_class["text"].map(key)

    pools: dict[str, set[str]] = {}
    for row in assignment.itertuples(index=False):
        pools.setdefault(row.entity, set()).add(row.image_file)

    # review_id -> the place it reviews, from the table the pool was built on.
    place = (
        two_class.drop_duplicates("id")[["id", "item", "label", "source"]]
        .rename(columns={"id": "review_id"})
    )
    joined = assignment.merge(place, on="review_id", how="left", suffixes=("", "_review"))

    def expected(row) -> str:
        if row["source"] == "ott" and row["item"]:
            return "ott-" + row["item"]
        if row["source"] == "soldner" and row["item"]:
            return "phone-" + slug(row["item"])
        return ""

    joined["expected_entity"] = joined.apply(expected, axis=1)
    checkable = joined.loc[joined["expected_entity"] != ""]
    agreement = (
        float((checkable["entity"] == checkable["expected_entity"]).mean()) if len(checkable) else float("nan")
    )
    prefix_ok = (
        sum(image.startswith(entity + "-") for image, entity in zip(joined["image_file"], joined["entity"]))
        / len(joined)
        if len(joined)
        else float("nan")
    )

    # Tie every frozen row back to its pool, by text, because the table has no id.
    mapping = frame.merge(
        two_class[["key", "id", "item"]].rename(columns={"id": "review_id", "item": "review_item"}),
        on="key",
        how="left",
    )
    mapping = mapping.merge(
        assignment[["review_id", "entity", "image_file", "license", "page_url"]],
        on="review_id",
        how="left",
    )
    entity_rows = frame.loc[frame["image_role"] == "shared_entity"]
    traced_rows = mapping.loc[mapping["image_role"] == "shared_entity"]
    scene_rows = frame.loc[frame["image_role"] == "shared_web"]

    entity_map = (
        traced_rows.loc[traced_rows["image_role"] == "shared_entity",
                        ["id", "domain", "label", "source", "image_224_path", "review_id", "review_item",
                         "entity", "image_file", "license", "page_url"]]
        .sort_values(["entity", "id"])
    )
    entity_map.to_csv(OUT / "entity_map.csv", index=False, encoding="utf-8-sig")

    per_image = entity_rows.groupby("image_224_path").size()
    scene_per_image = scene_rows.groupby("image_224_path").size()
    summary = {
        "shared_entity": {
            "rows": int(len(entity_rows)),
            "unique_images": int(entity_rows["image_224_path"].nunique()),
            "entities": int(assignment["entity"].nunique()),
            "images_per_entity": {
                "min": int(min(len(v) for v in pools.values())),
                "median": float(pd.Series([len(v) for v in pools.values()]).median()),
                "max": int(max(len(v) for v in pools.values())),
            },
            "rows_per_image": {
                "min": int(per_image.min()),
                "median": float(per_image.median()),
                "max": int(per_image.max()),
            },
            "rows_traced_to_entity": int(entity_map["entity"].notna().sum()),
            "trace_rate": round(float(entity_map["entity"].notna().mean()), 4),
            "assigned_entity_equals_reviewed_place": round(agreement, 4),
            "checkable_rows": int(len(checkable)),
            "image_filename_prefix_matches_entity": round(prefix_ok, 4),
            "matching": "entity level: the photo is of the hotel or phone model the review is about",
        },
        "shared_web": {
            "rows": int(len(scene_rows)),
            "unique_images": int(scene_rows["image_224_path"].nunique()),
            "by_domain": {str(k): int(v) for k, v in scene_rows["domain"].value_counts().items()},
            "rows_per_image": {
                "min": int(scene_per_image.min()),
                "median": float(scene_per_image.median()),
                "max": int(scene_per_image.max()),
            },
            "images_used_by_more_than_one_row": int((scene_per_image > 1).sum()),
            "matching": "scene level only: same food/hotel/ecommerce, not the same restaurant or product",
            "source_of_photos": "photos taken from real reviews in the same scene",
        },
        "table_columns": {
            "has_entity_column": False,
            "has_pool_column": False,
            "note": (
                "three_class.csv keeps only image_224_path, so the entity of a shared photo has to be "
                "recovered by joining text back to two_class.csv; the result is entity_map.csv."
            ),
        },
        "images_per_review": {
            "note": "every row has at most one image; image_path never carries a '|' separator",
            "rows_with_multiple": int(frame["image_path"].str.contains(r"\|").sum()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

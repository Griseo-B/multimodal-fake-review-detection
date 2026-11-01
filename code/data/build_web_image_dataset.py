"""Build the "human-written fake review + borrowed web image" dataset.

Motivation
----------
In the wild, fake reviewers rarely upload an on-scene photo of their own.
They copy/crawl a web image of the *same* hotel or phone model and post it
next to a fabricated review, so that the review looks "verified by a photo".
This script turns that real-world behaviour into an explicit, auditable
dataset design instead of treating it as a data defect.

Design
------
* Only entities for which we own a *per-entity* image pool and that have BOTH
  real reviews and human-written fake reviews are kept. 29 entities qualify
  (20 hotels from Ott et al. + 9 phone brands from Soldner et al.).
* Every review -- real or fake -- draws its image from the same per-entity
  pool.  The image therefore carries *no* label signal by construction, which
  is exactly the property a fake review has in reality: the picture is of the
  real thing, the words are the lie.
* Splits are grouped by entity, so no hotel / phone brand is seen in both
  train and test.

Outputs
-------
data/frozen/web_image_pair.csv          one row per (review, borrowed image)
data/frozen/web_image_pair_audit.json   audit of pools, labels, leakage
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
ENTITY_MAP = ROOT / "experiments" / "20261031_image_assignment" / "entity_map.csv"

KEEP_LABELS = ("real", "human_fake")
SEED = 20261031


def load() -> pd.DataFrame:
    d = pd.read_csv(FROZEN / "three_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    d = d[d["image_role"] == "shared_entity"].copy()
    ent = pd.read_csv(ENTITY_MAP, dtype=str, keep_default_na=False)
    ent = ent[["id", "entity", "license", "page_url", "image_file"]]
    d = d.merge(ent, on="id", how="inner")
    d = d[d["entity"].isin(ent["entity"])]
    d = d[d["label"].isin(KEEP_LABELS)]
    return d


def pick_paired_entities(d: pd.DataFrame) -> list[str]:
    piv = d.groupby(["entity", "label"]).size().unstack(fill_value=0)
    if "real" not in piv or "human_fake" not in piv:
        return []
    return sorted(piv[(piv["real"] > 0) & (piv["human_fake"] > 0)].index)


def entity_split(entities: list[str], domain_of: dict[str, str]) -> dict[str, str]:
    """Entity-level 70/15/15 split, stratified by domain."""
    rng = random.Random(SEED)
    out: dict[str, str] = {}
    for domain in sorted({domain_of[e] for e in entities}):
        group = sorted([e for e in entities if domain_of[e] == domain])
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        for i, e in enumerate(group):
            if i < n_test:
                out[e] = "test"
            elif i < n_test + n_val:
                out[e] = "val"
            else:
                out[e] = "train"
    return out


def audit(d: pd.DataFrame, entities: list[str], split: dict[str, str]) -> dict:
    per_entity = {}
    for e in entities:
        sub = d[d["entity"] == e]
        pool = sub["image_224_path"].nunique()
        mixed = int((sub.groupby("image_224_path")["label"].nunique() > 1).sum())
        per_entity[e] = {
            "domain": sub["domain"].iloc[0],
            "n_real": int((sub["label"] == "real").sum()),
            "n_human_fake": int((sub["label"] == "human_fake").sum()),
            "pool_size": int(pool),
            "images_used_by_both_labels": mixed,
            "split": split[e],
        }
    # how often the same image is reused inside one entity (real-world: the
    # same stolen photo gets posted by several accounts)
    reuse = d.groupby(["entity", "image_224_path"]).size()
    per_image_labels = d.groupby("image_224_path")["label"].nunique()
    shared_images = int((per_image_labels > 1).sum())
    total_images = int(len(per_image_labels))
    return {
        "n_rows": int(len(d)),
        "n_entities": len(entities),
        "labels": {k: int(v) for k, v in d["label"].value_counts().items()},
        "domains": {k: int(v) for k, v in d["domain"].value_counts().items()},
        "split_rows": {k: int(v) for k, v in d["split"].value_counts().items()},
        "split_entities": {k: int(v) for k, v in Counter(split.values()).items()},
        "license": {k: int(v) for k, v in d["license"].value_counts().items()},
        "entity_pool_mean": round(float(pd.Series({e: v["pool_size"] for e, v in per_entity.items()}).mean()), 2),
        "entity_pool_min": int(min(v["pool_size"] for v in per_entity.values())),
        "entity_pool_max": int(max(v["pool_size"] for v in per_entity.values())),
        "unique_images": total_images,
        "images_used_by_both_labels": shared_images,
        "pct_images_used_by_both_labels": round(shared_images / max(total_images, 1), 4),
        "reuse_per_image_median": float(reuse.median()),
        "per_entity": per_entity,
    }


def main() -> None:
    d = load()
    entities = pick_paired_entities(d)
    d = d[d["entity"].isin(entities)].copy()
    domain_of = d.drop_duplicates("entity").set_index("entity")["domain"].to_dict()
    split = entity_split(entities, domain_of)
    d["split_entity"] = d["entity"].map(split)
    d["binary_label"] = (d["label"] == "human_fake").astype(int)
    d["image_provenance"] = "borrowed_web_image"

    cols = [
        "id", "domain", "label", "binary_label", "entity", "split_entity",
        "source", "language", "text", "n_tokens",
        "image_path", "image_224_path", "image_role", "image_provenance",
        "license", "page_url",
    ]
    out = d[cols].sort_values(["domain", "entity", "label", "id"]).reset_index(drop=True)
    out.to_csv(FROZEN / "web_image_pair.csv", index=False, encoding="utf-8")

    a = audit(d, entities, split)
    a["design_note"] = (
        "Each review is paired with a web image of the SAME entity (hotel / phone brand). "
        "Real and human-written fake reviews of an entity share one image pool, so the image "
        "cannot encode the label; only the text distinguishes real from fake. This mirrors the "
        "real-world fake-review pattern of reusing crawled photos of the genuine item."
    )
    (FROZEN / "web_image_pair_audit.json").write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {len(out)} rows / {len(entities)} entities -> data/frozen/web_image_pair.csv")
    print("labels:", a["labels"])
    print("split rows:", a["split_rows"], "entities:", a["split_entities"])
    print("pool size mean/min/max:", a["entity_pool_mean"], a["entity_pool_min"], a["entity_pool_max"])
    print("per-entity balance:")
    print(out.groupby(["entity", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

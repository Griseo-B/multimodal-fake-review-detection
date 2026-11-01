"""Frozen benchmark: real review + own photo  vs  human fake + borrowed web photo.

This is the dataset the project actually needs. It reproduces the signal that
separates a genuine review from a bought one in the wild:

* a real review carries the customer's own snapshot  -> `paired_review`
* a human-written fake carries a web photo of the same thing, crawled by the
  spammer                                             -> `borrowed_*`

Pairing strength of the borrowed photo is recorded per row so results can be
broken down by how tight the correspondence really is:

    entity     the photo shows the same hotel / phone model the review discusses
    category   the photo shows the same cuisine or product category
    scene      the photo shows the right broad scene only

Splits are grouped by image identity, never by row, because borrowed photos are
reused across hundreds of reviews: splitting rows would put the same photo in
train and test and inflate every image metric.

Outputs
-------
data/frozen/web_image_benchmark.csv
data/frozen/web_image_benchmark_audit.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
SEED = 20261031
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def pairing_of(role: str) -> str:
    return {"shared_entity": "entity", "shared_web": "scene"}.get(role, "own")


def main() -> None:
    d = pd.read_csv(FROZEN / "three_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    reprise = pd.read_csv(FROZEN / "web_image_category.csv", dtype=str, keep_default_na=False)
    repair = reprise.set_index("id")[["image_path", "pairing", "category"]]

    real = d[(d["label"] == "real") & (d["image_role"] == "paired_review")].copy()
    real["pairing"] = "own"
    real["category"] = ""

    fake = d[(d["label"] == "human_fake") & (d["image_role"].isin(["shared_entity", "shared_web"]))].copy()
    fake["pairing"] = fake["image_role"].map(pairing_of)
    fake["category"] = ""
    # rows the re-pairing pass improved: adopt the new image and its strength
    hit = fake["id"].isin(repair.index)
    fake.loc[hit, "image_path"] = repair.loc[fake.loc[hit, "id"], "image_path"].to_numpy()
    fake.loc[hit, "pairing"] = repair.loc[fake.loc[hit, "id"], "pairing"].to_numpy()
    fake.loc[hit, "category"] = repair.loc[fake.loc[hit, "id"], "category"].to_numpy()

    table = pd.concat([real, fake], ignore_index=True)
    table = table[table["image_224_path"].astype(bool)].copy()
    table["binary_label"] = (table["label"] == "human_fake").astype(int)

    # split by what the model could memorise: the image file, or the entity
    ent = pd.read_csv(ROOT / "experiments" / "20261031_image_assignment" / "entity_map.csv",
                      dtype=str, keep_default_na=False).set_index("id")["entity"]
    table["entity"] = table["id"].map(ent).fillna("")
    image_key = table["image_224_path"].map(lambda p: Path(p).stem)
    table["split_group"] = [
        f"entity:{e}" if e else f"image:{img}"
        for e, img in zip(table["entity"], image_key)
    ]

    rng = random.Random(SEED)
    assign: dict[str, str] = {}
    for domain, sub in table.groupby("domain"):
        groups = sorted(sub["split_group"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_test, n_val = max(1, round(n * SPLITS["test"])), max(1, round(n * SPLITS["val"]))
        for i, g in enumerate(groups):
            assign[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    table["split"] = table["split_group"].map(assign)

    image_in_splits = table.groupby("image_224_path")["split"].nunique()
    overlapping = int((image_in_splits > 1).sum())

    cols = [
        "id", "domain", "label", "binary_label", "pairing", "category", "entity",
        "split", "split_group", "source", "language", "text", "n_tokens",
        "image_path", "image_224_path", "image_role",
    ]
    out = table[cols].sort_values(["domain", "label", "id"]).reset_index(drop=True)
    out.to_csv(FROZEN / "web_image_benchmark.csv", index=False, encoding="utf-8")

    audit = {
        "rows": int(len(out)),
        "labels": {str(k): int(v) for k, v in out["label"].value_counts().items()},
        "by_domain_label": {
            f"{dom}/{lab}": int(n)
            for (dom, lab), n in out.groupby(["domain", "label"]).size().items()
        },
        "pairing": {str(k): int(v) for k, v in out["pairing"].value_counts().items()},
        "pairing_x_label": {
            f"{pairing}/{lab}": int(n)
            for (pairing, lab), n in out.groupby(["pairing", "label"]).size().items()
        },
        "split_rows": {str(k): int(v) for k, v in out["split"].value_counts().items()},
        "unique_images": int(out["image_224_path"].nunique()),
        "images_in_more_than_one_split": overlapping,
        "split_rule": "grouped by entity for entity-paired rows, otherwise by image file",
        "design_note": (
            "Real reviews keep the customer's own photo; human-written fakes carry a borrowed "
            "web photo of the same entity / category / scene. The image therefore tells the "
            "model what the review is about, not whether the review is genuine."
        ),
        "known_confound": (
            "Real photos are amateur user snapshots while borrowed photos are polished "
            "web/merchant images, so low-level image statistics can separate the classes "
            "without any reasoning about consistency. Report image-only performance next to "
            "text+image performance so this shortcut stays visible."
        ),
    }
    (FROZEN / "web_image_benchmark_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"rows {len(out)}")
    print("labels", audit["labels"])
    print("pairing", audit["pairing"])
    print("pairing x label", audit["pairing_x_label"])
    print("by domain/label", audit["by_domain_label"])
    print("split rows", audit["split_rows"], "images", audit["unique_images"],
          "multi-split images", overlapping)


if __name__ == "__main__":
    main()

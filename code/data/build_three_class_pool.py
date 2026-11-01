"""Freeze two three-class experiment pools and repair the image split.

three_class.csv is a union table: seven protocol values, no protocol that
holds all three labels. Two pools are cut out of it, and the tables are not
modified.

pool A  hotel only, gold labels and this study's own generated AI reviews.
        real + human_fake + ai_fake, balanced enough to train three classes.
pool B  cross-source: public_dataset supplies real and ai_fake, while
        human_fake comes from different_origin and same_experiment. The
        contrasts in B are between sources, so results must be read with a
        leave-one-source check beside them.

Split repair. 1,412 images sit on more than one side of the split and 478 of
those also carry more than one label. Text has no leakage; images do. A
`split_purged` column puts every row that shares an image hash on one side,
chosen by the majority split of that image (train wins ties). Rows without an
image keep the original split. `image_cross_label` marks images shared by two
or more labels, which the image block must drop in any ablation.

Nothing is rebalanced here. Class weights belong in the model, not the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm3_common import ROOT  # noqa: E402

FROZEN = ROOT / "data" / "frozen"
TABLE = FROZEN / "three_class.csv"
HASH_CACHE = ROOT / "data" / "processed" / "image_hashes.json"
OUT_DIR = FROZEN

FIELDS = [
    "id", "subset", "domain", "label", "language", "protocol", "source",
    "generator", "group_id", "label_quality", "split", "split_purged",
    "text", "image_224_path", "image_role", "image_link", "image_md5",
    "image_cross_label", "n_tokens", "pos_noun", "pos_adj", "pos_adv",
    "pos_pron", "pos_verb",
]
SPLIT_ORDER = ["train", "val", "test"]


def file_md5(relative: str, cache: dict[str, str]) -> str:
    if relative in cache:
        return cache[relative]
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"missing image {relative}")
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    cache[relative] = digest
    return digest


def hash_images(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cache: dict[str, str] = {}
    if HASH_CACHE.is_file():
        cache = json.loads(HASH_CACHE.read_text(encoding="utf-8"))
    paths = sorted(path for path in frame["image_224_path"].unique() if path)
    before = len(cache)
    for index, relative in enumerate(paths, start=1):
        file_md5(relative, cache)
        if index % 5000 == 0:
            print(f"hash {index}/{len(paths)}", flush=True)
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    frame = frame.copy()
    frame["image_md5"] = frame["image_224_path"].map(lambda path: cache.get(path, "") if path else "")
    return frame, {"images": len(paths), "cache_before": before, "cache_after": len(cache)}


def image_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per image hash: split, label set, row count."""
    hashed = frame.loc[frame["image_md5"].astype(bool)]
    grouped = hashed.groupby("image_md5").agg(
        rows=("id", "size"),
        splits=("split", "nunique"),
        labels=("label", "nunique"),
        label_set=("label", lambda values: "|".join(sorted(set(values)))),
    )
    return grouped


def purge_splits(frame: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Put every image on one side of the split; keep rows without an image."""
    chosen: dict[str, str] = {}
    for image_hash, group in frame.loc[frame["image_md5"].astype(bool)].groupby("image_md5"):
        counts = group["split"].value_counts()
        for split in SPLIT_ORDER:  # train wins a tie, then val, then test
            if split in counts.index and counts[split] == counts.max():
                chosen[image_hash] = split
                break
        else:
            chosen[image_hash] = "train"
    frame = frame.copy()
    frame["image_cross_label"] = False
    frame.loc[frame["image_md5"].astype(bool), "image_cross_label"] = (
        frame.loc[frame["image_md5"].astype(bool), "image_md5"].map(groups["labels"] > 1)
    )
    already = frame["image_md5"].astype(bool)
    frame["split_purged"] = frame["split"]
    frame.loc[already, "split_purged"] = frame.loc[already, "image_md5"].map(chosen)
    leaky = groups.loc[groups["splits"] > 1]
    return frame, {
        "images_on_more_than_one_split": int(len(leaky)),
        "rows_in_leaky_images": int(leaky["rows"].sum()),
        "images_with_more_than_one_label": int((groups["labels"] > 1).sum()),
        "rows_in_multi_label_images": int(groups.loc[groups["labels"] > 1, "rows"].sum()),
        "rows_moved_by_purge": int((frame["split"] != frame["split_purged"]).sum()),
    }


def repair_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every split needs all three labels, otherwise macro-F1 is undefined."""
    frame = frame.copy()
    repaired = []
    for split in ("val", "test"):
        present = set(frame.loc[frame["split_purged"] == split, "label"])
        for label in ("real", "human_fake", "ai_fake"):
            if label in present:
                continue
            # A whole image group moves, so the repair cannot undo the purge.
            candidates = frame.loc[
                (frame["split_purged"] == "train")
                & (frame["label"] == label)
                & (~frame["image_cross_label"])
            ]
            if candidates.empty:
                candidates = frame.loc[(frame["split_purged"] == "train") & (frame["label"] == label)]
            if candidates.empty:
                continue
            key = "image_md5" if candidates["image_md5"].astype(bool).any() else "group_id"
            candidates = candidates.loc[candidates[key].astype(bool)]
            if candidates.empty:
                continue
            group = candidates[key].iloc[0]
            mask = (frame[key] == group) & (frame["split_purged"] == "train")
            frame.loc[mask, "split_purged"] = split
            repaired.append({"split": split, "label": label, "group": group, "rows": int(mask.sum())})
            present.add(label)
    return frame, {"repaired": repaired}


def counts(frame: pd.DataFrame, column: str) -> dict:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def cross(frame: pd.DataFrame, rows: list[str], column: str) -> dict:
    table = pd.crosstab([frame[name] for name in rows], frame[column])
    return {str(key): {str(k): int(v) for k, v in value.items()} for key, value in table.iterrows()}


def write_pool(name: str, frame: pd.DataFrame, spec: str, extra: dict) -> dict:
    frame = frame[FIELDS].reset_index(drop=True)
    output = OUT_DIR / f"three_class_pool_{name}.csv"
    frame.to_csv(output, index=False)
    audit = {
        "output": f"data/frozen/{output.name}",
        "spec": spec,
        "rows": int(len(frame)),
        "by_label": counts(frame, "label"),
        "by_domain": counts(frame, "domain"),
        "by_language": counts(frame, "language"),
        "by_source": counts(frame, "source"),
        "by_protocol": counts(frame, "protocol"),
        "by_generator": counts(frame[frame["generator"].astype(bool)], "generator"),
        "by_label_quality": counts(frame, "label_quality"),
        "by_split": counts(frame, "split"),
        "by_split_purged": counts(frame, "split_purged"),
        "label_x_domain": cross(frame, ["domain"], "label"),
        "label_x_generator": cross(frame[frame["label"] == "ai_fake"], ["generator"], "domain") if (frame["label"] == "ai_fake").any() else {},
        "with_image": int(frame["image_224_path"].astype(bool).sum()),
        "without_image": int((~frame["image_224_path"].astype(bool)).sum()),
        "by_image_role": counts(frame, "image_role"),
        "image_coverage_by_label": {
            str(label): {
                "rows": int(len(part)),
                "with_image": int(part["image_224_path"].astype(bool).sum()),
                "rate": round(float(part["image_224_path"].astype(bool).mean()), 4),
            }
            for label, part in frame.groupby("label")
        },
        "duplicate_normalized_text": int(
            frame["text"].fillna("").str.strip().str.casefold().duplicated().sum()
        ),
        "groups_crossing_split": int(
            frame.groupby("group_id")["split"].nunique().gt(1).sum()
        ),
        "groups_crossing_split_purged": int(
            frame.groupby("group_id")["split_purged"].nunique().gt(1).sum()
        ),
        "cross_label_images_in_pool": int(frame["image_cross_label"].sum()),
        "note": extra["note"],
    }
    audit.update(extra["audit"])
    (OUT_DIR / f"three_class_pool_{name}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(name, len(frame), json.dumps(audit["by_label"], ensure_ascii=False), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(TABLE))
    args = parser.parse_args()

    frame = pd.read_csv(args.table, dtype=str, keep_default_na=False, low_memory=False)
    print(f"rows {len(frame)}", flush=True)

    frame, hash_stats = hash_images(frame)
    groups = image_groups(frame)
    frame, purge_stats = purge_splits(frame, groups)
    frame, repair_stats = repair_labels(frame)

    hotel_gold = (frame["domain"] == "hotel") & frame["label_quality"].isin(
        ["gold", "generated_this_study"]
    )
    pool_a = frame.loc[hotel_gold].copy()

    public = frame["protocol"] == "public_dataset"
    human_gold = frame["protocol"].isin(["different_origin", "same_experiment"]) & (
        frame["label"] == "human_fake"
    )
    pool_b = frame.loc[public | human_gold].copy()

    common = {
        "hash": hash_stats,
        "purge": purge_stats,
        "repair": repair_stats,
    }
    audit_a = write_pool(
        "a",
        pool_a,
        "domain == hotel and label_quality in (gold, generated_this_study)",
        {
            "note": (
                "Hotel only. Three labels from gold sets plus this study's generated reviews. "
                "Single domain, so a leave-one-generator test is the honest generalisation check. "
                "Trips and collected platform reviews are left out because their basis is not gold."
            ),
            "audit": dict(common),
        },
    )
    audit_b = write_pool(
        "b",
        pool_b,
        "protocol == public_dataset or (protocol in (different_origin, same_experiment) and label == human_fake)",
        {
            "note": (
                "Cross-source. real and ai_fake are both from public_dataset; human_fake comes from "
                "two other protocol values. Any real-vs-human contrast is therefore also a "
                "source contrast and needs a leave-one-source reading beside it. rule_label "
                "(Chinese, rule basis) is excluded on purpose."
            ),
            "audit": dict(common),
        },
    )
    summary = {
        "pools": {"a": {"rows": audit_a["rows"], "by_label": audit_a["by_label"]}, "b": {"rows": audit_b["rows"], "by_label": audit_b["by_label"]}},
        "image_hash": hash_stats,
        "split_repair": purge_stats,
        "label_repair": repair_stats,
    }
    (OUT_DIR / "three_class_pool_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

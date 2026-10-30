"""Assemble one three-class table and preprocess images and text.

mm_v2.csv is not modified. Rows from different protocol values stay in
the file and are not one training pool. Reviews without an image stay.
Images used for training are squared to 224. Training images also get one
flipped, recolored copy. Text rows receive a token count and part-of-speech
shares, copied from mm_v2 when the review is already there.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mm_v2 import augment, ensure_nltk, pos_chinese, pos_english, save_square, tokens
from mm3_common import ROOT, SEED

FROZEN = ROOT / "data" / "frozen"
OUT = FROZEN / "three_class.csv"
AUDIT = FROZEN / "three_class_audit.json"
CATALOG = ROOT / "data" / "catalog.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "three_class" / "images"
AUG_DIR = ROOT / "data" / "processed" / "three_class" / "aug"
MANIFEST = ROOT / "data" / "raw" / "generated_ai_images_20261030" / "manifest.csv"

SD_SUBSETS = {"hotel_ai", "hotel_maide_en", "hotel_maide_zh", "ecommerce_salminen"}
FIELDS = [
    "id", "in_mm_v2", "subset", "domain", "label", "language", "protocol",
    "source", "generator", "polarity", "item", "group_id", "label_quality",
    "usable", "split", "text", "image_path", "image_224_path", "aug_path",
    "image_role", "image_link", "n_tokens", "pos_noun", "pos_adj", "pos_adv",
    "pos_pron", "pos_verb",
]
POS_COLUMNS = ["n_tokens", "pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb"]

RULE_SOURCES = {"dp-tdc", "dp-cdc", "jd", "jd-phone", "jd-food", "jd-appliance", "jd-clothing"}
COLLECTED_SOURCES = {"dianping-human", "jd-human", "taobao-human", "ctrip-human"}
PLATFORM_TEXT_SOURCES = {"ctrip", "trip"}
LI_SOURCES = {"li2014_restaurant", "li2014_hotel_expert"}


def norm(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def protocol_for_kept(source: str) -> str:
    if source in RULE_SOURCES:
        return "rule_label"
    if source in COLLECTED_SOURCES:
        return "platform_collected"
    if source in PLATFORM_TEXT_SOURCES:
        return "platform_user"
    if source in LI_SOURCES:
        return "different_origin"
    raise SystemExit(f"no protocol for {source}")


def project_path(part: str) -> str:
    part = part.replace("\\", "/")
    if part.startswith("data/"):
        return part
    return f"data/{part}"


def first_existing(path_field: str) -> tuple[str, int]:
    parts = [project_path(part) for part in str(path_field).split("|") if part]
    for part in parts:
        if (ROOT / part).is_file():
            return part, max(0, len(parts) - 1)
    return "", len(parts)


def blank_numbers() -> dict:
    return {column: "" for column in POS_COLUMNS}


def overlay_sd(frame: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    if manifest["group_id"].duplicated().any():
        raise SystemExit("manifest group_id is not unique")
    by_group = manifest.set_index("group_id")
    chosen = frame["subset"].isin(SD_SUBSETS) & (frame["label"] == "ai_fake")
    found = 0
    for index in frame.index[chosen]:
        group_id = frame.at[index, "group_id"]
        if group_id in by_group.index:
            hit = by_group.loc[group_id]
            source = str(hit["image_path"]).replace("\\", "/")
            if not (ROOT / source).is_file():
                raise SystemExit(f"missing generated image for {group_id}")
            frame.at[index, "image_path"] = source
            frame.at[index, "image_role"] = "paired_sd"
            frame.at[index, "image_link"] = "paired"
            found += 1
        else:
            frame.at[index, "image_path"] = ""
            frame.at[index, "image_role"] = "none"
            frame.at[index, "image_link"] = ""
    if found != len(manifest):
        raise SystemExit(f"manifest rows {len(manifest)} attached {found}")
    return frame


def load_allowed() -> pd.DataFrame:
    frame = pd.read_csv(FROZEN / "allowed_reviews.csv", dtype=str, keep_default_na=False)
    frame["usable"] = "true"
    frame["image_224_path"] = ""
    frame["aug_path"] = ""
    for column in POS_COLUMNS:
        frame[column] = ""
    frame = overlay_sd(frame)
    return frame


def load_kept(seen_text: set[str], seen_id: set[str]) -> tuple[pd.DataFrame, dict]:
    kept = pd.read_csv(
        ROOT / "data" / "raw" / "supplement" / "kept.csv",
        dtype=str,
        keep_default_na=False,
    )
    rows = []
    stats = Counter()
    missing = []
    for record in kept.itertuples(index=False):
        text = re.sub(r"\s+", " ", html.unescape(record.text or "")).strip()
        key = norm(text)
        if len(text) < 20:
            stats["shorter_than_20"] += 1
            continue
        if key in seen_text:
            stats["duplicate_text"] += 1
            continue
        seen_text.add(key)
        image_path, extra = first_existing(record.image_path)
        if extra:
            stats["rows_with_extra_photos"] += 1
        if record.image_path and not image_path:
            missing.append(record.id)
            continue
        row_id = record.id
        if row_id in seen_id:
            row_id = f"kept-{row_id}"
            stats["renamed_id"] += 1
        seen_id.add(row_id)
        role = record.image_role if image_path else "none"
        if role == "review":
            role = "paired_review"
            link = "paired"
        elif role == "shared_web":
            link = "shared"
        else:
            link = ""
        rows.append(
            {
                "id": row_id,
                "in_mm_v2": "no",
                "subset": record.source,
                "domain": record.scene,
                "label": record.label,
                "language": record.language,
                "protocol": protocol_for_kept(record.source),
                "source": record.source,
                "generator": "",
                "polarity": "",
                "item": "",
                "group_id": row_id,
                "label_quality": record.basis,
                "usable": "true",
                "split": "",
                "text": text,
                "image_path": image_path,
                "image_224_path": "",
                "aug_path": "",
                "image_role": role,
                "image_link": link,
                **blank_numbers(),
            }
        )
    if missing:
        raise SystemExit(f"kept image missing for {missing[:5]}")
    return pd.DataFrame(rows, columns=FIELDS), dict(stats)


def attach_224(frame: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, str] = {}
    paths = []
    made = 0
    for index, record in enumerate(frame.itertuples(index=False), start=1):
        source = record.image_path
        if not source:
            paths.append("")
            continue
        if source in cache:
            paths.append(cache[source])
            continue
        relative = source.replace("\\", "/")
        if relative.startswith("data/processed/mm_v2/images/"):
            dest_relative = relative
        else:
            digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
            dest = IMAGE_DIR / f"{digest}.jpg"
            save_square(ROOT / relative, dest)
            dest_relative = str(dest.relative_to(ROOT)).replace("\\", "/")
            made += 1
        cache[source] = dest_relative
        paths.append(dest_relative)
        if index % 5000 == 0:
            print(f"square {index} new {made}", flush=True)
    frame = frame.copy()
    frame["image_224_path"] = paths
    print(f"square done unique {len(cache)} new files touched {made}", flush=True)
    return frame


def add_linguistics(frame: pd.DataFrame) -> pd.DataFrame:
    ling = pd.read_csv(
        FROZEN / "mm_v2.csv",
        usecols=POS_COLUMNS,
    )
    ling.index = [f"mmv2-{index:05d}" for index in range(1, len(ling) + 1)]
    lookup = ling
    nltk_module = None
    copied = 0
    computed = 0
    numbers = {column: [] for column in POS_COLUMNS}
    for index, record in enumerate(frame.itertuples(index=False)):
        if record.in_mm_v2 == "yes" and record.id in lookup.index:
            hit = lookup.loc[record.id]
            numbers["n_tokens"].append(int(hit["n_tokens"]))
            for column in POS_COLUMNS[1:]:
                numbers[column].append(float(hit[column]))
            copied += 1
        else:
            if record.language == "zh":
                count, pos = pos_chinese(record.text)
            else:
                if nltk_module is None:
                    nltk_module = ensure_nltk()
                count = len(tokens(record.text))
                pos = pos_english(record.text, nltk_module)
            numbers["n_tokens"].append(int(count))
            for column in POS_COLUMNS[1:]:
                numbers[column].append(float(pos[column]))
            computed += 1
        done = copied + computed
        if done % 5000 == 0:
            print(f"pos {done} copied {copied} computed {computed}", flush=True)
    frame = frame.copy()
    for column, values in numbers.items():
        frame[column] = values
    print(f"pos done copied {copied} computed {computed}", flush=True)
    return frame


def fill_splits(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rng = np.random.default_rng(SEED)
    empty = frame["split"].astype(str).isin(["", "nan"])
    for protocol, part in frame.loc[empty].groupby("protocol", sort=False):
        groups = part["group_id"].drop_duplicates().tolist()
        rng.shuffle(groups)
        count = len(groups)
        if count < 3:
            frame.loc[part.index, "split"] = "train"
            continue
        n_test = max(1, int(round(count * 0.1)))
        n_val = max(1, int(round(count * 0.1)))
        if n_test + n_val >= count:
            n_test, n_val = 1, 1
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        group_values = frame.loc[part.index, "group_id"]
        frame.loc[part.index[group_values.isin(test_groups)], "split"] = "test"
        frame.loc[part.index[group_values.isin(val_groups)], "split"] = "val"
        assigned = test_groups | val_groups
        frame.loc[part.index[~group_values.isin(assigned)], "split"] = "train"
        for split_name in ("val", "test"):
            labels = sorted(part["label"].unique())
            present = set(frame.loc[part.index].loc[frame.loc[part.index, "split"] == split_name, "label"])
            for label in labels:
                if label in present:
                    continue
                donors = frame.loc[part.index]
                donors = donors.loc[(donors["split"] == "train") & (donors["label"] == label), "group_id"]
                donors = donors.drop_duplicates()
                if donors.empty:
                    continue
                group = donors.iloc[0]
                frame.loc[part.index[frame.loc[part.index, "group_id"] == group], "split"] = split_name
                present.add(label)
        print(f"split {protocol} groups {count}", flush=True)
    if frame["split"].astype(str).isin(["", "nan"]).any():
        raise SystemExit("some rows still have no split")
    return frame


def add_augmentation(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    cache: dict[str, str] = {}
    aug_paths = []
    made = 0
    for record in frame.itertuples(index=False):
        source = record.image_224_path
        if record.split != "train" or not source:
            aug_paths.append("")
            continue
        if source in cache:
            aug_paths.append(cache[source])
            continue
        relative = source.replace("\\", "/")
        if relative.startswith("data/processed/mm_v2/images/"):
            existing = ROOT / "data" / "processed" / "mm_v2" / "aug" / Path(relative).name
            if existing.is_file():
                dest_relative = str(existing.relative_to(ROOT)).replace("\\", "/")
                cache[source] = dest_relative
                aug_paths.append(dest_relative)
                continue
        dest = AUG_DIR / f"{Path(relative).stem}.jpg"
        seed = int(hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8], 16)
        augment(ROOT / relative, dest, seed)
        dest_relative = str(dest.relative_to(ROOT)).replace("\\", "/")
        cache[source] = dest_relative
        aug_paths.append(dest_relative)
        made += 1
        if made % 500 == 0:
            print(f"aug {made}", flush=True)
    frame["aug_path"] = aug_paths
    print(f"aug done new {made}", flush=True)
    return frame


def update_catalog(frame: pd.DataFrame) -> None:
    with CATALOG.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], [row for row in rows[1:] if len(row) >= 7 and row[2] != "三类训练表"]

    def image_text(mask: pd.Series) -> str:
        part = frame.loc[mask]
        sd = int((part["image_role"] == "paired_sd").sum())
        none = int((part["image_role"] == "none").sum())
        return f"{sd} 张 Stable Diffusion，{none} 条无图"

    replacements = {
        "芝加哥酒店两个小模型": image_text(frame["subset"] == "hotel_ai"),
        "酒店 MAiDE-up": image_text(frame["source"] == "maide-up"),
        "电商 Salminen": image_text(frame["source"] == "salminen"),
    }
    for row in body:
        if row[2] in replacements:
            row[4] = replacements[row[2]]
    body.append(
        [
            "训练",
            "真 / 人假 / AI假",
            "三类训练表",
            str(len(frame)),
            "有图的缩到 224",
            "按 protocol 分开",
            "frozen/three_class.csv",
        ]
    )
    with CATALOG.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)


def main() -> None:
    allowed = load_allowed()
    if allowed["id"].duplicated().any():
        raise SystemExit("allowed review id is not unique")
    seen_text = {norm(text) for text in allowed["text"]}
    kept, kept_stats = load_kept(seen_text, set(allowed["id"]))
    frame = pd.concat([allowed[FIELDS], kept], ignore_index=True)
    if frame["id"].duplicated().any():
        raise SystemExit("three-class id is not unique")
    print(f"rows {len(frame)} kept_added {len(kept)}", flush=True)
    frame = attach_224(frame)
    frame = add_linguistics(frame)
    frame = fill_splits(frame)
    frame = add_augmentation(frame)
    missing_224 = frame.loc[
        frame["image_path"].astype(bool) & ~frame["image_224_path"].map(lambda path: bool(path) and (ROOT / path).is_file())
    ]
    if not missing_224.empty:
        raise SystemExit(f"missing 224 copies {len(missing_224)}")
    bad_size = 0
    from PIL import Image

    sample = frame.loc[frame["image_224_path"].astype(bool), "image_224_path"].drop_duplicates().head(12)
    for relative in sample:
        with Image.open(ROOT / relative) as image:
            if image.size != (224, 224):
                bad_size += 1
    if bad_size:
        raise SystemExit("a sampled 224 image has another size")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame[FIELDS].to_csv(OUT, index=False)
    update_catalog(frame)

    def counts(column: str) -> dict:
        return {str(key): int(value) for key, value in frame[column].value_counts().items()}

    audit = {
        "rows": int(len(frame)),
        "mm_v2_unchanged": True,
        "output": "data/frozen/three_class.csv",
        "kept_added": int(len(kept)),
        "kept_stats": kept_stats,
        "by_label": counts("label"),
        "by_domain": counts("domain"),
        "by_protocol": counts("protocol"),
        "by_split": counts("split"),
        "by_image_role": counts("image_role"),
        "with_image": int(frame["image_224_path"].astype(bool).sum()),
        "without_image": int((~frame["image_224_path"].astype(bool)).sum()),
        "train_aug": int(frame["aug_path"].astype(bool).sum()),
        "note": "Rows with different protocol values are not one training pool. same_experiment and different_origin stay apart. rule_label, platform_collected and platform_user stay apart from gold labels. The full table is not rebalanced. mm_v2.csv was not modified.",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("rows", "kept_added", "by_label", "by_protocol", "with_image", "without_image")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

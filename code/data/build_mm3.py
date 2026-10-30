"""Assemble the food / hotel / ecommerce three-class multimodal table.

Images are stored as 224×224 JPEG files. Original width and height are kept
only as audit columns. The model scripts do not read them.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from mm3_common import (
    DOMAINS,
    EXPERIMENT,
    HOTEL_AI_CSV,
    IMAGE_DIR,
    LABELS,
    PER_CELL,
    ROOT,
    SEED,
    TABLE,
)
from template_banks import researcher_templates

PRODUCT_CATS = [
    "Home_and_Kitchen_5",
    "Electronics_5",
    "Clothing_Shoes_and_Jewelry_5",
    "Toys_and_Games_5",
    "Pet_Supplies_5",
    "Sports_and_Outdoors_5",
    "Tools_and_Home_Improvement_5",
]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def first_n(text: str, count: int = 10) -> str:
    return " ".join(tokens(text)[:count])


def save_square(image: Image.Image, dest: Path) -> tuple[int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    rgb = rgb.resize((224, 224), Image.Resampling.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(dest, "JPEG", quality=90)
    return width, height


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def sample_rows(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) < count:
        raise SystemExit(f"need {count} rows, found {len(frame)}")
    picked = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[picked].copy()


def sample_half(frame: pd.DataFrame, count: int, column: str, rng: np.random.Generator) -> pd.DataFrame:
    levels = sorted(frame[column].astype(str).unique())
    base, extra = divmod(count, len(levels))
    parts = []
    for index, level in enumerate(levels):
        take = base + (1 if index < extra else 0)
        part = frame.loc[frame[column].astype(str) == level]
        parts.append(sample_rows(part, take, rng))
    return pd.concat(parts, ignore_index=True)


def sample_across(frame: pd.DataFrame, column: str, count: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = {
        key: group.sample(frac=1, random_state=int(rng.integers(0, 1_000_000_000))).reset_index(drop=True)
        for key, group in frame.groupby(column)
    }
    keys = sorted(groups)
    picked = []
    cursor = 0
    while len(picked) < count:
        key = keys[cursor % len(keys)]
        group = groups[key]
        if len(group):
            picked.append(group.iloc[0])
            groups[key] = group.iloc[1:].reset_index(drop=True)
        cursor += 1
        if cursor > count * max(len(keys), 1) * 3:
            break
    if len(picked) < count:
        raise SystemExit(f"could not balance {column}: have {len(picked)} need {count}")
    return pd.DataFrame(picked)


def load_prefix_blocklist() -> set[str]:
    blocked = set()
    aigen = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv", usecols=["text"])
    ott = pd.read_csv(ROOT / "data" / "frozen" / "ott_hotel.csv", usecols=["text"])
    sales = pd.read_parquet(ROOT / "data" / "raw" / "salminen" / "train.parquet", columns=["text"])
    for series in (aigen["text"], ott["text"], sales["text"]):
        blocked.update(first_n(text) for text in series.fillna("").astype(str))
    blocked.discard("")
    return blocked


def food_rows(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv")
    frame = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    rows = []
    donors = []
    for label in ("real", "ai_fake"):
        part = frame.loc[frame["label"] == label]
        chosen = sample_rows(part, PER_CELL, rng)
        for record in chosen.itertuples(index=False):
            rows.append(
                {
                    "domain": "food",
                    "label": label,
                    "text": record.text,
                    "image_src": str(ROOT / record.image_path),
                    "group_id": f"aigen-{record.source_id}",
                    "source": "aigen",
                    "generator": "gpt-4-turbo+dalle-2" if label == "ai_fake" else "",
                    "polarity": "",
                    "rating": np.nan,
                }
            )
    real_ids = {row["group_id"] for row in rows if row["label"] == "real"}
    real_pool = frame.loc[frame["label"] == "real"]
    real_pool = real_pool.loc[~real_pool["source_id"].map(lambda value: f"aigen-{value}").isin(real_ids)]
    donors_frame = sample_rows(real_pool, PER_CELL, rng)
    for record in donors_frame.itertuples(index=False):
        donors.append(str(ROOT / record.image_path))
    return rows, donors


def hotel_text_rows(rng: np.random.Generator) -> list[dict]:
    frame = pd.read_csv(ROOT / "data" / "frozen" / "ott_hotel.csv")
    frame = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    rows = []
    for label in ("real", "human_fake"):
        part = frame.loc[frame["label"] == label]
        chosen = sample_half(part, PER_CELL, "polarity", rng)
        for record in chosen.itertuples(index=False):
            rows.append(
                {
                    "domain": "hotel",
                    "label": label,
                    "text": record.text,
                    "image_src": "",
                    "group_id": f"ott-{record.source_id}",
                    "source": "ott",
                    "generator": "",
                    "polarity": record.polarity,
                    "rating": np.nan,
                }
            )
    return rows


def hotel_ai_rows(blocked: set[str]) -> list[dict]:
    frame = pd.read_csv(HOTEL_AI_CSV)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.loc[frame["text"].map(lambda text: len(tokens(text)) >= 60)].copy()
    seen = set()
    rows = []
    for record in frame.itertuples(index=False):
        key = first_n(record.text)
        if key in blocked or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "domain": "hotel",
                "label": "ai_fake",
                "text": record.text,
                "image_src": "",
                "group_id": f"qwen-{len(rows)}",
                "source": "qwen2.5-0.5b",
                "generator": record.generator,
                "polarity": record.polarity,
                "rating": np.nan,
            }
        )
        if len(rows) == PER_CELL:
            break
    if len(rows) < PER_CELL:
        raise SystemExit(f"hotel AI reviews after the prefix check: {len(rows)} < {PER_CELL}")
    return rows


def ecommerce_rows(rng: np.random.Generator) -> list[dict]:
    frame = pd.read_parquet(
        ROOT / "data" / "raw" / "salminen" / "train.parquet",
        columns=["cat", "note", "label", "text"],
    )
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.loc[frame["cat"].isin(PRODUCT_CATS)].copy()
    frame = frame.loc[~frame["text"].str.contains("###", regex=False)].copy()
    frame = frame.loc[frame["text"].map(lambda text: len(tokens(text)) >= 30)].copy()
    rows = []
    for raw_label, label in (("real", "real"), ("fake", "ai_fake")):
        part = frame.loc[frame["label"] == raw_label]
        chosen = sample_across(part, "cat", PER_CELL, rng)
        for record in chosen.itertuples(index=False):
            rows.append(
                {
                    "domain": "ecommerce",
                    "label": label,
                    "text": record.text,
                    "image_src": "",
                    "group_id": f"salminen-{hashlib.sha1(record.text.encode('utf-8')).hexdigest()[:12]}",
                    "source": "salminen",
                    "generator": "gpt-2" if label == "ai_fake" else "",
                    "polarity": "",
                    "rating": float(record.note),
                }
            )
    return rows


def template_rows(donors: list[str]) -> list[dict]:
    rows = []
    food_index = 0
    for record in researcher_templates():
        row = {
            "domain": record["domain"],
            "label": record["label"],
            "text": record["text"],
            "image_src": "",
            "group_id": record["group_id"],
            "source": record["source"],
            "generator": "",
            "polarity": record["polarity"],
            "rating": np.nan,
        }
        if record["domain"] == "food":
            row["image_src"] = donors[food_index]
            food_index += 1
        rows.append(row)
    if food_index != PER_CELL:
        raise SystemExit(f"food template images {food_index} != {PER_CELL}")
    return rows


def cell_to_image(cell: object) -> Image.Image:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    if isinstance(raw, memoryview):
        raw = bytes(raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def parquet_images(path: Path, limit: int | None = None) -> list[Image.Image]:
    import pyarrow.parquet as pq

    images = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16, columns=["image"]):
        for cell in batch.column("image").to_pylist():
            images.append(cell_to_image(cell))
            if limit is not None and len(images) >= limit:
                return images
    return images


def materialize_pool(images: list[Image.Image], prefix: str) -> list[tuple[str, int, int]]:
    pool = []
    dest_dir = IMAGE_DIR / "pools"
    for index, image in enumerate(images):
        dest = dest_dir / f"{prefix}_{index:04d}.jpg"
        width, height = save_square(image, dest)
        pool.append((str(dest.relative_to(ROOT)).replace("\\", "/"), width, height))
    return pool


def assign_pool(text: str, pool: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return pool[int(digest[:8], 16) % len(pool)]


def ensure_product_pool() -> list[Image.Image]:
    dest = ROOT / "data" / "raw" / "abo_small" / "shard.parquet"
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        import urllib.request

        meta_url = (
            "https://datasets-server.huggingface.co/parquet?"
            "dataset=suvadityamuk/amazon-berkeley-objects&config=images_small"
        )
        with urllib.request.urlopen(meta_url, timeout=60) as response:
            payload = json.load(response)
        files = payload.get("parquet_files") or []
        if not files:
            raise SystemExit("ABO images_small has no parquet shard")
        first = min(files, key=lambda item: item.get("size") or 10**18)
        if (first.get("size") or 0) > 400_000_000:
            raise SystemExit(f"ABO shard is too large to fetch in this pass: {first.get('size')}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading product images {first['url']}", flush=True)
        urllib.request.urlretrieve(first["url"], dest)
    return parquet_images(dest, limit=240)


def attach_images(rows: list[dict], hotel_pool, product_pool) -> None:
    for row in rows:
        if row["image_src"] and row["domain"] == "food":
            stem = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:16]
            dest = IMAGE_DIR / f"food_{stem}.jpg"
            width, height = save_square(open_image(Path(row["image_src"])), dest)
            row["image_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")
            row["orig_width"] = width
            row["orig_height"] = height
            continue
        pool = hotel_pool if row["domain"] == "hotel" else product_pool
        path, width, height = assign_pool(row["text"], pool)
        row["image_path"] = path
        row["orig_width"] = width
        row["orig_height"] = height
        row["image_src"] = ""


def assign_splits(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frame = frame.copy()
    frame["split"] = "train"
    for domain in DOMAINS:
        for label in LABELS:
            mask = (frame["domain"] == domain) & (frame["label"] == label)
            groups = frame.loc[mask, "group_id"].drop_duplicates().tolist()
            rng.shuffle(groups)
            count = len(groups)
            n_test = max(1, int(round(count * 0.1)))
            n_val = max(1, int(round(count * 0.1)))
            if count >= 10:
                n_test = max(2, n_test)
            if n_test + n_val >= count:
                n_test, n_val = 1, 1
            test_groups = set(groups[:n_test])
            val_groups = set(groups[n_test : n_test + n_val])
            in_cell = frame.loc[mask, "group_id"]
            frame.loc[in_cell.index[in_cell.isin(test_groups)], "split"] = "test"
            frame.loc[in_cell.index[in_cell.isin(val_groups)], "split"] = "val"
    return frame


def main() -> None:
    if not HOTEL_AI_CSV.exists():
        raise SystemExit(f"missing {HOTEL_AI_CSV}; run generate_hotel_ai.py first")
    hotel_parquet = ROOT / "data" / "raw" / "hotel_images" / "train.parquet"
    if not hotel_parquet.exists():
        raise SystemExit(f"missing {hotel_parquet}")
    rng = np.random.default_rng(SEED)
    blocked = load_prefix_blocklist()
    food, donors = food_rows(rng)
    rows = food + hotel_text_rows(rng) + hotel_ai_rows(blocked) + ecommerce_rows(rng)
    rows.extend(template_rows(donors))
    texts = [row["text"] for row in rows]
    if len(texts) != len(set(texts)):
        raise SystemExit("duplicate full text inside the assembled table")
    owners: dict[str, str] = {}
    for row in rows:
        key = first_n(row["text"])
        if row["source"] == "researcher_template" and key in blocked:
            raise SystemExit("a template shares its first 10 words with a source corpus")
        previous = owners.get(key)
        if previous is not None and previous != row["group_id"]:
            raise SystemExit(f"first-10-word collision between {previous} and {row['group_id']}")
        owners[key] = row["group_id"]
    print("materializing image pools", flush=True)
    hotel_pool = materialize_pool(parquet_images(hotel_parquet), "hotel")
    product_pool = materialize_pool(ensure_product_pool()[:240], "product")
    if len(hotel_pool) < 30 or len(product_pool) < 30:
        raise SystemExit("image pool is too small")
    attach_images(rows, hotel_pool, product_pool)
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", [f"mm3_{index:05d}" for index in range(len(frame))])
    frame["usable"] = True
    frame = assign_splits(frame, rng)
    counts = frame.groupby(["domain", "label", "split"]).size().rename("n").reset_index()
    missing = [
        (domain, label)
        for domain in DOMAINS
        for label in LABELS
        if (frame["domain"].eq(domain) & frame["label"].eq(label)).sum() != PER_CELL
    ]
    if missing:
        raise SystemExit(f"cells not at {PER_CELL}: {missing}")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    keep = [
        "id", "domain", "label", "split", "text", "image_path", "orig_width", "orig_height",
        "group_id", "source", "generator", "polarity", "rating", "usable",
    ]
    frame[keep].to_csv(TABLE, index=False)
    audit = {
        "seed": SEED,
        "per_cell": PER_CELL,
        "rows": int(len(frame)),
        "saved_image_side": 224,
        "cells": counts.to_dict(orient="records"),
        "sources": frame["source"].value_counts().to_dict(),
        "licenses": {
            "aigen": "MIT",
            "ott": "CC BY-NC-SA 3.0; raw archive is not uploaded",
            "salminen": "CC BY 4.0; GPT-2 fakes are ai_fake, not human_fake",
            "hotel_images": "Onno/hotel-images card states no license; local only, not uploaded",
            "product_images": "Amazon Berkeley Objects mirror, CC BY 4.0",
            "templates": "written for this study; deceptive by construction; not crowd gold",
        },
        "prefix_rule": "hotel AI rows whose first 10 words appear in Ott, AiGen, or Salminen were dropped",
        "image_rule": "every file used by the model is 224x224; orig_width and orig_height are audit-only",
    }
    (EXPERIMENT / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(counts.to_string(index=False))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()

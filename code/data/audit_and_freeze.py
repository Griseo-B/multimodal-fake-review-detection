"""Audit local sources and write frozen subset tables.

Raw files are read only. Outputs go to data/frozen/.
"""

from __future__ import annotations

import csv
import hashlib
import random
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "frozen"
SEED = 42


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                handle.seek(16)
                width, height = struct.unpack(">II", handle.read(8))
                return int(width), int(height)
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker = handle.read(2)
                    if len(marker) < 2:
                        return None
                    while marker[0] != 0xFF:
                        marker = marker[1:2] + handle.read(1)
                    if marker[1] in (0xC0, 0xC1, 0xC2):
                        handle.read(3)
                        height, width = struct.unpack(">HH", handle.read(4))
                        return int(width), int(height)
                    if marker[1] == 0xD8 or marker[1] == 0x01 or 0xD0 <= marker[1] <= 0xD9:
                        continue
                    size_bytes = handle.read(2)
                    if len(size_bytes) < 2:
                        return None
                    size = struct.unpack(">H", size_bytes)[0]
                    handle.seek(size - 2, 1)
    except OSError:
        return None
    return None


def load_aigen() -> pd.DataFrame:
    rows = []
    split_files = {
        "train": RAW / "aigen_foodreview" / "data" / "train-00000-of-00003.parquet",
        "train_b": RAW / "aigen_foodreview" / "data" / "train-00001-of-00003.parquet",
        "train_c": RAW / "aigen_foodreview" / "data" / "train-00002-of-00003.parquet",
        "val": RAW / "aigen_foodreview" / "data" / "valid-00000-of-00001.parquet",
        "test": RAW / "aigen_foodreview" / "data" / "test-00000-of-00001.parquet",
    }
    frames = []
    train_parts = []
    for name, path in split_files.items():
        frame = pd.read_parquet(path, columns=["ID", "text", "label"])
        if name.startswith("train"):
            train_parts.append(frame)
        else:
            frame = frame.copy()
            frame["split"] = name
            frames.append(frame)
    train = pd.concat(train_parts, ignore_index=True)
    train["split"] = "train"
    data = pd.concat([train, *frames], ignore_index=True)
    data["text"] = data["text"].fillna("").map(lambda value: str(value).strip())
    data["label_name"] = data["label"].map({0: "real", 1: "ai_fake"})
    data["source_id"] = data["ID"].map(lambda value: str(int(value)) if pd.notna(value) else "")
    return data


def freeze_aigen(data: pd.DataFrame) -> dict:
    images = RAW / "aigen_foodreview" / "images"
    records = []
    size_by_label = Counter()
    missing_images = 0
    for row in data.itertuples(index=False):
        image_path = None
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = images / f"{row.source_id}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            missing_images += 1
            width = height = None
        else:
            parsed = image_size(image_path)
            if parsed is None:
                width = height = None
            else:
                width, height = parsed
                size_by_label[(row.label_name, width, height)] += 1
        records.append(
            {
                "subset": "aigen_food",
                "source_id": row.source_id,
                "split": row.split,
                "label": row.label_name,
                "text": row.text,
                "image_path": str(image_path.relative_to(ROOT)) if image_path else "",
                "image_width": width if width is not None else "",
                "image_height": height if height is not None else "",
            }
        )
    frame = pd.DataFrame.from_records(records)
    text_splits = defaultdict(set)
    for row in frame.itertuples(index=False):
        text_splits[row.text].add(row.split)
    cross = {text for text, splits in text_splits.items() if len(splits) > 1}
    rank = {"train": 0, "val": 1, "test": 2}
    frame = frame.sort_values(["text", "split", "source_id"], key=lambda col: col.map(rank) if col.name == "split" else col)
    seen = set()
    usable = []
    reason = []
    for row in frame.itertuples(index=False):
        if row.text in seen:
            usable.append(False)
            reason.append("duplicate_text_kept_in_earlier_split" if row.text in cross else "duplicate_text_within_split")
        elif not row.image_path:
            usable.append(False)
            reason.append("missing_image")
        else:
            usable.append(True)
            reason.append("")
            seen.add(row.text)
    frame["usable"] = usable
    frame["exclude_reason"] = reason
    frame.to_csv(OUT / "aigen_food.csv", index=False)
    sample = frame.sample(20, random_state=SEED)
    return {
        "n": len(frame),
        "usable": int(frame["usable"].sum()),
        "by_split": frame.loc[frame["usable"], "split"].value_counts().to_dict(),
        "by_label": frame.loc[frame["usable"], "label"].value_counts().to_dict(),
        "cross_split_texts": len(cross),
        "excluded": int((~frame["usable"]).sum()),
        "missing_images": missing_images,
        "size_by_label": size_by_label.most_common(8),
        "sample": sample,
    }


def read_ott() -> pd.DataFrame:
    root = RAW / "ott" / "op_spam_v1.4"
    rows = []
    for path in root.rglob("*.txt"):
        polarity = "negative" if "negative_polarity" in path.parts else "positive"
        label = "human_fake" if "deceptive_from_MTurk" in path.parts else "real"
        fold = next(part for part in path.parts if part.startswith("fold"))
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        rows.append(
            {
                "subset": "ott_hotel",
                "source_id": path.stem,
                "fold": fold,
                "polarity": polarity,
                "label": label,
                "text": text,
                "path": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def freeze_ott(frame: pd.DataFrame) -> dict:
    fold_to_split = {"fold1": "train", "fold2": "train", "fold3": "train", "fold4": "val", "fold5": "test"}
    frame = frame.copy()
    frame["split"] = frame["fold"].map(fold_to_split)
    text_splits = defaultdict(set)
    for row in frame.itertuples(index=False):
        text_splits[row.text].add(row.split)
    cross = {text for text, splits in text_splits.items() if len(splits) > 1}
    seen = set()
    usable = []
    reason = []
    for row in frame.sort_values(["text", "split", "source_id"]).itertuples(index=False):
        if row.text in seen:
            usable.append(False)
            reason.append("duplicate_text")
        else:
            usable.append(True)
            reason.append("")
            seen.add(row.text)
    ordered = frame.sort_values(["text", "split", "source_id"]).copy()
    ordered["usable"] = usable
    ordered["exclude_reason"] = reason
    ordered.to_csv(OUT / "ott_hotel.csv", index=False)
    return {
        "n": len(ordered),
        "usable": int(ordered["usable"].sum()),
        "by_label": ordered.loc[ordered["usable"], "label"].value_counts().to_dict(),
        "by_split": ordered.loc[ordered["usable"], "split"].value_counts().to_dict(),
        "cross_split_texts": len(cross),
        "sample": ordered.sample(20, random_state=SEED),
        "frame": ordered,
    }


def extract_gpt3() -> Path:
    dest = RAW / "op_spam_ai" / "gpt3"
    if any(dest.rglob("*.txt")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    base = RAW / "op_spam_ai" / "op_spam_AI-main"
    for name in ("generated_GPT3_positive.zip", "generated_GPT3_negative.zip"):
        with zipfile.ZipFile(base / name) as archive:
            archive.extractall(dest)
    return dest


def audit_gpt3(ott: pd.DataFrame) -> dict:
    if not (RAW / "op_spam_ai").exists():
        return {
            "status": "excluded_archive_removed",
            "n": 800,
            "prefix10_hits": 800,
            "note": "2026-10-26 local audit: 800/800 GPT-3 reviews share the first 10 words with the paired Ott review. Not frozen. The archive was deleted after that audit.",
        }
    dest = extract_gpt3()
    truthful = {
        (row.polarity, row.source_id[2:]): row.text
        for row in ott.itertuples(index=False)
        if row.label == "real"
    }
    rows = []
    prefix_hits = 0
    for path in dest.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        polarity = "negative" if "negative" in path.parts or path.stem.startswith("g_") and "neg" in str(path).lower() else None
        # Archives are split by the zip name, which remains in the extracted folder name.
        folder = str(path).lower()
        if "negative" in folder:
            polarity = "negative"
        elif "positive" in folder:
            polarity = "positive"
        key = (polarity, path.stem[2:] if path.stem.startswith("g_") else path.stem)
        paired = truthful.get(key, "")
        generated_words = text.split()
        paired_words = paired.split()
        hit = bool(paired_words) and generated_words[:10] == paired_words[:10]
        prefix_hits += int(hit)
        rows.append({"file": path.name, "polarity": polarity, "chars": len(text), "prefix10_matches_truthful": hit})
    sample_paths = sorted(dest.rglob("*.txt"))
    rng = random.Random(SEED)
    picked = rng.sample(sample_paths, 20) if len(sample_paths) >= 20 else sample_paths
    return {
        "n": len(rows),
        "prefix10_hits": prefix_hits,
        "sample_files": [path.name for path in picked],
        "empty": sum(1 for row in rows if row["chars"] == 0),
    }


def spot_text_file(path: Path, column_candidates: list[str]) -> dict:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    text_col = next((name for name in column_candidates if name in frame.columns), None)
    if text_col is None:
        text_col = next((name for name in frame.columns if frame[name].dtype == object), frame.columns[0])
    texts = frame[text_col].fillna("").map(str)
    empty = int((texts.str.strip() == "").sum())
    sample = frame.sample(min(20, len(frame)), random_state=SEED)
    return {
        "n": len(frame),
        "columns": list(frame.columns)[:12],
        "text_col": text_col,
        "empty": empty,
        "sample_ids": sample.index.astype(str).tolist()[:20],
    }


def write_sample_csv(name: str, frame: pd.DataFrame) -> None:
    keep = [col for col in ("source_id", "split", "label", "text") if col in frame.columns]
    out = frame[keep].copy()
    if "text" in out.columns:
        out["text_preview"] = out["text"].map(lambda value: str(value).replace("\n", " ")[:80])
        out = out.drop(columns=["text"])
    out.to_csv(OUT / name, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aigen = load_aigen()
    aigen_stats = freeze_aigen(aigen)
    write_sample_csv("aigen_spotcheck.csv", aigen_stats["sample"])
    ott = read_ott()
    ott_stats = freeze_ott(ott)
    write_sample_csv("ott_spotcheck.csv", ott_stats["sample"])
    gpt3_stats = audit_gpt3(ott_stats["frame"])
    optional_spots = {
        "ctrip_chnsenticorp": (RAW / "ctrip_chnsenticorp" / "ChnSentiCorp_htl_all.csv", ["review"]),
        "ctrip_trip_sample_2026": (RAW / "ctrip_trip_sample_2026" / "reviews.csv", ["review", "text", "content"]),
        "ecommerce_10cats": (RAW / "ecommerce_10cats" / "online_shopping_10_cats.csv", ["review", "text"]),
        "waimai_10k": (RAW / "waimai_10k" / "waimai_10k.csv", ["review"]),
        "jd_review": (RAW / "jd_review" / "train.parquet", ["text", "sentence"]),
        "jd_herbal": (RAW / "jd_herbal" / "train.parquet", ["review_text", "text", "review"]),
        "maide_up": (RAW / "maide_up" / "all_data.csv", ["Upside_Review"]),
    }
    spots = {}
    for name, (path, columns) in optional_spots.items():
        if path.exists():
            spots[name] = spot_text_file(path, columns)
        else:
            spots[name] = {"status": "not_on_disk", "reason": "not a gold label; raw file removed"}
    summary = {
        "aigen": {key: value for key, value in aigen_stats.items() if key != "sample"},
        "ott": {key: value for key, value in ott_stats.items() if key not in ("sample", "frame")},
        "gpt3": gpt3_stats,
        "spots": spots,
    }
    pd.Series({"ok": True}).to_json(OUT / "audit_summary.json")
    import json

    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

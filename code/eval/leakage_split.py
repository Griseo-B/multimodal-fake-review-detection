"""Does the image split leak, and does purging it change the number?

1,413 images sit on more than one side of the original split. This script counts
the overlap and fits the image block twice: once with the original split, once
with split_purged, where every image hash lives on one side. If the second
number is much lower, the first one was the image, not the review.

Also run: rows whose image is shared by two labels removed entirely. That is the
strict version the image block should report in the midterm text.

Outputs: overlap.json, metrics.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.common import ROOT  # noqa: E402

SEED = 42
IMAGE_BLOCK = (
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio",
    "saliency_offset", "third_rule_score", "border_energy_ratio", "lr_asymmetry",
    "clip_cosine", "sent_max_cosine", "sent_mean_cosine",
    "scene_prob", "scene_match", "entity_image_prob",
    "image_valence", "sentiment_synergy", "sentiment_agreement",
)
FEATURE_DIR = ROOT / "data" / "processed" / "features" / "20261008_public_dataset"


def load() -> pd.DataFrame:
    pool = pd.read_csv(ROOT / "data" / "frozen" / "three_class_pool_b.csv", dtype=str,
                       keep_default_na=False, low_memory=False)
    pool = pool.loc[pool["label"].isin(["real", "ai_fake"])].copy()
    for name in ("visual_features.parquet", "alignment_features.parquet", "scene_features.parquet"):
        path = FEATURE_DIR / name
        if path.is_file():
            pool = pool.merge(pd.read_parquet(path), on="id", how="left")
    pool = pool.loc[pool["image_md5"].astype(bool)].reset_index(drop=True)
    pool["y"] = (pool["label"] == "ai_fake").astype(int)
    return pool


def overlap(frame: pd.DataFrame, column: str) -> dict:
    train = set(frame.loc[frame[column] == "train", "image_md5"])
    test = frame.loc[frame[column] == "test"]
    shared = test["image_md5"].isin(train)
    return {
        "split_column": column,
        "test_rows": int(len(test)),
        "test_rows_sharing_an_image_with_train": int(shared.sum()),
        "share": round(float(shared.mean()), 4),
    }


def fit(frame: pd.DataFrame, split_column: str) -> dict | None:
    features = [name for name in IMAGE_BLOCK if name in frame.columns]
    train = frame.loc[frame[split_column] == "train"]
    test = frame.loc[frame[split_column] == "test"]
    if train["y"].nunique() < 2 or test["y"].nunique() < 2:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
    )
    model.fit(train[features].to_numpy(dtype=float), train["y"].to_numpy())
    score = model.predict_proba(test[features].to_numpy(dtype=float))[:, 1]
    prediction = (score >= 0.5).astype(int)
    return {
        "split_column": split_column,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": len(features),
        "macro_f1": round(float(f1_score(test["y"], prediction, average="macro")), 4),
        "auc": round(float(roc_auc_score(test["y"], score)), 4),
    }


def pool_overlap(name: str) -> dict:
    """Table-level overlap: no features needed, so every image row counts."""
    pool = pd.read_csv(ROOT / "data" / "frozen" / f"three_class_pool_{name}.csv", dtype=str,
                       keep_default_na=False, low_memory=False)
    pool = pool.loc[pool["image_md5"].astype(bool)]
    result = {"pool": name, "rows_with_image": int(len(pool))}
    for column in ("split", "split_purged"):
        train = set(pool.loc[pool[column] == "train", "image_md5"])
        test = pool.loc[pool[column] == "test"]
        result[f"test_rows_{column}"] = int(len(test))
        result[f"test_sharing_image_{column}"] = int(test["image_md5"].isin(train).sum())
    result["rows_with_cross_label_image"] = int(
        pool["image_cross_label"].astype(str).str.lower().isin(["true", "1"]).sum()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/20261013_leakage_split")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    frame = load()
    print(f"rows with an image and an image feature {len(frame)}", flush=True)
    overlaps = [overlap(frame, column) for column in ("split", "split_purged")]
    rows = [row for row in (fit(frame, "split"), fit(frame, "split_purged")) if row]
    clean = frame.loc[~frame["image_cross_label"].astype(str).str.lower().isin(["true", "1"])]
    rows.append({**fit(clean, "split_purged"), "variant": "purged_without_cross_label_images"})
    for row in rows:
        row.setdefault("variant", row["split_column"])
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    (out / "overlap.json").write_text(json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "contrast": "real vs ai_fake on public_dataset rows that own an image",
        "feature_block": "image only, 21 statistics",
        "overlap": overlaps,
        "pool_overlap": [pool_overlap(name) for name in ("a", "b")],
        "rows": rows,
        "note": (
            "The point of the second row is the difference from the first. The image block "
            "reads generator provenance, so a drop means part of the earlier number was the "
            "same picture appearing on both sides."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

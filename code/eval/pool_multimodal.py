"""Can the pools be trained as three-class multimodal, and what does the image add?

Every pool row with an image now has an image feature, including the shared
ones. That makes the question answerable with numbers instead of prose:

    paired coverage   how many rows per label actually own their image
    availability      how well `has_image` alone predicts the label
    text+has_image    how much of a multimodal gain is only "this row has a photo"
    text+image        three classes with image statistics, shared images included
    paired+text       three classes restricted to review-owned images

The last one is expected to fail. It is reported as a failure with the class
counts, because that is the finding: human_fake has no review-owned image
anywhere in the table, so a three-class multimodal fit on own images cannot
exist yet. The middle rows exist to stop the team from quoting a number that is
really the photo's provenance.

Outputs: coverage.csv, metrics.csv, per_class.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_three_class import (  # noqa: E402
    LABELS,
    balance_train,
    bootstrap_classifier,
    load_pool,
    make_model,
    scores_for,
)
from features.common import ROOT  # noqa: E402

SEED = 42
PAIRED_DIR = ROOT / "data" / "processed" / "features" / "20261008_public_dataset"
SHARED_DIR = ROOT / "data" / "processed" / "features" / "20261015_pool_shared_images"
TEXT_PARQUET = ROOT / "data" / "processed" / "features" / "20261008_all_protocols"
IMAGE_FILES = ("visual_features.parquet", "alignment_features.parquet", "scene_features.parquet")
IMAGE_COLUMNS = (
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio", "saliency_offset",
    "third_rule_score", "border_energy_ratio", "lr_asymmetry",
    "clip_cosine", "sent_max_cosine", "sent_mean_cosine",
    "scene_prob", "scene_match", "entity_image_prob", "image_valence",
    "sentiment_synergy", "sentiment_agreement",
)


def resolve(frame: pd.DataFrame, names) -> list[str]:
    """Find each name in the frame, tolerating merge suffixes and missing columns.

    load_pool merges the text parquet onto a table that already has pos_*
    columns read as strings, so pandas writes `pos_noun_x` / `pos_noun_y`.
    Picking by dtype keeps the numeric copy and never double counts.
    """
    found = []
    for name in names:
        for candidate in (name, f"{name}_y", f"{name}_x"):
            if candidate in frame.columns and pd.api.types.is_numeric_dtype(frame[candidate]):
                found.append(candidate)
                break
    return found


def text_feature_columns() -> list[str]:
    header = pd.read_parquet(TEXT_PARQUET / "text_features.parquet").columns.tolist()
    names = [name for name in header if name not in {"id", "opening_dup", "ngram_dup_ratio"}]
    return names + ["perplexity", "smoothness"]


def attach_images(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    parts = []
    for folder in (PAIRED_DIR, SHARED_DIR):
        for name in IMAGE_FILES:
            path = folder / name
            if path.is_file():
                parts.append(pd.read_parquet(path))
    if parts:
        image = parts[0]
        for part in parts[1:]:
            image = image.merge(part, on="id", how="outer")
        frame = frame.merge(image, on="id", how="left")
    frame["has_image"] = frame["has_image"].astype(bool) if "has_image" in frame else (
        frame["image_224_path"].astype(bool)
    )
    frame["has_paired_image"] = frame["image_role"].isin(["paired_review", "paired_dalle", "paired_sd"])
    return frame


def coverage_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        part = frame.loc[frame["label"] == label]
        if part.empty:
            continue
        rows.append(
            {
                "pool": name,
                "label": label,
                "rows": int(len(part)),
                "any_image": int(part["has_image"].sum()),
                "any_image_rate": round(float(part["has_image"].mean()), 4),
                "paired_image": int(part["has_paired_image"].sum()),
                "paired_image_rate": round(float(part["has_paired_image"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def run_setting(name: str, pool: str, block: str, frame: pd.DataFrame, columns: list[str],
                rng) -> tuple[dict | None, list[dict]]:
    train = frame.loc[frame["split_purged"] == "train"].reset_index(drop=True)
    test = frame.loc[frame["split_purged"] == "test"].reset_index(drop=True)
    classes = [label for label in LABELS if label in set(train["label"]) and label in set(test["label"])]
    if len(classes) < 3 or train.empty or test.empty:
        return None, []
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return None, []
    train = balance_train(train, "balanced")
    model = make_model("lightgbm", "balanced", len(classes))
    model.fit(train[columns].to_numpy(dtype=float), train["label"].to_numpy())
    probabilities = scores_for(model, test, columns, classes)
    prediction = np.array(classes)[probabilities.argmax(axis=1)]
    truth = test["label"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0
    )
    low, high = bootstrap_classifier(truth, prediction, rng)
    row = {
        "pool": pool,
        "setting": name,
        "block": block,
        "features": len(columns),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "classes": "|".join(classes),
        "macro_f1": round(float(f1_score(truth, prediction, average="macro")), 4),
        "f1_low": round(low, 4),
        "f1_high": round(high, 4),
        "accuracy": round(float((prediction == truth).mean()), 4),
    }
    detail = [
        {
            "pool": pool,
            "setting": name,
            "label": classes[index],
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index in range(len(classes))
    ]
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/20261015_pool_multimodal")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    coverage_frames, rows, detail_rows, blockers = [], [], [], []
    for pool in ("a", "b"):
        frame = attach_images(load_pool(pool))
        coverage_frames.append(coverage_table(frame, pool))
        text_columns = resolve(frame, text_feature_columns())
        image_columns = resolve(frame, IMAGE_COLUMNS)
        print(f"pool {pool} text {len(text_columns)} image {len(image_columns)}", flush=True)
        with_image = frame.loc[frame["has_image"]].reset_index(drop=True)
        paired_only = frame.loc[frame["has_paired_image"]].reset_index(drop=True)

        settings = [
            ("availability_only", "has_image", frame.assign(has_image=frame["has_image"].astype(int)), ["has_image"]),
            ("text(rows with image)", "text", with_image, text_columns),
            ("text+has_image", "text+flag", frame.assign(has_image=frame["has_image"].astype(int)), text_columns + ["has_image"]),
            ("text+image(shared ok)", "text+mm", with_image, text_columns + image_columns),
            ("image only(shared ok)", "mm", with_image, image_columns),
            ("text+image(paired only)", "text+mm", paired_only, text_columns + image_columns),
        ]
        for name, block, part, columns in settings:
            row, detail = run_setting(name, pool, block, part, columns, rng)
            if row is None:
                counts = {label: int((part["label"] == label).sum()) for label in LABELS}
                blockers.append({"pool": pool, "setting": name, "rows": int(len(part)), "by_label": counts,
                                 "reason": "a class is missing from train or test, so macro-F1 is undefined"})
                print(f"blocked {pool} {name} {counts}", flush=True)
                continue
            rows.append(row)
            detail_rows.extend(detail)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    coverage = pd.concat(coverage_frames, ignore_index=True)
    coverage.to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(out / "per_class.csv", index=False, encoding="utf-8-sig")

    summary = {
        "paired_image_rows_by_label": {
            row["pool"]: {"label": row["label"], "paired": row["paired_image"], "rows": row["rows"]}
            for row in coverage.to_dict("records")
        },
        "blocked": blockers,
        "rows": rows,
        "note": (
            "Every pool row with an image now carries an image feature, shared photos included. "
            "`availability_only` is the leak measure: it uses nothing but 'this row has a photo'. "
            "`text(rows with image)` is the control row for `text+image(shared ok)`: both are fitted "
            "on the same rows, so the difference is the image block and nothing else. Numbers on a "
            "subset are not comparable with the full-table ones in 20261016_pool_three_class. "
            "Every text+mm gain here comes from photographs of the hotel or product, not from a "
            "review-owned image, so it must be reported as provenance, not as fake-review evidence."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(coverage.to_string(index=False), flush=True)
    print(table.to_string(index=False) if not table.empty else "no rows", flush=True)


if __name__ == "__main__":
    main()

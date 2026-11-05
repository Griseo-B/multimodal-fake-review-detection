"""Does the image block measure fakery, or just synthetic versus photographic?

public_dataset holds two kinds of paired images: photos attached to real
reviews, DALL-E-2 images, and Stable Diffusion images generated in this study.
If the same image features separate DALL-E from Stable Diffusion with high
accuracy, they are reading a generator fingerprint. If they cannot, they are
reading the synthetic-versus-photographic gap. Either way the number says
something about the image pipeline, not about whether the review is honest.

Run after baseline_models.py; both write into the same experiment folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_models import IMAGE_BLOCK, fit_predict, load  # noqa: E402
from common import ROOT, TABLE  # noqa: E402
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
FOLDS = 5


def evaluate_block(frame: pd.DataFrame, block: str, model: str = "logreg") -> dict | None:
    frame = frame.reset_index(drop=True)
    features = [name for name in IMAGE_BLOCK if name in frame.columns]
    features = [name for name in features if frame[name].notna().sum() > 200]
    if len(features) < 3 or frame["y"].nunique() < 2:
        return None
    if frame["group_id"].eq("").any():
        frame.loc[frame["group_id"] == "", "group_id"] = frame.loc[frame["group_id"] == "", "id"]
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(frame))
    for train_index, test_index in splitter.split(frame, frame["y"], frame["group_id"]):
        scores[test_index] = fit_predict(frame.iloc[train_index], frame.iloc[test_index], features, model)
    return {
        "block": block,
        "model": model,
        "rows": int(len(frame)),
        "features": len(features),
        "positives": int(frame["y"].sum()),
        "auc": round(float(roc_auc_score(frame["y"], scores)), 4),
        "majority": round(float(max(frame["y"].mean(), 1 - frame["y"].mean())), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/20261008_public_dataset")
    parser.add_argument("--out", default="experiments/20261012_baseline")
    args = parser.parse_args()

    folder = ROOT / args.features
    frame = load(folder, "public_dataset", with_images=True)
    frame = frame.loc[frame["brightness"].notna()].reset_index(drop=True)

    meta = pd.read_csv(
        TABLE,
        usecols=["id", "generator", "image_role", "label"],
        dtype=str,
        keep_default_na=False,
    )
    frame = frame.merge(meta.drop(columns=["label"]), on="id", how="left")

    rows = []
    # generator names the text generator, so the image side is read from
    # image_role: paired_review is a photo, paired_dalle and paired_sd are fake.
    pairs = [
        ("real_vs_ai", frame.loc[frame["image_role"].isin(["paired_review", "paired_dalle", "paired_sd"])], {"paired_review": 0, "paired_dalle": 1, "paired_sd": 1}),
        ("dalle_vs_sd", frame.loc[frame["image_role"].isin(["paired_dalle", "paired_sd"])], {"paired_dalle": 1, "paired_sd": 0}),
        ("real_vs_sd", frame.loc[frame["image_role"].isin(["paired_review", "paired_sd"])], {"paired_review": 0, "paired_sd": 1}),
    ]
    for name, part, mapping in pairs:
        part = part.copy()
        part["y"] = part["image_role"].map(mapping)
        part = part.loc[part["y"].notna()].copy()
        part["y"] = part["y"].astype(int)
        if part["y"].nunique() < 2 or len(part) < 300:
            continue
        result = evaluate_block(part, name)
        if result:
            rows.append(result)

    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "image_audit.csv", index=False, encoding="utf-8-sig")
    document = {
        "question": "Is the image block reading a generator fingerprint or the synthetic-versus-photo gap?",
        "reading": "dalle_vs_sd at or above 0.90 means the block identifies the generator, "
                   "so real_vs_ai on images is not evidence that the review is dishonest.",
        "rows": table.to_dict("records"),
        "counts": {
            "generator": {str(k): int(v) for k, v in frame["generator"].value_counts().items()},
            "image_role": {str(k): int(v) for k, v in frame["image_role"].value_counts().items()},
        },
    }
    (out / "image_audit.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document["rows"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Can this feature set actually classify, or only rank single features?

Single-feature AUC says one number at a time. This fits a plain classifier on
the same features, so the answer to "is the feature set usable" has a number
too. Protocol stays inside the loop: rows of different protocols are never in
one fit, and folds are grouped on group_id so near-duplicate texts cannot sit
on both sides of a fold.

Two feature blocks are reported separately, because the image block only
exists for rows with a review-owned photo:

    text        the text table only
    text+mm     text plus image statistics, scene agreement and alignment

Reported: out-of-fold macro-F1 and AUC with a bootstrap interval, plus the
majority-class baseline so a weak number is recognisable as weak.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, TABLE  # noqa: E402

SEED = 42
FOLDS = 5
BOOTSTRAP = 500

TEXT_FILES = ("text_features.parquet", "perplexity.parquet", "smoothness.parquet")
IMAGE_FILES = (
    "visual_features.parquet",
    "alignment_features.parquet",
    "scene_features.parquet",
)
# Columns that describe the corpus rather than a single review. Keeping them
# in a classifier would report a number no deployment could reach.
CORPUS_LEVEL = {"opening_dup", "ngram_dup_ratio"}
# Features whose AUC was flagged as possibly carrying provenance rather than
# style: readability and perplexity diverge most between the two text
# generators. Dropping them is the lower bound, not the headline.
RISKY = {
    "perplexity", "flesch", "ari", "gunning_fog", "avg_word_len",
}
IMAGE_BLOCK = (
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio",
    "saliency_offset", "third_rule_score", "border_energy_ratio", "lr_asymmetry",
    "clip_cosine", "sent_max_cosine", "sent_mean_cosine",
    "scene_prob", "scene_match", "entity_image_prob",
    "image_valence", "sentiment_synergy", "sentiment_agreement",
)
ALWAYS_DROP = {
    "id", "domain", "label", "language", "protocol", "split", "group_id",
    "subset", "y",
}


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    size = len(labels)
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, size, size)
        chosen = labels[index]
        if chosen.min() == chosen.max():
            continue
        values.append(fast_auc(chosen, scores[index]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load(folder: Path, protocol: str, with_images: bool) -> pd.DataFrame:
    meta = pd.read_csv(TABLE, dtype=str, keep_default_na=False)
    meta = meta.loc[meta["protocol"] == protocol].reset_index(drop=True)
    files = TEXT_FILES + (IMAGE_FILES if with_images else ())
    keep_meta = ["id", "domain", "language", "label", "split", "group_id"]
    frame = meta[keep_meta].copy()
    for name in files:
        path = folder / name
        if path.is_file():
            frame = frame.merge(pd.read_parquet(path), on="id", how="left")
    return frame


def feature_columns(frame: pd.DataFrame, only: set[str] | None = None) -> list[str]:
    numeric = []
    for name in frame.columns:
        if name in ALWAYS_DROP or name in CORPUS_LEVEL:
            continue
        if only is not None and name not in only:
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        if frame[name].notna().sum() < 200:
            continue
        numeric.append(name)
    return numeric


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model: str) -> np.ndarray:
    if model == "logreg":
        pipeline = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        )
        pipeline.fit(train[features].to_numpy(dtype=float), train["y"].to_numpy())
        return pipeline.predict_proba(test[features].to_numpy(dtype=float))[:, 1]
    import lightgbm as lgb

    booster = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    booster.fit(train[features].to_numpy(dtype=float), train["y"].to_numpy())
    return booster.predict_proba(test[features].to_numpy(dtype=float))[:, 1]


def evaluate(
    frame: pd.DataFrame,
    protocol: str,
    block: str,
    model: str,
    rng,
    only: set[str] | None = None,
) -> dict | None:
    frame = frame.loc[frame["label"].isin(["real", "ai_fake", "human_fake"])].copy()
    fakes = [name for name in frame["label"].unique() if name != "real"]
    if not fakes:
        return None
    fake = fakes[0]
    frame = frame.loc[frame["label"].isin(["real", fake])].reset_index(drop=True)
    if frame["label"].nunique() < 2 or len(frame) < 300:
        return None
    frame["y"] = (frame["label"] == fake).astype(int)
    if "group_id" not in frame.columns or frame["group_id"].eq("").all():
        frame["group_id"] = frame["id"]
    frame.loc[frame["group_id"] == "", "group_id"] = frame.loc[frame["group_id"] == "", "id"]
    features = feature_columns(frame, only)
    if len(features) < 3:
        return None

    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(frame))
    started = time.perf_counter()
    for train_index, test_index in splitter.split(frame, frame["y"], frame["group_id"]):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        scores[test_index] = fit_predict(train, test, features, model)

    labels = frame["y"].to_numpy()
    predictions = (scores >= 0.5).astype(int)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1_fake = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_real = true_negative / (true_negative + false_negative) if true_negative + false_negative else 0.0
    recall_real = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if precision_real + recall_real else 0.0
    auc = fast_auc(labels, scores)
    low, high = bootstrap_auc(labels, scores, rng)
    majority = max(labels.mean(), 1 - labels.mean())
    return {
        "protocol": protocol,
        "contrast": f"real_vs_{fake}",
        "block": block,
        "model": model,
        "rows": int(len(frame)),
        "features": len(features),
        "positives": int(labels.sum()),
        "macro_f1": round(float((f1_fake + f1_real) / 2), 4),
        "f1_fake": round(float(f1_fake), 4),
        "f1_real": round(float(f1_real), 4),
        "auc": round(auc, 4),
        "auc_ci_low": round(low, 4),
        "auc_ci_high": round(high, 4),
        "majority_macro_f1": round(float(majority), 4),
        "seconds": round(time.perf_counter() - started, 1),
        "note": "out-of-fold, grouped on group_id; corpus-level duplicate features excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-features", default="data/processed/features/20261008_public_dataset")
    parser.add_argument("--fake-features", default="data/processed/features/20261008_all_protocols")
    parser.add_argument("--out", default="experiments/20261012_baseline")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    ai_folder = ROOT / args.ai_features
    text_frame = load(ai_folder, "public_dataset", with_images=False)
    mm_frame = load(ai_folder, "public_dataset", with_images=True)
    mm_frame = mm_frame.loc[mm_frame["brightness"].notna()] if "brightness" in mm_frame else mm_frame
    image_only = set(IMAGE_BLOCK)
    for model in ("logreg", "lightgbm"):
        for block, frame, only in (
            ("text", text_frame, None),
            ("text+mm", mm_frame, None),
            # Image block alone: this answers "is a DALL-E image separable from
            # a photo", which is a provenance question, not a review question.
            ("mm-only", mm_frame, image_only),
        ):
            result = evaluate(frame, "public_dataset", block, model, rng, only)
            if result:
                rows.append(result)
        cleaned = set(feature_columns(text_frame)) - RISKY
        result = evaluate(text_frame, "public_dataset", "text-sane", model, rng, cleaned)
        if result:
            rows.append(result)
        for domain in ("food", "ecommerce", "hotel"):
            part = mm_frame.loc[mm_frame["domain"] == domain]
            result = evaluate(part, f"public_dataset/{domain}", "text+mm", model, rng)
            if result:
                rows.append(result)
            # Matched text-only control on the same rows, so the domain rows can
            # be read as "image adds this much", not "this subset is easier".
            same_rows = text_frame.loc[text_frame["domain"] == domain]
            result = evaluate(same_rows, f"public_dataset/{domain}", "text", model, rng)
            if result:
                rows.append(result)

    fake_folder = ROOT / args.fake_features
    for protocol in ("different_origin", "same_experiment", "rule_label"):
        frame = load(fake_folder, protocol, with_images=False)
        for model in ("logreg", "lightgbm"):
            result = evaluate(frame, protocol, "text", model, rng)
            if result:
                rows.append(result)

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "folds": FOLDS,
        "bootstrap": BOOTSTRAP,
        "seed": SEED,
        "grouping": "StratifiedGroupKFold on group_id",
        "excluded": sorted(CORPUS_LEVEL) + ["raw image width, height, byte size"],
        "best": (
            table.sort_values("macro_f1", ascending=False)
            .head(8)[["protocol", "block", "model", "rows", "features", "macro_f1", "auc"]]
            .to_dict("records")
            if not table.empty
            else []
        ),
        "note": "macro-F1 is out-of-fold at threshold 0.5; the majority baseline is beside it.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False) if not table.empty else "no rows", flush=True)


if __name__ == "__main__":
    main()

"""Three-class baselines on the frozen pools, with the imbalance handled.

Two numbers decide whether the pools are trainable: the hotel-only pool (A) and
the cross-source pool (B). Both are text-only here. The image block exists only
for public_dataset rows, and pool A has an image for 899 of its 2,637 ai_fake
rows and none of its real rows, so a three-class multimodal fit is not possible
yet -- that is a finding, not a skipped step.

Because real:ai_fake:human_fake is about 3.2:3.1:1 in pool B and 1.4:1.4:1 in
pool A, three weighting schemes are reported side by side:

    plain       no weighting
    balanced    class_weight balanced in the model
    downsample  training real rows cut to 1.5x the largest other class

Training uses split_purged == train, evaluation uses split_purged == test, so
no image sits on both sides. A grouped 5-fold number is reported as well, for
stability; the test column is the headline.

Outputs: metrics.csv, per_class.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.common import ROOT  # noqa: E402

SEED = 42
FOLDS = 5
BOOTSTRAP = 500
LABELS = ["real", "human_fake", "ai_fake"]
CORPUS_LEVEL = {"opening_dup", "ngram_dup_ratio"}
DROP = {
    "id", "domain", "label", "language", "protocol", "split", "split_purged",
    "subset", "label_quality", "image_md5", "y", "group_id", "source",
    "generator",
}
FEATURE_DIR = ROOT / "data" / "processed" / "features" / "20261008_all_protocols"


def load_pool(name: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "frozen" / f"three_class_pool_{name}.csv", dtype=str,
                        keep_default_na=False, low_memory=False)
    for feature_file in ("text_features.parquet", "perplexity.parquet", "smoothness.parquet"):
        frame = frame.merge(
            pd.read_parquet(FEATURE_DIR / feature_file), on="id", how="left"
        )
    frame = frame.loc[frame["label"].isin(LABELS)].reset_index(drop=True)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for name in frame.columns:
        if name in DROP or name in CORPUS_LEVEL:
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        if frame[name].notna().sum() < 200:
            continue
        columns.append(name)
    return columns


def make_model(kind: str, weighting: str, n_classes: int):
    if kind == "logreg":
        weights = "balanced" if weighting == "balanced" else None
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight=weights, random_state=SEED),
        )
    import lightgbm as lgb

    if weighting == "balanced":
        weight = "balanced"
    else:
        weight = None
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=4,
        verbose=-1, class_weight=weight,
    )


def balance_train(train: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Cut every class down to 1.5x the smallest class, training rows only."""
    if weighting != "downsample":
        return train
    counts = train["label"].value_counts()
    if counts.empty:
        return train
    cap = int(counts.min() * 1.5)
    parts = []
    for _, part in train.groupby("label"):
        if len(part) > cap:
            part = part.sample(n=cap, random_state=SEED)
        parts.append(part)
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def scores_for(model, frame: pd.DataFrame, features: list[str], classes) -> np.ndarray:
    probabilities = model.predict_proba(frame[features].to_numpy(dtype=float))
    table = pd.DataFrame(probabilities, columns=list(model.classes_))
    for label in classes:
        if label not in table.columns:
            table[label] = 0.0
    return table[list(classes)].to_numpy()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def bootstrap_classifier(y_true: np.ndarray, y_pred: np.ndarray, rng) -> tuple[float, float]:
    values = []
    size = len(y_true)
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, size, size)
        if len(np.unique(y_true[index])) < 2:
            continue
        values.append(f1_score(y_true[index], y_pred[index], average="macro"))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_once(name: str, frame: pd.DataFrame, kind: str, weighting: str, rng) -> tuple[dict, list[dict]]:
    features = feature_columns(frame)
    train = frame.loc[frame["split_purged"] == "train"].reset_index(drop=True)
    test = frame.loc[frame["split_purged"] == "test"].reset_index(drop=True)
    train = balance_train(train, weighting)
    classes = [label for label in LABELS if label in set(train["label"])]
    if len(classes) < 3:
        return None, []

    started = time.perf_counter()
    model = make_model(kind, weighting, len(classes))
    model.fit(train[features].to_numpy(dtype=float), train["label"].to_numpy())
    probabilities = scores_for(model, test, features, classes)
    prediction = np.array(classes)[probabilities.argmax(axis=1)]
    truth = test["label"].to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0
    )
    low, high = bootstrap_classifier(truth, prediction, rng)
    majority = train["label"].value_counts().idxmax()
    majority_f1 = macro_f1(truth, np.repeat(majority, len(truth)))

    row = {
        "pool": name,
        "model": kind,
        "weighting": weighting,
        "features": len(features),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "macro_f1": round(macro_f1(truth, prediction), 4),
        "f1_low": round(low, 4),
        "f1_high": round(high, 4),
        "accuracy": round(float((prediction == truth).mean()), 4),
        "majority_macro_f1": round(majority_f1, 4),
        "majority_label": str(majority),
        "seconds": round(time.perf_counter() - started, 1),
        "split": "split_purged train/test",
    }
    per_class = [
        {
            "pool": name,
            "model": kind,
            "weighting": weighting,
            "label": classes[index],
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index in range(len(classes))
    ]
    return row, per_class


def cross_validated(name: str, frame: pd.DataFrame, kind: str, weighting: str) -> dict | None:
    """Grouped 5-fold over all rows of the pool, for stability only."""
    features = feature_columns(frame)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros((len(frame), 3))
    classes = np.array(LABELS)
    picked = np.zeros(len(frame), dtype=bool)
    for train_index, test_index in splitter.split(frame, frame["label"], frame["group_id"]):
        train = balance_train(frame.iloc[train_index], weighting)
        model = make_model(kind, weighting, 3)
        model.fit(train[features].to_numpy(dtype=float), train["label"].to_numpy())
        probabilities = scores_for(model, frame.iloc[test_index], features, LABELS)
        scores[test_index] = probabilities
        picked[test_index] = True
    if not picked.all():
        return None
    prediction = classes[scores.argmax(axis=1)]
    return {
        "pool": name,
        "model": kind,
        "weighting": weighting,
        "rows": int(len(frame)),
        "cv_macro_f1": round(macro_f1(frame["label"].to_numpy(), prediction), 4),
        "cv_note": "5-fold StratifiedGroupKFold on group_id, out-of-fold",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", default=["a", "b"])
    parser.add_argument("--models", nargs="+", default=["logreg", "lightgbm"])
    parser.add_argument("--weightings", nargs="+", default=["plain", "balanced", "downsample"])
    parser.add_argument("--out", default="experiments/20261016_pool_three_class")
    parser.add_argument("--cv", action="store_true", help="also run grouped 5-fold")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, per_class_rows, cv_rows = [], [], []
    for pool in args.pools:
        frame = load_pool(pool)
        print(f"pool {pool} rows {len(frame)} {frame['label'].value_counts().to_dict()}", flush=True)
        for kind in args.models:
            for weighting in args.weightings:
                row, detail = run_once(pool, frame, kind, weighting, rng)
                if not row:
                    print(f"skip {pool} {kind} {weighting}", flush=True)
                    continue
                rows.append(row)
                per_class_rows.extend(detail)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.cv:
                cv = cross_validated(pool, frame, kind, "balanced")
                if cv:
                    cv_rows.append(cv)
                    print(json.dumps(cv, ensure_ascii=False), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(out / "per_class.csv", index=False, encoding="utf-8-sig")
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(out / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "pools": args.pools,
        "folds_and_seed": {"test": "split_purged train/test", "cv": f"{FOLDS}-fold grouped, seed {SEED}", "bootstrap": BOOTSTRAP},
        "class_order": LABELS,
        "downsample_rule": "training rows of every class capped at 1.5x the smallest class",
        "best_per_pool": {
            pool: (
                table.loc[table["pool"] == pool].sort_values("macro_f1", ascending=False)
                .head(3)[["model", "weighting", "macro_f1", "f1_low", "f1_high", "majority_macro_f1"]]
                .to_dict("records")
            )
            for pool in args.pools
            if not table.loc[table["pool"] == pool].empty
        },
        "cv": cv_rows,
        "note": (
            "Text only. The image block covers public_dataset rows only, so no three-class "
            "multimodal fit exists on either pool yet. macro-F1 is reported against the "
            "majority-class baseline in the same table, because a number below that baseline "
            "means the weighting, not the features, is doing the work."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

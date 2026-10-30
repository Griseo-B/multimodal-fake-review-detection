"""TF-IDF text baselines on a frozen subset.

Fits the vectorizer on the train split only. Reports the untouched test split.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SEED = 42

POSITIVE = {
    "good", "great", "excellent", "amazing", "wonderful", "delicious", "tasty",
    "love", "loved", "best", "friendly", "perfect", "nice", "fresh", "clean",
    "recommend", "awesome", "pleasant", "comfortable", "beautiful", "favorite",
    "yummy", "outstanding", "helpful", "quick", "generous",
}
NEGATIVE = {
    "bad", "terrible", "awful", "horrible", "disgusting", "worst", "dirty",
    "rude", "slow", "bland", "cold", "stale", "poor", "disappointing",
    "overpriced", "mediocre", "nasty", "inedible", "unfriendly", "broken",
    "noisy", "smell", "smelly", "wait", "never",
}


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def length_feature(text: str) -> float:
    return float(len(word_tokens(text)))


def sentiment_feature(text: str) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    pos = sum(token in POSITIVE for token in tokens)
    neg = sum(token in NEGATIVE for token in tokens)
    return (pos - neg) / len(tokens)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    row = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
    }
    for label, prec, rec in zip(labels, precision, recall):
        row[f"precision_{label}"] = float(prec)
        row[f"recall_{label}"] = float(rec)
    return row


def fit_and_score(
    train_text: pd.Series,
    y_train: np.ndarray,
    test_text: pd.Series,
    y_test: np.ndarray,
    labels: list[str],
    with_side_features: bool,
) -> tuple[dict, TfidfVectorizer, LogisticRegression, StandardScaler | None]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    scaler = None
    if with_side_features:
        side_train = np.column_stack(
            [train_text.map(length_feature), train_text.map(sentiment_feature)]
        )
        side_test = np.column_stack(
            [test_text.map(length_feature), test_text.map(sentiment_feature)]
        )
        scaler = StandardScaler()
        side_train = scaler.fit_transform(side_train)
        side_test = scaler.transform(side_test)
        x_train = sparse.hstack([x_train, sparse.csr_matrix(side_train)])
        x_test = sparse.hstack([x_test, sparse.csr_matrix(side_test)])
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    model.fit(x_train, y_train)
    metrics = evaluate(y_test, model.predict(x_test), labels)
    return metrics, vectorizer, model, scaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / args.table)
    frame = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    labels = args.labels
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    rows = []
    for name, use_side in (("tfidf", False), ("tfidf_length_sentiment", True)):
        metrics, vectorizer, model, scaler = fit_and_score(
            train["text"], train["label"].to_numpy(), test["text"], test["label"].to_numpy(), labels, use_side
        )
        metrics["model"] = name
        metrics["split"] = "test"
        metrics["n_train"] = int(len(train))
        metrics["n_test"] = int(len(test))
        rows.append(metrics)
        import joblib

        joblib.dump(
            {"vectorizer": vectorizer, "model": model, "scaler": scaler, "labels": labels},
            out / f"model_{name}.joblib",
        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "split_counts.json").write_text(
        json.dumps(frame["split"].value_counts().to_dict(), indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

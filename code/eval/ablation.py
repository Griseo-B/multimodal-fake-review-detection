"""Ablation on features that were already extracted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
LABELS = ["real", "ai_fake"]


def row(name: str, dropped: str, y_true, y_pred) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    result = {
        "model": name,
        "dropped": dropped,
        "split": "test",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "status": "完成",
    }
    for label, prec, rec in zip(LABELS, precision, recall):
        result[f"precision_{label}"] = float(prec)
        result[f"recall_{label}"] = float(rec)
    return result


def predict(features: np.ndarray, train: np.ndarray, test: np.ndarray, labels: np.ndarray) -> np.ndarray:
    head = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    head.fit(features[train], labels[train])
    return head.predict(features[test])


def main() -> None:
    data = np.load(ROOT / "experiments" / "20261026_modality_rows" / "features.npz", allow_pickle=True)
    text = data["text"]
    image = data["image"]
    similarity = data["similarity"]
    split = data["split"]
    labels = data["label"]
    train = split == "train"
    test = split == "test"
    y_test = labels[test]
    specs = {
        "clip_fusion": (np.hstack([text, image, similarity]), "none"),
        "drop_image": (np.hstack([text, similarity]), "image"),
        "drop_similarity": (np.hstack([text, image]), "similarity"),
        "clip_text": (text, "image_and_similarity"),
    }
    rows = []
    for name, (features, dropped) in specs.items():
        rows.append(row(name, dropped, y_test, predict(features, train, test, labels)))
    text_metrics = pd.read_csv(ROOT / "experiments" / "20261026_aigen_tfidf" / "metrics.csv")
    for _, item in text_metrics.iterrows():
        dropped = "none" if item["model"] == "tfidf_length_sentiment" else "length_and_sentiment"
        rows.append(
            {
                "model": item["model"],
                "dropped": dropped,
                "split": "test",
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                "status": "完成",
                "precision_real": item["precision_real"],
                "recall_real": item["recall_real"],
                "precision_ai_fake": item["precision_ai_fake"],
                "recall_ai_fake": item["recall_ai_fake"],
            }
        )
    out = ROOT / "experiments" / "20261026_ablation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(rows)[["model", "dropped", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

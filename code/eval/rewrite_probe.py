"""Lexicon paraphrase of audited AI reviews, scored by an already trained model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N = 200

SYNONYMS = {
    "good": "decent",
    "great": "strong",
    "excellent": "superb",
    "amazing": "striking",
    "wonderful": "lovely",
    "delicious": "flavorful",
    "tasty": "appetizing",
    "love": "enjoy",
    "loved": "enjoyed",
    "best": "finest",
    "friendly": "cordial",
    "perfect": "ideal",
    "nice": "pleasant",
    "fresh": "crisp",
    "clean": "tidy",
    "recommend": "suggest",
    "awesome": "impressive",
    "beautiful": "attractive",
    "bad": "poor",
    "terrible": "dreadful",
    "awful": "unpleasant",
    "horrible": "grim",
    "dirty": "unclean",
    "rude": "impolite",
    "slow": "sluggish",
    "cold": "chilly",
    "really": "truly",
    "very": "quite",
    "the": "the",
    "restaurant": "eatery",
    "food": "meal",
    "service": "staff",
    "place": "spot",
    "hotel": "property",
    "room": "suite",
    "staff": "employees",
    "deliciously": "tastefully",
}


def rewrite(text: str) -> str:
    def replace(match: re.Match) -> str:
        word = match.group(0)
        key = word.lower()
        if key not in SYNONYMS or SYNONYMS[key] == key:
            return word
        swapped = SYNONYMS[key]
        if word[:1].isupper():
            swapped = swapped[:1].upper() + swapped[1:]
        return swapped

    rewritten = re.sub(r"[A-Za-z']+", replace, str(text))
    if rewritten == text:
        rewritten = "In short, " + text
    return rewritten


def score(bundle: dict, texts: pd.Series, y_true: np.ndarray) -> dict:
    matrix = bundle["vectorizer"].transform(texts)
    if bundle["scaler"] is not None:
        from code.train.text_baseline import length_feature, sentiment_feature

        side = np.column_stack([texts.map(length_feature), texts.map(sentiment_feature)])
        side = bundle["scaler"].transform(side)
        matrix = sparse.hstack([matrix, sparse.csr_matrix(side)])
    pred = bundle["model"].predict(matrix)
    labels = bundle["labels"]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(texts)),
        "predictions": pred,
    }


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv")
    usable = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    pool = usable.loc[(usable["split"] == "test") & (usable["label"] == "ai_fake")]
    sample = pool.sample(N, random_state=SEED).copy()
    sample["rewritten"] = sample["text"].map(rewrite)
    reals = usable.loc[(usable["split"] == "test") & (usable["label"] == "real"), ["source_id", "label", "text"]]
    bundle = joblib.load(ROOT / "experiments" / "20261026_aigen_tfidf" / "model_tfidf.joblib")
    original_text = pd.concat([sample["text"], reals["text"]], ignore_index=True)
    rewritten_text = pd.concat([sample["rewritten"], reals["text"]], ignore_index=True)
    y = pd.concat([sample["label"], reals["label"]], ignore_index=True).to_numpy()
    original = score(bundle, original_text, y)
    rewritten = score(bundle, rewritten_text, y)
    out = ROOT / "experiments" / "20261026_rewrite_tfidf"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        [
            {"version": "original", "accuracy": original["accuracy"], "macro_f1": original["macro_f1"], "n": original["n"], "n_rewritten_ai": N},
            {"version": "rewritten", "accuracy": rewritten["accuracy"], "macro_f1": rewritten["macro_f1"], "n": rewritten["n"], "n_rewritten_ai": N},
        ]
    )
    rows["macro_f1_drop"] = rows["macro_f1"] - original["macro_f1"]
    rows.loc[rows["version"] == "original", "macro_f1_drop"] = 0.0
    rows.to_csv(out / "metrics.csv", index=False)
    preview = sample[["source_id", "label", "text", "rewritten"]].copy()
    preview["pred_original"] = original["predictions"][:N]
    preview["pred_rewritten"] = rewritten["predictions"][:N]
    preview.to_csv(out / "rewritten_200.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "model": "experiments/20261026_aigen_tfidf/model_tfidf.joblib",
                "method": "fixed English synonym list; unchanged sentences receive the prefix 'In short, '",
                "original_macro_f1": original["macro_f1"],
                "rewritten_macro_f1": rewritten["macro_f1"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()

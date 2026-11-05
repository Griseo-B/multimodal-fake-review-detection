"""Baselines on the borrowed-web-image benchmark.

Answers three questions on the frozen split:

1. How separable are real (own photo) from human fake (borrowed web photo)
   reviews at all?
2. How much of that is text, and how much is the picture?
3. Is any image gain real, or is it the known confound that web photos look
   different from amateur snapshots? Reported per domain and per pairing
   strength so a shortcut has nowhere to hide.

    python code/eval/benchmark_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
FEAT = ROOT / "data" / "processed" / "features"
OUT = ROOT / "experiments" / "20261020_web_image_benchmark"

VISUAL = [
    "brightness", "saturation", "contrast", "lap_var", "edge_density", "colorfulness",
    "gray_entropy", "fft_high_ratio", "saliency_offset", "third_rule_score",
    "border_energy_ratio", "lr_asymmetry",
]
ALIGN = ["clip_cosine", "sent_max_cosine", "sent_mean_cosine"]
SCENE = ["scene_prob", "scene_match", "entity_image_prob", "image_valence",
         "sentiment_synergy", "sentiment_agreement"]
FEATURE_DIRS = [
    "20261008_public_dataset",
    "20261015_pool_shared_images",
    "20261019_web_image_pair",
    "20261020_web_image_benchmark",
]


def load() -> pd.DataFrame:
    d = pd.read_csv(FROZEN / "web_image_benchmark.csv", dtype=str, keep_default_na=False)
    parts = []
    for name in FEATURE_DIRS:
        base = FEAT / name
        v = pd.read_parquet(base / "visual_features.parquet")
        a = pd.read_parquet(base / "alignment_features.parquet")
        s = pd.read_parquet(base / "scene_features.parquet")
        parts.append(v.merge(a, on="id").merge(s, on="id"))
    feats = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    d = d.merge(feats, on="id", how="inner")
    d["y"] = d["binary_label"].astype(int)
    return d


def fit_text(train: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000)
    X = vec.fit_transform(train["text"])
    model = LogisticRegression(max_iter=3000, C=4.0)
    model.fit(X, train["y"])
    return vec, model


def fit_numeric(train: pd.DataFrame, cols: list[str]):
    imp = SimpleImputer(strategy="median").fit(train[cols].astype(float))
    sc = StandardScaler().fit(imp.transform(train[cols].astype(float)))
    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(sc.transform(imp.transform(train[cols].astype(float))), train["y"])
    return imp, sc, model


def fit_combined(train: pd.DataFrame, cols: list[str]):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000)
    Xt = vec.fit_transform(train["text"])
    imp = SimpleImputer(strategy="median").fit(train[cols].astype(float))
    sc = StandardScaler().fit(imp.transform(train[cols].astype(float)))
    Xn = csr_matrix(sc.transform(imp.transform(train[cols].astype(float))))
    model = LogisticRegression(max_iter=3000, C=4.0)
    model.fit(hstack([Xt, Xn]).tocsr(), train["y"])
    return vec, imp, sc, model


def scores(probe: str, test: pd.DataFrame, prob: np.ndarray) -> dict:
    y = test["y"].to_numpy()
    return {
        "probe": probe,
        "n": int(len(test)),
        "auc": round(float(roc_auc_score(y, prob)), 4) if len(set(y)) > 1 else None,
        "accuracy": round(float(accuracy_score(y, (prob >= 0.5).astype(int))), 4),
        "f1": round(float(f1_score(y, (prob >= 0.5).astype(int))), 4),
    }


def breakdown(probe: str, test: pd.DataFrame, prob: np.ndarray, key: str) -> list[dict]:
    rows = []
    frame = test.copy()
    frame["_p"] = prob
    for value, sub in frame.groupby(key):
        y = sub["y"].to_numpy()
        if len(set(y)) < 2:
            continue
        rows.append({
            "probe": probe,
            "slice": f"{key}={value}",
            "n": int(len(sub)),
            "auc": round(float(roc_auc_score(y, sub["_p"].to_numpy())), 4),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    train, val, test = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
    print(f"rows {len(d)} train {len(train)} val {len(val)} test {len(test)}")
    print("domain x label:\n" + pd.crosstab(d.domain, d.label).to_string())

    results: list[dict] = []

    vec, model = fit_text(train)
    p = model.predict_proba(vec.transform(test["text"]))[:, 1]
    results.append(scores("text_only", test, p))
    slices = breakdown("text_only", test, p, "domain")
    text_test_prob = p

    for name, cols in [
        ("image_visual_only", VISUAL),
        ("image_align_only", ALIGN),
        ("image_scene_only", SCENE),
        ("image_all", VISUAL + ALIGN + SCENE),
    ]:
        imp, sc, m = fit_numeric(train, cols)
        prob = m.predict_proba(sc.transform(imp.transform(test[cols].astype(float))))[:, 1]
        results.append(scores(name, test, prob))
        slices += breakdown(name, test, prob, "domain")

    cols = VISUAL + ALIGN + SCENE
    vec2, imp2, sc2, m2 = fit_combined(train, cols)
    Xt = vec2.transform(test["text"])
    Xn = csr_matrix(sc2.transform(imp2.transform(test[cols].astype(float))))
    prob = m2.predict_proba(hstack([Xt, Xn]).tocsr())[:, 1]
    results.append(scores("text_plus_image", test, prob))
    slices += breakdown("text_plus_image", test, prob, "domain")
    slices += breakdown("text_plus_image", test, prob, "pairing")

    table = pd.DataFrame(results)
    print("\n" + table.to_string(index=False))
    table.to_csv(OUT / "probe_scores.csv", index=False)
    slices_df = pd.DataFrame(slices)
    slices_df.to_csv(OUT / "probe_slices.csv", index=False)
    print("\nby slice:\n" + slices_df.to_string(index=False))

    report = {
        "rows": int(len(d)),
        "split_rows": {k: int(v) for k, v in d["split"].value_counts().items()},
        "domain_x_label": {
            f"{dom}/{lab}": int(n) for (dom, lab), n in d.groupby(["domain", "label"]).size().items()
        },
        "probes": results,
        "slices": slices,
        "delta_text_plus_image": round(
            float(table.loc[table.probe == "text_plus_image", "auc"].iloc[0]
                  - table.loc[table.probe == "text_only", "auc"].iloc[0]), 4),
        "note": (
            "Domain and label are correlated by construction (food supplies most real photos, "
            "ecommerce most fakes), so slice-level AUC is the honest number, not the pooled one."
        ),
    }
    (OUT / "baselines.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

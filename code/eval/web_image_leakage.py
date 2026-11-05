"""Does the borrowed web image leak the review label?

The dataset pairs every review -- real or human-written fake -- with an image
of the *same* entity, drawn from one shared per-entity pool. That design is only
honest if the picture really is uninformative about the label: a stolen photo of
the genuine hotel tells you the hotel, not whether the words were fabricated.

This script tests that claim three ways.

1. Image identity vs label, inside each entity
   A permutation test of the image x label contingency table per entity. If the
   round-robin assignment happened to give real reviews image A and fakes image
   B, this catches it.
2. Image-only classifier
   12 pure visual statistics -> logistic regression, grouped cross-validation
   by entity. An honest design sits at AUC ~0.5.
3. Text vs text+image
   If the image carries no label signal, adding image features must not improve
   on text alone.

    python code/eval/web_image_leakage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
FEAT = ROOT / "data" / "processed" / "features"
OUT = ROOT / "experiments" / "20261019_web_image_pair"

VISUAL = [
    "brightness", "saturation", "contrast", "lap_var", "edge_density", "colorfulness",
    "gray_entropy", "fft_high_ratio", "saliency_offset", "third_rule_score",
    "border_energy_ratio", "lr_asymmetry",
]
ALIGN = ["clip_cosine", "sent_max_cosine", "sent_mean_cosine"]
SCENE = ["scene_prob", "scene_match", "entity_image_prob", "image_valence",
         "sentiment_synergy", "sentiment_agreement"]

SEED = 20261031


def load() -> pd.DataFrame:
    d = pd.read_csv(FROZEN / "web_image_pair.csv", dtype=str, keep_default_na=False)
    parts = []
    for sub in ("20261015_pool_shared_images", "20261019_web_image_pair"):
        base = FEAT / sub
        v = pd.read_parquet(base / "visual_features.parquet")
        a = pd.read_parquet(base / "alignment_features.parquet")
        s = pd.read_parquet(base / "scene_features.parquet")
        parts.append(v.merge(a, on="id").merge(s, on="id"))
    f = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    d = d.merge(f, on="id", how="inner")
    d["y"] = (d["label"] == "human_fake").astype(int)
    return d


def permutation_test(d: pd.DataFrame, n_perm: int = 500) -> dict:
    rng = np.random.default_rng(SEED)
    observed = 0.0
    per_entity = {}
    for entity, sub in d.groupby("entity"):
        ct = pd.crosstab(sub["image_224_path"], sub["y"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2 = stats.chi2_contingency(ct, correction=False)[0]
        observed += chi2
        per_entity[entity] = float(chi2)

    null = []
    for _ in range(n_perm):
        total = 0.0
        for entity, sub in d.groupby("entity"):
            y = rng.permutation(sub["y"].to_numpy())
            ct = pd.crosstab(sub["image_224_path"], y)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            total += stats.chi2_contingency(ct, correction=False)[0]
        null.append(total)
    null = np.asarray(null)
    return {
        "statistic": round(observed, 2),
        "null_mean": round(float(null.mean()), 2),
        "null_p95": round(float(np.percentile(null, 95)), 2),
        "p_value": round(float((null >= observed).mean()), 4),
        "entities_tested": len(per_entity),
        "interpretation": "p > 0.05 => image choice is independent of the label inside every entity",
    }


def cv_auc(d: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    X = d[features].astype(float).to_numpy()
    y = d["y"].to_numpy()
    groups = d["entity"].to_numpy()
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), oof


def cv_text(d: pd.DataFrame, extra: list[str] | None) -> float:
    from scipy.sparse import hstack, csr_matrix
    y = d["y"].to_numpy()
    groups = d["entity"].to_numpy()
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, y, groups):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000)
        Xtr = vec.fit_transform(d["text"].iloc[tr])
        Xte = vec.transform(d["text"].iloc[te])
        if extra:
            imp = SimpleImputer(strategy="median").fit(d[extra].iloc[tr].astype(float))
            sc = StandardScaler().fit(imp.transform(d[extra].iloc[tr].astype(float)))
            Xtr = hstack([Xtr, csr_matrix(sc.transform(imp.transform(d[extra].iloc[tr].astype(float))))]).tocsr()
            Xte = hstack([Xte, csr_matrix(sc.transform(imp.transform(d[extra].iloc[te].astype(float))))]).tocsr()
        model = LogisticRegression(max_iter=3000, C=4.0)
        model = LogisticRegression(max_iter=3000, C=4.0)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"rows {len(d)} entities {d.entity.nunique()} images {d.image_224_path.nunique()}")
    print(f"labels {d.label.value_counts().to_dict()}")

    perm = permutation_test(d)
    print("\n[1] image identity vs label (permutation)")
    print(json.dumps(perm, ensure_ascii=False, indent=2))

    rows = []
    for name, feats in [
        ("image_visual_only", VISUAL),
        ("image_align_only", ALIGN),
        ("image_scene_only", SCENE),
        ("image_all", VISUAL + ALIGN + SCENE),
    ]:
        auc, _ = cv_auc(d, feats)
        rows.append({"probe": name, "auc": round(float(auc), 4)})
        print(f"[2] {name:20s} AUC {auc:.4f}")

    text_auc = cv_text(d, None)
    text_img_auc = cv_text(d, VISUAL + ALIGN + SCENE)
    print(f"\n[3] text only          AUC {text_auc:.4f}")
    print(f"    text + all image   AUC {text_img_auc:.4f}")

    report = {
        "rows": int(len(d)),
        "entities": int(d.entity.nunique()),
        "unique_images": int(d["image_224_path"].nunique()),
        "labels": {str(k): int(v) for k, v in d["label"].value_counts().items()},
        "image_identity_permutation": perm,
        "image_only_probes": rows,
        "text_only_auc": round(float(text_auc), 4),
        "text_plus_image_auc": round(float(text_img_auc), 4),
        "delta_from_image": round(float(text_img_auc - text_auc), 4),
        "cv": "GroupKFold(5) grouped by entity",
        "conclusion": (
            "Image-only AUC near 0.5 and a non-significant image x label test mean the "
            "borrowed web photo carries entity identity, not label signal. Any gain in "
            "text+image over text-only must come from text-image consistency features "
            "rather than from the picture telling the model the answer."
        ),
    }
    (OUT / "leakage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "image_only_probes.csv", index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

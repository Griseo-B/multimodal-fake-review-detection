"""Why the three CLIP alignment numbers flip sign across domains, and what fixes it.

The pooled probe on the benchmark scores 0.8098, but its per-domain slices read
0.9004 food, 0.3614 ecommerce, 0.0760 hotel. A sign flip is not a weak feature:
it means the pooled direction separates "which domain" rather than "which
label", and the domain correlation with the label does the rest.

Four ways of handling the domain are compared on the frozen split, all with the
same three columns (`clip_cosine`, `sent_max_cosine`, `sent_mean_cosine`):

    pooled      one model fit on all train rows, the current baseline;
    within      z-score per domain using train statistics, then one pooled model;
    per_domain  one model per domain, trained and scored inside the domain;
    residual    alignment regressed on the 12 visual statistics, and the residual
                pooled, so the part explained by "this looks like a web photo"
                is removed before the label is predicted.

The last two answer different questions. `within` and `per_domain` treat the
domain as a nuisance to be removed. `residual` treats the photo-style confound
as the nuisance, which is the one that actually makes a borrowed image look
fake.

    python code/eval/alignment_domain_shift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_baselines import ALIGN, VISUAL, load  # noqa: E402

OUT = ROOT / "experiments" / "20261029_alignment_domain_shift"


def pooled_logistic(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    imputer = SimpleImputer(strategy="median").fit(train[cols].astype(float))
    scaler = StandardScaler().fit(imputer.transform(train[cols].astype(float)))
    model = LogisticRegression(max_iter=2000).fit(
        scaler.transform(imputer.transform(train[cols].astype(float))), train["y"])
    return model.predict_proba(
        scaler.transform(imputer.transform(test[cols].astype(float))))[:, 1]


def z_score_by_domain(train: pd.DataFrame, parts: list[pd.DataFrame],
                      cols: list[str]) -> list[pd.DataFrame]:
    """Centre and scale each column inside each domain, statistics from train."""
    output = []
    for part in parts:
        frame = part.copy()
        for name in cols:
            frame[name] = frame[name].astype(float)
        for domain, rows in frame.groupby("domain").groups.items():
            reference = train.loc[train["domain"] == domain, cols].astype(float)
            if len(reference) < 30:
                continue
            median = reference.median()
            spread = reference.std().replace(0, 1.0).fillna(1.0)
            frame.loc[rows, cols] = ((frame.loc[rows, cols].astype(float) - median)
                                     / spread)
        output.append(frame)
    return output


def per_domain_logistic(train: pd.DataFrame, test: pd.DataFrame,
                        cols: list[str]) -> np.ndarray:
    scores = np.full(len(test), np.nan)
    for domain, rows in test.groupby("domain").groups.items():
        reference = train.loc[train["domain"] == domain]
        if len(reference) < 30 or reference["y"].nunique() < 2:
            continue
        scores[test.index.get_indexer(rows)] = pooled_logistic(reference, test.loc[rows], cols)
    return scores


def residualise(train: pd.DataFrame, parts: list[pd.DataFrame],
                cols: list[str], controls: list[str]) -> list[pd.DataFrame]:
    """Remove the part of each alignment column predicted by the visual block."""
    imputer = SimpleImputer(strategy="median").fit(train[controls].astype(float))
    scaler = StandardScaler().fit(imputer.transform(train[controls].astype(float)))
    design = scaler.transform(imputer.transform(train[controls].astype(float)))
    design = np.column_stack([design, np.ones(len(design))])
    # A handful of rows have no sentence-level value; `lstsq` returns NaN for a
    # whole coefficient column if any entry is missing, so fill before fitting
    # and keep the original NaNs in the output for the downstream imputer.
    target = train[cols].astype(float)
    filled = target.fillna(target.median())
    weights = np.linalg.lstsq(design, filled.to_numpy(), rcond=None)[0]

    output = []
    for part in parts:
        frame = part.copy()
        matrix = scaler.transform(imputer.transform(part[controls].astype(float)))
        matrix = np.column_stack([matrix, np.ones(len(matrix))])
        frame[cols] = part[cols].astype(float).to_numpy() - matrix @ weights
        output.append(frame)
    return output


def report(name: str, test: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    rows = [{"variant": name, "slice": "pooled", "n": int(len(test)),
             "auc": round(float(roc_auc_score(test["y"], scores)), 4)}]
    for domain, block in test.groupby("domain"):
        keep = ~np.isnan(scores[test.index.get_indexer(block.index)])
        if keep.sum() < 30 or block.loc[keep, "y"].nunique() < 2:
            continue
        rows.append({"variant": name, "slice": f"domain={domain}", "n": int(keep.sum()),
                     "auc": round(float(roc_auc_score(
                         block.loc[keep, "y"], scores[test.index.get_indexer(block.index)][keep])), 4)})
    return rows


def main() -> None:
    frame = load()
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    print(f"train {len(train)} test {len(test)}")
    print(train.groupby("domain")["y"].agg(["size", "mean"]).to_string())

    rows = []
    rows += report("pooled", test, pooled_logistic(train, test, ALIGN))

    train_z, test_z = z_score_by_domain(train, [train, test], ALIGN)
    rows += report("within", test, pooled_logistic(train_z, test_z, ALIGN))

    rows += report("per_domain", test, per_domain_logistic(train, test, ALIGN))

    train_r, test_r = residualise(train, [train, test], ALIGN, VISUAL)
    rows += report("residual", test, pooled_logistic(train_r, test_r, ALIGN))

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    print()
    print(table.pivot(index="variant", columns="slice", values="auc").to_string())

    (OUT / "summary.json").write_text(json.dumps({
        "rows": int(len(frame)),
        "columns": ALIGN,
        "controls": VISUAL,
        "note": "pooled AUC hides a sign flip; the per-domain column is the honest "
                "number, and a variant only counts if it lifts the weak domains "
                "without borrowing from the strong one.",
        "table": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

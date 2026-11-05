"""Where human fakes and AI fakes separate differently from real reviews.

No single protocol in the frozen table holds both fake types, so the two are
not compared head to head. Instead both are compared against the same anchor,
the real reviews of public_dataset, one feature at a time. The difference of
the two AUROCs answers the proposal's question: which features separate the
two fake types from real reviews in different ways.

Confound: the human fakes come from other protocols and other sources, so a
feature that differs between the two contrasts may be tracking source style
rather than fakery. The table is descriptive; nothing here is a classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, TABLE  # noqa: E402

FEATURE_FILES = ("text_features.parquet", "perplexity.parquet", "smoothness.parquet")
BOOTSTRAP = 500
SEED = 42


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def bootstrap_interval(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
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


def load(folder: Path, protocol: str) -> pd.DataFrame:
    meta = pd.read_csv(
        TABLE,
        usecols=["id", "domain", "label", "language", "protocol"],
        dtype=str,
        keep_default_na=False,
    )
    meta = meta.loc[meta["protocol"] == protocol].reset_index(drop=True)
    for name in FEATURE_FILES:
        path = folder / name
        if path.is_file():
            meta = meta.merge(pd.read_parquet(path), on="id", how="left")
    return meta


def compare(frame: pd.DataFrame, protocol: str, group: str) -> list[dict]:
    rows = []
    real = frame.loc[frame["label"] == "real"]
    fake = frame.loc[frame["label"] != "real"]
    if real.empty or fake.empty:
        return rows
    numeric = [
        name
        for name in frame.columns
        if name not in {"id", "domain", "label", "language", "protocol"}
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    rng = np.random.default_rng(SEED)
    for feature in numeric:
        real_values = pd.to_numeric(real[feature], errors="coerce").to_numpy(dtype=float)
        fake_values = pd.to_numeric(fake[feature], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(real_values)
        chosen = np.isfinite(fake_values)
        if keep.sum() < 100 or chosen.sum() < 100:
            continue
        real_clean = real_values[keep]
        fake_clean = fake_values[chosen]
        labels = np.concatenate([np.zeros(len(real_clean)), np.ones(len(fake_clean))])
        scores = np.concatenate([real_clean, fake_clean])
        auc = fast_auc(labels, scores)
        low, high = bootstrap_interval(labels, scores, rng)
        try:
            p_value = float(mannwhitneyu(real_clean, fake_clean, alternative="two-sided").pvalue)
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "protocol": protocol,
                "group": group,
                "label_fake": str(fake["label"].iloc[0]),
                "feature": feature,
                "n_real": int(len(real_clean)),
                "n_fake": int(len(fake_clean)),
                "auc": round(auc, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "median_real": round(float(np.median(real_clean)), 4),
                "median_fake": round(float(np.median(fake_clean)), 4),
                "mannwhitney_p": p_value,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-features", default="data/processed/features/20261008_public_dataset")
    parser.add_argument("--fake-features", default="data/processed/features/20261008_all_protocols")
    parser.add_argument("--out", default="experiments/20261010_differential")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ai_frame = load(ROOT / args.ai_features, "public_dataset")
    ai_frame = ai_frame.loc[ai_frame["label"].isin(["real", "ai_fake"])]
    rows = compare(ai_frame, "public_dataset", "all")
    for domain in ("food", "ecommerce", "hotel"):
        rows.extend(compare(ai_frame.loc[ai_frame["domain"] == domain], "public_dataset", domain))
    ai_table = pd.DataFrame(rows)

    fake_rows = []
    for protocol in ("different_origin", "same_experiment"):
        frame = load(ROOT / args.fake_features, protocol)
        fake_rows.extend(compare(frame, protocol, "all"))
        for domain in ("food", "ecommerce", "hotel"):
            fake_rows.extend(compare(frame.loc[frame["domain"] == domain], protocol, domain))
    fake_table = pd.DataFrame(fake_rows)

    ai_table.to_csv(out / "auc_real_vs_ai.csv", index=False, encoding="utf-8-sig")
    fake_table.to_csv(out / "auc_real_vs_human.csv", index=False, encoding="utf-8-sig")

    # Two contrasts share the real anchor, so the same feature compared on the
    # two fake types is a difference of differences, one row per feature.
    paired = []
    for _, human in fake_table.iterrows():
        match = ai_table.loc[
            (ai_table["group"] == human["group"]) & (ai_table["feature"] == human["feature"])
        ]
        if match.empty:
            continue
        ai = match.iloc[0]
        paired.append(
            {
                "group": human["group"],
                "feature": human["feature"],
                "auc_ai": ai["auc"],
                "auc_ai_ci": f"{ai['ci_low']}-{ai['ci_high']}",
                "auc_human": human["auc"],
                "auc_human_ci": f"{human['ci_low']}-{human['ci_high']}",
                "human_protocol": human["protocol"],
                "separation_ai": round(abs(ai["auc"] - 0.5), 4),
                "separation_human": round(abs(human["auc"] - 0.5), 4),
                "gap": round(abs(ai["auc"] - 0.5) - abs(human["auc"] - 0.5), 4),
            }
        )
    table = pd.DataFrame(paired)
    table["abs_gap"] = table["gap"].abs()
    table = table.sort_values(["group", "abs_gap"], ascending=[True, False])
    table.to_csv(out / "feature_gap.csv", index=False, encoding="utf-8-sig")

    summary = {
        "question": "Which features separate human fakes and AI fakes from real reviews differently?",
        "anchor": "real reviews of public_dataset",
        "confound": "human fakes come from different_origin and same_experiment, so a gap "
                    "may reflect source style rather than fakery. Descriptive only.",
        "rows": int(len(table)),
        "by_group": {str(k): int(v) for k, v in table.groupby("group").size().items()},
        "top_gaps": (
            table.groupby("group", group_keys=False)
            .head(5)[["group", "feature", "auc_ai", "auc_human", "gap"]]
            .to_dict("records")
            if not table.empty
            else []
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} rows {len(table)}", flush=True)


if __name__ == "__main__":
    main()

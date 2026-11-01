"""Single-feature separation between real and AI-generated reviews.

Every number is a description of one feature on one domain, not a model
score. Bootstrap intervals are reported because a three-decimal AUC without
an interval hides whether a gap is signal. Features whose interval sits at or
beyond 0.95 are flagged for a leakage check before they are quoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import AIGEN_RAW, ROOT, TABLE  # noqa: E402

BOOTSTRAP = 1000
LEAK = 0.95
FEATURE_FILES = {
    "text": "text_features.parquet",
    "perplexity": "perplexity.parquet",
    "smoothness": "smoothness.parquet",
    "visual": "visual_features.parquet",
    "align": "alignment_features.parquet",
    "scene": "scene_features.parquet",
}
TEXT_COLUMNS = [
    "n_words", "n_chars", "avg_word_len", "n_sents", "words_per_sentence",
    "ttr", "hapax_ratio", "excl_ratio", "question_ratio", "upper_word_ratio",
    "digit_ratio", "flesch", "ari", "gunning_fog",
    "pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb",
    "function_word_ratio", "first_person_ratio", "modal_ratio",
    "clause_marker_ratio", "passive_ratio", "comma_per_sentence",
    "sentiment", "sentiment_zh", "zh_idiom_ratio",
    "polarity_gap",
    "extreme_ratio", "intensifier_ratio", "number_ratio",
    "currency_ratio", "time_ratio", "measure_ratio", "entity_ratio",
    "opening_dup", "ngram_dup_ratio", "perplexity", "smoothness",
]
VISUAL_COLUMNS = [
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio",
    "saliency_offset", "third_rule_score", "border_energy_ratio", "lr_asymmetry",
]
ALIGN_COLUMNS = [
    "clip_cosine", "sent_max_cosine", "sent_mean_cosine",
    "scene_prob", "scene_match", "entity_image_prob",
    "image_valence", "sentiment_synergy", "sentiment_agreement",
]


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney form of AUC, with the score direction kept as given."""
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def bootstrap(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
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


def redundancy(
    frame: pd.DataFrame,
    features: list[str],
    group: str,
    rho_pair: float = 0.85,
    rho_cluster: float = 0.90,
) -> dict:
    """Spearman correlation on split=train, plus clusters of near-duplicates.

    Two features that move together are one piece of evidence, not two. The
    pairs below this threshold are dropped from the JSON, which is why the
    file stays small.
    """
    part = frame.loc[frame["split"] == "train"]
    present = [name for name in features if name in part.columns]
    values = part[present].apply(pd.to_numeric, errors="coerce")
    values = values.loc[:, values.notna().sum() >= 200]
    if values.shape[1] < 2:
        return {"group": group, "rows": int(len(values)), "pairs": [], "clusters": []}
    correlation = values.corr(method="spearman")
    pairs = []
    names = list(correlation.columns)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            rho = correlation.iloc[left, right]
            if np.isfinite(rho) and abs(rho) >= rho_pair:
                pairs.append(
                    {
                        "left": names[left],
                        "right": names[right],
                        "rho": round(float(rho), 4),
                        "abs": round(abs(float(rho)), 4),
                    }
                )
    pairs.sort(key=lambda item: item["abs"], reverse=True)

    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for item in pairs:
        if item["abs"] >= rho_cluster:
            left, right = find(item["left"]), find(item["right"])
            if left != right:
                parent[right] = left
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=len, reverse=True)
    return {
        "group": group,
        "rows": int(len(values)),
        "features": len(names),
        "rho_pair_threshold": rho_pair,
        "rho_cluster_threshold": rho_cluster,
        "pairs": pairs,
        "clusters": clusters,
    }


def load_features(folder: Path, frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame
    for name, filename in FEATURE_FILES.items():
        path = folder / filename
        if not path.is_file():
            print(f"missing {filename}, skipped", flush=True)
            continue
        part = pd.read_parquet(path)
        merged = merged.merge(part, on="id", how="left")
    return merged


def evaluate(
    frame: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    group: str,
    positive: str = "ai_fake",
) -> list[dict]:
    rows = []
    pair = frame.loc[frame["label"].isin(["real", positive])]
    if pair.empty:
        return rows
    labels = (pair["label"] == positive).to_numpy().astype(int)
    for feature in features:
        if feature not in pair.columns:
            continue
        values = pd.to_numeric(pair[feature], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        if keep.sum() < 30 or labels[keep].min() == labels[keep].max():
            continue
        chosen_labels = labels[keep]
        chosen_values = values[keep]
        auc = fast_auc(chosen_labels, chosen_values)
        low, high = bootstrap(chosen_labels, chosen_values, rng)
        rows.append(
            {
                "group": group,
                "contrast": f"real_vs_{positive}",
                "feature": feature,
                "n_real": int((chosen_labels == 0).sum()),
                "n_positive": int((chosen_labels == 1).sum()),
                "auc": round(auc, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "cliffs_delta": round(2 * auc - 1, 4),
                "mean_real": round(float(chosen_values[chosen_labels == 0].mean()), 4),
                "mean_positive": round(float(chosen_values[chosen_labels == 1].mean()), 4),
                "leak_flag": bool(min(auc, 1 - auc) <= 1 - LEAK),
            }
        )
    return rows


def aigen_check(frame: pd.DataFrame) -> dict:
    """Compare our readability and perplexity with the AiGen source columns."""
    ours = frame.loc[
        frame["subset"] == "food_aigen",
        ["id", "group_id", "flesch", "ari", "gunning_fog", "perplexity"],
    ].rename(
        columns={"flesch": "ours_flesch", "ari": "ours_ari",
                 "gunning_fog": "ours_fog", "perplexity": "ours_ppl"}
    )
    if ours.empty:
        return {"status": "no aigen rows"}
    source = []
    for name in ("train-00000-of-00003", "train-00001-of-00003", "train-00002-of-00003",
                 "valid-00000-of-00001", "test-00000-of-00001"):
        path = AIGEN_RAW / f"{name}.parquet"
        if path.is_file():
            source.append(
                pd.read_parquet(
                    path,
                    columns=["ID", "automated_readability_index", "flesch_reading_ease",
                             "gunning_fog", "ppl"],
                ).rename(
                    columns={"automated_readability_index": "src_ari",
                             "flesch_reading_ease": "src_flesch",
                             "gunning_fog": "src_fog", "ppl": "src_ppl"}
                )
            )
    if not source:
        return {"status": "raw parquet not on disk"}
    raw = pd.concat(source, ignore_index=True)
    raw["group_id"] = raw["ID"].map(lambda value: f"aigen-{int(value)}")
    merged = ours.merge(raw, on="group_id", how="inner")
    pairs = {
        "ari": ("ours_ari", "src_ari"),
        "flesch": ("ours_flesch", "src_flesch"),
        "gunning_fog": ("ours_fog", "src_fog"),
        "perplexity": ("ours_ppl", "src_ppl"),
    }
    result = {"status": "ok", "matched_rows": int(len(merged))}
    for name, (left, right) in pairs.items():
        both = merged[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if both.empty:
            continue
        result[name] = {
            "spearman": round(float(spearmanr(both[left], both[right]).statistic), 4),
            "mean_ours": round(float(both[left].mean()), 3),
            "mean_source": round(float(both[right].mean()), 3),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/20261030_public_dataset")
    parser.add_argument("--out", default="experiments/20261030_features")
    parser.add_argument("--protocol", default="public_dataset")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    folder = ROOT / args.features
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(
        TABLE,
        usecols=["id", "subset", "domain", "label", "language", "protocol", "split", "group_id"],
        dtype=str,
        keep_default_na=False,
    )
    meta = meta.loc[meta["protocol"] == args.protocol].reset_index(drop=True)
    frame = load_features(folder, meta)
    rng = np.random.default_rng(42)

    english = frame.loc[frame["language"] == "en"]
    positives = ["ai_fake"] if (frame["label"] == "ai_fake").any() else ["human_fake"]
    positive = positives[0]
    rows = evaluate(english, TEXT_COLUMNS, rng, "all", positive)
    for domain in ("food", "ecommerce", "hotel"):
        rows.extend(evaluate(english.loc[english["domain"] == domain], TEXT_COLUMNS, rng, domain, positive))
    rows.extend(evaluate(english.loc[english["domain"] == "food"], VISUAL_COLUMNS + ALIGN_COLUMNS, rng, "food", positive))
    rows.extend(evaluate(english.loc[english["domain"] == "ecommerce"], VISUAL_COLUMNS + ALIGN_COLUMNS, rng, "ecommerce", positive))
    rows.extend(evaluate(english.loc[english["domain"] == "hotel"], VISUAL_COLUMNS + ALIGN_COLUMNS, rng, "hotel", positive))

    chinese = frame.loc[frame["language"] == "zh"]
    chinese_table = pd.DataFrame(evaluate(chinese, TEXT_COLUMNS, rng, "zh", positive))
    if not chinese_table.empty:
        chinese_table.to_csv(out / f"feature_auc_zh{args.suffix}.csv", index=False, encoding="utf-8-sig")

    table = pd.DataFrame(rows)
    chinese_only = table.empty and not chinese_table.empty
    if chinese_only:
        table = chinese_table.copy()
    table.to_csv(out / f"feature_auc{args.suffix}.csv", index=False, encoding="utf-8-sig")

    flagged = (
        table.loc[table["leak_flag"], ["group", "feature", "auc", "ci_low", "ci_high"]]
        if "leak_flag" in table.columns
        else pd.DataFrame()
    )
    summary = {
        "protocol": args.protocol,
        "contrast": f"real_vs_{positive}",
        "features_dir": args.features,
        "rows": int(len(frame)),
        "rows_english": int(len(english)),
        "rows_chinese": int(len(chinese)),
        "bootstrap": BOOTSTRAP,
        "leak_threshold": LEAK,
        "chinese_only": chinese_only,
        "leak_flags": flagged.to_dict("records"),
        "top_features": (
            []
            if table.empty
            else table.assign(distance=(table["auc"] - 0.5).abs())
            .sort_values("distance", ascending=False)
            .head(12)[["group", "feature", "auc", "ci_low", "ci_high"]]
            .to_dict("records")
        ),
        "by_group": {
            str(key): int(value) for key, value in table.groupby("group").size().items()
        },
        "note": f"AUC compares real with {positive} only; rows of other labels are excluded.",
    }
    (out / f"summary{args.suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"aigen_check{args.suffix}.json").write_text(
        json.dumps(aigen_check(frame), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    text_and_image = TEXT_COLUMNS + VISUAL_COLUMNS + ALIGN_COLUMNS
    redundancy_report = {
        "note": "Spearman on split=train only. Pairs at or above 0.85 are listed; "
                "clusters join pairs at or above 0.90 and are counted as one piece of evidence.",
        "groups": [
            redundancy(english, text_and_image, "all"),
            redundancy(english.loc[english["domain"] == "food"], text_and_image, "food"),
            redundancy(english.loc[english["domain"] == "ecommerce"], text_and_image, "ecommerce"),
            redundancy(english.loc[english["domain"] == "hotel"], text_and_image, "hotel"),
        ],
    }
    (out / f"feature_redundancy{args.suffix}.json").write_text(
        json.dumps(redundancy_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out} rows {len(table)}", flush=True)
    if summary["leak_flags"]:
        print(json.dumps(summary["leak_flags"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

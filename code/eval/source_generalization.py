"""Is the number generalisation, or is it the source?

Generator and domain are confounded in this dataset: food AI is all
gpt-4-turbo+DALL-E-2, ecommerce AI is all gpt-2, hotel AI is gpt-4/qwen/tinyllama.
A "cross-domain" score is therefore also a "cross-generator" score. This script
says so with numbers instead of a caveat.

    leave-one-domain      pool B, train on two domains, test on the third
    leave-one-generator   pool A, train without one generator's AI rows,
                          test on that generator's AI rows

Reported per setting: macro-F1, the fake-class F1, and the drop from in-pool.
Outputs: metrics.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_three_class import LABELS, balance_train, feature_columns, make_model, load_pool, scores_for  # noqa: E402
from features.common import ROOT  # noqa: E402

SEED = 42


def evaluate_setting(name: str, split_kind: str, train: pd.DataFrame, test: pd.DataFrame,
                     kind: str = "lightgbm", weighting: str = "balanced") -> dict | None:
    classes = [label for label in LABELS if label in set(train["label"]) and label in set(test["label"])]
    if len(classes) < 2 or len(test) < 50:
        return None
    features = feature_columns(pd.concat([train, test]))
    train = balance_train(train, weighting)
    model = make_model(kind, weighting, len(classes))
    model.fit(train[features].to_numpy(dtype=float), train["label"].to_numpy())
    probabilities = scores_for(model, test, features, classes)
    prediction = np.array(classes)[probabilities.argmax(axis=1)]
    truth = test["label"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0
    )
    row = {
        "setting": name,
        "held_out": split_kind,
        "model": kind,
        "weighting": weighting,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "macro_f1": round(float(f1_score(truth, prediction, average="macro")), 4),
        "accuracy": round(float((prediction == truth).mean()), 4),
    }
    for index, label in enumerate(classes):
        row[f"f1_{label}"] = round(float(f1[index]), 4)
        row[f"support_{label}"] = int(support[index])
    row["classes_in_test"] = "|".join(classes)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/20261031_source_generalization")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []

    # --- leave one domain out, pool B ----------------------------------------
    pool_b = load_pool("b")
    in_pool_b = evaluate_setting(
        "in_pool", "none",
        pool_b.loc[pool_b["split_purged"] == "train"],
        pool_b.loc[pool_b["split_purged"] == "test"],
    )
    if in_pool_b:
        rows.append(in_pool_b)

    for domain in ("food", "ecommerce", "hotel"):
        held = pool_b.loc[pool_b["domain"] == domain]
        train = pool_b.loc[(pool_b["domain"] != domain) & (pool_b["split_purged"] == "train")]
        test = held.loc[held["split_purged"].isin(["val", "test"])]
        row = evaluate_setting(f"leave_one_domain/pool_b/{domain}", domain, train, test)
        if row:
            row["train_domains"] = "|".join(sorted(set(train["domain"])))
            rows.append(row)
            generators = sorted(set(held.loc[held["label"] == "ai_fake", "generator"]))
            details.append({
                "held_out_domain": domain,
                "ai_generators_in_test": generators,
                "ai_generators_in_train": sorted(set(train.loc[train["label"] == "ai_fake", "generator"])),
                "note": "the generator changes with the domain, so this is both contrasts at once",
            })
            print(json.dumps(row, ensure_ascii=False), flush=True)

    # --- leave one generator out, pool A -------------------------------------
    pool_a = load_pool("a")
    in_pool_a = evaluate_setting(
        "in_pool", "none",
        pool_a.loc[pool_a["split_purged"] == "train"],
        pool_a.loc[pool_a["split_purged"] == "test"],
    )
    if in_pool_a:
        rows.append(in_pool_a)

    generators = sorted(pool_a.loc[pool_a["label"] == "ai_fake", "generator"].unique())
    for generator in generators:
        ai_rows = pool_a.loc[pool_a["label"] == "ai_fake"]
        held = ai_rows.loc[ai_rows["generator"] == generator]
        others = pool_a.loc[(pool_a["label"] != "ai_fake") | (pool_a["generator"] != generator)]
        train = others.loc[others["split_purged"] == "train"]
        # Real and human rows come from the held-out side too, so the test set is
        # not just AI rows: a detector that calls everything fake must not score well.
        test = pd.concat([
            held,
            others.loc[others["split_purged"].isin(["val", "test"])].sample(
                n=min(len(held), int((others["split_purged"] != "train").sum())), random_state=SEED
            ),
        ])
        row = evaluate_setting(f"leave_one_generator/pool_a/{generator}", generator, train, test)
        if row:
            row["train_generators"] = "|".join(sorted(set(train.loc[train["label"] == "ai_fake", "generator"])))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "lightgbm, class_weight balanced",
        "split": "split_purged",
        "leave_one_domain": details,
        "rows": rows,
        "note": (
            "The leave-one-domain rows also change the generator, because generator and domain "
            "are confounded in this table. Read them as an upper-bound warning, not as a clean "
            "domain transfer result."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

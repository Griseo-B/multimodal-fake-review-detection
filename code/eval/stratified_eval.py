"""Does Chinese carry the score? Same runs, same split, read by language.

Reads `predictions.npz` written by `run_fusion.py`, so nothing is retrained
here; every number below belongs to a run that already appears in the fusion
table.

The pool A test split is 842 rows of which 182 are Chinese, and the Chinese
rows contain no human fakes at all. Chinese human fakes do exist in the master
table (7,692 of them), but every one carries `protocol = rule_label` and the
pools admit only `same_experiment` and `different_origin`. So "this is Chinese"
already rules out one of the three classes, and four of the five readability
columns are English formulas that stay blank on Chinese rows, which turns the
blank into the same flag.

Three readings per run:

    all          macro-F1 over the classes present, as reported elsewhere;
    language=xx  the same metric inside one language, so a model that only wins
                 on English is visible;
    zh_two_class Chinese rows collapsed to real-vs-ai, which is the only
                 question that slice can actually answer.

    python code/eval/stratified_eval.py --run experiments/20261102_seeds_05b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "20261102_stratified"
LABELS = ["real", "human_fake", "ai_fake"]


def per_class(target: np.ndarray, predicted: np.ndarray, classes: list[int]) -> dict:
    values = f1_score(target, predicted, average=None, labels=classes, zero_division=0)
    return {f"f1_{LABELS[c]}": round(float(v), 4) for c, v in zip(classes, values)}


def read_one(payload, key: str) -> dict:
    return {
        "id": payload[f"{key}|id"],
        "language": payload[f"{key}|language"],
        "label": payload[f"{key}|label"],
        "prediction": payload[f"{key}|probability"].argmax(1),
    }


def evaluate(run: str, seed: str, rows: dict) -> list[dict]:
    index = {name: position for position, name in enumerate(LABELS)}
    target = np.array([index[x] for x in rows["label"]])
    predicted = rows["prediction"]
    out = []

    classes = [c for c in range(3) if (target == c).any()]
    out.append({
        "run": run, "seed": seed, "slice": "all", "rows": int(len(target)),
        "classes": len(classes),
        "macro_f1": round(float(f1_score(target, predicted, average="macro",
                                         labels=classes, zero_division=0)), 4),
        "majority_macro_f1": round(float(np.bincount(target).max() / len(target)), 4),
        **per_class(target, predicted, classes),
    })

    for language in ("en", "zh"):
        pick = np.where(rows["language"] == language)[0]
        if len(pick) < 30:
            continue
        inside = target[pick]
        classes = [c for c in range(3) if (inside == c).any()]
        out.append({
            "run": run, "seed": seed, "slice": f"language={language}",
            "rows": int(len(pick)), "classes": len(classes),
            "macro_f1": round(float(f1_score(inside, predicted[pick], average="macro",
                                             labels=classes, zero_division=0)), 4),
            "majority_macro_f1": round(float(np.bincount(inside).max() / len(pick)), 4),
            **per_class(inside, predicted[pick], classes),
        })

    pick = np.where(rows["language"] == "zh")[0]
    if len(pick) >= 30:
        binary = (target[pick] == 2).astype(int)
        guess = (predicted[pick] == 2).astype(int)
        out.append({
            "run": run, "seed": seed, "slice": "zh_two_class",
            "rows": int(len(pick)), "classes": 2,
            "macro_f1": round(float(f1_score(binary, guess, average="macro")), 4),
            "majority_macro_f1": round(float(max(binary.mean(), 1 - binary.mean())), 4),
            "f1_real": round(float(f1_score(binary, guess, pos_label=0, zero_division=0)), 4),
            "f1_ai_fake": round(float(f1_score(binary, guess, pos_label=1, zero_division=0)), 4),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/20261102_seeds_05b")
    parser.add_argument("--seeds-vote", action="store_true",
                        help="add a majority vote across seeds per arm")
    args = parser.parse_args()

    source = ROOT / args.run
    OUT.mkdir(parents=True, exist_ok=True)
    payload = np.load(source / "predictions.npz", allow_pickle=False)
    keys = sorted({key.rsplit("|", 1)[0] for key in payload.files})
    print(f"{len(keys)} (pool, seed, run) arms in {source.name}")

    rows = []
    for key in keys:
        pool, seed, run = key.split("|")
        if "with_image" in pool:
            continue
        for row in evaluate(run, seed, read_one(payload, key)):
            rows.append({"pool": pool, **row})

    if args.seeds_vote:
        # Majority vote across seeds: what a small ensemble would give, and a
        # cleaner way to tell "unstable per seed" from "systematically wrong".
        arms = {}
        for key in keys:
            pool, seed, run = key.split("|")
            if "with_image" in pool:
                continue
            arms.setdefault((pool, run), []).append(read_one(payload, key))
        for (pool, run), passes in arms.items():
            if len(passes) < 2 or not all(np.array_equal(passes[0]["id"], p["id"])
                                          for p in passes):
                continue
            stacked = np.stack([p["prediction"] for p in passes])
            vote = np.array([np.bincount(stacked[:, i], minlength=3).argmax()
                             for i in range(stacked.shape[1])])
            merged = dict(passes[0], prediction=vote)
            for row in evaluate(f"{run} [vote]", "vote", merged):
                rows.append({"pool": pool, **row})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rows.csv", index=False, encoding="utf-8-sig")

    aggregate = (table.groupby(["pool", "run", "slice"])
                 .agg(rows=("rows", "max"), classes=("classes", "max"),
                      seeds=("seed", "nunique"),
                      macro_f1_mean=("macro_f1", "mean"),
                      macro_f1_std=("macro_f1", "std"),
                      majority=("majority_macro_f1", "max"))
                 .reset_index().round(4))
    aggregate.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(aggregate.to_string(index=False))

    (OUT / "summary.json").write_text(json.dumps({
        "source": str(source.relative_to(ROOT)),
        "note": "pool A test holds 182 Chinese rows and none of them is a human "
                "fake, so zh_two_class is the only honest reading of that slice",
        "table": aggregate.to_dict("records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

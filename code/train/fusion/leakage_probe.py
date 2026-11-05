"""What does the frozen LLM vector actually encode?

The fusion numbers rise sharply once the LLM block is added. Before that gain
is called detection, the same vector is asked to predict things that are not
the label: which dataset the review came from, which scene, which language,
which generator. If a linear probe recovers those at 0.9+, the vector is a
provenance fingerprint, and the gain on a same-source test split is expected
rather than informative. The handcrafted block is probed the same way as a
control, so the comparison is between two encoders, not between a model and
nothing.

    python code/train/fusion/leakage_probe.py --llm <run> --pool pool_b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build  # noqa: E402
from registry import TEXT_BLOCK, columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SEED = 42
FOLDS = 3
TARGETS = ["source", "domain", "language", "generator", "protocol"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def probe(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict | None:
    if len(np.unique(target)) < 2:
        return None
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(target), dtype=np.int64)
    for train, test in splitter.split(matrix, target, groups):
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        )
        pipeline.fit(matrix[train], target[train])
        scores[test] = pipeline.predict(matrix[test])
    return {
        "classes": int(len(np.unique(target))),
        "macro_f1": round(float(f1_score(target, scores, average="macro")), 4),
        "accuracy": round(float((scores == target).mean()), 4),
        "majority": round(float(np.bincount(target).max() / len(target)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="pool_b")
    parser.add_argument("--llm", required=True)
    parser.add_argument("--targets", nargs="+", default=["source", "domain", "language"])
    parser.add_argument("--out", default="experiments/20261021_fusion")
    args = parser.parse_args()

    data = build(args.pool, llm_path=ROOT / args.llm, with_images=False)
    frame = data["frame"]
    llm = data["blocks"]["llm"]["matrix"]
    hand_cols = [name for name in columns(TEXT_BLOCK) if name in frame.columns]
    hand = frame[hand_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    groups = frame["group_id"].to_numpy()

    rows = []
    for name in args.targets:
        if name not in frame.columns:
            continue
        values = frame[name].astype(str).to_numpy()
        keep = values != ""
        if keep.sum() < len(values):
            log(f"{name}: {int((~keep).sum())} rows have no value")
        if not keep.any():
            continue
        codes, uniques = pd.factorize(values[keep])
        for block, matrix in (("llm", llm[keep]), ("text_hand", hand[keep])):
            result = probe(matrix, codes, groups[keep])
            if result is None:
                continue
            rows.append({"pool": args.pool, "target": name, "block": block,
                         "values": int(len(uniques)), **result})
            log(f"{name} / {block}: macro-F1 {result['macro_f1']} "
                f"over {result['classes']} classes")

    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"leakage_{args.pool}.csv", index=False, encoding="utf-8-sig")
    (out / f"leakage_{args.pool}.json").write_text(
        json.dumps({"llm": args.llm, "pool": args.pool,
                    "note": "grouped 5-fold, group_id; a non-label target at high "
                            "macro-F1 means the block carries provenance",
                    "table": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

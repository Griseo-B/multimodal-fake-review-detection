"""Leave-one-domain-out and leave-one-generator-out for the fusion model.

The documented result is same-scene, same-source: pool B rests at 0.92 and
falls to 0.19-0.32 when a scene is held out, because generator and scene are
almost one-to-one in this table. That number has to be reproduced for the new
model, otherwise a fusion gain would be read as generalisation.

    python code/train/fusion/generalization.py --llm data/processed/features/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build  # noqa: E402
from run_fusion import (Config, LLM_BLOCK, SEED, TEXT_BLOCK, run_one,
                        subset)

ROOT = Path(__file__).resolve().parents[3]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def relabel(frame: pd.DataFrame, held: np.ndarray) -> pd.DataFrame:
    """Held-out rows become the test set; everything else keeps its split."""
    out = frame.copy()
    out.loc[held, "split"] = "test"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="pool_b")
    parser.add_argument("--field", default="domain")
    parser.add_argument("--llm", required=True)
    parser.add_argument("--out", default="experiments/20261022_fusion_generalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    config = Config()
    data = build(args.pool, llm_path=ROOT / args.llm, with_images=True)
    frame = data["frame"]
    log(f"{args.pool}: {len(frame)} rows, held-out field {args.field}")

    rows = []
    for value in sorted(frame[args.field].unique()):
        if str(value) == "":
            continue
        held = (frame[args.field] == value).to_numpy()
        if held.sum() < 200:
            log(f"skip {value}: only {int(held.sum())} rows")
            continue
        view = dict(data)
        view["frame"] = relabel(frame, held)
        row, _ = run_one(f"leave_out_{value}", view, [TEXT_BLOCK, LLM_BLOCK],
                         False, False, False, "gate", config, rng, device)
        rows.append({"pool": args.pool, "held_out": value, **row})
        log(f"{value}: macro-F1 {row['macro_f1']}")

    # Reference: the same model without holding anything out.
    row, _ = run_one("in_pool", data, [TEXT_BLOCK, LLM_BLOCK], False, False, False,
                     "gate", config, rng, device)
    rows.append({"pool": args.pool, "held_out": "none", **row})

    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"metrics_{args.pool}_{args.field}.csv", index=False,
                 encoding="utf-8-sig")
    summary = {
        "pool": args.pool,
        "field": args.field,
        "llm": args.llm,
        "config": asdict(config),
        "note": "held-out rows are the only test rows; the in-pool row is the "
                "reference and is not a generalisation number",
        "table": rows,
    }
    (out / f"summary_{args.pool}_{args.field}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table[["held_out", "macro_f1", "f1_real", "f1_human", "f1_ai"]].to_string(index=False))


if __name__ == "__main__":
    main()

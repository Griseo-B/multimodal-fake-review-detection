"""Freeze one LLM semantic vector per review.

The proposal asks that text and image land in one semantic space and that a
large model do the fusing. This step produces the text side of that space: a
frozen decoder-only LLM reads the review and the mean of its last hidden state
becomes a 896-dim (0.5B) or 1536-dim (1.5B) vector. Weights are never updated,
so the vector is a fixed transform and every downstream number is a property of
the fusion head, not of a fine-tuned generator.

Deliberate choices, all recorded in the run file:
    * raw review text only, no chat template: the vector should describe the
      review, not the instruction wrapper;
    * mean pooling over the attention mask, because the last token of a chat
      turn carries the assistant style rather than the review;
    * max 256 tokens, truncated from the right, which keeps the opening of the
      longer reviews where most of the template text sits.

    python code/train/fusion/encode_llm.py --pools pool_a pool_b benchmark
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "data" / "frozen"
OUT_ROOT = ROOT / "data" / "processed" / "features"

POOL_TABLES = {
    "pool_a": FROZEN / "three_class_pool_a.csv",
    "pool_b": FROZEN / "three_class_pool_b.csv",
    "benchmark": FROZEN / "web_image_benchmark.csv",
}
ALL_TABLE = FROZEN / "three_class.csv"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def union_text(pools: list[str]) -> pd.DataFrame:
    ids: set[str] = set()
    for pool in pools:
        frame = pd.read_csv(POOL_TABLES[pool], usecols=["id"], dtype=str)
        ids |= set(frame["id"])
    table = pd.read_csv(
        ALL_TABLE, usecols=["id", "text", "language"], dtype=str, keep_default_na=False
    )
    selected = table.loc[table["id"].isin(ids)].drop_duplicates("id").reset_index(drop=True)
    if len(selected) != len(ids):
        missing = len(ids) - len(selected)
        log(f"warning: {missing} ids are not in the union table and stay unencoded")
    return selected


@torch.no_grad()
def encode(
    texts: list[str],
    model_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).eval().to(device)
    width = int(model.config.hidden_size)
    output = np.zeros((len(texts), width), dtype=np.float16 if device.type == "cuda" else np.float32)

    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        output[start : start + len(chunk)] = pooled.float().cpu().numpy()
        if start % (batch_size * 20) == 0:
            done = start + len(chunk)
            rate = done / max(time.perf_counter() - started, 1e-6)
            eta = (len(texts) - done) / max(rate, 1e-6)
            log(f"{done}/{len(texts)} rows, {rate:.1f} rows/s, eta {eta / 60:.1f} min")

    meta = {
        "model": model_name,
        "hidden_size": width,
        "max_length": max_length,
        "pooling": "mean over attention mask, last hidden state",
        "rows": len(texts),
        "seconds": round(time.perf_counter() - started, 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    return output, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", default=["pool_a", "pool_b", "benchmark"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    frame = union_text(args.pools)
    if args.limit:
        frame = frame.head(args.limit)
    log(f"{len(frame)} unique reviews, model {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = frame["text"].astype(str).tolist()
    matrix, meta = encode(texts, args.model, device, args.max_length, args.batch_size)

    name = args.out or time.strftime("%Y%m%d") + "_llm_" + args.model.split("/")[-1]
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "llm_embedding.npz",
        id=frame["id"].to_numpy(dtype=str),
        embedding=matrix,
        language=frame["language"].to_numpy(dtype=str),
    )
    meta.update(
        {
            "pools": args.pools,
            "by_language": frame["language"].value_counts().to_dict(),
            "output": str(out.relative_to(ROOT)),
            "note": "frozen encoder; no fine-tuning; raw text, no chat template",
        }
    )
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

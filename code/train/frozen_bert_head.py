"""Frozen BERT encoder plus a classification head on 2,000 audited AiGen rows.

Internal control against TF-IDF on the same rows. The encoder stays frozen.
The table is marked in progress because this is not the full frozen split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
MODEL_NAME = "bert-base-uncased"
N = 2000
BATCH = 32
MAX_LENGTH = 128
LABELS = ["real", "ai_fake"]


def sample_rows(frame: pd.DataFrame, n: int | None) -> pd.DataFrame:
    usable = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    if n is None:
        return usable.reset_index(drop=True)
    parts = []
    for _, group in usable.groupby("split"):
        take = max(1, round(n * len(group) / len(usable)))
        parts.append(group.sample(min(take, len(group)), random_state=SEED))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=SEED)
    return sampled.reset_index(drop=True)


def encode(texts: list[str], tokenizer, encoder, device: torch.device) -> np.ndarray:
    vectors = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(
                texts[start : start + BATCH],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = encoder(**batch).last_hidden_state[:, 0, :].cpu().numpy()
            vectors.append(hidden)
            print(f"encoded {min(start + BATCH, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(vectors)


def metrics_row(name: str, y_true, y_pred, n_subset: int, n_train: int, n_test: int, status: str) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    row = {
        "model": name,
        "split": "test",
        "status": status,
        "n_subset": int(n_subset),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
    }
    for label, prec, rec in zip(LABELS, precision, recall):
        row[f"precision_{label}"] = float(prec)
        row[f"recall_{label}"] = float(rec)
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    out = ROOT / "experiments" / "20261026_aigen_bert_head"
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    limit = None if args.all else N
    status = "完成" if args.all else "进行中"
    id_name = "all_ids.csv" if args.all else "subset_ids.csv"
    frame = sample_rows(pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv"), limit)
    frame[["source_id", "split", "label"]].to_csv(out / id_name, index=False)
    print("loading", MODEL_NAME, "on", torch.cuda.get_device_name(0), "n", len(frame), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    vectors = encode(frame["text"].astype(str).tolist(), tokenizer, encoder, device)
    elapsed = time.perf_counter() - started
    train = frame["split"].to_numpy() == "train"
    test = frame["split"].to_numpy() == "test"
    head = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    head.fit(vectors[train], frame.loc[train, "label"])
    bert_pred = head.predict(vectors[test])
    y_test = frame.loc[test, "label"]
    saved = joblib.load(ROOT / "experiments" / "20261026_aigen_tfidf" / "model_tfidf.joblib")
    tfidf_pred = saved["model"].predict(saved["vectorizer"].transform(frame.loc[test, "text"]))
    rows = [
        metrics_row("tfidf_full_train", y_test, tfidf_pred, len(frame), int(train.sum()), int(test.sum()), status),
        metrics_row("frozen_bert_head", y_test, bert_pred, len(frame), int(train.sum()), int(test.sum()), status),
    ]
    metrics_name = "metrics_full.csv" if args.all else "metrics.csv"
    pd.DataFrame(rows).to_csv(out / metrics_name, index=False)
    timing_name = "timing_full.json" if args.all else "timing.json"
    (out / timing_name).write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "encode_sec": elapsed,
                "n": int(len(frame)),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "model": MODEL_NAME,
                "scope": "all_usable" if args.all else "subset_2000",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

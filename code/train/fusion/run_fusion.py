"""Experiment matrix for the section 3 detector.

Question: on the two frozen pools, does a large-model semantic vector plus a
gated multimodal fusion beat the handcrafted-feature baseline, and does a
two-level branch (real/fake, then human/AI) beat a flat three-class head?

Fixed for every row of the matrix:
    * split: the frozen `split` / `split_purged` column, never re-drawn;
    * early stopping on the validation split, test touched once;
    * macro-F1 at the 0.5/argmax decision, beside the majority-class baseline;
    * bootstrap 95% interval over 500 test resamples, seed 42;
    * corpus-level columns (`opening_dup`, `ngram_dup_ratio`) excluded, because
      they cannot be computed for one incoming review.

Ablations change one thing at a time: block set, stage structure, risk columns.

    python code/train/fusion/run_fusion.py --llm data/processed/features/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build  # noqa: E402
from model import GatedFusion, TwoLevelHead, parameter_count  # noqa: E402
from registry import (CLIP_IMAGE_BLOCK, CLIP_TEXT_BLOCK, IMAGE_BLOCK,  # noqa: E402
                      LLM_BLOCK, RESNET_BLOCK, TEXT_BLOCK, columns)

ROOT = Path(__file__).resolve().parents[3]
SEED = 42
BOOTSTRAP = 500
LABEL_ORDER = ["real", "human_fake", "ai_fake"]
LABEL_INDEX = {name: index for index, name in enumerate(LABEL_ORDER)}


@dataclass
class Config:
    seed: int = SEED
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 120
    patience: int = 15
    batch_size: int = 512


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def text_matrix(frame: pd.DataFrame, include_risk: bool) -> np.ndarray:
    cols = columns(TEXT_BLOCK, include_corpus=False, include_risk=include_risk)
    cols = [name for name in cols if name in frame.columns]
    return frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32), cols


def block_arrays(data: dict, blocks: list[str], include_risk: bool) -> tuple[dict, dict, dict]:
    frame = data["frame"]
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    widths: dict[str, int] = {}
    if TEXT_BLOCK in blocks:
        matrix, cols = text_matrix(frame, include_risk)
        # Missing values are kept; the scaler fills them with train medians and
        # appends the missing fraction, which carries the Chinese/English split.
        values[TEXT_BLOCK] = matrix
        masks[TEXT_BLOCK] = np.ones(len(frame), dtype=bool)
        widths[TEXT_BLOCK] = matrix.shape[1] + 1
    for name in blocks:
        if name == TEXT_BLOCK:
            continue
        payload = data["blocks"].get(name)
        if payload is None:
            continue
        matrix = payload["matrix"].astype(np.float32)
        if name == LLM_BLOCK:
            norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)
            matrix = matrix / norm
        values[name] = matrix
        masks[name] = payload["present"].astype(bool)
        widths[name] = matrix.shape[1]
    return values, masks, widths


SCALED_BLOCKS = {TEXT_BLOCK, IMAGE_BLOCK}


def fit_scaler(values: dict[str, np.ndarray], masks: dict[str, np.ndarray],
               train_index: np.ndarray) -> dict:
    """Medians and standard deviations from train rows only."""
    scaler = {}
    for name, matrix in values.items():
        if name not in SCALED_BLOCKS:
            scaler[name] = None
            continue
        train = matrix[train_index]
        median = np.nanmedian(train, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(train), median, train)
        std = filled.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        scaler[name] = (median, std)
    return scaler


def apply_scaler(values: dict[str, np.ndarray], scaler: dict) -> dict:
    out = {}
    for name, matrix in values.items():
        limits = scaler.get(name)
        if limits is None:
            out[name] = matrix
            continue
        median, std = limits
        # The fraction of missing columns is itself informative here: Chinese
        # rows have no English readability or perplexity by construction, and
        # dropping that fact would make the two languages look alike.
        filled = np.where(np.isnan(matrix), median, matrix)
        scaled = ((filled - median) / std).astype(np.float32)
        if name == TEXT_BLOCK:
            missing_fraction = np.isnan(matrix).mean(axis=1, keepdims=True).astype(np.float32)
            scaled = np.concatenate([scaled, missing_fraction], axis=1)
        out[name] = scaled
    return out


def to_tensors(values: dict[str, np.ndarray], masks: dict[str, np.ndarray], device):
    import torch

    blocks = {name: torch.tensor(matrix, dtype=torch.float32, device=device)
              for name, matrix in values.items()}
    mask_t = {name: torch.tensor(mask, device=device) for name, mask in masks.items()}
    return blocks, mask_t


def macro_f1(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    return float(f1_score(labels, predictions, average="macro", labels=list(range(n_classes)),
                          zero_division=0))


def bootstrap_ci(labels: np.ndarray, predictions: np.ndarray, n_classes: int,
                 rng: np.random.Generator) -> tuple[float, float]:
    values = []
    size = len(labels)
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, size, size)
        chosen = labels[index]
        if len(np.unique(chosen)) < 2:
            continue
        values.append(macro_f1(chosen, predictions[index], n_classes))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def majority_baseline(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    return float(counts.max() / counts.sum())


def train_torch(
    values: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    labels: np.ndarray,
    widths: dict[str, int],
    train_index: np.ndarray,
    val_index: np.ndarray,
    two_level: bool,
    binary: bool,
    config: Config,
    device,
) -> tuple[object, dict, np.ndarray]:
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    blocks, mask_t = to_tensors(values, masks, device)
    target = torch.tensor(labels, dtype=torch.long, device=device)
    stage1_target = (target != 0).long()

    if binary:
        model = GatedFusion(widths, config.hidden, config.dropout, n_classes=2)
    elif two_level:
        model = TwoLevelHead(widths, config.hidden, config.dropout)
    else:
        model = GatedFusion(widths, config.hidden, config.dropout, n_classes=3)
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    train_t = torch.tensor(train_index, device=device)
    val_t = torch.tensor(val_index, device=device)

    def scores(index_tensor):
        model.eval()
        with torch.no_grad():
            subset = {name: matrix[index_tensor] for name, matrix in blocks.items()}
            subset_mask = {name: mask[index_tensor] for name, mask in mask_t.items()}
            if binary:
                logits, _ = model(subset, subset_mask)
                return torch.softmax(logits, dim=1)[:, 1]
            if two_level:
                logits1, logits2, _, _ = model(subset, subset_mask)
                return TwoLevelHead.combine(logits1, logits2)
            logits, _ = model(subset, subset_mask)
            return torch.softmax(logits, dim=1)

    best_state, best_score, best_epoch, history = None, -1.0, -1, []
    n_classes = 2 if binary else 3
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(train_index)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = torch.tensor(order[start : start + config.batch_size], device=device)
            subset = {name: matrix[batch] for name, matrix in blocks.items()}
            subset_mask = {name: mask[batch] for name, mask in mask_t.items()}
            optimiser.zero_grad()
            if binary:
                logits, _ = model(subset, subset_mask)
                loss = nn.functional.cross_entropy(logits, target[batch])
            elif two_level:
                logits1, logits2, _, _ = model(subset, subset_mask)
                loss = nn.functional.cross_entropy(logits1, stage1_target[batch])
                fake = batch[(target[batch] != 0)]
                if len(fake):
                    full = {name: matrix[fake] for name, matrix in blocks.items()}
                    full_mask = {name: mask[fake] for name, mask in mask_t.items()}
                    _, kind_logits, _, _ = model(full, full_mask)
                    kind_target = (target[fake] == 2).long()
                    loss = loss + nn.functional.cross_entropy(kind_logits, kind_target)
            else:
                logits, _ = model(subset, subset_mask)
                loss = nn.functional.cross_entropy(logits, target[batch])
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)

        probabilities = scores(val_t).cpu().numpy()
        predictions = probabilities.argmax(1) if probabilities.ndim > 1 else (probabilities >= 0.5).astype(int)
        validation = macro_f1(labels[val_index], predictions, n_classes)
        history.append({"epoch": epoch, "loss": round(total / max(len(order), 1), 4),
                        "val_macro_f1": round(validation, 4)})
        if validation > best_score + 1e-4:
            best_score = validation
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probabilities = scores(torch.arange(len(labels), device=device)).cpu().numpy()
    attention = attention_table(model, blocks, mask_t, two_level, binary, device)
    info = {
        "parameters": parameter_count(model),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "val_macro_f1": round(best_score, 4),
        "history_tail": history[-3:],
        "attention": attention,
    }
    return model, info, test_probabilities


def attention_table(model, blocks, masks, two_level: bool, binary: bool, device) -> dict:
    """Mean gate weight per block, per predicted class, on all rows.

    These weights are the numbers the explainability stage will quote, so they
    are written next to the metrics rather than recomputed later.
    """
    import torch

    with torch.no_grad():
        if binary:
            logits, attention = model(blocks, masks)
            predictions = logits.argmax(1)
            names = list(getattr(model, "names"))
            table = {}
            for cls in range(2):
                rows = predictions == cls
                if rows.any():
                    table[f"class_{cls}"] = {name: round(float(attention[rows, i].mean()), 4)
                                             for i, name in enumerate(names)}
            return table
        if two_level:
            logits1, logits2, attention1, _ = model(blocks, masks)
            combined = TwoLevelHead.combine(logits1, logits2).argmax(1)
            names = list(model.stage1.names)
            table = {}
            for cls in range(3):
                rows = combined == cls
                if rows.any():
                    table[f"class_{cls}"] = {name: round(float(attention1[rows, i].mean()), 4)
                                             for i, name in enumerate(names)}
            return table
        logits, attention = model(blocks, masks)
        predictions = logits.argmax(1)
        names = list(getattr(model, "names"))
        table = {}
        for cls in range(3):
            rows = predictions == cls
            if rows.any():
                table[f"class_{cls}"] = {name: round(float(attention[rows, i].mean()), 4)
                                         for i, name in enumerate(names)}
        return table


def evaluate_predictions(labels: np.ndarray, probabilities, n_classes: int,
                         rng: np.random.Generator) -> dict:
    if probabilities.ndim == 1:
        predictions = (probabilities >= 0.5).astype(int)
        correct = np.where(predictions == 1, 1, 0)
        labels_binary = np.where(labels == 0, 0, 1)
        metrics = {
            "macro_f1": round(macro_f1(labels_binary, correct, 2), 4),
            "f1_real": round(float(f1_score(labels_binary, correct, pos_label=0, zero_division=0)), 4),
            "f1_fake": round(float(f1_score(labels_binary, correct, pos_label=1, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(labels_binary, probabilities)), 4),
            "majority_macro_f1": round(float(max(labels_binary.mean(), 1 - labels_binary.mean())), 4),
        }
        low, high = bootstrap_ci(labels_binary, correct, 2, rng)
        metrics["macro_f1_ci_low"] = round(low, 4)
        metrics["macro_f1_ci_high"] = round(high, 4)
        return metrics

    predictions = probabilities.argmax(1)
    per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)
    metrics = {
        "macro_f1": round(macro_f1(labels, predictions, 3), 4),
        "f1_real": round(float(per_class[0]), 4),
        "f1_human": round(float(per_class[1]), 4),
        "f1_ai": round(float(per_class[2]), 4),
        "majority_macro_f1": round(majority_baseline(labels), 4),
    }
    try:
        metrics["auc"] = round(float(roc_auc_score(labels, probabilities, multi_class="ovr",
                                                   average="macro")), 4)
    except ValueError:
        metrics["auc"] = float("nan")
    low, high = bootstrap_ci(labels, predictions, 3, rng)
    metrics["macro_f1_ci_low"] = round(low, 4)
    metrics["macro_f1_ci_high"] = round(high, 4)
    return metrics


def lightgbm_scores(values: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                    labels: np.ndarray, train_index: np.ndarray, test_index: np.ndarray,
                    binary: bool) -> np.ndarray:
    import lightgbm as lgb

    matrix = np.concatenate([matrix for matrix in values.values()], axis=1)
    matrix = np.where(np.isnan(matrix), 0.0, matrix)
    if binary:
        target = (labels != 0).astype(int)
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   subsample=0.9, colsample_bytree=0.9,
                                   random_state=SEED, n_jobs=4, verbose=-1)
        model.fit(matrix[train_index], target[train_index])
        return model.predict_proba(matrix[test_index])[:, 1]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                               random_state=SEED, n_jobs=4, verbose=-1)
    model.fit(matrix[train_index], labels[train_index])
    return model.predict_proba(matrix[test_index])


def run_one(label: str, data: dict, blocks: list[str], include_risk: bool,
            two_level: bool, binary: bool, family: str, config: Config,
            rng: np.random.Generator, device) -> dict:
    frame = data["frame"]
    labels = np.array([LABEL_INDEX[name] for name in frame["label"]], dtype=np.int64)
    train_index = np.where(frame["split"].eq("train"))[0]
    val_index = np.where(frame["split"].eq("val"))[0]
    test_index = np.where(frame["split"].eq("test"))[0]
    values, masks, widths = block_arrays(data, blocks, include_risk)
    scaler = fit_scaler(values, masks, train_index)
    scaled = apply_scaler(values, scaler)

    started = time.perf_counter()
    if family == "lightgbm":
        probabilities = lightgbm_scores(scaled, masks, labels, train_index, test_index, binary)
        info = {"parameters": None, "epochs_run": 0, "best_epoch": 0,
                "val_macro_f1": None, "attention": {}}
    else:
        _, info, all_probabilities = train_torch(
            scaled, masks, labels, widths, train_index, val_index,
            two_level, binary, config, device,
        )
        probabilities = all_probabilities[test_index]

    metrics = evaluate_predictions(labels[test_index], probabilities, 2 if binary else 3, rng)
    row = {
        "run": label,
        "family": family,
        "blocks": "+".join(blocks) + ("" if include_risk else " (no-risk)"),
        "two_level": two_level,
        "task": "binary" if binary else "three_class",
        "rows_train": int(len(train_index)),
        "rows_test": int(len(test_index)),
        "features": int(sum(widths.values())) if family == "lightgbm" else int(sum(widths.values())),
        "parameters": info.get("parameters"),
        "best_epoch": info.get("best_epoch"),
        "val_macro_f1": info.get("val_macro_f1"),
        "seconds": round(time.perf_counter() - started, 1),
        **metrics,
    }
    return row, info.get("attention", {})


def subset(data: dict, mask: np.ndarray) -> dict:
    """Cut a pool down to one domain or one label pair, keeping ids aligned."""
    frame = data["frame"].loc[mask].reset_index(drop=True)
    out = {
        "frame": frame,
        "blocks": {
            name: {
                "matrix": payload["matrix"][mask],
                "present": payload["present"][mask],
                "dim": payload["dim"],
            }
            for name, payload in data["blocks"].items()
        },
        "text_columns": data["text_columns"],
        "pool": data["pool"],
        "source_table": data["source_table"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="")
    parser.add_argument("--out", default="experiments/20261101_fusion")
    parser.add_argument("--pools", nargs="+", default=["pool_a", "pool_b"])
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    config = Config()
    llm_path = ROOT / args.llm if args.llm else None

    rows, attentions, meta = [], [], {}
    for pool in args.pools:
        data = build(pool, llm_path=llm_path, with_images=True)
        frame = data["frame"]
        log(f"{pool}: {len(frame)} rows, blocks {sorted(data['blocks'])}")
        meta[pool] = {
            "rows": int(len(frame)),
            "by_label": frame["label"].value_counts().to_dict(),
            "by_domain": frame["domain"].value_counts().to_dict(),
            "by_split": frame["split"].value_counts().to_dict(),
            "blocks": {name: {"dim": payload["dim"],
                              "present": int(payload["present"].sum())}
                       for name, payload in data["blocks"].items()},
        }

        matrix = [
            ("lgbm-text", [TEXT_BLOCK], False, "lightgbm"),
            ("gate-text", [TEXT_BLOCK], False, "gate"),
            ("gate-text+llm", [TEXT_BLOCK, LLM_BLOCK], False, "gate"),
            ("gate-text+llm+clip", [TEXT_BLOCK, LLM_BLOCK, CLIP_TEXT_BLOCK], False, "gate"),
        ]
        if llm_path is not None:
            matrix.insert(1, ("lgbm-text+llm", [TEXT_BLOCK, LLM_BLOCK], False, "lightgbm"))
        for label, blocks, include_risk, family in matrix:
            row, attention = run_one(label, data, blocks, include_risk, False, False,
                                     family, config, rng, device)
            rows.append({"pool": pool, **row})
            if attention:
                attentions.append({"pool": pool, "run": label, "attention": attention})

        # Two-level on the best block set, plus one ablation that drops the
        # risk columns so the report can show what readability/perplexity add.
        if llm_path is not None:
            for label, include_risk in (("twolevel-text+llm", False),
                                        ("twolevel-text+llm-noRisk", True)):
                row, attention = run_one(label, data, [TEXT_BLOCK, LLM_BLOCK],
                                         not include_risk, True, False, "gate",
                                         config, rng, device)
                rows.append({"pool": pool, **row})
                if attention:
                    attentions.append({"pool": pool, "run": label, "attention": attention})

        # Multimodal arm on rows that own an image. This subset is not random:
        # image presence is 23% / 76% / 75% across the three labels, so the row
        # is reported beside its text-only twin on the same rows.
        present = data["blocks"].get(IMAGE_BLOCK, {}).get("present")
        if present is not None and present.sum() > 500:
            sub = subset(data, present)
            for label, blocks in (("mm-text", [TEXT_BLOCK]),
                                  ("mm-text+llm", [TEXT_BLOCK, LLM_BLOCK]),
                                  ("mm-text+llm+image", [TEXT_BLOCK, LLM_BLOCK, IMAGE_BLOCK])):
                if LLM_BLOCK not in blocks and llm_path is None:
                    continue
                row, _ = run_one(label, sub, blocks, False, False, False,
                                 "gate", config, rng, device)
                rows.append({"pool": f"{pool}/with_image", **row})

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    (out / "attention.json").write_text(
        json.dumps(attentions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "config": asdict(config),
        "llm_features": str(llm_path.relative_to(ROOT)) if llm_path else "",
        "blocks": ["text_hand", "llm", "clip_text", "image_stats", "clip_image", "resnet"],
        "note": "corpus-level duplicate features excluded; test split used once; "
                "image rows are not a random subset",
        "pools": meta,
        "best": table.sort_values("macro_f1", ascending=False)
        .head(10)[["pool", "run", "blocks", "macro_f1", "f1_real", "f1_human", "f1_ai"]]
        .to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

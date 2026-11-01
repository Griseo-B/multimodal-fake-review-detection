"""Can a large model read the structured cards and name the right class?

Two probes, same model family, same prompt template:

    zero-shot   Qwen reads the bucketed feature description plus the review and
                the label with the highest likelihood wins. No training at all.
    lora        the same model with a rank-8 adapter on the attention projections,
                trained on pool A's train split, evaluated on its test split.

Both use the same bucketed description the cards carry (tercile cuts fit on
train only, one representative per redundant cluster), so the number answers a
specific question the proposal makes: is the structured description actually
readable by a large model, or only by a gradient-boosted tree?

    python code/train/fusion/llm_probe.py --pool pool_a --mode both
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
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "features"))

from data import build  # noqa: E402
from registry import TEXT_BLOCK, columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SEED = 42
LABEL_ORDER = ["real", "human_fake", "ai_fake"]
# One representative per redundant cluster, as in the cards: Flesch stands for
# ARI and Fog, TTR for hapax, and the duplicate counters are dropped because a
# single review cannot produce them.
TALK = ["perplexity", "flesch", "ttr", "pos_noun", "pos_adv", "pos_pron",
        "clause_marker_ratio", "sentiment", "extreme_ratio", "number_ratio",
        "smoothness", "n_words"]
PHRASES = {
    "perplexity": ("困惑度低", "困惑度居中", "困惑度高"),
    "flesch": ("可读性低", "可读性居中", "可读性高"),
    "ttr": ("用词重复", "用词多样性居中", "用词多样"),
    "pos_noun": ("名词偏少", "名词占比居中", "名词偏多"),
    "pos_adv": ("副词偏少", "副词占比居中", "副词偏多"),
    "pos_pron": ("代词偏少", "代词占比居中", "代词偏多"),
    "clause_marker_ratio": ("句式简单", "句式复杂度居中", "句式复杂"),
    "sentiment": ("情感偏负", "情感中性", "情感偏正"),
    "extreme_ratio": ("少有极端措辞", "极端措辞居中", "极端措辞多"),
    "number_ratio": ("细节数字少", "细节数字居中", "细节数字多"),
    "smoothness": ("句间跳跃", "句间连贯居中", "句间连贯"),
    "n_words": ("篇幅很短", "篇幅中等", "篇幅很长"),
}
INSTRUCTION = (
    "你是虚假评论审核员。下面给出一条评论的自检描述和原文。\n"
    "三个类别：real（真实用户评论）、human_fake（人工撰写的虚假评论）、"
    "ai_fake（AI 生成的虚假评论）。\n"
    "只回答一个类别代号：real 记作真，human_fake 记作人，ai_fake 记作机。\n"
)
LABELS = {"real": "real", "human_fake": "human_fake", "ai_fake": "ai_fake"}
# One token per class. Comparing multi-token strings by log-probability is
# length-biased ("real" is one token, the other two are three), and that bias
# alone was enough to collapse the probe onto the majority class.
ANSWER_TOKEN = {"real": "真", "human_fake": "人", "ai_fake": "机"}


@dataclass
class Config:
    seed: int = SEED
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 224
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def description(frame: pd.DataFrame, train_index: np.ndarray) -> list[str]:
    """Bucketed one-liner per row, cuts fit on train rows only."""
    parts = []
    for name in TALK:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        train = values.iloc[train_index].dropna()
        if len(train) < 30:
            continue
        low, high = np.quantile(train, [1 / 3, 2 / 3])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        parts.append((name, values.to_numpy(dtype=float), float(low), float(high)))
    rows = []
    for row in range(len(frame)):
        words = []
        for name, values, low, high in parts:
            value = values[row]
            if not np.isfinite(value):
                continue
            bucket = 0 if value <= low else (2 if value >= high else 1)
            words.append(PHRASES[name][bucket])
        rows.append("；".join(words))
    return rows


def build_prompt(summary: str, text: str) -> str:
    return (f"{INSTRUCTION}自检描述：{summary}\n评论原文：{text}\n类别：")


def label_scores(model, tokenizer, prompts: list[str], device, batch_size: int = 8) -> np.ndarray:
    """Likelihood of each label as the next tokens; the argmax is the answer.

    Scoring three continuations is cheaper and more stable than generating and
    parsing, and it removes the failure mode where the model writes a sentence
    that names two classes.
    """
    label_ids = [tokenizer.encode(ANSWER_TOKEN[name], add_special_tokens=False)
                 for name in LABEL_ORDER]
    scores = np.zeros((len(prompts), len(LABEL_ORDER)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                               max_length=384).to(device)
            outputs = model(**tokens).logits
            lengths = tokens["attention_mask"].sum(1) - 1
            log_probs = torch.log_softmax(outputs.float(), dim=-1)
            for row in range(len(chunk)):
                position = int(lengths[row])
                for index, ids in enumerate(label_ids):
                    value = 0.0
                    for offset, token_id in enumerate(ids):
                        if position + offset >= log_probs.shape[1]:
                            break
                        value += float(log_probs[row, position + offset, token_id])
                    # Sum, not mean: the three labels differ in token count
                    # ("real" is one token, the other two are three), and a
                    # per-token average systematically prefers the shortest.
                    scores[start + row, index] = value
    return scores


def evaluate(name: str, labels: np.ndarray, scores: np.ndarray, rows: int) -> dict:
    from sklearn.metrics import confusion_matrix

    predictions = scores.argmax(1)
    per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return {
        "run": name,
        "rows": rows,
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
        "f1_real": round(float(per_class[0]), 4),
        "f1_human": round(float(per_class[1]), 4),
        "f1_ai": round(float(per_class[2]), 4),
        "accuracy": round(float((predictions == labels).mean()), 4),
        "majority_macro_f1": round(float(np.bincount(labels).max() / len(labels)), 4),
        "confusion": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="pool_a")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", default="both", choices=["zero", "lora", "both"])
    parser.add_argument("--zero-shot-rows", type=int, default=300)
    parser.add_argument("--out", default="experiments/20261101_llm_probe")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build(args.pool, with_images=False)
    frame = data["frame"]
    labels = np.array([LABEL_ORDER.index(name) for name in frame["label"]], dtype=np.int64)
    train_index = np.where(frame["split"].eq("train"))[0]
    val_index = np.where(frame["split"].eq("val"))[0]
    test_index = np.where(frame["split"].eq("test"))[0]
    summary = description(frame, train_index)
    prompts = [build_prompt(summary[i], frame["text"].iloc[i]) for i in range(len(frame))]
    log(f"{args.pool}: {len(frame)} rows, test {len(test_index)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).eval().to(device)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(SEED)

    if args.mode in ("zero", "both"):
        sample = test_index if len(test_index) <= args.zero_shot_rows else rng.choice(
            test_index, args.zero_shot_rows, replace=False)
        started = time.perf_counter()
        scores = label_scores(model, tokenizer, [prompts[i] for i in sample], device)
        rows.append({**evaluate("zero-shot", labels[sample], scores, len(sample)),
                     "seconds": round(time.perf_counter() - started, 1)})
        log(f"zero-shot done in {rows[-1]['seconds']}s")

    if args.mode in ("lora", "both"):
        from peft import LoraConfig, get_peft_model

        adapter = LoraConfig(
            r=Config().lora_r,
            lora_alpha=Config().lora_alpha,
            lora_dropout=Config().lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, adapter)
        model.print_trainable_parameters()
        config = Config()
        optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config.lr)
        # Left padding keeps the label at the end of every row, which is what
        # num_logits_to_keep slices; right padding would put padding after the
        # label and teach the adapter to answer after pad tokens.
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        started = time.perf_counter()
        model.train()
        step = 0
        for epoch in range(1, config.epochs + 1):
            order = np.random.default_rng(config.seed + epoch).permutation(train_index)
            for start in range(0, len(order), config.batch_size):
                index = order[start : start + config.batch_size]
                batch = [prompts[i] for i in index]
                targets = [ANSWER_TOKEN[LABEL_ORDER[labels[i]]] for i in index]
                full = [text + target for text, target in zip(batch, targets)]
                encoded = tokenizer(full, return_tensors="pt", padding=True,
                                    truncation=True, max_length=config.max_length).to(device)
                lengths = [len(tokenizer(target, add_special_tokens=False)["input_ids"])
                           for target in targets]
                width = max(lengths)
                # The label occupies the last `length` tokens of each row; with
                # left padding the tail is always real text.
                target = encoded["input_ids"][:, -width:].clone()
                for row, length in enumerate(lengths):
                    if length < width:
                        target[row, : width - length] = -100
                outputs = model(input_ids=encoded["input_ids"],
                                attention_mask=encoded["attention_mask"],
                                num_logits_to_keep=width + 1)
                logits = outputs.logits[:, :width, :]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                    ignore_index=-100,
                )
                (loss / config.grad_accum).backward()
                if (start // config.batch_size + 1) % config.grad_accum == 0:
                    optimiser.step()
                    optimiser.zero_grad()
                    step += 1
            log(f"epoch {epoch} done, {step} steps, {time.perf_counter() - started:.0f}s")
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        model.eval()
        scores = label_scores(model, tokenizer, [prompts[i] for i in test_index], device)
        rows.append({**evaluate("lora", labels[test_index], scores, len(test_index)),
                     "seconds": round(time.perf_counter() - started, 1)})
        model.save_pretrained(out / "lora_adapter")

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    summary_doc = {
        "model": args.model,
        "pool": args.pool,
        "config": asdict(Config()),
        "description_fields": TALK,
        "prompt": INSTRUCTION,
        "rows": int(len(frame)),
        "note": "label likelihood rather than free generation; description buckets "
                "fit on train only; corpus-level fields excluded",
        "table": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

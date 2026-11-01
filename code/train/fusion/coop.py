"""Cross-modal prompt learning: a learnable CLIP context, then a fusion head.

The proposal's "cross-modal prompt learning + feature concatenation" in its
smallest honest form:

    1. CLIP is frozen. Only a context of N learnable vectors in front of each
       class name is trained (CoOp). Two classes: a genuine review photo, a
       machine-generated review photo.
    2. The two class logits are then concatenated with the handcrafted text
       features and a logistic head decides real versus AI fake.

What this can and cannot show is decided by the data, not the code: the images
in the food pool are amateur photos of real dishes against DALL-E renderings,
so a good number here says the two image sources look different. That is a
provenance signal. The row is reported beside its text-only twin so the report
never claims the image adds fake-review evidence.

    python code/train/fusion/coop.py --pool pool_b --domain food
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
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "features"))

from data import build  # noqa: E402
from registry import TEXT_BLOCK, columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SEED = 42
N_CTX = 16
CLASS_NAMES = ["a genuine photo taken by a customer", "a machine generated promotional photo"]
LABEL_ORDER = ["real", "human_fake", "ai_fake"]
LABEL_INDEX = {name: index for index, name in enumerate(LABEL_ORDER)}


@dataclass
class Config:
    seed: int = SEED
    n_ctx: int = N_CTX
    epochs: int = 12
    lr: float = 1e-4
    batch_size: int = 512
    logit_scale: float = 100.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    sys.path.insert(0, str(ROOT / "code" / "data"))
    from features_mm3 import CLIP_MODEL

    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
    except OSError:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        model = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval().to(device)
    return processor, model


class CoOp(torch.nn.Module):
    def __init__(self, clip, n_ctx: int, device):
        super().__init__()
        self.clip = clip
        self.n_ctx = n_ctx
        width = clip.text_model.config.hidden_size
        # Small init: a random context would start far from any word and the
        # frozen image tower would see noise for the first several steps.
        self.context = torch.nn.Parameter(torch.randn(n_ctx, width, device=device) * 0.02)
        self.class_ids = None

    def tokenise(self, processor, device):
        placeholder = " ".join(["X"] * self.n_ctx)
        prompts = [f"{placeholder} {name}" for name in CLASS_NAMES]
        tokens = processor(text=prompts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=77)
        self.class_ids = {key: value.to(device) for key, value in tokens.items()}
        # Start from a real phrase rather than noise: "a photo of a" repeated
        # lands inside the region the frozen tower already understands, and the
        # learned context moves from there instead of from a random point.
        init = processor(text="a photo of a " * 4, return_tensors="pt",
                         truncation=True, max_length=self.n_ctx + 2)
        init_ids = init["input_ids"][0, 1:-1][: self.n_ctx].to(device)
        with torch.no_grad():
            init_embedding = self.clip.text_model.embeddings.token_embedding(init_ids)
        if len(init_embedding) < self.n_ctx:
            pad = init_embedding[-1:].expand(self.n_ctx - len(init_embedding), -1)
            init_embedding = torch.cat([init_embedding, pad], dim=0)
        self.context.data.copy_(init_embedding)

    def text_features(self) -> torch.Tensor:
        ids = self.class_ids["input_ids"]
        mask = self.class_ids["attention_mask"]
        embedding = self.clip.text_model.embeddings.token_embedding(ids)
        # Position 0 is <start>; the context replaces the next n_ctx tokens,
        # which is where the placeholders were tokenised.
        context = self.context.unsqueeze(0).expand(len(CLASS_NAMES), -1, -1).to(embedding.dtype)
        embedding = torch.cat([embedding[:, :1, :], context, embedding[:, 1 + self.n_ctx :, :]], dim=1)
        return self._encode(embedding, ids, mask)

    def _encode(self, embedding: torch.Tensor, ids: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Run the frozen text tower on pre-built token embeddings.

        transformers 4.46's CLIPTextTransformer takes input_ids, not
        inputs_embeds, so the three layers it wraps are called directly. The
        causal mask is the one it would have built itself.
        """
        tower = self.clip.text_model
        from transformers.modeling_attn_mask_utils import (
            _create_4d_causal_attention_mask, _prepare_4d_attention_mask)

        hidden = tower.embeddings(inputs_embeds=embedding)
        causal = _create_4d_causal_attention_mask(hidden.shape[:2], hidden.dtype,
                                                  device=hidden.device)
        attn = _prepare_4d_attention_mask(mask, hidden.dtype)
        encoded = tower.encoder(inputs_embeds=hidden, attention_mask=attn,
                                causal_attention_mask=causal,
                                output_attentions=False, output_hidden_states=False,
                                return_dict=True)[0]
        last = tower.final_layer_norm(encoded)
        # Mirrors the frozen tower's own pooling rule: this checkpoint carries
        # the legacy eos_token_id=2, so the end-of-text position is the largest
        # token id (padding uses that same id, and argmax returns the first).
        if tower.eos_token_id == 2:
            index = ids.argmax(dim=-1)
        else:
            index = (ids == tower.eos_token_id).int().argmax(dim=-1)
        pooled = last[torch.arange(last.shape[0], device=last.device), index]
        return self.clip.text_projection(pooled)


def frozen_prompt_features(processor, clip, device) -> torch.Tensor:
    tokens = processor(text=CLASS_NAMES, return_tensors="pt", padding=True, truncation=True)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        return clip.get_text_features(**tokens)


def train_coop(clip, processor, images: np.ndarray, labels: np.ndarray,
               train_index: np.ndarray, val_index: np.ndarray, config: Config,
               device) -> tuple[CoOp, dict]:
    model = CoOp(clip, config.n_ctx, device).to(device)
    model.tokenise(processor, device)
    for parameter in clip.parameters():
        parameter.requires_grad = False

    image_t = torch.tensor(images, dtype=torch.float32, device=device)
    label_t = torch.tensor(labels, dtype=torch.long, device=device)
    optimiser = torch.optim.AdamW([model.context], lr=config.lr, weight_decay=1e-4)
    best, best_epoch, history = -1.0, -1, []
    rng = np.random.default_rng(config.seed)

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_index)
        for start in range(0, len(order), config.batch_size):
            batch = torch.tensor(order[start : start + config.batch_size], device=device)
            text = model.text_features()
            logits = config.logit_scale * image_t[batch] @ text.t()
            loss = torch.nn.functional.cross_entropy(logits, label_t[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            text = model.text_features()
            scores = (config.logit_scale * image_t[val_index] @ text.t()).softmax(1).cpu().numpy()
        predictions = scores.argmax(1)
        score = float(f1_score(labels[val_index], predictions, average="macro"))
        history.append({"epoch": epoch, "val_macro_f1": round(score, 4),
                        "val_auc": round(float(roc_auc_score(labels[val_index], scores[:, 1])), 4)})
        if epoch % 2 == 0:
            log(f"epoch {epoch}: val macro-F1 {score:.4f}, "
                f"val AUC {history[-1]['val_auc']:.4f}")
        if score > best:
            best, best_epoch = score, epoch
            best_context = model.context.detach().clone()
    model.context.data = best_context
    model.eval()
    return model, {"best_epoch": best_epoch, "val_macro_f1": round(best, 4),
                   "epochs_run": len(history), "history": history[-3:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="pool_b")
    parser.add_argument("--domain", default="food")
    parser.add_argument("--out", default="experiments/20261101_coop")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build(args.pool, with_images=True)
    frame = data["frame"]
    keep = (frame["domain"] == args.domain) & (frame["label"].isin(["real", "ai_fake"]))
    keep &= data["blocks"]["clip_image"]["present"]
    data = {
        "frame": frame.loc[keep].reset_index(drop=True),
        "blocks": {name: {"matrix": payload["matrix"][keep.to_numpy()],
                          "present": payload["present"][keep.to_numpy()],
                          "dim": payload["dim"]}
                   for name, payload in data["blocks"].items()},
    }
    frame = data["frame"]
    labels = np.array([0 if name == "real" else 1 for name in frame["label"]], dtype=np.int64)
    log(f"{args.pool}/{args.domain}: {len(frame)} rows, "
        f"real {int((labels == 0).sum())}, ai_fake {int((labels == 1).sum())}")

    images = data["blocks"]["clip_image"]["matrix"]
    images = images / np.clip(np.linalg.norm(images, axis=1, keepdims=True), 1e-8, None)
    train_index = np.where(frame["split"].eq("train"))[0]
    val_index = np.where(frame["split"].eq("val"))[0]
    test_index = np.where(frame["split"].eq("test"))[0]

    processor, clip = load_clip(device)
    config = Config(epochs=args.epochs)
    rows = []
    rng = np.random.default_rng(SEED)

    def score(name: str, probabilities: np.ndarray, note: str) -> None:
        predictions = (probabilities >= 0.5).astype(int)
        values = []
        for _ in range(500):
            index = rng.integers(0, len(test_index), len(test_index))
            if len(np.unique(labels[test_index][index])) < 2:
                continue
            values.append(f1_score(labels[test_index][index], predictions[index],
                                   average="macro"))
        rows.append({
            "run": name,
            "rows_train": int(len(train_index)),
            "rows_test": int(len(test_index)),
            "macro_f1": round(float(f1_score(labels[test_index], predictions, average="macro")), 4),
            "macro_f1_ci_low": round(float(np.percentile(values, 2.5)), 4),
            "macro_f1_ci_high": round(float(np.percentile(values, 97.5)), 4),
            "auc": round(float(roc_auc_score(labels[test_index], probabilities)), 4),
            "note": note,
        })

    # 1. Frozen prompt, no training: the reference the learned context must beat.
    with torch.no_grad():
        text = frozen_prompt_features(processor, clip, device)
        image_t = torch.tensor(images, dtype=torch.float32, device=device)
        frozen = (config.logit_scale * image_t[test_index] @ text.t()).softmax(1).cpu().numpy()
    score("clip-frozen-prompt", frozen[:, 1], "hand-written class prompts, zero training")

    # 2. Learned context.
    model, info = train_coop(clip, processor, images, labels, train_index, val_index,
                             config, device)
    with torch.no_grad():
        text = model.text_features()
        image_t = torch.tensor(images, dtype=torch.float32, device=device)
        coop = (config.logit_scale * image_t[test_index] @ text.t()).softmax(1).cpu().numpy()
    score("coop-learned-context", coop[:, 1], f"ctx {config.n_ctx}, best epoch {info['best_epoch']}")

    # 3. Prompt logits concatenated with handcrafted text features.
    text_cols = [name for name in columns(TEXT_BLOCK, include_corpus=False) if name in frame.columns]
    hand = frame[text_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    train_full = np.where(frame["split"].isin(["train", "val"]))[0]
    with torch.no_grad():
        text = model.text_features()
        train_logits = (config.logit_scale * image_t[train_full] @ text.t()).cpu().numpy()
        test_logits = (config.logit_scale * image_t[test_index] @ text.t()).cpu().numpy()
    concat_train = np.concatenate([train_logits, hand[train_full]], axis=1)
    concat_test = np.concatenate([test_logits, hand[test_index]], axis=1)
    head = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    head.fit(concat_train, labels[train_full])
    score("coop+hand-fusion", head.predict_proba(concat_test)[:, 1],
          "learned prompt logits concatenated with handcrafted text features")

    # 4. Text-only twin on the same rows: the number the prompt arm has to beat
    #    before anyone can call the image informative.
    text_head = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    text_head.fit(hand[train_full], labels[train_full])
    score("text-only-twin", text_head.predict_proba(hand[test_index])[:, 1],
          "same rows, handcrafted text features only")

    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"metrics_{args.domain}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "config": asdict(config),
        "pool": args.pool,
        "domain": args.domain,
        "rows": int(len(frame)),
        "class_names": CLASS_NAMES,
        "training": info,
        "note": "image source (amateur photo vs DALL-E) is the separable factor; "
                "the text-only twin on the same rows is the control",
        "table": rows,
    }
    (out / f"summary_{args.domain}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Score the existing 200 synonym rewrites with the frozen BERT and CLIP heads.

Heads are refit on the original train split. Rewritten sentences are not added
to training. Images stay the original files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from code.train.frozen_bert_head import MODEL_NAME, encode

SEED = 42
LABELS = ["real", "ai_fake"]


def scores(y_true, y_pred) -> tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
    )


def fit_predict(features: np.ndarray, train: np.ndarray, y_train, eval_features: np.ndarray) -> np.ndarray:
    head = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    head.fit(features[train], y_train)
    return head.predict(eval_features)


def main() -> None:
    device = torch.device("cuda")
    frame = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv")
    frame = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].reset_index(drop=True)
    frame["source_id"] = frame["source_id"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    rewritten = pd.read_csv(ROOT / "experiments" / "20261026_rewrite_tfidf" / "rewritten_200.csv")
    rewritten["source_id"] = rewritten["source_id"].astype(str)
    id_to_index = {source_id: index for index, source_id in enumerate(frame["source_id"])}
    ai_index = np.array([id_to_index[source_id] for source_id in rewritten["source_id"]])
    real_index = frame.index[(frame["split"] == "test") & (frame["label"] == "real")].to_numpy()
    eval_index = np.concatenate([ai_index, real_index])
    y_eval = frame.loc[eval_index, "label"].to_numpy()
    train = frame["split"].to_numpy() == "train"
    y_train = frame.loc[train, "label"]

    data = np.load(ROOT / "experiments" / "20261026_modality_rows" / "features.npz")
    text = data["text"]
    image = data["image"]
    print("encoding rewritten CLIP text", flush=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tokens = clip_processor(
        text=rewritten["rewritten"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        new_text = clip_model.get_text_features(**tokens)
        new_text = new_text.cpu().numpy() if torch.is_tensor(new_text) else new_text.text_embeds.cpu().numpy()
    text_rewritten = text.copy()
    text_rewritten[ai_index] = new_text
    image_norm = image / np.clip(np.linalg.norm(image, axis=1, keepdims=True), 1e-8, None)
    text_norm = text_rewritten / np.clip(np.linalg.norm(text_rewritten, axis=1, keepdims=True), 1e-8, None)
    similarity = np.sum(image_norm * text_norm, axis=1, keepdims=True)
    original_similarity = data["similarity"]

    rows = []
    specs = {
        "clip_text": (text, text_rewritten),
        "clip_image": (image, image),
        "clip_fusion": (
            np.hstack([text, image, original_similarity]),
            np.hstack([text_rewritten, image, similarity]),
        ),
    }
    for name, (original_features, rewritten_features) in specs.items():
        original_pred = fit_predict(original_features, train, y_train, original_features[eval_index])
        rewritten_pred = fit_predict(original_features, train, y_train, rewritten_features[eval_index])
        original_acc, original_f1 = scores(y_eval, original_pred)
        rewritten_acc, rewritten_f1 = scores(y_eval, rewritten_pred)
        rows.append({"model": name, "version": "original", "accuracy": original_acc, "macro_f1": original_f1, "n": len(y_eval)})
        rows.append(
            {
                "model": name,
                "version": "rewritten",
                "accuracy": rewritten_acc,
                "macro_f1": rewritten_f1,
                "macro_f1_drop": rewritten_f1 - original_f1,
                "n": len(y_eval),
            }
        )

    print("encoding BERT", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    train_vectors = encode(frame.loc[train, "text"].tolist(), tokenizer, encoder, device)
    original_eval_text = frame.loc[ai_index, "text"].tolist() + frame.loc[real_index, "text"].tolist()
    rewritten_eval_text = rewritten["rewritten"].tolist() + frame.loc[real_index, "text"].tolist()
    original_eval = encode(original_eval_text, tokenizer, encoder, device)
    rewritten_eval = encode(rewritten_eval_text, tokenizer, encoder, device)
    head = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    head.fit(train_vectors, y_train)
    original_acc, original_f1 = scores(y_eval, head.predict(original_eval))
    rewritten_acc, rewritten_f1 = scores(y_eval, head.predict(rewritten_eval))
    rows.append({"model": "frozen_bert_head", "version": "original", "accuracy": original_acc, "macro_f1": original_f1, "n": len(y_eval)})
    rows.append(
        {
            "model": "frozen_bert_head",
            "version": "rewritten",
            "accuracy": rewritten_acc,
            "macro_f1": rewritten_f1,
            "macro_f1_drop": rewritten_f1 - original_f1,
            "n": len(y_eval),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "experiments" / "20261026_rewrite_tfidf" / "metrics_models.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

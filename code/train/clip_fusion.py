"""Frozen CLIP features on the full audited AiGen split.

Images are resized by the CLIP processor before encoding. Raw width and height
are used only for the size-shortcut check, not as model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
MODEL_NAME = "openai/clip-vit-base-patch32"
LABELS = ["real", "ai_fake"]
BATCH = 16


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv")
    frame = frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame.reset_index(drop=True)


def as_tensor(output: torch.Tensor | object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"unexpected CLIP output: {type(output)}")


def metrics_row(name: str, y_true, y_pred) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    row = {
        "model": name,
        "split": "test",
        "status": "完成",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
    }
    for label, prec, rec in zip(LABELS, precision, recall):
        row[f"precision_{label}"] = float(prec)
        row[f"recall_{label}"] = float(rec)
    return row


def fit_head(x_train: np.ndarray, y_train, x_test: np.ndarray, y_test, scale: bool) -> dict:
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    head = LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED)
    head.fit(x_train, y_train)
    return metrics_row("pending", y_test, head.predict(x_test))


def encode(frame: pd.DataFrame, processor, model, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    image_parts = []
    text_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), BATCH):
            chunk = frame.iloc[start : start + BATCH]
            images = [Image.open(ROOT / path).convert("RGB") for path in chunk["image_path"]]
            pixels = processor(images=images, return_tensors="pt")
            tokens = processor(
                text=chunk["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            pixels = {key: value.to(device) for key, value in pixels.items()}
            tokens = {key: value.to(device) for key, value in tokens.items()}
            image_parts.append(as_tensor(model.get_image_features(**pixels)).cpu().numpy())
            text_parts.append(as_tensor(model.get_text_features(**tokens)).cpu().numpy())
            for image in images:
                image.close()
            print(f"encoded {min(start + BATCH, len(frame))}/{len(frame)}", flush=True)
    return np.vstack(image_parts), np.vstack(text_parts)


def main() -> None:
    out = ROOT / "experiments" / "20261026_modality_rows"
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    frame = load_frame()
    train = frame["split"].to_numpy() == "train"
    test = frame["split"].to_numpy() == "test"
    y_train = frame.loc[train, "label"]
    y_test = frame.loc[test, "label"]
    size = frame[["image_width", "image_height"]].to_numpy(dtype=np.float32)
    rows = [fit_head(size[train], y_train, size[test], y_test, scale=True)]
    rows[0]["model"] = "raw_size_only"
    print("loading", MODEL_NAME, "on", torch.cuda.get_device_name(0), flush=True)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image_feat, text_feat = encode(frame, processor, model, device)
    image_norm = image_feat / np.clip(np.linalg.norm(image_feat, axis=1, keepdims=True), 1e-8, None)
    text_norm = text_feat / np.clip(np.linalg.norm(text_feat, axis=1, keepdims=True), 1e-8, None)
    similarity = np.sum(image_norm * text_norm, axis=1, keepdims=True)
    named = {
        "clip_text": text_feat,
        "clip_image": image_feat,
        "clip_similarity": similarity,
        "clip_fusion": np.hstack([text_feat, image_feat, similarity]),
    }
    for name, features in named.items():
        row = fit_head(features[train], y_train, features[test], y_test, scale=False)
        row["model"] = name
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(
        out / "features.npz",
        image=image_feat.astype(np.float32),
        text=text_feat.astype(np.float32),
        similarity=similarity.astype(np.float32),
        width=frame["image_width"].to_numpy(),
        height=frame["image_height"].to_numpy(),
        split=frame["split"].to_numpy(),
        label=frame["label"].to_numpy(),
    )
    (out / "timing.json").write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "model": MODEL_NAME,
                "n": int(len(frame)),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "processor_size": getattr(processor, "crop_size", None) or getattr(processor.image_processor, "crop_size", None),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Sentence-level semantic smoothness under CLIP.

Smoothness is the mean cosine between neighbouring sentence embeddings. It is
computed for every row with at least two sentences, with or without an image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, sentences  # noqa: E402

sys.path.insert(0, str(ROOT / "code" / "data"))
from features_mm3 import CLIP_MODEL  # noqa: E402

BATCH = 32


def load_clip(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
    except OSError:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        model = CLIPModel.from_pretrained(CLIP_MODEL)
    model.eval().to(device)
    return processor, model


def embed_texts(processor, model, texts: list[str], device: torch.device) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            tokens = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model.get_text_features(**tokens)
            tensor = output if torch.is_tensor(output) else output.text_embeds
            parts.append(tensor.detach().cpu().numpy())
    if not parts:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def smoothness_column(frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    processor, model = load_clip(device)
    owners: list[int] = []
    chunk: list[str] = []
    for index, text in enumerate(frame["text"].astype(str).tolist()):
        for sentence in sentences(text):
            owners.append(index)
            chunk.append(sentence)
    values = np.full(len(frame), np.nan, dtype=np.float32)
    if chunk:
        embed = embed_texts(processor, model, chunk, device)
        norm = embed / np.clip(np.linalg.norm(embed, axis=1, keepdims=True), 1e-8, None)
        owner_array = np.asarray(owners)
        for index in range(len(frame)):
            selected = norm[owner_array == index]
            if len(selected) >= 2:
                values[index] = float(np.mean(np.sum(selected[:-1] * selected[1:], axis=1)))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def smoothness_frame(frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": frame["id"].to_numpy(), "smoothness": smoothness_column(frame, device)}
    )

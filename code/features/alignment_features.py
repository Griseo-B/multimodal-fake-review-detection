"""Image-text alignment under CLIP for rows with a review-owned image.

Three numbers: full-review cosine, the best matching sentence, and the mean
over sentences. Sentence-level values use at most eight sentences per review,
the same cap as the 10-27 experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import project_path, sentences  # noqa: E402
from smoothness import embed_texts, load_clip  # noqa: E402

COLUMNS = ["clip_cosine", "sent_max_cosine", "sent_mean_cosine"]


def _normalise(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-8, None)


def embed_images(processor, model, paths: list[Path], device: torch.device) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(paths), 16):
            batch = []
            for path in paths[start : start + 16]:
                with Image.open(path) as handle:
                    batch.append(handle.convert("RGB"))
            pixels = processor(images=batch, return_tensors="pt")
            pixels = {key: value.to(device) for key, value in pixels.items()}
            output = model.get_image_features(**pixels)
            tensor = output if torch.is_tensor(output) else output.image_embeds
            parts.append(tensor.detach().cpu().numpy())
    if not parts:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def alignment_frame(frame: pd.DataFrame, device: torch.device, progress: int = 500):
    processor, model = load_clip(device)
    texts = frame["text"].astype(str).tolist()
    paths = [project_path(value) for value in frame["image_file"].tolist()]
    text_embed = _normalise(embed_texts(processor, model, texts, device))
    image_embed = _normalise(embed_images(processor, model, paths, device))
    cosine = np.sum(text_embed * image_embed, axis=1)

    owners: list[int] = []
    chunk: list[str] = []
    for index, text in enumerate(texts):
        for sentence in sentences(text):
            owners.append(index)
            chunk.append(sentence)
    best = np.full(len(frame), np.nan, dtype=np.float32)
    average = np.full(len(frame), np.nan, dtype=np.float32)
    if chunk:
        sentence_embed = _normalise(embed_texts(processor, model, chunk, device))
        owner_array = np.asarray(owners)
        similarity = np.sum(sentence_embed * image_embed[owner_array], axis=1)
        for index in range(len(frame)):
            selected = similarity[owner_array == index]
            if selected.size:
                best[index] = float(selected.max())
                average[index] = float(selected.mean())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    table = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "clip_cosine": cosine.astype(np.float32),
            "sent_max_cosine": best,
            "sent_mean_cosine": average,
        }
    )
    embeddings = {
        "clip_text": text_embed.astype(np.float16),
        "clip_image": image_embed.astype(np.float16),
    }
    return table[["id", *COLUMNS]], embeddings

"""Text, vision, and image-text features for the frozen multimodal table.

Scalers and the TF-IDF matrix are fit later, on the train split only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

from mm3_common import FEATURE_DIR, ROOT, SEED, TABLE

PPL_MODEL = "distilgpt2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
POSITIVE = {
    "good", "great", "excellent", "amazing", "wonderful", "delicious", "tasty",
    "love", "loved", "best", "friendly", "perfect", "nice", "fresh", "clean",
    "recommend", "awesome", "pleasant", "comfortable", "beautiful", "favorite",
    "outstanding", "helpful", "quick", "generous", "again",
}
NEGATIVE = {
    "bad", "terrible", "awful", "horrible", "disgusting", "worst", "dirty",
    "rude", "slow", "bland", "cold", "stale", "poor", "disappointing",
    "overpriced", "mediocre", "nasty", "broken", "noisy", "smell", "never",
    "wrong", "late", "missing", "leaks", "wobbles",
}
HAND = [
    "n_words", "avg_word_len", "ttr", "excl_ratio", "upper_ratio", "flesch",
    "sentiment", "pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb",
    "perplexity", "smoothness", "rating", "bright", "sat", "contrast",
    "lap_var", "edge_density", "clip_cosine",
]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word)
    count = max(1, len(groups))
    if word.endswith("e") and count > 1:
        count -= 1
    return count


def flesch(text: str) -> float:
    words = tokens(text)
    if not words:
        return float("nan")
    sentences = max(1, len(re.findall(r"[.!?]", text)))
    syllable_count = sum(syllables(word) for word in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllable_count / len(words))


def ensure_nltk():
    import nltk

    for resource in (
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            continue
    return nltk


def pos_ratios(text: str, nltk_module) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return {key: float("nan") for key in ("pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb")}
    try:
        tags = [tag for _, tag in nltk_module.pos_tag(words)]
    except Exception:
        tags = []
    total = max(1, len(tags))

    def share(prefix: str) -> float:
        return sum(tag.startswith(prefix) for tag in tags) / total

    return {
        "pos_noun": share("NN"),
        "pos_adj": share("JJ"),
        "pos_adv": share("RB"),
        "pos_pron": share("PR"),
        "pos_verb": share("VB"),
    }


def text_stats(frame: pd.DataFrame, nltk_module) -> np.ndarray:
    rows = []
    for text, rating in zip(frame["text"], frame["rating"]):
        words = tokens(text)
        letters = [char for char in text if char.isalpha()]
        unique = len(set(words))
        sentiment = 0.0
        if words:
            sentiment = (sum(word in POSITIVE for word in words) - sum(word in NEGATIVE for word in words)) / len(words)
        stats = {
            "n_words": float(len(words)),
            "avg_word_len": float(np.mean([len(word) for word in words])) if words else float("nan"),
            "ttr": float(unique / len(words)) if words else float("nan"),
            "excl_ratio": float(text.count("!") / max(1, len(text))),
            "upper_ratio": float(sum(char.isupper() for char in letters) / max(1, len(letters))),
            "flesch": float(flesch(text)),
            "sentiment": float(sentiment),
            "rating": float(rating) if pd.notna(rating) else float("nan"),
        }
        stats.update(pos_ratios(text, nltk_module))
        rows.append(stats)
    return pd.DataFrame(rows)


def visual_stats(path: Path) -> dict[str, float]:
    image = Image.open(ROOT / path).convert("RGB")
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return {
        "bright": float(value.mean()),
        "sat": float(saturation.mean()),
        "contrast": float(value.std()),
        "lap_var": float(lap.var()),
        "edge_density": float((magnitude > 0.08).mean()),
    }


def perplexities(texts: list[str], device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(PPL_MODEL).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scores = []
    for start in range(0, len(texts), 8):
        chunk = texts[start : start + 8]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        with torch.no_grad():
            logits = model(**encoded).logits[:, :-1, :]
        shifted = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shifted.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted.shape)
        counts = (shifted != -100).sum(dim=1).clamp(min=1)
        scores.extend(torch.exp(loss.sum(dim=1) / counts).cpu().tolist())
        if start % 160 == 0:
            print(f"perplexity {start}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float32)


def as_numpy(output: torch.Tensor | object) -> np.ndarray:
    tensor = output if torch.is_tensor(output) else None
    if tensor is None:
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                tensor = value
                break
    if tensor is None:
        raise TypeError(f"unexpected CLIP output: {type(output)}")
    return tensor.detach().cpu().numpy()


def encode_clip(frame: pd.DataFrame, device: torch.device):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()
    text_parts = []
    image_parts = []
    with torch.no_grad():
        for start in range(0, len(frame), 16):
            chunk = frame.iloc[start : start + 16]
            images = [Image.open(ROOT / path).convert("RGB") for path in chunk["image_path"]]
            pixels = processor(images=images, return_tensors="pt")
            tokens = processor(
                text=chunk["text"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            pixels = {key: value.to(device) for key, value in pixels.items()}
            tokens = {key: value.to(device) for key, value in tokens.items()}
            image_parts.append(as_numpy(model.get_image_features(**pixels)))
            text_parts.append(as_numpy(model.get_text_features(**tokens)))
            for image in images:
                image.close()
            if start % 160 == 0:
                print(f"clip {start}/{len(frame)}", flush=True)
    text_embed = np.concatenate(text_parts)
    image_embed = np.concatenate(image_parts)
    text_norm = text_embed / np.clip(np.linalg.norm(text_embed, axis=1, keepdims=True), 1e-8, None)
    image_norm = image_embed / np.clip(np.linalg.norm(image_embed, axis=1, keepdims=True), 1e-8, None)
    cosine = np.sum(text_norm * image_norm, axis=1)
    sentence_owner = []
    sentence_text = []
    for index, text in enumerate(frame["text"].tolist()):
        parts = [part.strip() for part in re.split(r"[.!?]+", text) if len(part.split()) >= 4][:4]
        for part in parts:
            sentence_owner.append(index)
            sentence_text.append(part)
    sentence_embed = np.zeros((len(sentence_text), text_embed.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(sentence_text), 32):
            tokens = processor(
                text=sentence_text[start : start + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            batch = as_numpy(model.get_text_features(**tokens))
            sentence_embed[start : start + len(batch)] = batch
    sentence_norm = sentence_embed / np.clip(np.linalg.norm(sentence_embed, axis=1, keepdims=True), 1e-8, None)
    smoothness = np.full(len(frame), np.nan, dtype=np.float32)
    owners = np.asarray(sentence_owner)
    for index in range(len(frame)):
        chosen = sentence_norm[owners == index]
        if len(chosen) >= 2:
            smoothness[index] = float(np.mean(np.sum(chosen[:-1] * chosen[1:], axis=1)))
    del model
    torch.cuda.empty_cache()
    return text_norm.astype(np.float32), image_norm.astype(np.float32), cosine.astype(np.float32), smoothness


def encode_resnet(frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    parts = []
    with torch.no_grad():
        for start in range(0, len(frame), 32):
            images = [transform(Image.open(ROOT / path).convert("RGB")) for path in frame["image_path"].iloc[start : start + 32]]
            batch = torch.stack(images).to(device)
            parts.append(model(batch).cpu().numpy())
            if start % 160 == 0:
                print(f"resnet {start}/{len(frame)}", flush=True)
    embed = np.concatenate(parts).astype(np.float32)
    norms = np.clip(np.linalg.norm(embed, axis=1, keepdims=True), 1e-8, None)
    del model
    torch.cuda.empty_cache()
    return (embed / norms).astype(np.float32)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(TABLE)
    frame["text"] = frame["text"].fillna("").astype(str)
    nltk_module = ensure_nltk()
    print("text stats", flush=True)
    stats = text_stats(frame, nltk_module)
    print("visual stats", flush=True)
    visual = pd.DataFrame([visual_stats(Path(path)) for path in frame["image_path"]])
    print("perplexity", flush=True)
    stats["perplexity"] = perplexities(frame["text"].tolist(), device)
    print("clip", flush=True)
    text_embed, image_embed, cosine, smoothness = encode_clip(frame, device)
    stats["smoothness"] = smoothness
    stats["clip_cosine"] = cosine
    hand = pd.concat([stats.reset_index(drop=True), visual.reset_index(drop=True)], axis=1)
    missing = [name for name in HAND if name not in hand.columns]
    if missing:
        raise SystemExit(f"missing features: {missing}")
    print("resnet", flush=True)
    resnet_embed = encode_resnet(frame, device)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_DIR / "features.npz",
        id=frame["id"].to_numpy(),
        clip_text=text_embed,
        clip_image=image_embed,
        resnet=resnet_embed,
        hand=hand[HAND].to_numpy(dtype=np.float32),
        hand_names=np.asarray(HAND),
    )
    print(f"wrote {FEATURE_DIR / 'features.npz'} rows={len(frame)}")


if __name__ == "__main__":
    main()

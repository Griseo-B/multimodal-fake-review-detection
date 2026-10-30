"""Hotel AI reviews for the same 20 Chicago hotels as Ott.

The prompt uses the hotel name, polarity, and topic words that occur in the
real Ott reviews of that hotel. It does not include any review sentence.
A candidate is dropped when any 10-word window already appears in Ott, or when
its first 10 words already appear in Ott, AiGen, Salminen, or MAiDE-up.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mm3_common import ROOT, SEED

OUT = ROOT / "data" / "raw" / "hotel_ai" / "reviews_v2.csv"
MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TARGET = 420
MAX_ATTEMPTS = 8000
HOTELS = {
    "affinia": "Affinia Chicago",
    "allegro": "Hotel Allegro Chicago",
    "amalfi": "Amalfi Hotel Chicago",
    "ambassador": "Ambassador East Hotel",
    "conrad": "Conrad Chicago",
    "fairmont": "Fairmont Chicago",
    "hardrock": "Hard Rock Hotel Chicago",
    "hilton": "Hilton Chicago",
    "homewood": "Homewood Suites Chicago Downtown",
    "hyatt": "Hyatt Regency Chicago",
    "intercontinental": "InterContinental Chicago",
    "james": "The James Chicago",
    "knickerbocker": "Millennium Knickerbocker Hotel",
    "monaco": "Hotel Monaco Chicago",
    "omni": "Omni Chicago Hotel",
    "palmer": "The Palmer House Hilton",
    "sheraton": "Sheraton Chicago",
    "sofitel": "Sofitel Chicago",
    "swissotel": "Swissotel Chicago",
    "talbott": "The Talbott Hotel",
}
ASPECTS = (
    "room", "bed", "bathroom", "staff", "service", "location", "breakfast",
    "lobby", "view", "price", "clean", "noise", "restaurant", "parking",
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def windows(text: str, width: int = 10) -> set[str]:
    words = tokens(text)
    if len(words) < width:
        return set()
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)}


def first_n(text: str, count: int = 10) -> str:
    return " ".join(tokens(text)[:count])


def hotel_key(source_id: str) -> str:
    parts = str(source_id).split("_")
    return parts[1] if len(parts) >= 2 else ""


def load_guards() -> tuple[set[str], set[str], dict[str, list[str]]]:
    blocked = set()
    ott = pd.read_csv(ROOT / "data" / "frozen" / "ott_hotel.csv", usecols=["source_id", "label", "text", "usable"])
    usable = ott["usable"].astype(str).isin(["True", "true", "1"])
    ott_text = ott.loc[usable, "text"].fillna("").astype(str)
    seen = set()
    for text in ott_text:
        blocked.add(first_n(text))
        seen.update(windows(text))
    aigen = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv", usecols=["text"])
    sales = pd.read_parquet(ROOT / "data" / "raw" / "salminen" / "train.parquet", columns=["text"])
    maide = pd.read_csv(ROOT / "data" / "raw" / "maide_up" / "all_data.csv", usecols=["Upside_Review", "Downside_Review"])
    for series in (aigen["text"], sales["text"], maide["Upside_Review"], maide["Downside_Review"]):
        for text in series.fillna("").astype(str):
            blocked.add(first_n(text))
    blocked.discard("")
    aspects: dict[str, list[str]] = {}
    real = ott.loc[usable & (ott["label"] == "real")].copy()
    for key, frame in real.groupby(real["source_id"].map(hotel_key)):
        blob = " ".join(frame["text"].fillna("").astype(str)).lower()
        found = [word for word in ASPECTS if blob.count(word) >= 5]
        aspects[key] = found[:4] or ["room", "staff", "location"]
    return blocked, seen, aspects


def prompt_for(hotel: str, polarity: str, topics: list[str], kind: str) -> str:
    feeling = "a positive stay" if polarity == "positive" else "a disappointing stay"
    topic_text = ", ".join(topics)
    if kind == "chat":
        return (
            f"Write a first-person hotel review of {hotel}. "
            f"The stay should read as {feeling}. "
            f"You may talk about these topics in your own words: {topic_text}. "
            "Use 80 to 120 words. Do not add a title. Do not quote another review. "
            "Do not say that the review was generated."
        )
    return (
        f"Hotel review of {hotel}. The guest describes {feeling}. "
        f"Topics: {topic_text}.\nReview:"
    )


def clean_text(text: str) -> str:
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    marker = "Review:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text.strip(" \"'")


def mentions_hotel(text: str, key: str, hotel: str) -> bool:
    words = set(tokens(text))
    name_words = [word for word in tokens(hotel) if word not in {"hotel", "chicago", "the"} and len(word) > 3]
    return key in words or any(word in words for word in name_words)


def acceptable(text: str, blocked: set[str], seen: set[str], used_first: set[str], key: str, hotel: str) -> bool:
    words = tokens(text)
    if not 60 <= len(words) <= 160:
        return False
    if "i" not in words or not mentions_hotel(text, key, hotel):
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters or sum(char.isascii() and char.isalpha() for char in letters) / len(letters) < 0.9:
        return False
    lowered = text.lower()
    if "as an ai" in lowered or "language model" in lowered or "first-person hotel review" in lowered:
        return False
    if "she said" in lowered or "he said" in lowered or text.count('"') > 2:
        return False
    opening = first_n(text)
    if not opening or opening in blocked or opening in used_first:
        return False
    spans = windows(text)
    if not spans or spans & seen:
        return False
    return True


def generate_batch(model, tokenizer, prompts: list[str], device: torch.device, kind: str) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = prompts
    if kind == "chat":
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
        )
    texts = []
    width = inputs["input_ids"].shape[1]
    for row in output:
        texts.append(clean_text(tokenizer.decode(row[width:], skip_special_tokens=True)))
    return texts


def run_model(
    model_name: str,
    blocked: set[str],
    seen: set[str],
    aspects: dict[str, list[str]],
    limit: int,
    device: torch.device,
    kept: list[dict] | None = None,
    save=None,
) -> list[dict]:
    kind = "base" if model_name == "distilgpt2" else "chat"
    print("loading", model_name, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    names = list(HOTELS)
    kept = list(kept or [])
    used_first = {first_n(row["text"]) for row in kept}
    for row in kept:
        seen.update(windows(row["text"]))
    attempts = len(kept)
    target = limit
    batch_size = 8 if device.type == "cuda" else 1
    while len(kept) < target and attempts < MAX_ATTEMPTS:
        take = min(batch_size, target - len(kept) + batch_size)
        jobs = []
        for offset in range(take):
            index = attempts + offset
            key = names[index % len(names)]
            polarity = "positive" if (index // len(names)) % 2 == 0 else "negative"
            jobs.append((key, polarity))
        attempts += len(jobs)
        prompts = [prompt_for(HOTELS[key], polarity, aspects.get(key, ["room", "staff"]), kind) for key, polarity in jobs]
        try:
            texts = generate_batch(model, tokenizer, prompts, device, kind)
        except RuntimeError as exc:
            print("batch failed", exc, flush=True)
            continue
        for (key, polarity), text in zip(jobs, texts):
            if not acceptable(text, blocked, seen, used_first, key, HOTELS[key]):
                continue
            opening = first_n(text)
            used_first.add(opening)
            seen.update(windows(text))
            kept.append(
                {
                    "text": text,
                    "hotel": HOTELS[key],
                    "hotel_key": key,
                    "polarity": polarity,
                    "generator": model_name,
                    "word_count": len(tokens(text)),
                    "seed": "hotel name, polarity, topic words from real Ott reviews of this hotel",
                }
            )
            if len(kept) >= target:
                break
        if save is not None:
            save(kept)
        print(f"{model_name} kept {len(kept)}/{target} attempts {attempts}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=TARGET)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocked, seen, aspects = load_guards()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.out.exists():
        existing = pd.read_csv(args.out).to_dict("records")
        print(f"resume {args.out} n={len(existing)}", flush=True)

    def write(rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(args.out, index=False)

    rows = existing
    for model_name in args.models:
        mine = [row for row in rows if row.get("generator") == model_name]
        others = [row for row in rows if row.get("generator") != model_name]
        updated = run_model(
            model_name,
            blocked,
            seen,
            aspects,
            args.limit,
            device,
            kept=mine,
            save=lambda kept_rows, base=others: write(base + kept_rows),
        )
        rows = others + updated
        write(rows)
    print(f"wrote {args.out} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

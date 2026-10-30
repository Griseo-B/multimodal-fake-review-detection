"""AI reviews for phones that already have a same-experiment human review.

The prompt is the phone name, polarity, and four fixed topic words. Review
sentences are not included. A candidate is dropped when any 10-word window
already appears in the phone's human reviews or in text already kept.
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SOLDNER = ROOT / "data" / "raw" / "human_written_fake" / "soldner2022" / "review_data.csv"
BLOCK = ROOT / "data" / "frozen" / "two_class.csv"
OUT = ROOT / "data" / "raw" / "phone_ai" / "reviews.csv"
MODELS = (
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("tinyllama-1.1b", str(ROOT / "data" / "raw" / "hotel_ai" / "tinyllama")),
)
TOPICS = ["battery", "camera", "screen", "software"]
FIELDS = ["text", "item", "brand", "phone_type", "polarity", "generator", "word_count"]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def windows(text: str) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[index : index + 10]) for index in range(0, max(0, len(words) - 9))}


def prompt_for(name: str, polarity: str) -> str:
    feeling = "a positive experience" if polarity == "pos" else "a disappointing experience"
    topics = ", ".join(TOPICS)
    return (
        f"Write a first-person review of the {name} phone. "
        f"The experience should read as {feeling}. "
        f"You may talk about these topics in your own words: {topics}. "
        "Use 80 to 120 words. Do not add a title. Do not quote another review. "
        "Do not say that the review was generated."
    )


def acceptable(text: str, name: str, blocked: set[tuple[str, ...]], openings: set[str]) -> bool:
    words = tokens(text)
    if not 60 <= len(words) <= 160:
        return False
    if not any(word in {"i", "i'm", "ive", "i've", "i'd", "i'll"} for word in words):
        return False
    if not any(part in words for part in tokens(name) if len(part) > 2):
        return False
    lowered = text.lower()
    if "as an ai" in lowered or "language model" in lowered:
        return False
    opening = first_n(text)
    if not opening or opening in openings:
        return False
    spans = windows(text)
    return bool(spans) and spans.isdisjoint(blocked)


def choose_phones() -> list[tuple[str, str, str]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    with SOLDNER.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["data_source_label"] != "prolific":
                continue
            key = (row["brand"], row["phone_type"])
            if row["veracity_label"] == "truthful" and row["ownership_label"] == "owner":
                groups[key].add("real")
            elif row["veracity_label"] == "deceptive":
                groups[key].add("human_fake")
    ready = [key for key, labels in groups.items() if "real" in labels and "human_fake" in labels]
    ready.sort()
    random.Random(42).shuffle(ready)
    chosen = ready[:20]
    return [(brand, phone, f"{brand} {phone}") for brand, phone in chosen]


def first_n(text: str) -> str:
    return " ".join(tokens(text)[:10])


def blocked_for(names: set[str]) -> tuple[set[tuple[str, ...]], set[str]]:
    blocked: set[tuple[str, ...]] = set()
    with SOLDNER.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = f"{row['brand']} {row['phone_type']}"
            if name in names:
                blocked.update(windows(row["review_text"]))
    openings: set[str] = set()
    with BLOCK.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            opening = first_n(row["text"])
            if opening:
                openings.add(opening)
    return blocked, openings


def generate_one(model, tokenizer, prompt: str, device: torch.device) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
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
    width = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][width:], skip_special_tokens=True).strip()


def load_kept() -> list[dict]:
    if not OUT.exists():
        return []
    with OUT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(kept: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    phones = choose_phones()
    names = {name for _, _, name in phones}
    blocked, openings = blocked_for(names)
    kept = load_kept()
    done = {(row["generator"], row["item"], row["polarity"]) for row in kept}
    for row in kept:
        blocked.update(windows(row["text"]))
        openings.add(first_n(row["text"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [
        (brand, phone, name, polarity, short, model_name)
        for short, model_name in MODELS
        for brand, phone, name in phones
        for polarity in ("pos", "neg")
        if (short, name, polarity) not in done
    ]
    if not jobs:
        print("wrote", len(kept), OUT, flush=True)
        return
    short, model_name = jobs[0][4], jobs[0][5]
    batch = [job for job in jobs if job[4] == short][:10]
    print("loading", model_name, "jobs", len(batch), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for brand, phone, name, polarity, short, _model_name in batch:
        text = ""
        for attempt in range(2):
            candidate = generate_one(model, tokenizer, prompt_for(name, polarity), device)
            if acceptable(candidate, name, blocked, openings):
                text = candidate
                break
            print("retry", short, name, polarity, attempt, flush=True)
        if not text:
            kept.append(
                {
                    "text": "",
                    "item": name,
                    "brand": brand,
                    "phone_type": phone,
                    "polarity": polarity,
                    "generator": short,
                    "word_count": "0",
                }
            )
            print("dropped", short, name, polarity, flush=True)
        else:
            blocked.update(windows(text))
            openings.add(first_n(text))
            kept.append(
                {
                    "text": text,
                    "item": name,
                    "brand": brand,
                    "phone_type": phone,
                    "polarity": polarity,
                    "generator": short,
                    "word_count": str(len(tokens(text))),
                }
            )
            print("kept", len(kept), short, name, polarity, flush=True)
        save(kept)
    print("batch_done", len(kept), flush=True)


if __name__ == "__main__":
    main()

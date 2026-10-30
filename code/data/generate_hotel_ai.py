"""Generate hotel AI reviews from the hotel name and polarity only.

The prompt does not include any existing review. Build step drops a row if its
first ten words already appear in Ott, AiGen, or Salminen.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mm3_common import HOTEL_AI_CSV, PER_CELL, ROOT, SEED

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
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


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def prompt_for(hotel: str, polarity: str) -> str:
    feeling = "a positive stay" if polarity == "positive" else "a disappointing stay"
    return (
        f"Write a first-person hotel review of {hotel} in Chicago. "
        f"The stay should read as {feeling}. "
        "Use 80 to 120 words. Do not add a title. Do not quote another review. "
        "Do not say that the review was generated."
    )


def generate_one(model, tokenizer, hotel: str, polarity: str, device: torch.device) -> str:
    messages = [{"role": "user", "content": prompt_for(hotel, polarity)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    for _ in range(3):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"\s+", " ", text)
        if 60 <= word_count(text) <= 160:
            return text
    return text


def main() -> None:
    print("loading", MODEL_NAME, flush=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    names = list(HOTELS.values())
    rows = []
    target = PER_CELL + 30
    for index in range(target):
        hotel = names[index % len(names)]
        polarity = "positive" if index % 2 == 0 else "negative"
        text = generate_one(model, tokenizer, hotel, polarity, device)
        rows.append(
            {
                "text": text,
                "hotel": hotel,
                "polarity": polarity,
                "generator": MODEL_NAME,
                "word_count": word_count(text),
            }
        )
        print(f"generated {index + 1}/{target} words={word_count(text)}", flush=True)
    HOTEL_AI_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(HOTEL_AI_CSV, index=False)
    print(f"wrote {HOTEL_AI_CSV} n={len(rows)}")


if __name__ == "__main__":
    main()

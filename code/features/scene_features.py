"""Image-text agreement beyond a whole-review cosine.

Three things the proposal asks for that a global CLIP cosine cannot answer:

scene_match    does the photo show the thing the review names. The image is
               scored zero-shot against a small label set for its domain, and
               the winning label counts as matched when the review mentions
               that label. This is the operational stand-in for entity
               consistency: no NER model is installed, so labels come from a
               fixed keyword list per domain.
sentiment_synergy  is the photo's appeal pointing the same way as the text
               polarity. Image valence comes from a two-prompt softmax.
entity_image_prob  the image score of the labels the review actually names,
               which is stricter than scene_match.

Everything reuses embeddings.npz, so no image is decoded again.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, TABLE  # noqa: E402
from smoothness import embed_texts, load_clip  # noqa: E402

# keyword -> prompt. The keyword is what the review would say; the prompt is
# what CLIP sees, so the two can differ in wording.
SCENES = {
    "food": {
        "pizza": "a photo of a pizza",
        "burger": "a photo of a hamburger",
        "sushi": "a photo of sushi",
        "noodle": "a photo of a bowl of noodles",
        "pasta": "a photo of a plate of pasta",
        "steak": "a photo of a steak",
        "salad": "a photo of a salad",
        "soup": "a photo of a bowl of soup",
        "rice": "a photo of a rice dish",
        "chicken": "a photo of a chicken dish",
        "fish": "a photo of a fish dish",
        "seafood": "a photo of a seafood dish",
        "shrimp": "a photo of a shrimp dish",
        "dessert": "a photo of a dessert",
        "cake": "a photo of a slice of cake",
        "ice cream": "a photo of ice cream",
        "coffee": "a photo of a cup of coffee",
        "tea": "a photo of a cup of tea",
        "beer": "a photo of a glass of beer",
        "wine": "a photo of a glass of wine",
        "bread": "a photo of bread",
        "cheese": "a photo of cheese",
        "fruit": "a photo of fruit",
        "breakfast": "a photo of a breakfast plate",
        "restaurant": "a photo of a restaurant interior",
        "table": "a photo of a set table",
        "menu": "a photo of a menu",
    },
    "hotel": {
        "room": "a photo of a hotel room",
        "bed": "a photo of a bed",
        "bathroom": "a photo of a bathroom",
        "pool": "a photo of a swimming pool",
        "lobby": "a photo of a hotel lobby",
        "balcony": "a photo of a balcony",
        "breakfast": "a photo of a breakfast buffet",
        "view": "a photo of a scenic view",
        "beach": "a photo of a beach",
        "gym": "a photo of a hotel gym",
        "restaurant": "a photo of a hotel restaurant",
        "garden": "a photo of a garden",
        "building": "a photo of a hotel building exterior",
        "shower": "a photo of a shower",
        "towel": "a photo of folded towels",
    },
    "ecommerce": {
        "phone": "a photo of a smartphone",
        "laptop": "a photo of a laptop",
        "computer": "a photo of a computer",
        "headphone": "a photo of headphones",
        "earphone": "a photo of earphones",
        "speaker": "a photo of a speaker",
        "camera": "a photo of a camera",
        "watch": "a photo of a watch",
        "shoe": "a photo of shoes",
        "boot": "a photo of boots",
        "bag": "a photo of a bag",
        "shirt": "a photo of a shirt",
        "dress": "a photo of a dress",
        "jacket": "a photo of a jacket",
        "pants": "a photo of trousers",
        "tablet": "a photo of a tablet computer",
        "keyboard": "a photo of a keyboard",
        "mouse": "a photo of a computer mouse",
        "charger": "a photo of a charger",
        "cable": "a photo of a cable",
        "battery": "a photo of a battery",
        "toy": "a photo of a toy",
        "book": "a photo of a book",
        "bottle": "a photo of a bottle",
    },
}
VALENCE_PROMPTS = {
    "food": ("an appetizing appealing photo of delicious food",
             "an unappetizing unappealing photo of bad food"),
    "hotel": ("an appealing attractive photo of a pleasant place",
              "an unappealing unattractive photo of an unpleasant place"),
    "ecommerce": ("an appealing attractive photo of a desirable product",
                  "an unappealing unattractive photo of an undesirable product"),
}
DEFAULT_VALENCE = (
    "an appealing attractive photo",
    "an unappealing unattractive photo",
)
BATCH = 32
LOGIT_SCALE = 100.0
COLUMNS = [
    "scene_prob", "scene_match", "entity_image_prob", "entity_count",
    "image_valence", "sentiment_synergy", "sentiment_agreement",
]


def _normalise(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-8, None)


def mentioned(text: str, keywords: list[str]) -> list[int]:
    """Indices of keywords named in the review, matched on word boundaries."""
    lowered = str(text).lower()
    hits = []
    for index, keyword in enumerate(keywords):
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            hits.append(index)
    return hits


def embed_prompts(processor, model, prompts: list[str], device) -> np.ndarray:
    return _normalise(embed_texts(processor, model, prompts, device))


def scene_frame(frame: pd.DataFrame, embeddings_path: Path, device) -> tuple[pd.DataFrame, dict]:
    payload = np.load(embeddings_path, allow_pickle=True)
    ids = payload["id"].astype(str)
    labels_seen: dict[str, dict] = {}
    image = _normalise(payload["clip_image"].astype(np.float32))
    order = pd.Index(ids)
    lookup = {value: position for position, value in enumerate(order)}

    processor, model = load_clip(device)
    rows = np.full((len(frame), len(COLUMNS)), np.nan, dtype=np.float32)
    missing = 0
    for row_index, record in enumerate(frame.itertuples(index=False)):
        position = lookup.get(str(record.id))
        if position is None:
            missing += 1
            continue
        domain = str(record.domain)
        scenes = SCENES.get(domain) or SCENES["food"]
        keywords = list(scenes.keys())
        if domain not in labels_seen:
            labels_seen[domain] = keywords
            prompts = embed_prompts(processor, model, [scenes[key] for key in keywords], device)
            positives, negatives = VALENCE_PROMPTS.get(domain, DEFAULT_VALENCE)
            valence = embed_prompts(processor, model, [positives, negatives], device)
            labels_seen[domain] = {"keywords": keywords, "scene": prompts, "valence": valence}
        cached = labels_seen[domain]
        vector = image[position]

        scene_scores = LOGIT_SCALE * (cached["scene"] @ vector)
        shifted = scene_scores - scene_scores.max()
        probability = np.exp(shifted)
        probability /= probability.sum()
        best = int(probability.argmax())
        hits = mentioned(record.text, cached["keywords"])
        valence_scores = LOGIT_SCALE * (cached["valence"] @ vector)
        valence_probability = np.exp(valence_scores - valence_scores.max())
        valence_probability /= valence_probability.sum()

        rows[row_index, 0] = float(probability[best])
        rows[row_index, 1] = float(best in hits)
        if hits:
            rows[row_index, 2] = float(probability[hits].max())
            rows[row_index, 3] = float(len(hits))
        rows[row_index, 4] = float(valence_probability[0])
        rows[row_index, 5] = float(np.sign(record.text_sentiment) * (valence_probability[0] - 0.5))
        rows[row_index, 6] = float(valence_probability[0] > 0.5)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows, columns=COLUMNS, index=frame.index)
    table.insert(0, "id", frame["id"].to_numpy())
    notes = {
        "rows_with_embedding": int(len(frame) - missing),
        "rows_missing_embedding": int(missing),
        "no_entity_named": int(np.isnan(rows[:, 3]).sum()),
        "logit_scale": LOGIT_SCALE,
        "note": "scene_match and entity_image_prob use a fixed keyword list per "
                "domain, not a trained NER model. A row with no keyword named "
                "leaves entity_image_prob empty by design.",
    }
    return table, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/20261008_public_dataset")
    parser.add_argument("--protocol", default="public_dataset")
    args = parser.parse_args()

    folder = ROOT / args.features
    embeddings_path = folder / "embeddings.npz"
    if not embeddings_path.is_file():
        raise SystemExit(f"{embeddings_path} not found; run the align part first")
    alignment = pd.read_parquet(folder / "alignment_features.parquet")
    meta = pd.read_csv(
        TABLE,
        usecols=["id", "domain", "language", "protocol", "text"],
        dtype=str,
        keep_default_na=False,
    )
    meta = meta.loc[meta["protocol"] == args.protocol]
    text = pd.read_parquet(folder / "text_features.parquet", columns=["id", "sentiment", "sentiment_zh"])
    frame = (
        alignment.merge(meta, on="id", how="left")
        .merge(text, on="id", how="left")
        .reset_index(drop=True)
    )
    frame["text_sentiment"] = frame["sentiment"].fillna(frame["sentiment_zh"]).fillna(0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, notes = scene_frame(frame, embeddings_path, device)
    table.to_parquet(folder / "scene_features.parquet", index=False)
    run_path = folder / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    run.setdefault("notes", {})["scene"] = notes
    run["parts"] = sorted(set(run.get("parts", [])) | {"scene"})
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote scene features {table.shape}", flush=True)


if __name__ == "__main__":
    main()

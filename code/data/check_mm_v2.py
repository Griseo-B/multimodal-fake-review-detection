"""Spot-check the frozen v2 table and write a balanced Chicago hotel subset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "frozen" / "mm_v2.csv"
HOTEL3 = ROOT / "data" / "frozen" / "mm_v2_hotel3.csv"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def main() -> None:
    frame = pd.read_csv(TABLE)
    food = frame.loc[frame["subset"] == "food_aigen"].iloc[0]
    image = Image.open(ROOT / food["image_path"])
    print("image", image.size, food["image_role"], int(food["orig_width"]), int(food["orig_height"]))
    print("aug", int(frame["aug_path"].fillna("").astype(bool).sum()))
    print("en pos", frame.loc[frame["language"] == "en", ["pos_noun", "pos_verb"]].mean().round(3).to_dict())
    print("zh pos", frame.loc[frame["language"] == "zh", ["pos_noun", "pos_verb"]].mean().round(3).to_dict())
    ai = frame.loc[frame["subset"] == "hotel_ai"]
    openings = set(frame.loc[frame["subset"] == "hotel_ott", "text"].map(lambda text: " ".join(tokens(text)[:10])))
    print("ai generators", ai["generator"].value_counts().to_dict())
    print("ai polarity", ai["polarity"].value_counts().to_dict())
    print("opening overlap", int(ai["text"].map(lambda text: " ".join(tokens(text)[:10])).isin(openings).sum()))
    print("templates", int((frame["source"] == "researcher_template").sum()))

    rng = np.random.default_rng(42)
    generators = sorted(ai["generator"].dropna().unique())
    per_model = 420 // max(len(generators), 1)
    sampled = []
    for generator in generators:
        part = ai.loc[ai["generator"] == generator]
        chosen = []
        for polarity in ("positive", "negative"):
            side = part.loc[part["polarity"] == polarity]
            take = min(len(side), per_model // 2)
            if take:
                chosen.extend(rng.choice(side.index.to_numpy(), size=take, replace=False).tolist())
        short = per_model - len(chosen)
        if short > 0:
            rest = part.drop(index=chosen, errors="ignore")
            if len(rest):
                chosen.extend(rng.choice(rest.index.to_numpy(), size=min(short, len(rest)), replace=False).tolist())
        sampled.append(frame.loc[chosen])
    picked = sampled or [ai]
    ott = frame.loc[frame["subset"] == "hotel_ott"]
    for label in ("real", "human_fake"):
        for polarity in ("positive", "negative"):
            part = ott.loc[(ott["label"] == label) & (ott["polarity"] == polarity)]
            choice = rng.choice(part.index.to_numpy(), size=210, replace=False)
            picked.append(frame.loc[choice])
    hotel = pd.concat(picked, ignore_index=True)
    hotel["split"] = "train"
    for label in hotel["label"].unique():
        for polarity in hotel.loc[hotel["label"] == label, "polarity"].fillna("").unique():
            mask = (hotel["label"] == label) & (hotel["polarity"].fillna("") == polarity)
            index = hotel.loc[mask].index.to_numpy()
            rng.shuffle(index)
            n_test = max(1, int(round(len(index) * 0.1)))
            n_val = max(1, int(round(len(index) * 0.1)))
            hotel.loc[index[:n_test], "split"] = "test"
            hotel.loc[index[n_test : n_test + n_val], "split"] = "val"
    hotel.to_csv(HOTEL3, index=False)
    print("hotel3", hotel.groupby(["label", "split"]).size().to_dict(), "n", len(hotel))


if __name__ == "__main__":
    main()

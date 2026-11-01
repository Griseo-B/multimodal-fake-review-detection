"""Perplexity under distilgpt2 for the English rows.

Chinese rows are left as NaN on purpose: the table holds 1,740 Chinese reviews
from one subset, and mixing them into an English perplexity comparison would
read as a language effect. The reason code is recorded in run.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, tokens  # noqa: E402

sys.path.insert(0, str(ROOT / "code" / "data"))
from features_mm3 import perplexities  # noqa: E402

MIN_WORDS = 5


def perplexity_column(texts: list[str], languages: list[str], device: torch.device) -> tuple[np.ndarray, dict]:
    values = np.full(len(texts), np.nan, dtype=np.float32)
    keep = [
        index
        for index, (text, language) in enumerate(zip(texts, languages))
        if language == "en" and len(tokens(text)) >= MIN_WORDS
    ]
    skipped = {
        "not_english": int(sum(language != "en" for language in languages)),
        "too_short": int(
            sum(
                language == "en" and len(tokens(text)) < MIN_WORDS
                for text, language in zip(texts, languages)
            )
        ),
        "computed": len(keep),
    }
    if keep:
        values[keep] = perplexities([texts[index] for index in keep], device)
    return values, skipped


def perplexity_frame(frame: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, dict]:
    values, skipped = perplexity_column(
        frame["text"].astype(str).tolist(),
        frame["language"].astype(str).tolist(),
        device,
    )
    table = pd.DataFrame({"id": frame["id"].to_numpy(), "perplexity": values})
    return table, skipped

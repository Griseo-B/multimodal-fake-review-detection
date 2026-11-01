"""Text features for the three-class table.

Readability follows the definitions in code/data/features_mm3.py. Part-of-speech
shares are read from the frozen table instead of being recomputed, so they match
byte for byte. Template features are computed inside one subset and split.

Chinese rows go through jieba and a seed lexicon; English-only formulas and
lexicons stay NaN there and the Chinese columns stay NaN on English rows, so the
two never end up in the same average.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from common import (
    CLAUSE_MARKERS,
    CURRENCY,
    EXTREME,
    FUNCTION_WORDS,
    FIRST_PERSON,
    INTENSIFIERS,
    LATIN_WORD,
    MEASURE_WORDS,
    MODALS,
    NEGATIVE,
    PASSIVE,
    POSITIVE,
    TIME_WORDS,
    ZH_CLAUSE_MARKERS,
    ZH_NEGATIVE,
    ZH_POSITIVE,
    ari,
    flesch,
    gunning_fog,
    has_cjk,
    split_sentences,
    tokens,
)

SHINGLE = 10
# Noun/verb/adj/adv/pronoun shares are already in the frozen table, computed
# when the table was built. Word-class distribution is the "part-of-speech"
# feature the proposal asks for, and reading it through keeps the numbers
# identical to data/frozen.
POS_COLUMNS = ["pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb"]
COLUMNS = [
    "n_words", "n_chars", "avg_word_len", "n_sents", "words_per_sentence",
    "ttr", "hapax_ratio", "excl_ratio", "question_ratio", "upper_word_ratio",
    "digit_ratio", "flesch", "ari", "gunning_fog",
    *POS_COLUMNS,
    "function_word_ratio", "first_person_ratio", "modal_ratio",
    "clause_marker_ratio", "passive_ratio", "comma_per_sentence",
    "sentiment", "sentiment_zh", "zh_idiom_ratio",
    "polarity_gap", "polarity_sign",
    "extreme_ratio", "intensifier_ratio", "number_ratio", "currency_ratio",
    "time_ratio", "measure_ratio", "entity_ratio", "opening_dup",
    "ngram_dup_ratio",
]
MULTIPLIERS = np.array(
    [0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
     0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5, 0x94D049BB133111EB,
     0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3,
     0x589965CC75374CC3],
    dtype=np.uint64,
)


def _words(text: str) -> list[str]:
    """ASCII words for English rows; jieba segments rows with Chinese.

    A row counts as Chinese when it has no ASCII word at all. Mixed rows keep
    the ASCII branch so the English columns stay comparable.
    """
    words = tokens(text)
    if words:
        return words
    import jieba

    return [word for word in jieba.lcut(text) if word.strip() and not word.isspace()]


def _shingle_hashes(words: list[str], vocab: dict[str, int]) -> np.ndarray:
    """Deterministic 64-bit hash for every ten-token window of one review."""
    if len(words) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter(
        (vocab.setdefault(word, len(vocab)) for word in words),
        dtype=np.uint64,
        count=len(words),
    )
    windows = len(words) - SHINGLE + 1
    hashed = np.zeros(windows, dtype=np.uint64)
    # uint64 overflow is intentional; the same numbers appear on every run.
    with np.errstate(over="ignore"):
        for offset in range(SHINGLE):
            hashed += ids[offset : offset + windows] * MULTIPLIERS[offset]
    return hashed


def _text_row(text: str) -> dict:
    words = _words(text)
    raw_words = LATIN_WORD.findall(text)
    chinese = has_cjk(text)
    english = bool(tokens(text)) and not chinese
    count = len(words)
    sentence_list = split_sentences(text, minimum_words=1)
    sentence_count = max(1, len(sentence_list))
    characters = sum(len(word) for word in words)
    numeric = sum(word.replace(".", "", 1).isdigit() for word in words)
    frequencies = Counter(words)
    hapax = sum(1 for value in frequencies.values() if value == 1)
    positive = sum(word in POSITIVE for word in words)
    negative = sum(word in NEGATIVE for word in words)
    zh_positive = sum(word in ZH_POSITIVE for word in words)
    zh_negative = sum(word in ZH_NEGATIVE for word in words)
    capitalised = sum(
        token[:1].isupper()
        for sentence in sentence_list
        for token in sentence.split()[1:]
    )

    def share(number: float) -> float:
        return number / count if count else float("nan")

    def english_share(number: float) -> float:
        return share(number) if english else float("nan")

    def readable(function) -> float:
        return float("nan") if chinese else function(text)

    return {
        "n_words": count,
        "n_chars": characters,
        "avg_word_len": characters / count if count else float("nan"),
        "n_sents": sentence_count,
        "words_per_sentence": count / sentence_count,
        "ttr": len(frequencies) / count if count else float("nan"),
        "hapax_ratio": share(hapax),
        "excl_ratio": text.count("!") / max(1, len(text)),
        "question_ratio": text.count("?") / max(1, len(text)),
        "upper_word_ratio": share(
            sum(word.isupper() and len(word) > 1 for word in raw_words)
        ),
        "digit_ratio": share(numeric),
        "flesch": readable(flesch),
        "ari": readable(ari),
        "gunning_fog": readable(gunning_fog),
        # Closed-class density, first person, and modals: the "word
        # frequency" side of the common text features.
        "function_word_ratio": share(sum(word in FUNCTION_WORDS for word in words))
        if english
        else float("nan"),
        "first_person_ratio": share(sum(word in FIRST_PERSON for word in words))
        if english
        else float("nan"),
        "modal_ratio": share(sum(word in MODALS for word in words)) if english else float("nan"),
        # Sentence complexity: subordinators, passive voice, commas per sentence.
        "clause_marker_ratio": (
            share(sum(word in CLAUSE_MARKERS for word in words))
            if english
            else (share(sum(word in ZH_CLAUSE_MARKERS for word in words)) if chinese else float("nan"))
        ),
        "passive_ratio": (
            len(PASSIVE.findall(text)) / sentence_count if english else float("nan")
        ),
        "comma_per_sentence": text.count(",") / sentence_count,
        "sentiment": share(positive - negative) if english else float("nan"),
        "sentiment_zh": share(zh_positive - zh_negative) if chinese else float("nan"),
        "zh_idiom_ratio": (
            share(sum(len(word) == 4 and all("\u4e00" <= c <= "\u9fff" for c in word)
                      for word in words))
            if chinese
            else float("nan")
        ),
        "extreme_ratio": english_share(sum(word in EXTREME for word in words)),
        "intensifier_ratio": english_share(sum(word in INTENSIFIERS for word in words)),
        "number_ratio": share(numeric),
        "currency_ratio": english_share(sum(word in CURRENCY for word in words)),
        "time_ratio": english_share(sum(word in TIME_WORDS for word in words)),
        "measure_ratio": english_share(sum(word in MEASURE_WORDS for word in words)),
        "entity_ratio": english_share(capitalised),
    }


POLARITY_SIGN = {"pos": 1, "positive": 1, "neg": -1, "negative": -1, "neu": 0}


def _polarity_columns(frame: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
    """Does the text polarity agree with the review's own polarity?

    `polarity` is the star class the review was filed under, so a review filed
    as positive but written with negative words is the review-level version of
    a rating deviation. A user's own rating history is not in the table, so
    this is the only part of 评分偏差 the data can support. Only 12,951 rows
    carry a polarity; the rest stay NaN.
    """
    if "polarity" not in frame.columns:
        return pd.DataFrame(
            {"polarity_gap": np.nan, "polarity_sign": np.nan}, index=frame.index
        )
    sign = frame["polarity"].map(POLARITY_SIGN).astype(float)
    text_sign = np.sign(pd.to_numeric(sentiment, errors="coerce").fillna(0.0))
    known = sign.notna() & (sign != 0) & (text_sign != 0)
    gap = pd.Series(np.nan, index=frame.index, dtype=float)
    gap[known] = (text_sign[known] != sign[known]).astype(float)
    return pd.DataFrame({"polarity_gap": gap, "polarity_sign": sign})


def _template_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Opening reuse and ten-token repetition inside each subset and split.

    Grouping by split as well as subset keeps test rows from borrowing
    duplicate counts from train. The value still describes the corpus rather
    than one review, so it is not deployable for a single incoming review.
    """
    opening = frame["text"].map(lambda text: " ".join(_words(text)[:SHINGLE]))
    frames = []
    for _, part in frame.groupby(["subset", "split"], sort=False):
        counts = opening.loc[part.index].value_counts()
        opening_dup = opening.loc[part.index].map(counts) - 1
        vocab: dict[str, int] = {}
        per_row = [_shingle_hashes(_words(text), vocab) for text in part["text"].tolist()]
        lengths = np.fromiter((len(item) for item in per_row), dtype=np.int64, count=len(per_row))
        ratio = np.zeros(len(part), dtype=np.float64)
        if lengths.sum():
            flat = np.concatenate(per_row)
            _, inverse, occurrences = np.unique(flat, return_inverse=True, return_counts=True)
            repeated = occurrences[inverse] > 1
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            for position in range(len(part)):
                start, stop = offsets[position], offsets[position + 1]
                if stop > start:
                    ratio[position] = repeated[start:stop].mean()
        frames.append(
            pd.DataFrame(
                {"opening_dup": opening_dup.to_numpy(), "ngram_dup_ratio": ratio},
                index=part.index,
            )
        )
    return pd.concat(frames).reindex(frame.index)


def text_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_text_row(str(text)) for text in frame["text"].tolist()]
    out = pd.DataFrame(rows, index=frame.index)
    template = _template_columns(frame)
    out["opening_dup"] = template["opening_dup"].fillna(0).astype(int)
    out["ngram_dup_ratio"] = template["ngram_dup_ratio"].astype(float)
    sentiment = out["sentiment"].fillna(out["sentiment_zh"])
    polarity = _polarity_columns(frame, sentiment)
    out["polarity_gap"] = polarity["polarity_gap"].to_numpy()
    out["polarity_sign"] = polarity["polarity_sign"].to_numpy()
    for name in POS_COLUMNS:
        if name in frame.columns:
            out[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy()
    return out.reindex(columns=COLUMNS)

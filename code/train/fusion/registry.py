"""One dictionary for every feature in the fusion model.

Before this file the two consumers disagreed: `code/features/baseline_models.py`
fit on 39 numeric columns while `code/features/build_cards.py` wrote 40 fields,
and neither marked which columns describe the corpus rather than the review.
The registry is the single place where that is written down, so a new column
cannot enter the model without an entry saying what it is.

Fields
    block       which input group the column belongs to; the fusion model gates
                over blocks, so a block boundary is also an attention unit.
    kind        per_review can be computed for one incoming review;
                corpus_level needs the rest of the corpus and cannot be served.
    modality    text / image / align / llm.
    language    en, zh, or all. A NaN here means the column is not defined for
                that language, not that the review scored zero.
    risk        True when the column separated real and AI text through how the
                dataset was built (readability and perplexity of one generator)
                rather than through fakery. Reported separately, never silently
                dropped.
    note        one line for the report.
"""

from __future__ import annotations

from dataclasses import dataclass

TEXT_BLOCK = "text_hand"
LLM_BLOCK = "llm"
CLIP_TEXT_BLOCK = "clip_text"
IMAGE_BLOCK = "image_stats"
CLIP_IMAGE_BLOCK = "clip_image"
RESNET_BLOCK = "resnet"

BLOCKS = [
    TEXT_BLOCK, LLM_BLOCK, CLIP_TEXT_BLOCK,
    IMAGE_BLOCK, CLIP_IMAGE_BLOCK, RESNET_BLOCK,
]


@dataclass(frozen=True)
class Feature:
    name: str
    block: str
    kind: str = "per_review"
    modality: str = "text"
    language: str = "all"
    risk: bool = False
    note: str = ""


def _text(name, language="all", risk=False, note="", kind="per_review"):
    return Feature(name, TEXT_BLOCK, kind, "text", language, risk, note)


# Readability and perplexity are flagged: on food, Flesch reached AUC 0.040 and
# on ecommerce, perplexity 0.030. Both are single-generator artefacts.
TEXT_FEATURES = [
    _text("n_words", note="length; deployable"),
    _text("n_chars", note="length; deployable"),
    _text("avg_word_len", risk=True, note="same evidence as the readability trio"),
    _text("n_sents"),
    _text("words_per_sentence"),
    _text("ttr", note="lexical diversity"),
    _text("hapax_ratio", note="one-off words"),
    _text("excl_ratio"),
    _text("question_ratio"),
    _text("upper_word_ratio"),
    _text("digit_ratio"),
    _text("flesch", risk=True, note="food AUC 0.040; generator artefact"),
    _text("ari", risk=True, note="syllable based, collinear with Flesch"),
    _text("gunning_fog", risk=True, note="same cluster as ARI"),
    _text("pos_noun", note="read from the frozen table"),
    _text("pos_adj", note="read from the frozen table"),
    _text("pos_adv", note="read from the frozen table"),
    _text("pos_pron", note="read from the frozen table"),
    _text("pos_verb", note="read from the frozen table"),
    _text("function_word_ratio", language="en"),
    _text("first_person_ratio", language="en"),
    _text("modal_ratio", language="en"),
    _text("clause_marker_ratio"),
    _text("passive_ratio", language="en", note="no signal, AUC 0.514"),
    _text("comma_per_sentence"),
    _text("sentiment", language="en"),
    _text("sentiment_zh", language="zh"),
    _text("zh_idiom_ratio", language="zh"),
    _text("polarity_gap", note="review-level rating deviation; AUC 0.45, no use"),
    _text("polarity_sign"),
    _text("extreme_ratio", language="en"),
    _text("intensifier_ratio", language="en"),
    _text("number_ratio"),
    _text("currency_ratio", language="en"),
    _text("time_ratio", language="en"),
    _text("measure_ratio", language="en"),
    _text("entity_ratio", language="en", note="capitalised words, not true NER"),
    _text("opening_dup", kind="corpus_level", note="cannot be computed for one review"),
    _text("ngram_dup_ratio", kind="corpus_level", note="corpus-level repetition"),
    _text("perplexity", language="en", risk=True, note="distilgpt2; ecommerce AUC 0.030"),
    _text("smoothness", note="mean cosine between neighbouring sentences"),
]

IMAGE_FEATURES = [
    Feature("brightness", IMAGE_BLOCK, modality="image"),
    Feature("saturation", IMAGE_BLOCK, modality="image"),
    Feature("contrast", IMAGE_BLOCK, modality="image"),
    Feature("lap_var", IMAGE_BLOCK, modality="image"),
    Feature("edge_density", IMAGE_BLOCK, modality="image"),
    Feature("colorfulness", IMAGE_BLOCK, modality="image"),
    Feature("gray_entropy", IMAGE_BLOCK, modality="image"),
    Feature("fft_high_ratio", IMAGE_BLOCK, modality="image"),
    Feature("saliency_offset", IMAGE_BLOCK, modality="image"),
    Feature("third_rule_score", IMAGE_BLOCK, modality="image"),
    Feature("border_energy_ratio", IMAGE_BLOCK, modality="image"),
    Feature("lr_asymmetry", IMAGE_BLOCK, modality="image"),
    Feature("scene_prob", IMAGE_BLOCK, modality="align"),
    Feature("scene_match", IMAGE_BLOCK, modality="align"),
    Feature("entity_image_prob", IMAGE_BLOCK, modality="align"),
    Feature("entity_count", IMAGE_BLOCK, modality="align"),
    Feature("image_valence", IMAGE_BLOCK, modality="align"),
    Feature("sentiment_synergy", IMAGE_BLOCK, modality="align"),
    Feature("sentiment_agreement", IMAGE_BLOCK, modality="align"),
    Feature("clip_cosine", IMAGE_BLOCK, modality="align"),
    Feature("sent_max_cosine", IMAGE_BLOCK, modality="align"),
    Feature("sent_mean_cosine", IMAGE_BLOCK, modality="align"),
]

REGISTRY: dict[str, Feature] = {
    item.name: item for item in TEXT_FEATURES + IMAGE_FEATURES
}

# Width is filled when the embedding file is read; 512 and 2048 are fixed by
# the encoders, the LLM width depends on which checkpoint was used.
EMBEDDING_BLOCKS = {
    LLM_BLOCK: {"dim": None, "modality": "text"},
    CLIP_TEXT_BLOCK: {"dim": 512, "modality": "text"},
    CLIP_IMAGE_BLOCK: {"dim": 512, "modality": "image"},
    RESNET_BLOCK: {"dim": 2048, "modality": "image"},
}


def columns(block: str, include_corpus: bool = False, include_risk: bool = True) -> list[str]:
    """Column names for one block, filtered by what a deployment could serve."""
    out = []
    for item in REGISTRY.values():
        if item.block != block:
            continue
        if item.kind == "corpus_level" and not include_corpus:
            continue
        if item.risk and not include_risk:
            continue
        out.append(item.name)
    return out


def describe() -> dict:
    """Serialisable form for run.json."""
    return {
        name: {
            "block": item.block,
            "kind": item.kind,
            "modality": item.modality,
            "language": item.language,
            "risk": item.risk,
            "note": item.note,
        }
        for name, item in REGISTRY.items()
    }

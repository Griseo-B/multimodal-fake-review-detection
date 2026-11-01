"""Turn the numeric feature tables into one JSON line per review.

A card holds the same features the detector will later see, split into
low / mid / high buckets. Bucket cuts are fit on split=train only, per
domain and language, and stored in schema.json. Cards carry no label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, TABLE  # noqa: E402

TEXT_FIELDS = [
    "perplexity", "flesch", "ari", "gunning_fog", "ttr", "n_words",
    "pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb",
    "clause_marker_ratio", "sentiment", "sentiment_zh", "extreme_ratio",
    "number_ratio", "entity_ratio", "opening_dup", "ngram_dup_ratio",
    "smoothness",
]
IMAGE_FIELDS = [
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio",
    "saliency_offset", "third_rule_score", "border_energy_ratio", "lr_asymmetry",
    "image_valence", "sentiment_synergy",
]
ALIGN_FIELDS = [
    "clip_cosine", "sent_max_cosine", "sent_mean_cosine",
    "scene_prob", "scene_match", "entity_image_prob",
]
LOW_HIGH = (1 / 3, 2 / 3)
# Polarity is signed data, not a spread: terciles on a mostly-zero column
# would label neutral reviews as negative. These use the sign instead.
SIGN_BUCKET = {"sentiment", "sentiment_zh"}
DESCRIPTIONS = {
    "perplexity": "Perplexity under distilgpt2; lower means more predictable text.",
    "flesch": "Flesch reading ease; higher means easier text.",
    "ari": "Automated Readability Index; higher means harder text.",
    "gunning_fog": "Gunning Fog index; share of words with three or more syllables.",
    "ttr": "Type-token ratio; lexical diversity.",
    "n_words": "Word count.",
    "pos_noun": "Noun share of tokens, read from the frozen table.",
    "pos_adj": "Adjective share of tokens, read from the frozen table.",
    "pos_adv": "Adverb share of tokens, read from the frozen table.",
    "pos_pron": "Pronoun share of tokens, read from the frozen table.",
    "pos_verb": "Verb share of tokens, read from the frozen table.",
    "sentiment": "Lexicon polarity in [-1, 1].",
    "extreme_ratio": "Share of extreme or absolutist words.",
    "number_ratio": "Share of numeric tokens; a proxy for concrete detail.",
    "entity_ratio": "Share of capitalised non-initial words; a proxy for named entities.",
    "opening_dup": "Other reviews in the same subset that start with the same ten words.",
    "ngram_dup_ratio": "Share of ten-word windows repeated elsewhere in the same subset.",
    "smoothness": "Mean cosine between neighbouring sentence embeddings.",
    "brightness": "Mean HSV value.",
    "saturation": "Mean HSV saturation.",
    "contrast": "Standard deviation of HSV value.",
    "lap_var": "Variance of the Laplacian; sharpness.",
    "edge_density": "Share of pixels with a strong gradient.",
    "colorfulness": "Hasler-Suesstrunk colourfulness metric.",
    "gray_entropy": "Normalised entropy of the grey histogram.",
    "fft_high_ratio": "High-frequency energy share; texture uniformity.",
    "clip_cosine": "CLIP cosine between the whole review and its image.",
    "sent_max_cosine": "Best sentence-level CLIP cosine against the image.",
    "sent_mean_cosine": "Mean sentence-level CLIP cosine against the image.",
    "clause_marker_ratio": "Share of subordinators and coordinators; sentence complexity.",
    "sentiment_zh": "Polarity from a seed Chinese lexicon in [-1, 1]; empty on English rows.",
    "saliency_offset": "Distance of the gradient-energy centroid from the image centre.",
    "third_rule_score": "Largest energy share among the four thirds windows.",
    "border_energy_ratio": "Share of gradient energy in the outer border band.",
    "lr_asymmetry": "Mean absolute difference against the mirror image; low means symmetric.",
    "image_valence": "Two-prompt CLIP appeal score of the photo in [0, 1].",
    "sentiment_synergy": "Text polarity times image appeal minus 0.5; positive means aligned.",
    "scene_prob": "Zero-shot CLIP confidence in the winning scene label.",
    "scene_match": "1 when the winning scene label is named in the review.",
    "entity_image_prob": "Image score of the strongest entity the review names; empty if none.",
}
# Chinese phrases for the one-line description. Buckets are the only input, so
# the sentence carries no value the numbers do not already carry.
PHRASES = {
    "perplexity": ("困惑度低", "困惑度居中", "困惑度高"),
    "flesch": ("可读性低", "可读性居中", "可读性高"),
    "ari": ("文本偏难", "难度居中", "文本偏易"),
    "gunning_fog": ("长词占比高", "长词占比居中", "长词占比低"),
    "ttr": ("用词重复", "用词多样性居中", "用词多样"),
    "pos_noun": ("名词偏少", "名词占比居中", "名词偏多"),
    "pos_adv": ("副词偏少", "副词占比居中", "副词偏多"),
    "pos_pron": ("代词偏少", "代词占比居中", "代词偏多"),
    "sentiment": ("情感偏负", "情感中性", "情感偏正"),
    "sentiment_zh": ("情感偏负", "情感中性", "情感偏正"),
    "extreme_ratio": ("少有极端措辞", "极端措辞居中", "极端措辞多"),
    "number_ratio": ("细节数字少", "细节数字居中", "细节数字多"),
    "entity_ratio": ("少提具体名物", "具体名物居中", "常提具体名物"),
    "clause_marker_ratio": ("句式简单", "句式复杂度居中", "句式复杂"),
    "ngram_dup_ratio": ("无明显套句", "有少量套句", "套句多"),
    "opening_dup": ("开头无重复", "开头少量重复", "开头大量重复"),
    "smoothness": ("句间跳跃", "句间连贯居中", "句间连贯"),
    "brightness": ("画面偏暗", "亮度居中", "画面偏亮"),
    "colorfulness": ("色彩寡淡", "色彩居中", "色彩浓"),
    "contrast": ("对比弱", "对比居中", "对比强"),
    "lap_var": ("画面偏糊", "清晰度居中", "画面锐利"),
    "fft_high_ratio": ("纹理均匀", "纹理居中", "纹理杂乱"),
    "saliency_offset": ("主体居中", "主体位置居中", "主体偏离中心"),
    "clip_cosine": ("图文相关低", "图文相关居中", "图文相关高"),
    "scene_match": ("图文场景不一致", "图文场景部分一致", "图文场景一致"),
    "image_valence": ("图片观感差", "图片观感居中", "图片观感好"),
}


def cuts(series: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 30:
        return None
    low, high = np.quantile(values, LOW_HIGH)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return float(low), float(high)


def bucket(value: float, limits: tuple[float, float] | None) -> str | None:
    if limits is None or not np.isfinite(value):
        return None
    low, high = limits
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def sign_bucket(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value < 0:
        return "low"
    if value > 0:
        return "high"
    return "mid"


# One feature per redundant cluster, so the sentence does not repeat itself:
# flesch stands for ari and gunning_fog, ttr for hapax, and so on.
TEXT_TALK = [
    "perplexity", "flesch", "ttr", "pos_noun", "pos_adv", "pos_pron",
    "clause_marker_ratio", "sentiment", "sentiment_zh", "extreme_ratio",
    "number_ratio", "ngram_dup_ratio", "smoothness",
]
IMAGE_TALK = ["colorfulness", "contrast", "lap_var", "image_valence"]
ALIGN_TALK = ["clip_cosine", "scene_match"]


def describe(card: dict) -> str:
    """Two short Chinese sentences from buckets, for prompt use.

    Buckets are the only input, so the sentence carries no number the card does
    not already carry, and it never mentions a label.
    """

    def phrase(section: str, name: str) -> str | None:
        item = (card.get(section) or {}).get(name)
        text = PHRASES.get(name)
        if not item or not text:
            return None
        bucket_name = item.get("bucket")
        if bucket_name not in ("low", "mid", "high"):
            return None
        return text[{"low": 0, "mid": 1, "high": 2}[bucket_name]]

    text_parts = [phrase("text", name) for name in TEXT_TALK]
    image_parts = [phrase("image", name) for name in IMAGE_TALK]
    image_parts += [phrase("alignment", name) for name in ALIGN_TALK]
    sentences = []
    if any(text_parts):
        sentences.append("文本：" + "、".join(part for part in text_parts if part) + "。")
    if any(image_parts):
        sentences.append("图文：" + "、".join(part for part in image_parts if part) + "。")
    return "".join(sentences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+",
                        default=["data/processed/features/20261030_public_dataset"])
    parser.add_argument("--out", default="data/processed/features/cards")
    parser.add_argument("--table", default="")
    parser.add_argument("--protocol", default="public_dataset",
                        help="keep one protocol; 'any' keeps every protocol")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--name", default="public_dataset",
                        help="file stem under --out and the label in schema.json")
    args = parser.parse_args()

    folders = [ROOT / part for part in args.features]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table_path = ROOT / args.table if args.table else TABLE
    header = pd.read_csv(table_path, nrows=0).columns
    usecols = [name for name in
               ["id", "subset", "domain", "label", "language", "protocol", args.split_column]
               if name in header]
    frame = pd.read_csv(table_path, usecols=usecols, dtype=str, keep_default_na=False)
    if args.protocol != "any":
        frame = frame.loc[frame["protocol"] == args.protocol].reset_index(drop=True)
    if args.split_column not in frame.columns:
        frame[args.split_column] = "train"
    for filename in ("text_features.parquet", "perplexity.parquet", "smoothness.parquet",
                     "visual_features.parquet", "alignment_features.parquet",
                     "scene_features.parquet"):
        parts = [pd.read_parquet(folder / filename) for folder in folders
                 if (folder / filename).is_file()]
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        merged["id"] = merged["id"].astype(str)
        frame = frame.merge(merged.drop_duplicates("id", keep="first"), on="id", how="left")

    available = [name for name in TEXT_FIELDS + IMAGE_FIELDS + ALIGN_FIELDS if name in frame.columns]
    train = frame.loc[frame[args.split_column] == "train"]
    schema: dict[str, dict] = {}
    for (domain, language), part in train.groupby(["domain", "language"]):
        key = f"{domain}|{language}"
        schema[key] = {}
        for name in available:
            if name in SIGN_BUCKET:
                continue
            limits = cuts(part[name]) if name in part.columns else None
            if limits is not None:
                schema[key][name] = {"low": round(limits[0], 4), "high": round(limits[1], 4)}

    written = 0
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame.itertuples(index=False):
            key = f"{record.domain}|{record.language}"
            limits_for_row = schema.get(key, {})
            text = {}
            image = {}
            align = {}
            for name in TEXT_FIELDS:
                if name not in available:
                    continue
                raw_value = getattr(record, name)
                if name in SIGN_BUCKET:
                    value = None if pd.isna(raw_value) else round(float(raw_value), 4)
                    text[name] = {"value": value, "bucket": sign_bucket(float(raw_value))}
                    continue
                limits = limits_for_row.get(name)
                if limits is None or pd.isna(raw_value):
                    value = None if pd.isna(raw_value) else round(float(raw_value), 4)
                    text[name] = {"value": value, "bucket": None}
                else:
                    value = round(float(raw_value), 4)
                    text[name] = {"value": value, "bucket": bucket(value, tuple(limits.values()))}
            has_image = pd.notna(getattr(record, "brightness", np.nan)) if "brightness" in available else False
            if has_image:
                for name in IMAGE_FIELDS:
                    limits = limits_for_row.get(name)
                    value = round(float(getattr(record, name)), 4)
                    image[name] = {"value": value,
                                   "bucket": bucket(value, tuple(limits.values()) if limits else None)}
                for name in ALIGN_FIELDS:
                    raw_value = getattr(record, name)
                    limits = limits_for_row.get(name)
                    if limits is None or pd.isna(raw_value):
                        value = None if pd.isna(raw_value) else round(float(raw_value), 4)
                        align[name] = {"value": value, "bucket": None}
                    else:
                        value = round(float(raw_value), 4)
                        align[name] = {"value": value, "bucket": bucket(value, tuple(limits.values()))}
            card = {
                "id": record.id,
                "domain": record.domain,
                "language": record.language,
                "subset": record.subset,
                "text": text,
            }
            if image:
                card["image"] = image
                card["alignment"] = align
            card["description"] = describe(card)
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
            written += 1

    document = {
        "features_dir": [str(Path(part)) for part in args.features],
        "cards": written,
        "source_table": str(table_path.relative_to(ROOT)),
        "protocol": args.protocol,
        "split_column": args.split_column,
        "labels_included": False,
        "threshold_source": "split=train, terciles per domain and language; "
                            "sentiment and sentiment_zh use the sign instead",
        "fields": {name: DESCRIPTIONS[name] for name in available if name in DESCRIPTIONS},
        "thresholds": schema,
    }
    (out / f"{args.name}_schema.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.name == "public_dataset":
        # Older notes and reports point at cards/schema.json; keep that name live.
        (out / "schema.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {written} cards to {out}", flush=True)


if __name__ == "__main__":
    main()

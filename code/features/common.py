"""Shared constants and helpers for the feature extraction stage.

Text helpers are imported from code/data/features_mm3.py so the readability
and tokenisation definitions stay identical to the 10-27 experiment. Two
readability formulas that file does not have, ARI and Gunning-Fog, are added
here.

Chinese rows get their own handling: jieba segmentation, a seed sentiment
lexicon, and a character-based sentence rule. English-only formulas stay NaN
on Chinese text, because a handful of Latin tokens would otherwise produce
readability values near 115.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "data"))

from features_mm3 import NEGATIVE, POSITIVE, flesch, syllables, tokens  # noqa: E402

SEED = 42
TABLE = ROOT / "data" / "frozen" / "three_class.csv"
FEATURE_ROOT = ROOT / "data" / "processed" / "features"
AIGEN_RAW = ROOT / "data" / "raw" / "aigen_foodreview" / "data"

LABELS = ["real", "human_fake", "ai_fake"]
# Only "paired" means the image belongs to this review. brand_pool and
# assignment attach an existing brand or Wikimedia photo to a review, so the
# image says nothing about that review and must stay out of image features.
PAIRED_LINKS = {"paired"}
IMAGE_COLUMNS = ("image_224_path", "image_path")

SENTENCE_SPLIT = re.compile(r"[.!?。！？\n]+")
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z']+")
# Chinese sentences need a character floor, not a token floor: after jieba a
# clause like "很好吃" is one token.
CJK_MIN_CHARS = 8

TIME_WORDS = {
    "today", "yesterday", "tonight", "morning", "afternoon", "evening",
    "week", "weekend", "month", "night", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "hour", "minute", "day", "year",
}
MEASURE_WORDS = {
    "minute", "minutes", "hour", "hours", "km", "mile", "miles", "meter",
    "meters", "gram", "grams", "kg", "ml", "liter", "liters", "ounce",
    "ounces", "inch", "inches", "cm", "mm", "degree", "degrees", "percent",
}
CURRENCY = {"$", "usd", "eur", "rmb", "cny", "dollar", "dollars", "euro",
            "euros", "yuan", "pound", "pounds"}
EXTREME = {
    "absolutely", "totally", "extremely", "completely", "perfectly",
    "amazing", "amazingly", "perfect", "terrible", "horrible", "awful",
    "worst", "best", "never", "always", "everyone", "everything", "nothing",
    "incredible", "incredibly", "fantastic", "disgusting", "delicious",
    "outstanding", "exceptional", "unbelievable", "ridiculous", "favorite",
}
INTENSIFIERS = {"very", "really", "so", "too", "quite", "highly", "super",
                "extremely", "totally", "absolutely", "completely"}

# Closed-class words. Written out here instead of using the nltk corpus,
# which is not installed on this machine; the list is the standard 120-word
# closed class and is checked for duplicates at import.
FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our",
                "ours", "ourselves"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should",
          "will", "would", "ought"}

# Subordinators and coordinators; their density is the "sentence complexity"
# the proposal asks for, next to clause length.
CLAUSE_MARKERS = {
    "because", "although", "though", "while", "whereas", "unless", "since",
    "if", "when", "whenever", "after", "before", "until", "that", "which",
    "who", "whom", "whose", "where", "whether", "despite", "however",
    "therefore", "moreover", "furthermore", "nevertheless", "besides",
    "and", "but", "or", "yet",
}
PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b",
    re.IGNORECASE,
)
# Seed lexicon for Chinese. It is small on purpose: no Chinese sentiment
# package is installed, and a hand-written list of common review words is
# enough to get a polarity sign. Coverage is recorded in run.json.
ZH_POSITIVE = {
    "好吃", "美味", "不错", "很好", "推荐", "满意", "干净", "舒适", "方便",
    "实惠", "划算", "新鲜", "精致", "热情", "周到", "贴心", "喜欢", "惊喜",
    "值得", "舒服", "漂亮", "大气", "快捷", "专业", "友好", "耐心", "整洁",
    "香", "嫩", "鲜", "棒", "赞", "完美", "优秀", "出色", "便利", "丰富",
    "宽敞", "安静", "用心", "细致", "地道", "入味", "划算", "超值",
}
ZH_NEGATIVE = {
    "难吃", "很差", "失望", "糟糕", "脏", "吵", "慢", "贵", "坑", "差",
    "烂", "恶心", "异味", "不新鲜", "冷漠", "敷衍", "拖沓", "后悔", "投诉",
    "退款", "欺骗", "虚假", "劣质", "破旧", "拥挤", "嘈杂", "粗糙", "油腻",
    "咸", "太甜", "生", "焦", "脏乱", "难闻", "不推荐", "一般", "鸡肋",
    "态度差", "不卫生", "上当", "坑人", "缩水", "变质", "硬", "干", "淡",
}
ZH_CLAUSE_MARKERS = {
    "因为", "所以", "但是", "但", "而且", "虽然", "如果", "不过", "然后",
    "并且", "以及", "而", "却", "即使", "由于", "因此", "另外", "同时",
    "于是", "反而",
}


def has_cjk(text: str) -> bool:
    return bool(CJK.search(str(text)))


def project_path(part: str) -> Path:
    return ROOT / str(part).replace("\\", "/")


def image_path_of(row) -> str:
    """Prefer the 224 copy; fall back to the original when it is missing."""
    for column in IMAGE_COLUMNS:
        value = str(getattr(row, column, "") or "").strip()
        if value and project_path(value).is_file():
            return value
    return ""


def split_sentences(text: str, minimum_words: int = 4) -> list[str]:
    """Clause list, with an English token floor and a Chinese character floor."""
    parts = [part.strip() for part in SENTENCE_SPLIT.split(str(text)) if part.strip()]
    kept = []
    for part in parts:
        if has_cjk(part):
            if len(CJK.findall(part)) >= CJK_MIN_CHARS:
                kept.append(part)
        elif len(part.split()) >= minimum_words:
            kept.append(part)
    return kept


def sentences(text: str, minimum_words: int = 4, limit: int = 8) -> list[str]:
    return split_sentences(text, minimum_words)[:limit]


def ari(text: str) -> float:
    """Automated Readability Index; mirrors textstat.automated_readability_index."""
    words = tokens(text)
    if not words:
        return float("nan")
    characters = sum(len(word) for word in words)
    sentence_count = max(1, len(split_sentences(text, minimum_words=0)))
    return 4.71 * (characters / len(words)) + 0.5 * (len(words) / sentence_count) - 21.43


def gunning_fog(text: str) -> float:
    """Gunning Fog; complex words carry three or more syllables."""
    words = tokens(text)
    if not words:
        return float("nan")
    complex_words = sum(syllables(word) >= 3 for word in words)
    sentence_count = max(1, len(split_sentences(text, minimum_words=0)))
    return 0.4 * (len(words) / sentence_count + 100 * complex_words / len(words))


def label_binary(label: str) -> int:
    """1 for ai_fake, 0 for real, NaN for human_fake (not used in this stage)."""
    if label == "ai_fake":
        return 1
    if label == "real":
        return 0
    return -1

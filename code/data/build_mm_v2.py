"""Build the larger frozen multimodal table from audited public sets.

Researcher-written templates are not included. Hotel and ecommerce rows stay
text-only until every class in that slice has its own image. Food images are
resized to 224 and the training split gets one flipped, recolored copy.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from mm3_common import ROOT, SEED

OUT = ROOT / "data" / "frozen" / "mm_v2.csv"
AUDIT = ROOT / "data" / "frozen" / "mm_v2_audit.json"
IMAGE_DIR = ROOT / "data" / "processed" / "mm_v2" / "images"
AUG_DIR = ROOT / "data" / "processed" / "mm_v2" / "aug"
HOTEL_AI = ROOT / "data" / "raw" / "hotel_ai" / "reviews_v2.csv"
PRODUCT_CATS = [
    "Home_and_Kitchen_5",
    "Electronics_5",
    "Clothing_Shoes_and_Jewelry_5",
    "Toys_and_Games_5",
    "Pet_Supplies_5",
    "Sports_and_Outdoors_5",
    "Tools_and_Home_Improvement_5",
]
COLUMNS = [
    "subset", "domain", "label", "language", "text", "image_path", "aug_path",
    "orig_width", "orig_height", "image_role", "group_id", "source", "generator",
    "polarity", "label_quality", "split", "n_tokens", "pos_noun", "pos_adj",
    "pos_adv", "pos_pron", "pos_verb",
]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def first_n(text: str, count: int = 10) -> str:
    return " ".join(tokens(text)[:count])


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["usable"].astype(str).isin(["True", "true", "1"])].copy()


def blank_row() -> dict:
    return {column: "" for column in COLUMNS}


def save_square(source: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        rgb.save(dest, "JPEG", quality=90)


def augment(source: Path, dest: Path, seed: int) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with Image.open(source) as image:
        rgb = image.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
        rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(0.9, 1.1)))
        rgb = ImageEnhance.Color(rgb).enhance(float(rng.uniform(0.9, 1.1)))
        rgb.save(dest, "JPEG", quality=90)


def assign_within(frame: pd.DataFrame, by_cols: list[str], group_col: str, rng: np.random.Generator) -> pd.DataFrame:
    frame = frame.copy()
    frame["split"] = "train"
    for _, part in frame.groupby(by_cols, dropna=False):
        groups = part[group_col].drop_duplicates().tolist()
        rng.shuffle(groups)
        count = len(groups)
        n_test = max(1, int(round(count * 0.1)))
        n_val = max(1, int(round(count * 0.1)))
        if n_test + n_val >= count:
            n_test, n_val = 1, 1
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        group_values = frame.loc[part.index, group_col]
        frame.loc[group_values.index[group_values.isin(test_groups)], "split"] = "test"
        frame.loc[group_values.index[group_values.isin(val_groups)], "split"] = "val"
    return frame


def repair_split(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Move one group so validation and test each contain every label that exists."""
    frame = frame.copy()
    labels = sorted(frame["label"].unique())
    for split in ("val", "test"):
        for label in labels:
            present = set(frame.loc[frame["split"] == split, "label"])
            if label in present:
                continue
            donors = frame.loc[(frame["split"] == "train") & (frame["label"] == label), group_col].drop_duplicates()
            if donors.empty:
                continue
            group = donors.iloc[0]
            frame.loc[frame[group_col] == group, "split"] = split
    return frame


def food_rows() -> list[dict]:
    frame = usable(pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv"))
    rows = []
    for record in frame.itertuples(index=False):
        source = ROOT / str(record.image_path)
        dest = IMAGE_DIR / f"food_{record.source_id}.jpg"
        save_square(source, dest)
        row = blank_row()
        row.update(
            {
                "subset": "food_aigen",
                "domain": "food",
                "label": record.label,
                "language": "en",
                "text": str(record.text),
                "image_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "orig_width": int(record.image_width),
                "orig_height": int(record.image_height),
                "image_role": "paired_dalle" if record.label == "ai_fake" else "paired_review",
                "group_id": f"aigen-{record.source_id}",
                "source": "aigen",
                "generator": "gpt-4-turbo+dalle-2" if record.label == "ai_fake" else "",
                "polarity": "",
                "label_quality": "gold",
                "split": record.split,
            }
        )
        rows.append(row)
        if len(rows) % 2000 == 0:
            print(f"food images {len(rows)}", flush=True)
    return rows


def hotel_ott_rows() -> list[dict]:
    frame = usable(pd.read_csv(ROOT / "data" / "frozen" / "ott_hotel.csv"))
    rows = []
    for record in frame.itertuples(index=False):
        row = blank_row()
        row.update(
            {
                "subset": "hotel_ott",
                "domain": "hotel",
                "label": record.label,
                "language": "en",
                "text": str(record.text),
                "image_role": "none",
                "group_id": f"ott-{record.source_id}",
                "source": "ott",
                "generator": "",
                "polarity": record.polarity,
                "label_quality": "gold",
                "split": record.split,
            }
        )
        rows.append(row)
    return rows


def hotel_ai_rows(blocked_windows: set[str]) -> list[dict]:
    path = HOTEL_AI if HOTEL_AI.exists() else HOTEL_AI.with_name("reviews_v2_qwen.csv")
    if not path.exists():
        print("hotel AI file missing, slice skipped", flush=True)
        return []
    frame = pd.read_csv(path)
    rows = []
    for index, record in enumerate(frame.itertuples(index=False)):
        text = str(record.text)
        spans = {" ".join(tokens(text)[start : start + 10]) for start in range(max(0, len(tokens(text)) - 9))}
        if spans & blocked_windows:
            continue
        row = blank_row()
        row.update(
            {
                "subset": "hotel_ai",
                "domain": "hotel",
                "label": "ai_fake",
                "language": "en",
                "text": text,
                "image_role": "none",
                "group_id": f"gen-{record.generator}-{record.hotel_key}-{index}",
                "source": "generated_this_study",
                "generator": record.generator,
                "polarity": record.polarity,
                "label_quality": "generated_this_study",
                "split": "",
            }
        )
        rows.append(row)
    if not rows:
        return []
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    table = assign_within(table, ["generator", "polarity"], "group_id", rng)
    return table.to_dict("records")


def ott_windows() -> set[str]:
    frame = usable(pd.read_csv(ROOT / "data" / "frozen" / "ott_hotel.csv", usecols=["text", "usable"]))
    seen = set()
    for text in frame["text"].fillna("").astype(str):
        words = tokens(text)
        seen.update(" ".join(words[index : index + 10]) for index in range(max(0, len(words) - 9)))
    return seen


def join_review(upside: object, downside: object) -> str:
    parts = []
    for value in (upside, downside):
        if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
            parts.append(value.strip())
    return " ".join(parts)


def maide_rows() -> list[dict]:
    frame = pd.read_csv(ROOT / "data" / "raw" / "maide_up" / "all_data.csv")
    rows = []
    for record in frame.to_dict("records"):
        language = "zh" if record["Review_Language"] == "Chinese" else "en" if record["Review_Language"] == "English" else ""
        if not language:
            continue
        text = join_review(record["Upside_Review"], record["Downside_Review"])
        if language == "zh" and len(text.strip()) < 20:
            continue
        if language == "en" and len(tokens(text)) < 30:
            continue
        row = blank_row()
        row.update(
            {
                "subset": "hotel_maide_zh" if language == "zh" else "hotel_maide_en",
                "domain": "hotel",
                "label": "ai_fake" if int(record["source"]) == 1 else "real",
                "language": language,
                "text": text,
                "image_role": "none",
                "group_id": f"maide-{language}-{record['Hotel Name']}",
                "source": "maide-up",
                "generator": "gpt-4" if int(record["source"]) == 1 else "",
                "polarity": str(record["Sentiment"]).lower(),
                "label_quality": "gold",
                "split": "",
            }
        )
        rows.append(row)
    if not rows:
        return []
    table = pd.DataFrame(rows)
    # itertuples renames 'Hotel Name' to Hotel_Name. Group ids currently share a hotel,
    # so reviews of one hotel stay together. Add a row key after the split is assigned.
    rng = np.random.default_rng(SEED)
    parts = []
    for subset, part in table.groupby("subset"):
        part = assign_within(part, ["subset"], "group_id", rng)
        part = repair_split(part, "group_id")
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    table["group_id"] = [f"{group}-{index}" for index, group in enumerate(table["group_id"])]
    return table.to_dict("records")


def ecommerce_rows() -> list[dict]:
    frame = pd.read_parquet(ROOT / "data" / "raw" / "salminen" / "train.parquet", columns=["cat", "label", "text"])
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame.loc[frame["cat"].isin(PRODUCT_CATS)].copy()
    frame = frame.loc[~frame["text"].str.contains("###", regex=False)].copy()
    frame = frame.loc[frame["text"].map(lambda text: len(tokens(text)) >= 30)].copy()
    frame["opening"] = frame["text"].map(first_n)
    rows = []
    for record in frame.itertuples(index=False):
        label = "ai_fake" if record.label == "fake" else "real"
        digest = hashlib.sha1(record.text.encode("utf-8")).hexdigest()[:16]
        row = blank_row()
        row.update(
            {
                "subset": "ecommerce_salminen",
                "domain": "ecommerce",
                "label": label,
                "language": "en",
                "text": record.text,
                "image_role": "none",
                "group_id": f"salminen-{digest}",
                "source": "salminen",
                "generator": "gpt-2" if label == "ai_fake" else "",
                "polarity": "",
                "label_quality": "gold",
                "split": "",
                "_opening": record.opening,
                "_cat": record.cat,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    table = assign_within(table, ["_cat"], "_opening", rng)
    return table.drop(columns=["_opening", "_cat"]).to_dict("records")


def ensure_nltk():
    import nltk

    for resource in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            continue
    return nltk


def pos_english(text: str, nltk_module) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text)
    empty = {key: 0.0 for key in ("pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb")}
    if not words:
        return empty
    tags = [tag for _, tag in nltk_module.pos_tag(words)]
    total = max(1, len(tags))

    def share(prefix: str) -> float:
        return sum(tag.startswith(prefix) for tag in tags) / total

    return {
        "pos_noun": share("NN"),
        "pos_adj": share("JJ"),
        "pos_adv": share("RB"),
        "pos_pron": share("PR"),
        "pos_verb": share("VB"),
    }


def pos_chinese(text: str) -> tuple[int, dict[str, float]]:
    import jieba.posseg as pseg

    pairs = list(pseg.cut(text))
    total = max(1, len(pairs))
    flags = [flag for _, flag in pairs]

    def share(prefixes: tuple[str, ...]) -> float:
        return sum(any(flag.startswith(prefix) for prefix in prefixes) for flag in flags) / total

    return len(pairs), {
        "pos_noun": share(("n",)),
        "pos_adj": share(("a",)),
        "pos_adv": share(("d",)),
        "pos_pron": share(("r",)),
        "pos_verb": share(("v",)),
    }


def add_linguistics(frame: pd.DataFrame) -> pd.DataFrame:
    nltk_module = ensure_nltk()
    counts = []
    pos_rows = []
    for index, record in enumerate(frame.itertuples(index=False)):
        if record.language == "zh":
            count, pos = pos_chinese(str(record.text))
        else:
            count = len(tokens(record.text))
            pos = pos_english(str(record.text), nltk_module)
        counts.append(count)
        pos_rows.append(pos)
        if index and index % 5000 == 0:
            print(f"pos {index}", flush=True)
    frame = frame.copy()
    frame["n_tokens"] = counts
    pos_frame = pd.DataFrame(pos_rows)
    for column in pos_frame.columns:
        frame[column] = pos_frame[column].to_numpy()
    return frame


def add_augmentation(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aug_paths = []
    for record in frame.itertuples(index=False):
        if record.split != "train" or not record.image_path:
            aug_paths.append("")
            continue
        source = ROOT / record.image_path
        dest = AUG_DIR / Path(record.image_path).name
        seed = int(hashlib.sha256(record.group_id.encode("utf-8")).hexdigest()[:8], 16)
        augment(source, dest, seed)
        aug_paths.append(str(dest.relative_to(ROOT)).replace("\\", "/"))
    frame["aug_path"] = aug_paths
    return frame


def contingency(frame: pd.DataFrame) -> dict:
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return {}
    report = {}
    for subset, part in frame.groupby("subset"):
        table = pd.crosstab(part["split"], part["label"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            report[subset] = {"table": table.to_dict()}
            continue
        chi2, p_value, _, _ = chi2_contingency(table)
        report[subset] = {"chi2": float(chi2), "p": float(p_value), "table": table.to_dict()}
    return report


def checklist(frame: pd.DataFrame) -> list[dict]:
    def count(subset: str, label: str | None = None) -> int:
        part = frame.loc[frame["subset"] == subset]
        if label is not None:
            part = part.loc[part["label"] == label]
        return int(len(part))

    food_images = int(((frame["subset"] == "food_aigen") & frame["image_path"].astype(bool)).sum())
    hotel_ai = count("hotel_ai")
    generators = sorted(frame.loc[frame["subset"] == "hotel_ai", "generator"].dropna().unique())
    return [
        {
            "item": "餐饮、酒店、电商都有样本",
            "status": "met" if set(frame["domain"]) >= {"food", "hotel", "ecommerce"} else "unmet",
            "evidence": {key: int(value) for key, value in frame.groupby("domain").size().items()},
        },
        {
            "item": "真实、用户虚假、AI 虚假三类出现在同一酒店场景",
            "status": "met" if hotel_ai and count("hotel_ott", "real") and count("hotel_ott", "human_fake") else "unmet",
            "evidence": {
                "hotel_real": count("hotel_ott", "real"),
                "hotel_human_fake": count("hotel_ott", "human_fake"),
                "hotel_ai": hotel_ai,
                "generators": generators,
            },
        },
        {
            "item": "餐饮和电商也有用户生成虚假金标",
            "status": "unmet",
            "evidence": "公开金标没有进入本表。Li 2014 餐饮众包集约 200 条，作者页本次没有打开。电商侧没有找到可下载的众包人假集。",
        },
        {
            "item": "带图的中文真评，以及国内水军文本",
            "status": "unmet",
            "evidence": "携程样本无图文件。ScienceDB 京东评论已下载，论文按 11 条规则建库，留在表外。中文只收入 MAiDE-up 酒店真 / GPT-4 文本。",
        },
        {
            "item": "AI 文本来自一种以上模型，且不用真实评论原句做开头",
            "status": "met" if len(generators) >= 2 else "partial",
            "evidence": {
                "food": "gpt-4-turbo，公开集",
                "ecommerce": "gpt-2，公开集",
                "hotel_maide": "gpt-4，公开集，酒店与 Ott 不是同一批",
                "hotel_chicago": generators,
            },
        },
        {
            "item": "AI 评论有配套生成图，且三类图片不共用",
            "status": "partial",
            "evidence": f"餐饮 AI 图是 DALL·E，与真评图一一对应，共 {food_images} 张已缩到 224。酒店和电商整切片没有配图，避免只有部分类别有图。",
        },
        {
            "item": "去重、分词、词性、尺寸统一、训练增强",
            "status": "partial",
            "evidence": "全表按正文去重。英文用 nltk 词性，中文用 jieba。餐饮图为 224。训练集餐饮图有水平翻转和颜色扰动。归一化是读取时 RGB/255 再按 CLIP 均值方差，没有把浮点写进文件。",
        },
    ]


def main() -> None:
    print("collect", flush=True)
    rows = food_rows() + hotel_ott_rows() + ecommerce_rows() + maide_rows()
    rows.extend(hotel_ai_rows(ott_windows()))
    frame = pd.DataFrame(rows)
    before = len(frame)
    frame = frame.drop_duplicates("text").copy()
    dropped = before - len(frame)
    print(f"rows {len(frame)} dropped exact text {dropped}", flush=True)
    frame = add_linguistics(frame)
    frame = add_augmentation(frame)
    frame = frame[COLUMNS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    audit = {
        "rows": int(len(frame)),
        "dropped_exact_text": int(dropped),
        "by_subset_label": {
            f"{subset}|{label}": int(count)
            for (subset, label), count in frame.groupby(["subset", "label"]).size().items()
        },
        "by_split": frame.groupby(["subset", "split"]).size().astype(int).to_dict(),
        "images": int(frame["image_path"].astype(bool).sum()),
        "augmented": int(frame["aug_path"].astype(bool).sum()),
        "chi2_split_label": contingency(frame),
        "image_norm": "224 RGB JPEG. At read time divide by 255 and apply CLIP mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711).",
        "checklist": checklist(frame),
    }
    # groupby with a tuple key becomes a string key that json can store
    audit["by_split"] = {f"{subset}|{split}": int(count) for (subset, split), count in frame.groupby(["subset", "split"]).size().items()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

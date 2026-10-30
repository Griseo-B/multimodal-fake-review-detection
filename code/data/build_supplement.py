"""Keep newly checked reviews and write a short catalog.

Gold labels, human annotation, and stated rule scores are accepted.
Sentiment-only labels and model relabeling are not.
See data/templates/标注规范.md.
"""

from __future__ import annotations

import csv
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "supplement"
KEPT = RAW / "kept.csv"
CATALOG = ROOT / "data" / "catalog.csv"
CHECK = RAW / "check.json"
ALLOWED = ROOT / "data" / "frozen" / "allowed_reviews.csv"
ZIP = RAW / "li-2014_original.zip"

KEEP_FIELDS = [
    "id", "label", "scene", "language", "basis", "source", "text", "image_path", "image_role",
]

# Human-written fakes have no photo of their own. Pair each one with one
# already downloaded photo from the same scene. The file stays a real web
# photo shared across labels, not a forged image.
SCENE_PHOTO_SOURCES = {
    "food": ("dianping-human",),
    "hotel": ("ctrip-human",),
    "ecommerce": ("jd-human", "taobao-human"),
}


def norm(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def existing_texts() -> set[str]:
    seen = set()
    with ALLOWED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seen.add(norm(row["text"]))
    return seen


def read_zip() -> list[dict]:
    rows = []
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt") or name.startswith("__MACOSX"):
                continue
            parts = [part for part in name.split("/") if part]
            if "doctor" in parts:
                continue
            if "deceptive_expert" not in parts and "restaurant" not in parts:
                continue
            text = archive.read(name).decode("utf-8", "replace").strip()
            if len(text) < 20:
                continue
            if "restaurant" in parts:
                scene = "food"
                label = "human_fake" if "deceptive_MTurk" in parts else "real"
                source = "li2014_restaurant"
            else:
                scene = "hotel"
                label = "human_fake"
                source = "li2014_hotel_expert"
            rows.append(
                {
                    "id": "",
                    "label": label,
                    "scene": scene,
                    "language": "en",
                    "basis": "gold",
                    "source": source,
                    "text": re.sub(r"\s+", " ", text).strip(),
                    "image_path": "",
                    "_key": norm(text),
                }
            )
    return rows


def dianping_rows() -> tuple[list[dict], int]:
    base = RAW / "fakereview_data_set"
    rows = []
    short = 0
    for meta_name, text_name, source in (
        ("Dp-TDC_metadata.csv", "Dp-TDC_text.csv", "dp-tdc"),
        ("Dp-CDC_metadata.csv", "Dp-CDC_text.csv", "dp-cdc"),
    ):
        texts = {}
        with (base / text_name).open(encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                texts[record["id"]] = re.sub(r"\s+", " ", record.get("text") or "").strip()
        with (base / meta_name).open(encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                text = texts.get(record["id"], "")
                if len(text) < 20:
                    short += 1
                    continue
                pictures = int(float(record.get("picture") or 0))
                rows.append(
                    {
                        "id": f"{source}-{record['id']}",
                        "label": "human_fake" if record["label"] == "1" else "real",
                        "scene": "food",
                        "language": "zh",
                        "basis": "rule",
                        "source": source,
                        "text": text,
                        "image_path": "",
                        "_key": norm(text),
                        "_images": pictures,
                    }
                )
    return rows, short


def jd_rows() -> tuple[list[dict], int]:
    """JD labels follow Gu et al. 2024, Table 2: 0 is fake, 1 is real."""
    root = RAW / "jd_scidb"
    specs = [
        (root / "京东虚假评论数据集" / "train.txt", "jd", "train"),
        (root / "京东虚假评论数据集" / "dev.txt", "jd", "dev"),
        (root / "京东虚假评论数据集" / "test.txt", "jd", "test"),
        (root / "跨领域实验数据集" / "手机.txt", "jd-phone", "cross"),
        (root / "跨领域实验数据集" / "食品.txt", "jd-food", "cross"),
        (root / "跨领域实验数据集" / "家电.txt", "jd-appliance", "cross"),
        (root / "跨领域实验数据集" / "服装.txt", "jd-clothing", "cross"),
    ]
    rows = []
    short = 0
    for path, source, split in specs:
        raw = path.read_bytes()
        try:
            text_file = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text_file = raw.decode("gb18030")
        index = 0
        for line in text_file.splitlines():
            if "\t" not in line:
                continue
            label, rest = line.split("\t", 1)
            body = re.sub(r"\s+", " ", rest.split("\t", 1)[0]).strip()
            index += 1
            if len(body) < 20:
                short += 1
                continue
            rows.append(
                {
                    "id": f"{source}-{split}-{index:05d}",
                    "label": "human_fake" if label.strip() == "0" else "real",
                    "scene": "ecommerce",
                    "language": "zh",
                    "basis": "rule",
                    "source": source,
                    "text": body,
                    "image_path": "",
                    "_key": norm(body),
                }
            )
    return rows, short


def ctrip_rows() -> list[dict]:
    path = RAW / "ctrip_trip_2026.csv"
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            text = re.sub(r"\s+", " ", record.get("reviewText") or "").strip()
            if len(text) < 20:
                continue
            language = record.get("language") or ""
            if language in {"CN", "zh"}:
                language = "zh"
            rows.append(
                {
                    "id": f"ctrip-{record['reviewId']}",
                    "label": "real",
                    "scene": "hotel",
                    "language": language,
                    "basis": "platform_user",
                    "source": "ctrip" if record.get("source") == "ctrip" else "trip",
                    "text": text,
                    "image_path": "",
                    "_key": norm(text),
                    "_images": int(float(record.get("imagesCount") or 0)),
                }
            )
    return rows


def _collected_row(source: str, scene: str, review: dict, index: int, prefix: str) -> dict | None:
    text = re.sub(r"\s+", " ", review.get("text") or "").strip()
    if not text:
        return None
    files = [name for name in (review.get("photo_files") or []) if name]
    image_path = "|".join(f"{prefix}/{name}" for name in files)
    review_id = review.get("id") or review.get("review_id") or f"{index:05d}"
    return {
        "id": f"{source}-{review_id}",
        "label": "real",
        "scene": scene,
        "language": "zh",
        "basis": "human",
        "source": source,
        "text": text,
        "image_path": image_path,
        "_key": norm(text),
        "_images": len(files),
    }


def collected_rows() -> list[dict]:
    """Platform reviews the project lead checked as real on 2026-10-29."""
    rows: list[dict] = []
    day = RAW / "collected_20261029"

    taobao = json.loads((day / "taobao_reviews.json").read_text(encoding="utf-8"))
    for index, review in enumerate(taobao["reviews"], start=1):
        row = _collected_row(
            "taobao-human",
            "ecommerce",
            review,
            index,
            "raw/supplement/collected_20261029/taobao_images",
        )
        if row:
            rows.append(row)

    jingdong = json.loads((RAW / "collected_jd_20261029" / "reviews.json").read_text(encoding="utf-8"))
    jd_index = 0
    for product in jingdong["products"]:
        for review in product["reviews"]:
            jd_index += 1
            row = _collected_row(
                "jd-human",
                "ecommerce",
                review,
                jd_index,
                "raw/supplement/collected_jd_20261029",
            )
            if row:
                rows.append(row)
    small_jd = json.loads((day / "jd_reviews.json").read_text(encoding="utf-8"))
    for index, review in enumerate(small_jd["reviews"], start=1):
        row = _collected_row(
            "jd-human",
            "ecommerce",
            review,
            index,
            "raw/supplement/collected_20261029/jd_images",
        )
        if row:
            rows.append(row)

    for folder in ("dianping_auto", "dianping_auto_2"):
        payload = json.loads((day / folder / "dianping_reviews.json").read_text(encoding="utf-8"))
        dp_index = 0
        for shop in payload["shops"]:
            for review in shop["reviews"]:
                dp_index += 1
                row = _collected_row(
                    "dianping-human",
                    "food",
                    review,
                    dp_index,
                    f"raw/supplement/collected_20261029/{folder}/images",
                )
                if row:
                    rows.append(row)
    small_dp = json.loads((day / "dianping_reviews.json").read_text(encoding="utf-8"))
    for index, review in enumerate(small_dp["reviews"], start=1):
        row = _collected_row(
            "dianping-human",
            "food",
            review,
            index,
            "raw/supplement/collected_20261029/dianping_images",
        )
        if row:
            rows.append(row)

    ctrip = json.loads((RAW / "collected_ctrip_20261029" / "reviews.json").read_text(encoding="utf-8"))
    ctrip_index = 0
    for hotel in ctrip["hotels"]:
        for review in hotel["reviews"]:
            ctrip_index += 1
            row = _collected_row(
                "ctrip-human",
                "hotel",
                review,
                ctrip_index,
                "raw/supplement/collected_ctrip_20261029",
            )
            if row:
                rows.append(row)
    return rows


def attach_shared_web_images(rows: list[dict]) -> None:
    pools: dict[str, list[str]] = {scene: [] for scene in SCENE_PHOTO_SOURCES}
    seen: dict[str, set[str]] = {scene: set() for scene in SCENE_PHOTO_SOURCES}
    data_root = ROOT / "data"
    for row in rows:
        paths = [part for part in (row.get("image_path") or "").split("|") if part]
        row["image_role"] = "review" if paths else "none"
        if row.get("label") != "real":
            continue
        for scene, sources in SCENE_PHOTO_SOURCES.items():
            if row.get("source") not in sources:
                continue
            for part in paths:
                if part in seen[scene] or not (data_root / part).is_file():
                    continue
                seen[scene].add(part)
                pools[scene].append(part)
    for scene in pools:
        pools[scene].sort()
    cursor = {scene: 0 for scene in pools}
    for row in rows:
        if row.get("label") != "human_fake" or row.get("image_path"):
            continue
        pool = pools.get(row.get("scene") or "") or []
        if not pool:
            continue
        scene = row["scene"]
        row["image_path"] = pool[cursor[scene] % len(pool)]
        cursor[scene] += 1
        row["image_role"] = "shared_web"


def count_of(by: Counter, source: str, label: str) -> int:
    return sum(count for (src, lab, _scene), count in by.items() if src == source and lab == label)


def main() -> None:
    dianping, dianping_short = dianping_rows()
    jingdong, jingdong_short = jd_rows()
    collected = collected_rows()
    incoming = read_zip() + ctrip_rows() + dianping + jingdong + collected
    labels: dict[str, str] = {}
    conflict_keys: set[str] = set()
    for row in incoming:
        previous = labels.get(row["_key"])
        if previous is None:
            labels[row["_key"]] = row["label"]
        elif previous != row["label"]:
            conflict_keys.add(row["_key"])

    seen = existing_texts()
    fresh = []
    dropped = Counter()
    conflicts = Counter()
    for row in incoming:
        if row["_key"] in conflict_keys:
            conflicts[row["source"]] += 1
            continue
        if row["_key"] in seen:
            dropped[row["source"]] += 1
            continue
        seen.add(row["_key"])
        fresh.append(row)
    for index, row in enumerate(fresh, start=1):
        if not row["id"]:
            row["id"] = f"{row['source']}-{index:05d}"
    attach_shared_web_images(fresh)
    KEPT.parent.mkdir(parents=True, exist_ok=True)
    with KEPT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fresh)

    claimed_images = sum(row.get("_images", 0) > 0 for row in fresh if row["source"] in {"ctrip", "trip"})
    dianping_pictures = sum(row.get("_images", 0) > 0 for row in fresh if row["source"] in {"dp-tdc", "dp-cdc"})
    by = Counter((row["source"], row["label"], row["scene"]) for row in fresh)
    dp_fake = count_of(by, "dp-tdc", "human_fake") + count_of(by, "dp-cdc", "human_fake")
    dp_real = count_of(by, "dp-tdc", "real") + count_of(by, "dp-cdc", "real")
    jd_sources = {"jd", "jd-phone", "jd-food", "jd-appliance", "jd-clothing"}
    jd_fake = sum(count_of(by, source, "human_fake") for source in jd_sources)
    jd_real = sum(count_of(by, source, "real") for source in jd_sources)
    ctrip_n = count_of(by, "ctrip", "real") + count_of(by, "trip", "real")
    shared_web = Counter(row["scene"] for row in fresh if row.get("image_role") == "shared_web")
    human_sources = ("taobao-human", "jd-human", "dianping-human", "ctrip-human")
    human_n = {source: count_of(by, source, "real") for source in human_sources}
    human_files = {
        source: sum(row.get("_images", 0) for row in fresh if row["source"] == source)
        for source in human_sources
    }
    human_with_files = {
        source: sum(row.get("_images", 0) > 0 for row in fresh if row["source"] == source)
        for source in human_sources
    }
    check = {
        "kept_rows": len(fresh),
        "duplicates_left_out": dict(dropped),
        "conflicting_labels_left_out": dict(conflicts),
        "shorter_than_20_left_out": {"dianping": dianping_short, "jd": jingdong_short},
        "kept": {f"{source}|{label}|{scene}": count for (source, label, scene), count in sorted(by.items())},
        "ctrip_rows_that_mention_images_but_have_no_file": claimed_images,
        "dianping_rows_with_picture_count_but_no_file": dianping_pictures,
        "text_only_reviews_allowed": {
            "decided": "2026-10-29",
            "note": "一部分评论可以没有图片。没有文件的行留在表里，不为了凑图去补。",
        },
        "ecommerce_platforms": {
            "decided": "2026-10-30",
            "note": "放弃淘宝平台，不再继续采集。电商用已经采集的天猫 1 个商品和京东。已写入的天猫 20 条保留。",
        },
        "human_fake_shared_web": {
            "food": shared_web["food"],
            "hotel": shared_web["hotel"],
            "ecommerce": shared_web["ecommerce"],
            "note": "一条人写假评配一张已经下载的同场景网上实拍。图记为共用，不是这条评论自己的图，也不是伪造图。",
        },
        "rejected": [
            {"name": "ChatGPT 重标的垃圾评论", "reason": "模型重标，且不是消费评论的真假"},
            {"name": "Li 2014 医生评论", "reason": "不在申报书的酒店、餐饮、电商里"},
            {"name": "罗马乌尔都语 RU-FRDC", "reason": "档案只写从网上直接推断，没有写明人工逐条标注"},
            {"name": "京东包里的骚扰短信和微博谣言", "reason": "不是消费评论"},
        ],
        "proposal": [
            {"item": "一套数据里有真、人假、AI 假", "status": "达到", "evidence": "原有表、Li 2014 人写评论，加上大众点评和京东的规则标注"},
            {"item": "覆盖酒店、餐饮、电商", "status": "达到", "evidence": "场景是覆盖，不按场景拆开要求"},
            {"item": "携程、淘宝等带图真评", "status": "达到", "evidence": f"2026-10-30 放弃淘宝平台，不再继续采集。电商用已经采集的天猫和京东：天猫 1 个商品 {human_n['taobao-human']} 条（{human_with_files['taobao-human']} 条有图，{human_files['taobao-human']} 张），京东 20 个商品 {human_n['jd-human']} 条（{human_with_files['jd-human']} 条有图，{human_files['jd-human']} 张）。携程 20 家酒店 {human_n['ctrip-human']} 条（{human_with_files['ctrip-human']} 条有图，{human_files['ctrip-human']} 张）。大众点评 126 家店 {human_n['dianping-human']} 条（{human_with_files['dianping-human']} 条有图，{human_files['dianping-human']} 张）。一部分评论可以没有图片。"},
            {"item": "Ott 之外的国内水军", "status": "达到", "evidence": f"大众点评规则标注人假 {dp_fake} 条、真评 {dp_real} 条已写入。京东规则标注人假 {jd_fake} 条、真评 {jd_real} 条已写入。人写假评各配一张已经下载的同场景网上实拍，记为共用，不是伪造图。同一句出现相反标签的没有写入"},
            {"item": "AI 评论配 DALL·E 或 Stable Diffusion 图", "status": "达到", "evidence": "餐饮已有 DALL·E。酒店和电商没有原图的 AI 评论，每个子集前一半配了 Stable Diffusion，后一半不配图。清单在 data/raw/generated_ai_images_20261030/manifest.csv。"},
        ],
    }
    CHECK.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    catalog = [
        ["已有", "真 / AI假", "餐饮 AiGen", "20128", "一一对应", "金标", "frozen/allowed_reviews.csv"],
        ["已有", "真 / 人假", "酒店 Ott", "1596", "共用实拍", "金标", "frozen/allowed_reviews.csv"],
        ["已有", "AI假", "芝加哥酒店两个小模型", "840", "一半 Stable Diffusion，一半无图", "生成", "frozen/allowed_reviews.csv"],
        ["已有", "真 / AI假", "酒店 MAiDE-up", "3064", "AI 评论前一半配 Stable Diffusion", "金标", "frozen/allowed_reviews.csv"],
        ["已有", "真 / AI假", "电商 Salminen", "15978", "AI 评论前一半配 Stable Diffusion", "金标", "frozen/allowed_reviews.csv"],
        ["已有", "真 / 人假", "书评、手机、四城、荷兰语等", "9705", "部分共用实拍", "金标", "frozen/allowed_reviews.csv"],
        ["已有", "AI假", "手机", "59", "56 条有同品牌照片", "生成", "frozen/allowed_reviews.csv"],
        ["新收", "人假", "餐饮 Li 2014 众包", str(count_of(by, "li2014_restaurant", "human_fake")), "共用网上实拍", "金标", "raw/supplement/kept.csv"],
        ["新收", "真", "餐饮 Li 2014 顾客", str(count_of(by, "li2014_restaurant", "real")), "无", "金标", "raw/supplement/kept.csv"],
        ["新收", "人假", "酒店 Li 2014 员工", str(count_of(by, "li2014_hotel_expert", "human_fake")), "共用网上实拍", "金标", "raw/supplement/kept.csv"],
        ["新收", "真", "携程 / Trip.com", str(ctrip_n), "无文件", "平台用户评论", "raw/supplement/kept.csv"],
        ["新收", "人假", "大众点评十城与成都", str(dp_fake), "共用网上实拍", "规则标注", "raw/supplement/kept.csv"],
        ["新收", "真", "大众点评十城与成都", str(dp_real), "只有张数", "规则标注", "raw/supplement/kept.csv"],
        ["新收", "人假", "京东商品", str(jd_fake), "共用网上实拍", "规则标注", "raw/supplement/kept.csv"],
        ["新收", "真", "京东商品", str(jd_real), "无", "规则标注", "raw/supplement/kept.csv"],
        ["新收", "真", "天猫 1 个商品", str(human_n["taobao-human"]), f"{human_files['taobao-human']} 张文件", "人工标注", "raw/supplement/collected_20261029"],
        ["新收", "真", "京东 20 个商品", str(human_n["jd-human"]), f"{human_files['jd-human']} 张文件", "人工标注", "raw/supplement/collected_jd_20261029"],
        ["新收", "真", "大众点评 126 家店", str(human_n["dianping-human"]), f"{human_files['dianping-human']} 张文件", "人工标注", "raw/supplement/collected_20261029"],
        ["新收", "真", "携程 20 家酒店", str(human_n["ctrip-human"]), f"{human_files['ctrip-human']} 张文件", "人工标注", "raw/supplement/collected_ctrip_20261029"],
        ["不收", "", "模型重标垃圾评论", "", "", "不是人工标注", ""],
        ["不收", "", "Li 2014 医生", "", "", "不在申报场景", "原始包里仍留着"],
        ["不收", "", "罗马乌尔都语评论", "5150", "", "未写明人工标注", "raw/supplement/ru-frdc-2022.csv"],
        ["不收", "", "骚扰短信与微博谣言", "", "", "不是消费评论", "raw/supplement/jd_scidb"],
    ]
    audit_path = ROOT / "data" / "frozen" / "three_class_audit.json"
    if audit_path.exists():
        three_rows = json.loads(audit_path.read_text(encoding="utf-8"))["rows"]
        catalog.append(
            ["训练", "真 / 人假 / AI假", "三类训练表", str(three_rows), "有图的缩到 224", "按 protocol 分开", "frozen/three_class.csv"]
        )
    with CATALOG.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["去向", "标签", "内容", "条数", "图", "依据", "文件"])
        writer.writerows(catalog)
    print(json.dumps({"kept_rows": len(fresh), "duplicates_left_out": dict(dropped), "conflicting_labels_left_out": dict(conflicts), "kept": check["kept"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

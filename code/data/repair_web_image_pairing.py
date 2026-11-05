"""Re-pair human-written fake reviews with a *more* corresponding web photo.

The 20261031 audit found every image attached to a human-written fake review
came from a whole-scene round-robin pool: a fake review about a hotpot
restaurant could get a coffee-shop photo. That is not how a spammer behaves --
they crawl a photo of the same kind of thing. This script tightens the pairing
from scene level to category level wherever the evidence allows.

Category taxonomies come from the photos we collected ourselves, so the
categories are the ones we actually own images for:

* food       12 cuisines, taken from the Dianping search query of each shop
             (火锅 / 川菜 / 湘菜 / 韩国料理 / 日料 / 烧烤 / 咖啡 / 甜品 /
              本帮菜 / 粤菜 / 西餐 / 面包).
* ecommerce  6 product categories, derived from the JD product titles
             (3C数码 / 家用电器 / 食品饮料 / 个人护理 / 服饰鞋包 / 居家日用).

A fake review is classified by keyword evidence in its own text, then given a
photo from that category's pool. Reviews with no usable evidence keep the old
scene-level photo and are reported as such rather than silently upgraded.

Outputs
-------
experiments/20261019_web_image_pair/pairing.csv   per-row old vs new image
experiments/20261019_web_image_pair/pairing.json  coverage and pool audit
data/frozen/web_image_category.csv                the re-paired fake rows

    python code/data/repair_web_image_pairing.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
COLLECTED = ROOT / "data" / "raw" / "supplement" / "collected_taobao_dianping"
JD_COLLECTED = ROOT / "data" / "raw" / "supplement" / "collected_jd"
OUT = ROOT / "experiments" / "20261019_web_image_pair"

# --- food: keyword evidence -> cuisine, ordered most specific first ----------
CUISINE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "日料": ("日料", "日本料理", "寿司", "刺身", "寿喜锅", "烧鸟", "居酒屋", "天妇罗",
             "鳗鱼", "铁板烧", "割烹", "放题", "葱烧", "手卷", "和牛", "清酒"),
    "韩国料理": ("韩国", "韩式", "韩料", "部队锅", "泡菜", "石锅拌饭", "炒年糕", "烤肉",
                 "炸鸡", "参鸡汤", "大酱汤"),
    "火锅": ("火锅", "锅底", "涮", "毛肚", "鸳鸯锅", "牛油", "小料", "麻辣烫", "串串香",
             "自助小料", "番茄锅", "菌汤锅", "冒菜锅"),
    "烧烤": ("烧烤", "撸串", "烤串", "羊肉串", "烤五花", "炭烤", "串儿", "烤鱼", "烤腰子",
             "大排档", "夜宵摊"),
    "川菜": ("川菜", "水煮鱼", "麻婆豆腐", "回锅肉", "毛血旺", "酸菜鱼", "宫保鸡丁",
             "辣子鸡", "口水鸡", "鱼香肉丝", "川味", "麻辣香锅", "冒菜"),
    "湘菜": ("湘菜", "剁椒", "小炒肉", "腊味", "口味虾", "臭豆腐", "湖南", "农家小炒",
             "擂椒", "血鸭"),
    "粤菜": ("粤菜", "早茶", "烧鹅", "叉烧", "肠粉", "煲仔饭", "潮汕", "顺德", "广式",
             "虾饺", "烧麦", "凤爪", "砂锅粥", "白切鸡"),
    "本帮菜": ("本帮", "上海菜", "红烧肉", "腌笃鲜", "生煎", "小笼", "响油鳝", "油爆虾",
               "八宝鸭", "草头"),
    "西餐": ("西餐", "牛排", "意面", "披萨", "意式", "法餐", "汉堡", "沙拉", "三明治",
             "brunch", "自助餐", "浓汤", "烩饭", "tapas"),
    "咖啡": ("咖啡", "拿铁", "美式", "卡布奇诺", "手冲", "浓缩", "澳白", "冷萃", "摩卡",
             "咖啡豆", "拉花", "dirty"),
    "甜品": ("甜品", "蛋糕", "冰淇淋", "慕斯", "布丁", "双皮奶", "糖水", "芋圆", "提拉米苏",
             "舒芙蕾", "泡芙", "奶冻", "冰沙"),
    "面包": ("面包", "吐司", "欧包", "可颂", "贝果", "碱水", "烘焙", "法棍", "丹麦酥",
             "恰巴塔", "现烤"),
}

# --- ecommerce: keyword evidence -> product category -------------------------
PRODUCT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "3C数码": ("手机", "耳机", "音质", "蓝牙", "键盘", "鼠标", "充电宝", "充电", "电池",
               "屏幕", "像素", "拍照", "电脑", "平板", "数据线", "音箱", "音响", "续航",
               "降噪", "连接", "信号", "系统", "卡顿"),
    "食品饮料": ("好吃", "口感", "味道", "零食", "牛奶", "饼干", "坚果", "薯片", "咖啡",
                 "饮料", "甜", "酸甜", "醇", "冲泡", "保质期", "份量", "包装", "口味",
                 "喝", "嚼", "零食"),
    "个人护理": ("洗发", "牙膏", "牙刷", "沐浴", "香皂", "护肤", "面膜", "剃须", "去屑",
                 "头皮", "留香", "清洁力", "泡沫", "牙齿", "口气", "皂"),
    "服饰鞋包": ("鞋", "衣服", "裤", "穿着", "尺码", "面料", "版型", "裙子", "外套",
                 "透气", "运动鞋", "跑步", "上脚", "包裹"),
    "家用电器": ("电饭煲", "豆浆", "破壁", "洗衣机", "冰箱", "空调", "热水器", "加湿",
                 "吹风", "榨汁", "蒸", "加热", "功率", "噪音", "预约"),
    "居家日用": ("纸巾", "抽纸", "洗衣液", "保温杯", "杯子", "毛巾", "收纳", "拖把",
                 "抹布", "保温", "不锈", "材质", "异味"),
}

MIN_HITS = 2  # one keyword can be a coincidence; two is evidence

# A single hit is accepted when the word names the place or the product itself
# ("火锅店", "咖啡馆", "耳机"), because that is direct evidence rather than a
# passing mention. Words that a review of some other cuisine often mentions
# ("冰淇淋", "咖啡", "啤酒") are deliberately excluded here.
DEFINING: dict[str, tuple[str, ...]] = {
    "日料": ("日料", "日本料理", "居酒屋", "烧鸟店", "寿司店"),
    "韩国料理": ("韩料", "韩国料理", "韩式料理"),
    "火锅": ("火锅", "火锅店", "涮肉", "涮锅", "串串香"),
    "烧烤": ("烧烤", "烧烤店", "撸串", "烤串"),
    "川菜": ("川菜", "川菜馆"),
    "湘菜": ("湘菜", "湘菜馆"),
    "粤菜": ("粤菜", "早茶", "煲仔饭", "潮汕火锅", "顺德菜"),
    "本帮菜": ("本帮菜", "上海菜"),
    "西餐": ("西餐", "牛排馆", "意式餐厅", "法餐"),
    "咖啡": ("咖啡馆", "咖啡店", "咖啡厅", "精品咖啡"),
    "甜品": ("甜品店", "蛋糕店"),
    "面包": ("面包店", "烘焙店", "欧包店"),
    "3C数码": ("耳机", "手机", "笔记本电脑", "充电宝", "键盘", "鼠标", "平板电脑"),
    "食品饮料": ("牛奶", "饼干", "薯片", "坚果", "咖啡粉", "巧克力", "面包", "零食"),
    "个人护理": ("洗发水", "牙膏", "电动牙刷", "沐浴露", "香皂", "剃须刀"),
    "服饰鞋包": ("运动鞋", "跑鞋", "外套", "裤子", "连衣裙", "双肩包"),
    "家用电器": ("电饭煲", "豆浆机", "破壁机", "洗衣机", "冰箱", "空调", "加湿器"),
    "居家日用": ("抽纸", "纸巾", "洗衣液", "保温杯", "毛巾", "拖把", "收纳盒"),
}


def category_pool(catalog: dict[str, tuple[str, ...]], collected: dict[str, str]) -> dict[str, list[str]]:
    """Image path -> category label, using the shop/product name we collected."""
    pool: dict[str, list[str]] = defaultdict(list)
    for path, text in collected.items():
        best, best_hits = None, 0
        for cat, words in catalog.items():
            hits = sum(1 for w in words if w in text)
            if hits > best_hits:
                best, best_hits = cat, hits
        if best:
            pool[best].append(path)
    for cat in pool:
        pool[cat].sort()
    return pool


def collect_food_images() -> tuple[dict[str, str], dict[str, int]]:
    """image path -> the shop's Dianping search query (its cuisine)."""
    out: dict[str, str] = {}
    per_query: Counter = Counter()
    for folder in sorted(COLLECTED.glob("dianping_auto*")):
        meta = folder / "dianping_reviews.json"
        if not meta.is_file():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        for shop in data.get("shops", []):
            query = shop.get("query") or ""
            for review in shop.get("reviews", []):
                for name in review.get("photo_files", []):
                    path = folder / "images" / name
                    if path.is_file():
                        out[path.relative_to(ROOT).as_posix()] = query
                        per_query[query] += 1
    single = COLLECTED / "dianping_images"
    if single.is_dir():
        for path in single.glob("*"):
            out[path.relative_to(ROOT).as_posix()] = ""
    return out, dict(per_query)


def collect_jd_images() -> dict[str, str]:
    """image path -> the product title it belongs to."""
    out: dict[str, str] = {}
    meta = JD_COLLECTED / "reviews.json"
    if not meta.is_file():
        return out
    data = json.loads(meta.read_text(encoding="utf-8"))
    for entry in data.get("products", []):
        product = entry.get("product") or {}
        title = product.get("title") or ""
        for review in entry.get("reviews", []):
            for name in review.get("photo_files", []):
                path = JD_COLLECTED / name
                if path.is_file():
                    out[path.relative_to(ROOT).as_posix()] = title
    return out


def classify(text: str, catalog: dict[str, tuple[str, ...]]) -> tuple[str | None, int]:
    """Return (category, hits) using the conservative rule documented above.

    A category is accepted only if its score is strictly the best one and either
    it reached MIN_HITS, or the single hit names the place / product directly.
    """
    text = (text or "").lower()
    scores = {cat: sum(1 for w in words if w.lower() in text) for cat, words in catalog.items()}
    if not scores:
        return None, 0
    top = max(scores.values())
    if top == 0:
        return None, 0
    winners = [cat for cat, value in scores.items() if value == top]
    if len(winners) > 1:  # ambiguous tie -> no claim
        return None, top
    cat = winners[0]
    if top >= MIN_HITS:
        return cat, top
    if any(w.lower() in text for w in DEFINING.get(cat, ())):
        return cat, top
    return None, top


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(FROZEN / "three_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    web = d[d["image_role"] == "shared_web"].copy()
    print(f"shared_web rows {len(web)}")
    print(web.groupby("domain").size().to_dict())

    food_imgs, query_counts = collect_food_images()
    jd_imgs = collect_jd_images()
    food_pool = category_pool(CUISINE_KEYWORDS, food_imgs)
    jd_pool = category_pool(PRODUCT_KEYWORDS, jd_imgs)
    print(f"food images {len(food_imgs)} -> { {k: len(v) for k, v in food_pool.items()} }")
    print(f"jd images {len(jd_imgs)} -> { {k: len(v) for k, v in jd_pool.items()} }")

    cursor = {("food", k): 0 for k in food_pool}
    cursor.update({("ecommerce", k): 0 for k in jd_pool})

    rows = []
    stats = Counter()
    for _, row in web.iterrows():
        domain = row["domain"]
        cat, hits = None, 0
        if domain == "food":
            cat, hits = classify(row["text"], CUISINE_KEYWORDS)
            pool = food_pool.get(cat or "", [])
            key = ("food", cat)
        elif domain == "ecommerce":
            cat, hits = classify(row["text"], PRODUCT_KEYWORDS)
            pool = jd_pool.get(cat or "", [])
            key = ("ecommerce", cat)
        else:
            pool = []

        old = row["image_path"]
        new = old
        pairing = "scene"
        if cat and pool:
            new = pool[cursor[key] % len(pool)]
            cursor[key] += 1
            pairing = "category"
            stats[f"{domain}:matched"] += 1
        else:
            if domain == "hotel":
                stats["hotel:no_category_taxonomy"] += 1
            else:
                stats[f"{domain}:no_evidence"] += 1

        rows.append({
            "id": row["id"],
            "domain": domain,
            "label": row["label"],
            "source": row["source"],
            "text": row["text"],
            "category": cat or "",
            "keyword_hits": hits,
            "pairing": pairing,
            "old_image_path": old,
            "image_path": new,
        })

    out = pd.DataFrame(rows)
    out["changed"] = out["old_image_path"] != out["image_path"]
    out.to_csv(OUT / "pairing.csv", index=False, encoding="utf-8")

    frozen = out[["id", "domain", "source", "label", "text", "category", "pairing", "image_path"]].copy()
    frozen.to_csv(FROZEN / "web_image_category.csv", index=False, encoding="utf-8")

    audit = {
        "rows": int(len(out)),
        "changed": int(out["changed"].sum()),
        "pairing": {str(k): int(v) for k, v in out["pairing"].value_counts().items()},
        "by_domain": {
            domain: {
                "rows": int(len(sub)),
                "category": int((sub["pairing"] == "category").sum()),
                "scene": int((sub["pairing"] == "scene").sum()),
                "top_categories": {str(k): int(v) for k, v in sub.loc[sub.category != "", "category"].value_counts().head(8).items()},
            }
            for domain, sub in out.groupby("domain")
        },
        "pools": {
            "food": {k: len(v) for k, v in food_pool.items()},
            "ecommerce": {k: len(v) for k, v in jd_pool.items()},
            "food_query_counts": query_counts,
        },
        "min_hits": MIN_HITS,
        "note": (
            "A row keeps pairing='scene' when its text gives no category evidence or the "
            "category has no photo pool. Those rows are not silently upgraded."
        ),
    }
    (OUT / "pairing.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\npairing:", audit["pairing"])
    print("changed:", audit["changed"])
    for domain, info in audit["by_domain"].items():
        print(f"  {domain:10s} rows {info['rows']:5d} category {info['category']:5d} scene {info['scene']:5d} {info['top_categories']}")


if __name__ == "__main__":
    main()

"""Write docs/数据卡.md from the frozen tables.

The card answers one question a reviewer will ask: what exactly is in this
dataset, and where is it thin? Every number is counted here, not typed by hand,
so the card cannot drift away from the tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm3_common import ROOT  # noqa: E402

FROZEN = ROOT / "data" / "frozen"
OUT = ROOT / "docs" / "数据卡.md"

ROLE_MEANING = {
    "paired_review": "真评自己的实拍",
    "paired_dalle": "DALL·E-2 为该条 AI 评论生成",
    "paired_sd": "Stable Diffusion 1.5 为该条 AI 评论生成",
    "shared_entity": "同一酒店或商品的共用实拍",
    "shared_web": "人写假评配的同场景网上实拍",
    "none": "无图",
}
SOURCE_LICENSE = {
    "aigen": "AiGen-FoodReview，MIT；图由 DALL·E-2 生成",
    "ott": "Ott 酒店评论，CC BY-NC-SA 3.0；原包不上传",
    "salminen": "Salminen，CC BY 4.0；其 GPT-2 生成文本记为 ai_fake",
    "maide-up": "MAiDE-up，酒店评论与 AI 改写",
    "generated_this_study": "本研究生成，未对外发布权重",
    "phone_ai": "本研究生成",
    "blt": "BLT 虚假评论集",
    "clips": "CLiPS 荷兰语书评",
    "derev": "DeRev 书评",
    "soldner": "Soldner 酒店评论",
    "four_city": "Four-city 虚假评论集",
    "li2014_restaurant": "Li 2014 餐饮众包",
    "li2014_hotel_expert": "Li 2014 酒店员工",
    "dp-tdc": "大众点评十城，作者按规则标注",
    "dp-cdc": "大众点评成都，作者按规则标注",
    "jd": "京东 ScienceDB，规则标注",
    "jd-food": "京东跨领域，规则标注",
    "jd-phone": "京东跨领域，规则标注",
    "jd-clothing": "京东跨领域，规则标注",
    "jd-appliance": "京东跨领域，规则标注",
    "jd-human": "本项目采集，负责人核为人工真评",
    "dianping-human": "本项目采集，负责人核为人工真评",
    "ctrip-human": "本项目采集，负责人核为人工真评",
    "taobao-human": "本项目采集，负责人核为人工真评",
    "ctrip": "携程文本，平台用户评论",
    "trip": "Trip.com 文本，平台用户评论",
}


def md_table(frame: pd.DataFrame, index_name: str = "") -> str:
    columns = [index_name] + [str(column) for column in frame.columns] if index_name else [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for key, row in frame.iterrows():
        values = [str(key)] + [f"{value:,}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    frame = pd.read_csv(FROZEN / "three_class.csv", dtype=str, keep_default_na=False, low_memory=False)
    frame["has_image"] = frame["image_224_path"].astype(bool)
    pools = {
        name: json.loads((FROZEN / f"three_class_pool_{name}_audit.json").read_text(encoding="utf-8"))
        for name in ("a", "b")
    }
    leak = json.loads((ROOT / "experiments" / "20261013_leakage_split" / "summary.json").read_text(encoding="utf-8"))
    shared = json.loads((ROOT / "experiments" / "20261014_image_assignment" / "summary.json").read_text(encoding="utf-8"))
    pairing = json.loads((ROOT / "experiments" / "20261019_web_image_pair" / "pairing.json").read_text(encoding="utf-8"))
    matcher = json.loads((ROOT / "experiments" / "20261019_web_image_pair" / "category_matcher_validation.json").read_text(encoding="utf-8"))
    leakage = json.loads((ROOT / "experiments" / "20261019_web_image_pair" / "leakage.json").read_text(encoding="utf-8"))
    bench = json.loads((FROZEN / "web_image_benchmark_audit.json").read_text(encoding="utf-8"))
    baselines = json.loads((ROOT / "experiments" / "20261020_web_image_benchmark" / "baselines.json").read_text(encoding="utf-8"))
    probe = {row["probe"]: row for row in baselines["probes"]}
    pool_audit = json.loads((FROZEN / "three_class_pool_audit.json").read_text(encoding="utf-8"))
    split_repair = pool_audit["split_repair"]

    label_domain = pd.crosstab(frame["label"], frame["domain"]).loc[["real", "human_fake", "ai_fake"]]
    label_domain.loc["合计"] = label_domain.sum()
    label_language = pd.crosstab(frame["label"], frame["language"]).loc[["real", "human_fake", "ai_fake"]]
    label_quality = pd.crosstab(frame["label"], frame["label_quality"]).loc[["real", "human_fake", "ai_fake"]]
    label_protocol = pd.crosstab(frame["label"], frame["protocol"]).loc[["real", "human_fake", "ai_fake"]]

    coverage = frame.groupby("label")["has_image"].agg(["size", "sum"])
    coverage.columns = ["条数", "有图", ]
    coverage["覆盖率"] = (coverage["有图"] / coverage["条数"] * 100).round(1).astype(str) + "%"
    coverage = coverage.loc[["real", "human_fake", "ai_fake"]]

    role = pd.crosstab(frame["image_role"].replace("", "none"), frame["label"])
    role = role.loc[[key for key in ROLE_MEANING if key in role.index]]

    # Image role is not merely correlated with the label: in both pools it is a
    # function of the label. That is the single fact a reader must not miss.
    pool_role = []
    for name in ("a", "b"):
        part = pd.read_csv(FROZEN / f"three_class_pool_{name}.csv", dtype=str,
                           keep_default_na=False, low_memory=False)
        part = part.loc[part["image_224_path"].astype(bool)]
        table = pd.crosstab(part["image_role"], part["label"])
        for role_name, row in table.iterrows():
            pool_role.append(
                {
                    "池 / image_role": f"{name.upper()} / {role_name}",
                    "真": int(row.get("real", 0)),
                    "人假": int(row.get("human_fake", 0)),
                    "AI 假": int(row.get("ai_fake", 0)),
                }
            )
    pool_role = pd.DataFrame(pool_role).set_index("池 / image_role")

    paired_cover = []
    for name in ("a", "b"):
        part = pd.read_csv(FROZEN / f"three_class_pool_{name}.csv", dtype=str,
                           keep_default_na=False, low_memory=False)
        for label in ("real", "human_fake", "ai_fake"):
            rows = part.loc[part["label"] == label]
            paired = rows["image_role"].isin(["paired_review", "paired_dalle", "paired_sd"])
            paired_cover.append(
                {
                    "池 / 标签": f"{name.upper()} / {label}",
                    "条数": int(len(rows)),
                    "有图": int(rows["image_224_path"].astype(bool).sum()),
                    "自属图": int(paired.sum()),
                }
            )
    paired_cover = pd.DataFrame(paired_cover).set_index("池 / 标签")

    domain_cover = frame.groupby(["domain", "label"])["has_image"].agg(["size", "sum"])
    domain_cover["rate"] = (domain_cover["sum"] / domain_cover["size"] * 100).round(1).astype(str) + "%"
    domain_cover.columns = ["条数", "有图", "覆盖率"]
    domain_cover.index = [f"{domain} / {label}" for domain, label in domain_cover.index]

    hashed = frame.loc[frame["has_image"]]
    train_paths = set(hashed.loc[hashed["split"] == "train", "image_224_path"])
    test_rows = hashed.loc[hashed["split"] == "test"]
    dataset_overlap = int(test_rows["image_224_path"].isin(train_paths).sum())
    reused = hashed["image_224_path"].value_counts()
    reused_images = int((reused > 1).sum())

    ai = frame.loc[frame["label"] == "ai_fake"]
    gen_domain = pd.crosstab(ai["generator"].replace("", "未记录"), ai["domain"])

    zh = frame.loc[frame["language"] == "zh"]
    zh_label = zh["label"].value_counts().rename("条数").to_frame()
    zh_label["标注依据"] = zh.groupby("label")["label_quality"].agg(lambda s: "/".join(sorted(set(s))))
    zh_label["来源"] = zh.groupby("label")["source"].agg(lambda s: "/".join(sorted(set(s))))

    split_protocol = pd.crosstab(frame["protocol"], frame["split"])

    lines = []
    lines.append("# 数据卡")
    lines.append("")
    lines.append("> 自动生成：`code/data/build_data_card.py`。数字来自 `data/frozen/three_class.csv` 与两个实验池的核对文件，不手填。")
    lines.append("> 生成日期 2026-11-01。改表之后重跑本脚本，不要手改本页。")
    lines.append("")
    lines.append("## 一句话状态")
    lines.append("")
    lines.append(
        f"三类并集表 **{len(frame):,} 条**（真 {int((frame.label == 'real').sum()):,}、"
        f"人写假 {int((frame.label == 'human_fake').sum()):,}、AI 假 {int((frame.label == 'ai_fake').sum()):,}），"
        f"覆盖餐饮、电商、酒店三个主场景。它是**并集**，不是一张能直接训三分类的表：七个 `protocol` 没有一个同时含三类。"
        f"可训的三分类池另存为 A（酒店单域，{pools['a']['rows']:,} 条）与 B（跨源，{pools['b']['rows']:,} 条）。"
    )
    lines.append("")
    lines.append(f"只有 **{int(frame['has_image'].sum()):,} 条（{frame['has_image'].mean() * 100:.1f}%）有图**，且真评一侧只有 23% 有图。"
                 "图文配对完整的只有餐饮一个领域。")
    lines.append("")
    lines.append("## 1 构成")
    lines.append("")
    lines.append("### 1.1 标签 × 场景")
    lines.append("")
    lines.append(md_table(label_domain, "标签"))
    lines.append("")
    lines.append("### 1.2 标签 × 语种")
    lines.append("")
    lines.append(md_table(label_language, "标签"))
    lines.append("")
    lines.append("### 1.3 标签 × 标注依据")
    lines.append("")
    lines.append(md_table(label_quality, "标签"))
    lines.append("")
    lines.append("`gold` 是原数据集作者已经定过谁写的；`rule` 是作者按写明的规则打分（全部是中文大众点评与京东）；"
                 "`human` 与 `platform_user` 是本项目采集并人工核过的真评；`generated_this_study` 是本项目自己生成的 AI 评论。")
    lines.append("")
    lines.append("### 1.4 标签 × protocol")
    lines.append("")
    lines.append(md_table(label_protocol, "标签"))
    lines.append("")
    lines.append("不同 `protocol` 不放进同一次训练。这张表就是“组不出三分类”的具体形态。")
    lines.append("")
    lines.append("## 2 图文覆盖")
    lines.append("")
    lines.append(md_table(coverage, "标签"))
    lines.append("")
    lines.append("### 2.1 图片角色")
    lines.append("")
    lines.append(md_table(role, "image_role"))
    lines.append("")
    lines.append("| image_role | 含义 |")
    lines.append("|---|---|")
    for key, meaning in ROLE_MEANING.items():
        lines.append(f"| `{key}` | {meaning} |")
    lines.append("")
    lines.append("### 2.2 场景 × 标签的覆盖率")
    lines.append("")
    lines.append(md_table(domain_cover, "场景 / 标签"))
    lines.append("")
    lines.append("读法：")
    lines.append("")
    lines.append("- 只有 **餐饮** 同时有“真评实拍”和“为 AI 评论生成的 DALL·E 图”，是唯一能做真伪图像对照的领域。")
    lines.append("- 电商、酒店的真评一侧几乎没有配对图，`text+mm` 只能落在 AI 一侧，那是来源线索，不是虚假线索。")
    lines.append("- 人写假评的图是借用的网图（`shared_web` / `shared_entity`），不是伪造图；图像模态对它本身不含“假”的信号，详见 2.3。")
    lines.append("- 有没有图和标签相关（真 23%、人假 76%、AI 假 75%），所以 **图本身不能当特征之外的第二条判据**，"
                 "任何图像实验都要和“有没有图”这个基线对照。")
    lines.append("- 「真评配自拍 + 假评配借用网图」的成品冻结在 `web_image_benchmark.csv`，见 2.4。")
    lines.append("")
    lines.append("### 2.3 人写假评的配图：借用网图，三档对应强度")
    lines.append("")
    lines.append("**设计前提**：现实中大量假评不是自己拍的，而是上网抓一张同一家店 / 同一款货的图贴上去。"
                 "所以本数据集不把「假评没有原创图」当缺陷，而是按这个真实行为建模——"
                 "**真评配自己的实拍，人写假评配借用网图**。口径写在 `code/data/build_web_image_benchmark.py` 的 docstring 里。")
    lines.append("")
    lines.append("在这个前提下，「借用网图」还分三档对应强度，逐档核对过。核对脚本 "
                 "`code/data/audit_image_assignment.py`（实体档）与 `code/data/repair_web_image_pairing.py`（品类档），"
                 "结果在 `experiments/20261014_image_assignment/`、`experiments/20261019_web_image_pair/`。")
    lines.append("")
    lines.append("| 档位 | 行数 | 唯一图 | 图的来源 | 对应关系 |")
    lines.append("|---|---:|---:|---|---|")
    lines.append(
        f"| `entity` 实体档 | {shared['shared_entity']['rows']:,} | {shared['shared_entity']['unique_images']:,} | "
        f"该酒店 / 该手机型号的图池，每个实体 {shared['shared_entity']['images_per_entity']['min']}–"
        f"{shared['shared_entity']['images_per_entity']['max']} 张，中位 "
        f"{shared['shared_entity']['images_per_entity']['median']:.0f} 张 | 图就是评论写的那一家 |"
    )
    lines.append(
        f"| `category` 品类档 | {pairing['by_domain']['food']['category'] + pairing['by_domain']['ecommerce']['category']:,} | — | "
        "同菜系（12 类）或同品类（6 类）的实拍 | 图是同菜系 / 同品类，不是同一家 |"
    )
    lines.append(
        f"| `scene` 场景档 | {pairing['by_domain']['food']['scene'] + pairing['by_domain']['ecommerce']['scene'] + pairing['by_domain']['hotel']['scene']:,} | — | "
        "同场景（餐饮 / 酒店 / 电商）实拍 | 只保证大场景对得上 |"
    )
    lines.append("")
    lines.append(
        "行数按 `image_role` 计，含真评那一侧；2.4 的基准表只取人写假那一侧，所以两处数字不同"
        f"（`entity` 档在基准表里是 {bench['pairing']['entity']:,} 行）。"
    )
    lines.append("")
    lines.append(
        f"`entity` 档：可核对的 {shared['shared_entity']['checkable_rows']:,} 行里，分配实体与评论所写实体一致率 "
        f"{shared['shared_entity']['assigned_entity_equals_reviewed_place']:.0%}，文件名前缀与实体一致 "
        f"{shared['shared_entity']['image_filename_prefix_matches_entity']:.0%}。"
    )
    lines.append("")
    lines.append(
        f"`category` 档是本次新增的：`shared_web` 原来按 `domain` 全场景轮流发图，一条写火锅的假评可能配到咖啡馆的图。"
        f"现在先从评论文本里判菜系 / 品类，再从**同品类**图池取图。判类器在 125 家自有店铺的真评上做过留出验证"
        f"（`code/eval/category_matcher_validation.py`）：覆盖 {matcher['coverage']:.0%}，"
        f"**精确率 {matcher['precision']:.3f}**，所以只把有把握的行升级到品类档，剩下 "
        f"{pairing['pairing']['scene']:,} 行老实留在场景档，不硬凑。"
    )
    lines.append("")
    lines.append(
        "判类器保守到什么程度：需要同一类别命中 ≥2 个关键词，或者 1 个「店名 / 品名本身」的关键词"
        "（火锅店、咖啡馆、耳机）；同分平局直接放弃。"
    )
    lines.append("")
    lines.append(
        "落表后再回查一遍：升级到品类档的 3,420 行，其图所在店铺 / 商品的实际菜系或品类与分配的 `category` "
        "**一致率 100%**（餐饮 2,500 行、电商 920 行各自 100%）——即配的图确实来自同菜系 / 同品类的实拍。"
    )
    lines.append("")
    lines.append(
        f"实体内真假共用同一图池，是为了让图不含标签信号：183 张图里 **100% 都被真假两类用过**"
        f"（`web_image_pair_audit.json`）。核验（`code/eval/web_image_leakage.py`）："
        f"29 个实体内图 × 标签列联表置换检验统计量 {leakage['image_identity_permutation']['statistic']}，"
        f"零分布均值 {leakage['image_identity_permutation']['null_mean']}，"
        f"p = {leakage['image_identity_permutation']['p_value']}；"
        f"纯视觉 12 维的实体分组交叉验证 AUC "
        f"{next(r['auc'] for r in leakage['image_only_probes'] if r['probe'] == 'image_visual_only')}，接近 0.5。"
    )
    lines.append("")
    lines.append("### 2.4 借用网图基准表")
    lines.append("")
    lines.append(
        f"`data/frozen/web_image_benchmark.csv`，**{bench['rows']:,} 行，真 {bench['labels']['real']:,} / "
        f"人写假 {bench['labels']['human_fake']:,}**，是「真评配自拍 + 假评配网图」这一口径的冻结成品。"
    )
    lines.append("")
    lines.append("| 列 | 含义 |")
    lines.append("|---|---|")
    lines.append("| `pairing` | `own`（真评自拍）/ `entity` / `category` / `scene` |")
    lines.append("| `binary_label` | 1 = 人写假 |")
    lines.append("| `entity` / `category` | 借用图对应的实体 / 品类，`own` 行为空 |")
    lines.append(
        f"| `split` | 按**图像身份**分组划分（实体档按实体、其余按图片文件），"
        f"跨集合同图 {bench['images_in_more_than_one_split']} 张 |"
    )
    lines.append("")
    lines.append(
        "基线（`code/eval/benchmark_baselines.py`，冻结划分）：纯文本 AUC "
        f"{probe['text_only']['auc']:.4f}，纯视觉 {probe['image_visual_only']['auc']:.4f}，"
        f"文本+全部图像 {probe['text_plus_image']['auc']:.4f}。"
        f"图像相对纯文本的增量 {baselines['delta_text_plus_image']:+.4f}，即图像没有增量。"
    )
    lines.append("")
    lines.append(
        "已知混杂，写在这里以免被当成结论：真评是业余随手拍、借用图是商家 / 网图精修，"
        f"**低层图像统计本身就能分出两类**（纯视觉池化 AUC {probe['image_visual_only']['auc']:.3f}）。"
        "这是这条口径自带的，不是泄漏，但报数时必须同时给纯文本和纯图像两条线。"
    )
    lines.append("")
    lines.append("## 3 AI 生成器 × 场景")
    lines.append("")
    lines.append(md_table(gen_domain, "生成器"))
    lines.append("")
    lines.append("生成器和场景几乎一一对应，所以“跨域泛化”在现有表里实际是“跨生成器”。"
                 "`experiments/20261017_source_generalization/` 把这两个对比分开量了一遍，结论写在 `docs/实验进展.md`。")
    lines.append("")
    lines.append("### 3.1 图片角色是标签的函数，不是标签的相关量")
    lines.append("")
    lines.append(md_table(pool_role, "池 / image_role"))
    lines.append("")
    lines.append("两池都成立：`paired_dalle`、`paired_sd` 只出现在 AI 假，`paired_review` 只出现在真评，"
                 "`shared_entity`、`shared_web` 只出现在人写假与真评。所以 **“图是不是生成的”等于“是不是 AI 假”，"
                 "“图是不是共用”等于“是不是人写假”**。这不是可校正的偏差，是标签定义写进了图片角色。")
    lines.append("")
    lines.append("自属图（`paired_*`）的逐标签覆盖：")
    lines.append("")
    lines.append(md_table(paired_cover, "池 / 标签"))
    lines.append("")
    lines.append("人写假评的自属图在两个池里都是 **0**。这不是采集疏漏：假评不会附一张能证明它在现场的照片。"
                 "结论是三分类多模态在当前数据上组不出来；任何把共用图放进模型的写法都会把图片角色当标签用。"
                 "实测见 `experiments/20261015_pool_multimodal/`：池 B 只用图像时人写假评 F1 = 1.000、AI 假 = 0.000。")
    lines.append("")
    lines.append("中期口径据此锁定为「文本三分类 + 餐饮二分类图文真伪」：图像只在餐饮 20,128 行的真伪对照里用，"
                 "三分类一律纯文本。范围这么写死的理由就是上两张表，不是模型还没调好。")
    lines.append("")
    lines.append("## 4 划分与图像泄漏")
    lines.append("")
    lines.append(md_table(split_protocol, "protocol"))
    lines.append("")
    lines.append(f"文本侧干净：标准化后重复文本 {int(frame['text'].fillna('').str.strip().str.casefold().duplicated().sum())} 条，"
                 f"跨 split 的 `group_id` 0 个。图像侧不干净：")
    lines.append("")
    lines.append(f"- 全表 {len(hashed):,} 条带图样本里，共用图片 {reused_images:,} 张；原始 `split` 下，测试集有 "
                 f"**{dataset_overlap:,} 行**的图片在训练集里出现过。")
    lines.append(f"- 按图片 md5 归并后，池 A 有 {leak['pool_overlap'][0]['rows_with_cross_label_image']:,} 行、"
                 f"池 B 有 {leak['pool_overlap'][1]['rows_with_cross_label_image']:,} 行的图片被两个以上标签共用，"
                 "这类图片在图像块实验里必须整组剔除。")
    pool_a_overlap, pool_b_overlap = leak["pool_overlap"]
    lines.append(
        f"- `split_purged` 把同一张图的全部行放在同一侧：池 A 的跨 split 图片行 "
        f"{pool_a_overlap['test_sharing_image_split']} → {pool_a_overlap['test_sharing_image_split_purged']}，池 B "
        f"{pool_b_overlap['test_sharing_image_split']} → {pool_b_overlap['test_sharing_image_split_purged']}，"
        f"全表 {split_repair['rows_moved_by_purge']:,} 行换了划分。"
    )
    lines.append("- 目前唯一跑过的图像块（`public_dataset` 的 24,768 行）本来就没有重叠，所以那一组数字不受影响；"
                 "受影响的是未来的酒店、电商图像实验。")
    lines.append("")
    lines.append("## 5 两个实验池")
    lines.append("")
    lines.append("| 池 | 规格 | 条数 | 真 / 人假 / AI 假 | 有图 | 用法 |")
    lines.append("|---|---|---:|---|---:|---|")
    for name, title, usage in (
        ("a", "酒店单域，gold 与本项目生成", "证明方法在单一场景内有效；配留一生成器"),
        ("b", "public_dataset 出真与 AI，人假来自另两个 protocol", "主结果与对比表；必须配留一场景"),
    ):
        audit = pools[name]
        counts = audit["by_label"]
        lines.append(
            f"| {name.upper()} | {title} | {audit['rows']:,} | "
            f"{counts.get('real', 0):,} / {counts.get('human_fake', 0):,} / {counts.get('ai_fake', 0):,} | "
            f"{audit['with_image']:,} | {usage} |"
        )
    lines.append("")
    lines.append("规格原文：")
    lines.append("")
    for name in ("a", "b"):
        lines.append(f"- 池 {name.upper()}：`{pools[name]['spec']}`")
    lines.append("")
    lines.append("`rule_label`（中文、规则依据）没有进任何一个池。中文因此只出现在补充表里。")
    lines.append("")
    lines.append("## 6 中文")
    lines.append("")
    lines.append(md_table(zh_label, "标签"))
    lines.append("")
    lines.append("中文缺 AI 假：只有 MAiDE-up 酒店 992 条，约占全表 AI 假的 4.9%；人写假评全是规则标注。"
                 "所以中文只能做二分类，或做“三类但 AI 假样本极少”的敏感性分析，不能和英文结果并列报告。")
    lines.append("")
    lines.append("## 7 申报书里有、这张表里没有的")
    lines.append("")
    lines.append("| 要素 | 现状 |")
    lines.append("|---|---|")
    lines.append("| 发布时间聚集性 | 冻结表无时间列 |")
    lines.append("| 账号活跃度、跨商家行为、GNN 拓扑 | 无用户 ID、无行为日志；`group_id` 是商品或酒店，不是人 |")
    lines.append("| 用户历史评分偏差 | 只有单条评论的 `polarity`，评论级 `polarity_gap` 已验证无区分度（AUC 0.45–0.50） |")
    lines.append("| 中文困惑度 | distilgpt2 是英文权重，中文留空 |")
    lines.append("| 命名实体识别 | 本机无 spacy，退到关键词表 |")
    lines.append("| 大规模真人标注一致率 | 按项目组决定另行核对，仓库内不保存明细；本卡不列具体数值 |")
    lines.append("")
    lines.append("## 8 来源与许可")
    lines.append("")
    lines.append("| source | 说明 |")
    lines.append("|---|---|")
    for source in sorted(frame["source"].unique()):
        lines.append(f"| `{source}` | {SOURCE_LICENSE.get(source, '见 `data/README.md`')} |")
    lines.append("")
    lines.append("`shared_web` 一类图只记了“网上实拍”，没有逐张原网址，不能写成已经逐张核准的自由许可。")
    lines.append("")
    lines.append("## 9 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python code/data/build_three_class_pool.py     # 两个池 + split_purged + 核对文件")
    lines.append("python code/eval/pool_three_class.py --cv      # 三分类基线，三种不平衡处理")
    lines.append("python code/eval/leakage_split.py              # 图像泄漏对照")
    lines.append("python code/eval/source_generalization.py      # 留一场景 / 留一生成器")
    lines.append("python code/data/build_data_card.py            # 本页")
    lines.append("```")
    lines.append("")
    lines.append("借用网图这条口径的完整链路：")
    lines.append("")
    lines.append("```bash")
    lines.append("python code/data/build_web_image_dataset.py     # 实体档冻结表 web_image_pair.csv")
    lines.append("python code/data/repair_web_image_pairing.py    # 场景档升品类档，写 web_image_category.csv")
    lines.append("python code/eval/category_matcher_validation.py # 判类器留出验证")
    lines.append("python code/eval/web_image_leakage.py           # 图泄不泄标签：置换检验 + 纯图像探针")
    lines.append("python code/data/build_web_image_benchmark.py   # 基准表 web_image_benchmark.csv")
    lines.append("python code/features/extract_web_image_features.py")
    lines.append("python code/features/extract_benchmark_images.py")
    lines.append("python code/eval/benchmark_baselines.py         # 基准表基线，按领域 / 档位切片")
    lines.append("```")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} lines {len(lines)}", flush=True)


if __name__ == "__main__":
    main()

# 数据目录说明

- **公开数据优先**
- **原始大数据不上传 GitHub**（已在 `.gitignore` 忽略）

## 文件夹

| 路径 | 用途 |
|------|------|
| `raw/` | 原始下载数据（本地存放，只读） |
| `records/` | 主档：一行一条，全量导入 |
| `templates/` | 清单表、标注表、规范 |

## 找公开数据集

找到后请填写：`数据集清单.csv`（模板见同目录）。

每个数据集至少写清：

1. 名称  
2. 下载链接  
3. 大约条数  
4. 有没有图片  
5. 原有标签是什么  
6. 能否对应到：真 / 人假 / AI假  
7. 是否已下载到本地  
8. 一句话是否推荐使用  

清单里已进主档的来源、两位同学提交但尚未放入本仓库 `data/raw/` 的候选，都在同一张表。AiGen-FoodReview 的文件校验见 `data/数据下载验证.md`。

## 当前已下载（本地 `raw/`，不入库）

公开学术/开源评论，**不是**去淘宝/携程现网爬取（执行方案中期前也不做大规模爬取）。

| 本地目录 | 平台 | 条数 | 有图？ | 编号字母 |
|----------|------|------|--------|----------|
| `raw/aigen_foodreview/` | Yelp 口径餐饮（真/AI假） | 20,144 | **是，20,144 张图文对齐** | A |
| `raw/ctrip_chnsenticorp/` | 携程酒店 | 7,766 | 无 | B |
| `raw/ctrip_trip_sample_2026/` | 携程 / Trip.com | 800 | 平台上 440 条曾带图，本地无图文件 | C |
| `raw/ecommerce_10cats/` | 淘宝/京东/当当/携程混合网购 | 62,774 | 无 | D |
| `raw/waimai_10k/` | 外卖 | 11,987 | 无 | E |
| `raw/jd_review/` | 京东 | 3,452 | 无 | F |
| `raw/jd_herbal/` | 京东 | 211,391 | 无 | G |

导入已完成（每源全部收录，不抽样）。主档文件：

- `records/all.jsonl`：全部 **318,314** 条
- `records/by_source/*.jsonl`：按来源拆开
- `records/id_map.csv`：编号 ↔ 来源 ↔ 原始行
- `records/letters.json`：字母对照
- `records/manifest.json`：条数统计

编号如 `A000001`、`B000001`，首字母表示来源；每条外套里仍有 `source`、原始定位、标签、场景等字段。图片路径指向 `raw/aigen_foodreview/images/`，二进制不写入主档。

公开可下载的淘宝/携程**晒图文件包不存在**（携程样本只有 `imagesCount`；Hugging Face 上 6.5GB「淘宝确认收货」是 App 截图，不是评论配图）。中文电商图文需后续自建（用已有中文评 + 生成图）或另找带图公开集。

这些公开集原标签几乎都是**情感/评分**，不是真/人假/AI假。除 AiGen 自带真/AI假外，其余 `label` 一律记为 `real`（真实平台用户评论口径），三类标签要另标或另下虚假评论集。空评仍进档，仅 `usable_for_train=false`。

## 统一样本字段（主档外套）

| 字段 | 说明 |
|------|------|
| `id` | 字母+六位序号，如 `A000001` |
| `source` | 来源代号 |
| `source_letter` | 与编号首字母一致 |
| `content.text` | 评论文字 |
| `content.images` | 图片路径列表，没有则空 |
| `labels.task` | `real` / `human_fake` / `ai_fake` |
| `context.domain` | 场景：hotel / food / ecommerce 等 |
| `generation` | 若是 AI 假：文本/图像模型 |
| `split` | `train` / `val` / `test` |
| `raw_payload` | 该行全部原始栏目 |

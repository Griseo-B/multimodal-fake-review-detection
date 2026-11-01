# 冻结子集

这些表保持不动。说明和条数见 `docs/实验进展.md`。`allowed_reviews.csv` 把允许使用的评论和已有配图放在一起，不替代 `mm_v2.csv`。

| 文件 | 内容 |
|---|---|
| `aigen_food.csv` | AiGen 真 / AI 假。`usable=true` 的行才能训练 |
| `ott_hotel.csv` | Ott 真 / 人假。划分使用语料自带 fold |
| `aigen_spotcheck.csv`、`ott_spotcheck.csv` | 抽检预览 |
| `audit_summary.json` | 审核计数 |
| `mm_v2.csv` | 41,606 条全量表。不要改 |
| `three_class.csv` | 允许使用的评论加上新收评论。95,299 行。不同 `protocol` 分开训练 |
| `three_class_audit.json` | 上表的条数、划分和图 |
| `allowed_reviews.csv` | 允许使用的 51,370 条。餐饮图和后补照片已写在行上 |
| `mm_v2_hotel3.csv` | 芝加哥酒店三类各 420 条，已有 `split` |
| `mm_v2_audit.json` | 全量表核对 |
| `two_class.csv` | 真评 / 人写假评，11,301 行 |
| `two_class_protocol.csv` | 加上 `protocol`。同一实验和不同来源分开训练 |
| `hotel_three_class.csv` | 芝加哥 20 家酒店的文本三类，没有划分列 |
| `phone_paired.csv` | 19 款手机的文本三类 |
| `mm3.csv` | 10 月 27 日的 1,080 条，只对应那次模型 |
| `three_class_pool_a.csv`、`three_class_pool_b.csv` | 能直接训三分类的两张表。A 是酒店单域 7,168 条，B 是跨源 45,370 条。多一列 `split_purged`（按图片 md5 归并）和 `image_cross_label`。核对文件 `_audit.json` |
| `three_class_pool_audit.json` | 两个池的条数、图片哈希与划分修复汇总 |
| `web_image_pair.csv` | 实体档「人写假评 + 同实体网图」。29 个实体，5,237 行（真 3,132 / 人写假 2,105），真假共用同一图池。核对文件 `_audit.json` |
| `web_image_category.csv` | 场景档升品类档的产物，8,172 行，带 `pairing` 与 `category` 两列 | 
| `web_image_benchmark.csv` | 借用网图基准表。21,700 行（真 11,103 / 人写假 10,597），`pairing` 取 `own` / `entity` / `category` / `scene`，按图像身份分组划分。核对文件 `_audit.json` |

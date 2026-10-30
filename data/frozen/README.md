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

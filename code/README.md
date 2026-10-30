# 代码

| 目录 | 用途 |
|---|---|
| `data/` | 审核、清洗、建表、配图 |
| `train/` | 训练 |
| `eval/` | 评估 |
| `explain/` | 特征贡献 |

随机种子是 42。大数据和权重不上传 GitHub。哪些脚本还要留，写在 `docs/实验进展.md` 的「代码」一节。

当前入口：

- `python code/data/audit_and_freeze.py`
- `python code/data/build_mm_v2.py`
- `python code/data/build_allowed_reviews.py`
- `python code/data/build_supplement.py`
- `python code/data/build_three_class.py`
- `python code/train/text_baseline.py --table data/frozen/aigen_food.csv --out experiments/20261026_aigen_tfidf --labels real ai_fake`

`text_baseline.py` 只读带 `usable` 和 `split` 的表。

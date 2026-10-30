# 数据

训练和复现用的表在 `frozen/`。原始文件在 `raw/`，缩放后的图在 `processed/`。大文件不上传 GitHub。进展和取舍写在 `docs/实验进展.md`。

## 现在用的

| 路径 | 用途 |
|---|---|
| `frozen/three_class.csv` | 三类训练表，95,299 条。带 `protocol` 和 224 图。不要和 `mm_v2.csv` 混着改 |
| `frozen/mm_v2.csv` | 全量表，41,606 条。不要改 |
| `frozen/allowed_reviews.csv` | 允许使用的评论合在一张表里，并带上已有图片路径 |
| `frozen/mm_v2_hotel3.csv` | 芝加哥酒店三类平衡子集，已有划分 |
| `frozen/two_class.csv`、`two_class_protocol.csv` | 真评 / 人写假评，以及能否混训 |
| `frozen/hotel_three_class.csv`、`phone_paired.csv` | 酒店、手机的文本三类 |
| `frozen/aigen_food.csv`、`ott_hotel.csv` | 审核后的两个子集 |
| `frozen/mm3.csv` | 10 月 27 日的 1,080 条，只对应那次模型 |
| `raw/aigen_foodreview/` | 餐饮原图和原表。MIT |
| `raw/ott/` | Ott 酒店文本。CC BY-NC-SA 3.0 |
| `raw/maide_up/`、`raw/salminen/` | 酒店和电商文本 |
| `raw/human_written_fake/` | 人写假评来源 |
| `raw/hotel_ai/*.csv`、`raw/phone_ai/` | 本研究已经生成的文本 |
| `raw/human_fake_images/` | 酒店和手机后补照片及分配清单 |
| `raw/abo_small/`、`raw/hotel_images/` | 只为重做 `mm3.csv`。酒店图没有许可证，不上传 |
| `raw/models/clip-vit-base-patch32/` | 重跑图像特征时用 |
| `processed/mm_v2/`、`processed/mm3/`、`processed/three_class/` | 已经缩到 224 的图 |

标签和能否进入数据集，以 `templates/标注规范.md` 为准。金标、人工标注和按规则打分都可以用。总览在 `catalog.csv`。本次补进的评论在 `raw/supplement/kept.csv`。

## AiGen 校验

三个 CSV 的 MD5 与 Zenodo v1 一致。

| 文件 | MD5 |
|---|---|
| train.csv | `a4afa633ef14d4a8bae7fb4781e0d6ec` |
| val.csv | `d6d18216e7cabf9a9b956309d4d8e06f` |
| test.csv | `c41a361eafa9800e3e8be3362a0101f4` |
| images.zip | `c4e484d8f709d3943b7aff2f48f581c7` |

来源：<https://zenodo.org/records/10511456>

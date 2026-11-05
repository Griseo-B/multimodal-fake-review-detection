"""Image features for every row of the borrowed-web-image dataset.

`data/frozen/web_image_pair.csv` holds 5,237 reviews over 183 unique web
images. The 20261015_pool_shared_images pass already covered 2,901 of them;
this script fills the remaining 2,336 (all ecommerce real reviews) and writes a
self-contained feature set for the whole table so the leakage test can run on
the complete dataset in one place.

    python code/features/extract_web_image_features.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FEATURE_ROOT, ROOT  # noqa: E402

PREVIOUS = FEATURE_ROOT / "20261015_pool_shared_images"
OUT = FEATURE_ROOT / "20261019_web_image_pair"
TEXT_FEATURES = FEATURE_ROOT / "20261008_all_protocols" / "text_features.parquet"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_frame() -> pd.DataFrame:
    d = pd.read_csv(
        ROOT / "data" / "frozen" / "web_image_pair.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    covered = pd.read_parquet(PREVIOUS / "visual_features.parquet", columns=["id"])
    missing = d.loc[~d["id"].isin(set(covered["id"]))].copy()
    missing["image_file"] = missing["image_224_path"]
    missing["image_link"] = "assignment"
    return missing.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_frame()
    log(f"rows {len(frame)} unique images {frame['image_file'].nunique()}")
    log(f"by domain {frame['domain'].value_counts().to_dict()}")

    timings: dict[str, float] = {}

    started = time.perf_counter()
    from visual_features import STAT_COLUMNS, visual_features

    stats, resnet = visual_features(frame, device)
    stats = stats[STAT_COLUMNS].copy()
    stats.insert(0, "id", frame["id"].to_numpy())
    stats.to_parquet(OUT / "visual_features.parquet", index=False)
    timings["visual"] = round(time.perf_counter() - started, 1)
    log(f"visual {stats.shape} {timings['visual']}s")

    started = time.perf_counter()
    from alignment_features import alignment_frame

    align, clip_embeddings = alignment_frame(frame, device)
    align.to_parquet(OUT / "alignment_features.parquet", index=False)
    timings["align"] = round(time.perf_counter() - started, 1)
    log(f"align {align.shape} {timings['align']}s")

    np.savez_compressed(
        OUT / "embeddings.npz",
        id=np.asarray(frame["id"].tolist(), dtype=str),
        resnet=resnet,
        **clip_embeddings,
    )

    started = time.perf_counter()
    from scene_features import scene_frame

    text = pd.read_parquet(TEXT_FEATURES, columns=["id", "sentiment", "sentiment_zh"])
    scene_input = (
        align.merge(
            frame[["id", "domain", "label", "language", "image_role", "image_link", "text"]],
            on="id",
            how="left",
        )
        .merge(text, on="id", how="left")
        .reset_index(drop=True)
    )
    scene_input["text_sentiment"] = (
        scene_input["sentiment"].fillna(scene_input["sentiment_zh"]).fillna(0.0)
    )
    scene, scene_notes = scene_frame(scene_input, OUT / "embeddings.npz", device)
    scene.to_parquet(OUT / "scene_features.parquet", index=False)
    timings["scene"] = round(time.perf_counter() - started, 1)
    log(f"scene {scene.shape} {timings['scene']}s")

    run = {
        "purpose": "features for rows the 20261015_pool_shared_images pass did not cover",
        "rows": int(len(frame)),
        "unique_images": int(frame["image_file"].nunique()),
        "by_domain": {str(k): int(v) for k, v in frame["domain"].value_counts().items()},
        "by_label": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "timings_sec": timings,
        "notes": {"scene": scene_notes},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()

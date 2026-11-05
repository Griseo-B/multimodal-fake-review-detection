"""Image features for the pool rows the public_dataset run never covered.

The 20261008_public_dataset pass used PAIRED_LINKS == {"paired"}, i.e. only
images that belong to the review itself. Every row still missing an image
feature in pool A and pool B is a shared image: `assignment` (one photo reused
across reviews of the same hotel or product) or `shared` (a same-scene web photo
attached to a human-written fake). There is no other gap, so this script is the
whole remainder: 2,616 pool-A rows over 875 files and 2,905 pool-B rows over
712 files.

The features themselves are identical to the earlier pass. The point of writing
them separately, with `image_role` and `image_link` kept in the output, is that
a shared image must never be read as evidence about that one review: for a
human_fake row the photo is a real photo of the same hotel, and for a real row
it is somebody else's photo of the same hotel. Any model given these features is
given hotel identity, not a fake signal.

    python code/features/extract_pool_images.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FEATURE_ROOT, ROOT, project_path  # noqa: E402

PARTS = ["visual", "align", "scene"]
COVERED = FEATURE_ROOT / "20261008_public_dataset"
TEXT_FEATURES = FEATURE_ROOT / "20261008_all_protocols" / "text_features.parquet"
POOLS = ("a", "b")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_rows() -> pd.DataFrame:
    frames = []
    for name in POOLS:
        frames.append(
            pd.read_csv(
                ROOT / "data" / "frozen" / f"three_class_pool_{name}.csv",
                dtype=str,
                keep_default_na=False,
                low_memory=False,
            )
        )
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    frame = frame.loc[frame["image_224_path"].astype(bool)].copy()
    covered = set(pd.read_parquet(COVERED / "visual_features.parquet", columns=["id"])["id"])
    frame = frame.loc[~frame["id"].isin(covered)].copy()
    frame["image_file"] = frame["image_224_path"]
    frame = frame.reset_index(drop=True)
    log(f"rows {len(frame)} unique images {frame['image_file'].nunique()}")
    log(f"by role {frame['image_role'].value_counts().to_dict()}")
    log(f"by label {frame['label'].value_counts().to_dict()}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FEATURE_ROOT / "20261015_pool_shared_images"))
    parser.add_argument("--parts", default="visual,align,scene")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        raise SystemExit(f"unknown parts: {unknown}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_rows()
    timings: dict[str, float] = {}
    notes: dict[str, object] = {}

    def done(name: str) -> bool:
        return (out / f"{name}.parquet").is_file() and not args.force

    embeddings_path = out / "embeddings.npz"
    embeddings: dict[str, np.ndarray] = {}

    if "visual" in parts and not done("visual_features"):
        started = time.perf_counter()
        from visual_features import STAT_COLUMNS, visual_features

        stats, resnet = visual_features(frame, device)
        stats = stats[STAT_COLUMNS].copy()
        stats.insert(0, "id", frame["id"].to_numpy())
        stats.to_parquet(out / "visual_features.parquet", index=False)
        embeddings["resnet"] = resnet
        timings["visual"] = round(time.perf_counter() - started, 1)
        log(f"visual {stats.shape} {timings['visual']}s")
    elif "visual" in parts:
        log("visual already written")

    if "align" in parts and not done("alignment_features"):
        started = time.perf_counter()
        from alignment_features import alignment_frame

        table, clip_embeddings = alignment_frame(frame, device)
        table.to_parquet(out / "alignment_features.parquet", index=False)
        embeddings.update(clip_embeddings)
        timings["align"] = round(time.perf_counter() - started, 1)
        notes["align"] = {
            "nan_cosine": int(table["clip_cosine"].isna().sum()),
            "nan_sentence": int(table["sent_max_cosine"].isna().sum()),
        }
        log(f"align {table.shape} {timings['align']}s")
    elif "align" in parts:
        log("alignment already written")

    if embeddings:
        # Merge with anything already on disk so a partial rerun does not drop
        # the ResNet matrix when only align is recomputed.
        existing: dict[str, np.ndarray] = {}
        if embeddings_path.is_file():
            payload = np.load(embeddings_path, allow_pickle=True)
            existing = {key: payload[key] for key in payload.files if key != "id"}
        merged = {**existing, **embeddings}
        np.savez_compressed(
            embeddings_path,
            id=np.asarray(frame["id"].tolist(), dtype=str),
            **merged,
        )
        log(f"embeddings {sorted(merged)}")

    if "scene" in parts and not done("scene_features"):
        started = time.perf_counter()
        from scene_features import scene_frame

        alignment = pd.read_parquet(out / "alignment_features.parquet")
        text = pd.read_parquet(TEXT_FEATURES, columns=["id", "sentiment", "sentiment_zh"])
        scene_input = (
            alignment.merge(
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
        table, scene_notes = scene_frame(scene_input, embeddings_path, device)
        table.to_parquet(out / "scene_features.parquet", index=False)
        timings["scene"] = round(time.perf_counter() - started, 1)
        notes["scene"] = scene_notes
        log(f"scene {table.shape} {timings['scene']}s")
    elif "scene" in parts:
        log("scene already written")

    run_path = out / "run.json"
    previous: dict = {}
    if run_path.is_file():
        try:
            previous = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    timings = {**previous.get("timings_sec", {}), **timings}
    notes = {**previous.get("notes", {}), **notes}

    run = {
        "purpose": "image features for pool rows whose image is shared, not review-owned",        "rows": int(len(frame)),
        "unique_images": int(frame["image_file"].nunique()),
        "by_label": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "by_role": {str(k): int(v) for k, v in frame["image_role"].value_counts().items()},
        "by_link": {str(k): int(v) for k, v in frame["image_link"].value_counts().items()},
        "by_domain": {str(k): int(v) for k, v in frame["domain"].value_counts().items()},
        "by_pool": {
            name: int(frame["id"].isin(
                set(pd.read_csv(
                    ROOT / "data" / "frozen" / f"three_class_pool_{name}.csv",
                    usecols=["id"], dtype=str, keep_default_na=False,
                )["id"])
            ).sum())
            for name in POOLS
        },
        "parts": sorted(set(previous.get("parts", [])) | set(parts)),
        "timings_sec": timings,
        "notes": notes,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": 42,
        "frozen_table_modified": False,
        "warning": (
            "shared images belong to the hotel or product, not to the review. "
            "A feature computed here is evidence about the picture, never about "
            "whether this review is fake."
        ),
    }
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

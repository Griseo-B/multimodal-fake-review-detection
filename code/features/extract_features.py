"""Run the feature extraction for one protocol.

    python code/features/extract_features.py --protocol public_dataset \
        --out data/processed/features/20261030_public_dataset

Each part writes its own file, so a rerun skips finished parts unless --force
is given. Image parts only see rows whose image belongs to that review.
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

from common import FEATURE_ROOT, PAIRED_LINKS, TABLE, image_path_of  # noqa: E402

PARTS = ["text", "perplexity", "smoothness", "visual", "align"]
USE_COLUMNS = [
    "id", "subset", "domain", "label", "language", "protocol", "split",
    "text", "image_path", "image_224_path", "image_role", "image_link",
    "n_tokens", "pos_noun", "pos_adj", "pos_adv", "pos_pron", "pos_verb",
    "polarity",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_table(protocol: str, limit: int) -> pd.DataFrame:
    frame = pd.read_csv(TABLE, usecols=USE_COLUMNS, dtype=str, keep_default_na=False)
    if protocol != "all":
        frame = frame.loc[frame["protocol"] == protocol].copy()
    if limit:
        frame = frame.head(limit).copy()
    frame["image_file"] = [image_path_of(row) for row in frame.itertuples(index=False)]
    frame = frame.reset_index(drop=True)
    log(f"rows {len(frame)} protocol {protocol}")
    return frame


def image_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame["image_link"].isin(PAIRED_LINKS) & frame["image_file"].astype(bool)
    return frame.loc[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="public_dataset")
    parser.add_argument("--out", default="")
    parser.add_argument("--parts", default="text,perplexity,smoothness,visual,align")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    # Rows already covered by an earlier feature folder are skipped, so a
    # table that grew after the first run only costs the new rows.
    parser.add_argument("--skip-covered", action="append", default=[])
    args = parser.parse_args()

    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        raise SystemExit(f"unknown parts: {unknown}")

    out = Path(args.out) if args.out else FEATURE_ROOT / f"{time.strftime('%Y%m%d')}_{args.protocol}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_table(args.protocol, args.limit)
    if args.skip_covered:
        covered: set[str] = set()
        for folder in args.skip_covered:
            path = Path(folder) / "text_features.parquet"
            if path.is_file():
                covered |= set(pd.read_parquet(path, columns=["id"])["id"].astype(str))
        before = len(frame)
        frame = frame.loc[~frame["id"].astype(str).isin(covered)].reset_index(drop=True)
        log(f"skip-covered removed {before - len(frame)} rows, {len(frame)} left")
    embeddings: dict[str, np.ndarray] = {}
    embedding_ids: list[str] = []
    timings: dict[str, float] = {}
    notes: dict[str, dict] = {}

    def done(name: str) -> bool:
        return (out / f"{name}.parquet").is_file() and not args.force

    if "text" in parts and not done("text_features"):
        started = time.perf_counter()
        from text_features import text_features

        table = text_features(frame)
        table.insert(0, "id", frame["id"].to_numpy())
        table.to_parquet(out / "text_features.parquet", index=False)
        timings["text"] = round(time.perf_counter() - started, 1)
        log(f"text features {table.shape} {timings['text']}s")
    elif "text" in parts:
        log("text features already written")

    if "perplexity" in parts and not done("perplexity"):
        started = time.perf_counter()
        from perplexity import perplexity_frame

        table, skipped = perplexity_frame(frame, device)
        table.to_parquet(out / "perplexity.parquet", index=False)
        timings["perplexity"] = round(time.perf_counter() - started, 1)
        notes["perplexity"] = skipped
        log(f"perplexity {len(table)} {timings['perplexity']}s skipped {skipped}")
    elif "perplexity" in parts:
        log("perplexity already written")

    if "smoothness" in parts and not done("smoothness"):
        started = time.perf_counter()
        from smoothness import smoothness_frame

        table = smoothness_frame(frame, device)
        table.to_parquet(out / "smoothness.parquet", index=False)
        timings["smoothness"] = round(time.perf_counter() - started, 1)
        notes["smoothness"] = {"nan": int(table["smoothness"].isna().sum())}
        log(f"smoothness {len(table)} {timings['smoothness']}s")
    elif "smoothness" in parts:
        log("smoothness already written")

    paired = image_rows(frame)
    log(f"rows with a review-owned image {len(paired)}")

    if "visual" in parts and paired.empty is False:
        if not done("visual_features"):
            started = time.perf_counter()
            from visual_features import visual_features

            stats, resnet = visual_features(paired, device)
            stats.insert(0, "id", paired["id"].to_numpy())
            stats.to_parquet(out / "visual_features.parquet", index=False)
            embeddings["resnet"] = resnet
            timings["visual"] = round(time.perf_counter() - started, 1)
            log(f"visual features {stats.shape} {timings['visual']}s")
        else:
            log("visual features already written")

    if "align" in parts and paired.empty is False:
        if not done("alignment_features"):
            started = time.perf_counter()
            from alignment_features import alignment_frame

            table, clip_embeddings = alignment_frame(paired, device)
            table.to_parquet(out / "alignment_features.parquet", index=False)
            embeddings.update(clip_embeddings)
            timings["align"] = round(time.perf_counter() - started, 1)
            notes["align"] = {
                "nan_cosine": int(table["clip_cosine"].isna().sum()),
                "nan_sentence": int(table["sent_max_cosine"].isna().sum()),
            }
            log(f"alignment {table.shape} {timings['align']}s")
        else:
            log("alignment already written")

    if embeddings:
        np.savez_compressed(
            out / "embeddings.npz",
            id=np.asarray(paired["id"].tolist(), dtype=str),
            **embeddings,
        )

    run_path = out / "run.json"
    previous: dict = {}
    if run_path.is_file():
        try:
            previous = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    timings = {**previous.get("timings_sec", {}), **timings}
    notes = {**previous.get("notes", {}), **notes}
    written_parts = sorted(set(previous.get("parts", [])) | set(parts))

    run = {
        "protocol": args.protocol,
        "table": str(TABLE.relative_to(TABLE.parents[2])),
        "rows": int(len(frame)),
        "rows_with_paired_image": int(len(paired)),
        "by_label": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "by_domain": {str(k): int(v) for k, v in frame["domain"].value_counts().items()},
        "by_language": {str(k): int(v) for k, v in frame["language"].value_counts().items()},
        "by_split": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
        "parts": written_parts,
        "timings_sec": timings,
        "notes": notes,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": 42,
        "frozen_table_modified": False,
    }
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

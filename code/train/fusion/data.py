"""Assemble one modelling table from the frozen pools and cached feature files.

The frozen tables are read-only here. Every block is left-joined on `id`, and
a missing block is recorded as a mask rather than imputed, because the reason a
row has no image is not random (real reviews have images 23% of the time,
human fakes 76%).

Pool definitions stay exactly as `code/data/build_three_class_pool.py` wrote
them:
    pool_a      hotel only, gold and this study's generation, 7,168 rows
    pool_b      public_dataset for real/AI plus the two human-fake protocols
    benchmark   borrowed-web-photo table, binary real vs human fake
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import IMAGE_BLOCK, columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "data" / "processed" / "features"

TEXT_DIRS = [
    FEATURES / "20261030_all_protocols",
    FEATURES / "20261101_missing_text",
    FEATURES / "20261030_public_dataset",
]
IMAGE_DIRS = [
    FEATURES / "20261030_public_dataset",
    FEATURES / "20261031_pool_shared_images",
    FEATURES / "20261101_web_image_benchmark",
]

TABLES = {
    "pool_a": ROOT / "data" / "frozen" / "three_class_pool_a.csv",
    "pool_b": ROOT / "data" / "frozen" / "three_class_pool_b.csv",
    "benchmark": ROOT / "data" / "frozen" / "web_image_benchmark.csv",
}

META_COLUMNS = [
    "id", "label", "domain", "language", "protocol", "group_id",
    "image_role", "generator", "source", "text",
]


def load_meta(pool: str) -> pd.DataFrame:
    table = TABLES[pool]
    header = pd.read_csv(table, nrows=0).columns
    usecols = [name for name in META_COLUMNS if name in header]
    frame = pd.read_csv(table, usecols=usecols, dtype=str, keep_default_na=False)
    if pool == "benchmark":
        # The benchmark table is binary by construction; keep the same label
        # vocabulary so downstream code does not branch.
        frame["label"] = np.where(frame["label"].eq("human_fake"), "human_fake", "real")
        frame["protocol"] = "borrowed_web_photo"
        split = pd.read_csv(table, usecols=["id", "split"], dtype=str)
        frame = frame.merge(split, on="id", how="left")
        frame["image_role"] = "own_or_borrowed"
        frame["generator"] = ""
        frame["source"] = frame.get("source", "")
        frame["domain"] = frame.get("domain", "")
        frame["group_id"] = frame.get("group_id", frame["id"])
        frame["language"] = frame.get("language", "")
    else:
        split_column = "split_purged" if "split_purged" in header else "split"
        split = pd.read_csv(table, usecols=["id", split_column], dtype=str)
        frame = frame.merge(split.rename(columns={split_column: "split"}), on="id", how="left")
    for column in ("generator", "source", "group_id", "domain", "language"):
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.loc[frame["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    frame["group_id"] = frame["group_id"].where(frame["group_id"].ne(""), frame["id"])
    return frame


def _concat(directory_list: list[Path], filename: str) -> pd.DataFrame:
    parts = []
    for folder in directory_list:
        path = folder / filename
        if path.is_file():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    table["id"] = table["id"].astype(str)
    # First copy wins: the all-protocol run predates the pilot folder and uses
    # the same code, so duplicates are identical apart from float noise.
    return table.drop_duplicates("id", keep="first")


def load_text_blocks() -> pd.DataFrame:
    text = _concat(TEXT_DIRS, "text_features.parquet")
    perplexity = _concat(TEXT_DIRS, "perplexity.parquet")
    smoothness = _concat(TEXT_DIRS, "smoothness.parquet")
    out = text
    for extra in (perplexity, smoothness):
        if extra.empty:
            continue
        columns = [name for name in extra.columns if name == "id" or name not in out.columns]
        out = out.merge(extra[columns], on="id", how="left")
    return out


def load_image_blocks() -> pd.DataFrame:
    visual = _concat(IMAGE_DIRS, "visual_features.parquet")
    alignment = _concat(IMAGE_DIRS, "alignment_features.parquet")
    scene = _concat(IMAGE_DIRS, "scene_features.parquet")
    out = visual
    for extra in (alignment, scene):
        if extra.empty:
            continue
        columns = [name for name in extra.columns if name == "id" or name not in out.columns]
        out = out.merge(extra[columns], on="id", how="left")
    return out


def load_embeddings() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Clip and ResNet embeddings, keyed by block name."""
    store: dict[str, list[np.ndarray]] = {"clip_text": [], "clip_image": [], "resnet": []}
    ids: list[np.ndarray] = []
    for folder in IMAGE_DIRS:
        path = folder / "embeddings.npz"
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True)
        block_ids = payload["id"].astype(str)
        ids.append(block_ids)
        for name in store:
            store[name].append(payload[name].astype(np.float32))
    if not ids:
        return {}
    flat_ids = np.concatenate(ids)
    _, first = np.unique(flat_ids, return_index=True)
    first.sort()
    out = {}
    for name, parts in store.items():
        matrix = np.concatenate(parts)[first]
        out[name] = (flat_ids[first], matrix)
    return out


def load_llm_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return payload["id"].astype(str), payload["embedding"].astype(np.float32)


def attach_embeddings(frame: pd.DataFrame, source: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the matrix aligned to frame order plus a presence mask."""
    ids, matrix = source
    lookup = {value: position for position, value in enumerate(ids)}
    index = np.array([lookup.get(str(value), -1) for value in frame["id"]], dtype=np.int64)
    present = index >= 0
    aligned = np.zeros((len(frame), matrix.shape[1]), dtype=np.float32)
    aligned[present] = matrix[index[present]]
    return aligned, present


def build(pool: str, llm_path: Path | None = None, with_images: bool = True) -> dict:
    meta = load_meta(pool)
    text = load_text_blocks()
    text_columns = [name for name in text.columns if name != "id"]
    frame = meta.merge(text, on="id", how="left")
    image = pd.DataFrame()
    if with_images:
        image = load_image_blocks()
        merge_columns = [name for name in image.columns if name == "id" or name not in frame.columns]
        frame = frame.merge(image[merge_columns], on="id", how="left")

    embeddings = load_embeddings()
    blocks: dict[str, dict] = {}
    if with_images:
        # Low-level, composition, scene and alignment columns live in the frame
        # after the merge; they become one gate group of their own.
        image_columns = [name for name in columns(IMAGE_BLOCK) if name in frame.columns]
        if image_columns:
            matrix = frame[image_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float32)
            present = ~np.isnan(matrix).all(axis=1)
            blocks[IMAGE_BLOCK] = {"matrix": matrix, "present": present,
                                   "dim": matrix.shape[1]}
    for name, source in embeddings.items():
        matrix, present = attach_embeddings(frame, source)
        blocks[name] = {"matrix": matrix, "present": present, "dim": matrix.shape[1]}
    if llm_path is not None:
        candidate = Path(llm_path)
        if candidate.is_dir():
            candidate = candidate / "llm_embedding.npz"
        if candidate.is_file():
            matrix, present = attach_embeddings(frame, load_llm_embeddings(candidate))
            blocks["llm"] = {"matrix": matrix, "present": present, "dim": matrix.shape[1]}

    # Sorting makes the row order independent of how the frozen table was
    # written; the same permutation must be applied to every block.
    perm = np.argsort(frame["id"].to_numpy(), kind="stable")
    ordered = frame.iloc[perm].reset_index(drop=True)
    for name, block in blocks.items():
        block["matrix"] = block["matrix"][perm]
        block["present"] = block["present"][perm]
    return {
        "frame": ordered,
        "blocks": blocks,
        "text_columns": text_columns,
        "pool": pool,
        "source_table": str(TABLES[pool].relative_to(ROOT)),
    }


if __name__ == "__main__":
    for name in TABLES:
        data = build(name)
        frame = data["frame"]
        print(f"== {name}: {len(frame)} rows")
        print(frame["label"].value_counts().to_dict())
        print(frame["domain"].value_counts().to_dict())
        print(frame["split"].value_counts().to_dict())
        for block, payload in data["blocks"].items():
            print(f"   {block}: dim {payload['dim']}, present {int(payload['present'].sum())}")

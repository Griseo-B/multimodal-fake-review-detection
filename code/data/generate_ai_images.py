"""Paired Stable Diffusion 1.5 images for half of the AI reviews that had no picture.

Within each subset, rows are sorted by group_id. The first half gets an image.
The second half stays without one. Food AiGen already has DALL-E images and is
left unchanged. This script does not edit mm_v2.csv.
"""

from __future__ import annotations

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "frozen" / "mm_v2.csv"
OUT = ROOT / "data" / "raw" / "generated_ai_images_20261030"
IMAGE_DIR = OUT / "images"
SMALL_DIR = ROOT / "data" / "processed" / "generated_ai_images_20261030"
MANIFEST = OUT / "manifest.csv"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LOCAL_MODEL = ROOT / "data" / "raw" / "models" / "stable-diffusion-v1-5"
WEIGHTS = {
    "text_encoder/model.safetensors": 492_265_874,
    "unet/diffusion_pytorch_model.safetensors": 3_438_167_540,
    "vae/diffusion_pytorch_model.safetensors": 334_643_276,
}
CONFIGS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
)

HOTELS = {
    "affinia": "Affinia Chicago",
    "allegro": "Hotel Allegro Chicago",
    "amalfi": "Amalfi Hotel Chicago",
    "ambassador": "Ambassador East Hotel",
    "conrad": "Conrad Chicago",
    "fairmont": "Fairmont Chicago",
    "hardrock": "Hard Rock Hotel Chicago",
    "hilton": "Hilton Chicago",
    "homewood": "Homewood Suites Chicago Downtown",
    "hyatt": "Hyatt Regency Chicago",
    "intercontinental": "InterContinental Chicago",
    "james": "The James Chicago",
    "knickerbocker": "Millennium Knickerbocker Hotel",
    "monaco": "Hotel Monaco Chicago",
    "omni": "Omni Chicago Hotel",
    "palmer": "The Palmer House Hilton",
    "sheraton": "Sheraton Chicago",
    "sofitel": "Sofitel Chicago",
    "swissotel": "Swissotel Chicago",
    "talbott": "The Talbott Hotel",
}

VISUAL = (
    ("早餐", "a hotel breakfast table"),
    ("浴室", "a hotel bathroom"),
    ("浴缸", "a hotel bathroom"),
    ("大堂", "a hotel lobby"),
    ("前台", "a hotel front desk"),
    ("床", "a hotel bed"),
    ("房间", "a hotel guest room"),
    ("景观", "a view from a hotel window"),
    ("breakfast", "a hotel breakfast table"),
    ("bathroom", "a hotel bathroom"),
    ("lobby", "a hotel lobby"),
    ("bed", "a hotel bed"),
    ("room", "a hotel guest room"),
    ("view", "a view from a hotel window"),
    ("pool", "a hotel swimming pool"),
)

FIELDS = [
    "group_id", "subset", "domain", "text_generator", "prompt", "seed",
    "image_path", "image_224_path", "image_role", "image_generator",
]


def hotel_name(subset: str, group_id: str) -> str:
    if subset == "hotel_ai":
        parts = group_id.split("-")
        key = parts[-2] if parts and parts[-1].isdigit() else parts[-1]
        return HOTELS.get(key, key.replace("_", " "))
    body = group_id.split("-", 2)[-1]
    name, _, tail = body.rpartition("-")
    return name if tail.isdigit() and name else body


def visual_phrase(text: str) -> str:
    lowered = text.lower()
    for key, phrase in VISUAL:
        if key in text or key in lowered:
            return phrase
    return "a hotel guest room"


def product_phrase(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)
    kept = [word for word in words if word.lower() not in {"the", "and", "for", "with", "this", "that", "have", "had", "was", "were", "are", "bought", "gift"}]
    phrase = " ".join(kept[:8]) or "consumer product"
    return phrase


def prompt_for(row: pd.Series) -> str:
    if row["domain"] == "hotel":
        place = hotel_name(row["subset"], row["group_id"])
        return (
            f"a realistic photograph of {visual_phrase(str(row['text']))} at {place}, "
            "natural light, no text, no watermark, no logo"
        )
    return (
        f"a realistic product photograph of {product_phrase(str(row['text']))}, "
        "on a plain table, soft light, no text, no watermark, no logo"
    )


def file_stem(group_id: str) -> str:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:16]
    return digest


def download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    done = temp.stat().st_size if temp.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if done:
        headers["Range"] = f"bytes={done}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        if done and status == 200:
            done = 0
        length = int(response.headers.get("Content-Length") or 0)
        total = done + length if status == 206 else length
        mode = "ab" if done and status == 206 else "wb"
        with temp.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total and done // (64 * 1024 * 1024) != (done - len(chunk)) // (64 * 1024 * 1024):
                    print(f"  {dest.name} {done // (1024 * 1024)}/{total // (1024 * 1024)} MB", flush=True)
    temp.replace(dest)


def ensure_model() -> Path:
    for name in CONFIGS:
        dest = LOCAL_MODEL / name
        if dest.exists() and dest.stat().st_size > 20:
            continue
        url = f"https://hf-mirror.com/{MODEL_ID}/resolve/main/{name}"
        print(f"download {name}", flush=True)
        download_file(url, dest)
    for name, expected in WEIGHTS.items():
        dest = LOCAL_MODEL / name
        if dest.exists() and dest.stat().st_size == expected:
            continue
        if dest.exists():
            dest.unlink()
        url = f"https://hf-mirror.com/{MODEL_ID}/resolve/main/{name}"
        print(f"download {name}", flush=True)
        download_file(url, dest)
        if dest.stat().st_size != expected:
            raise RuntimeError(f"{name} size {dest.stat().st_size} != {expected}")
    index_path = LOCAL_MODEL / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop("safety_checker", None)
    index.pop("feature_extractor", None)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return LOCAL_MODEL


def keep_image_half(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for subset, group in frame.groupby("subset", sort=False):
        group = group.sort_values("group_id")
        keep = (len(group) + 1) // 2
        print(f"half {subset} {keep}/{len(group)}", flush=True)
        chosen.append(group.iloc[:keep])
    if not chosen:
        return frame.iloc[0:0]
    return pd.concat(chosen, ignore_index=True)


def load_pending() -> pd.DataFrame:
    frame = pd.read_csv(
        TABLE,
        usecols=["subset", "domain", "label", "text", "image_role", "group_id", "generator"],
    )
    pending = frame[(frame["label"] == "ai_fake") & (frame["image_role"] == "none")].copy()
    pending = pending.sort_values(["subset", "group_id"]).reset_index(drop=True)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subset", default="")
    args = parser.parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    pending = load_pending()
    if args.subset:
        pending = pending[pending["subset"] == args.subset]
    pending = keep_image_half(pending)
    done = set()
    if MANIFEST.exists():
        done = set(pd.read_csv(MANIFEST)["group_id"].astype(str))
    todo = pending[~pending["group_id"].astype(str).isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"pending {len(pending)} todo {len(todo)}", flush=True)
    if todo.empty:
        return

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        ensure_model(),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if not MANIFEST.exists():
        pd.DataFrame(columns=FIELDS).to_csv(MANIFEST, index=False)

    started = time.perf_counter()
    for index, record in enumerate(todo.itertuples(index=False), start=1):
        image_started = time.perf_counter()
        stem = file_stem(record.group_id)
        raw_path = IMAGE_DIR / f"{stem}.jpg"
        small_path = SMALL_DIR / f"{stem}.jpg"
        prompt = prompt_for(pd.Series(record._asdict()))
        seed = int(hashlib.sha1(record.group_id.encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt,
            negative_prompt="text, watermark, logo, cartoon, collage, blurry",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
        ).images[0]
        image.save(raw_path, "JPEG", quality=90)
        small = image.convert("RGB")
        small.thumbnail((224, 224))
        canvas = Image.new("RGB", (224, 224), (255, 255, 255))
        canvas.paste(small, ((224 - small.width) // 2, (224 - small.height) // 2))
        canvas.save(small_path, "JPEG", quality=90)
        row = {
            "group_id": record.group_id,
            "subset": record.subset,
            "domain": record.domain,
            "text_generator": record.generator,
            "prompt": prompt,
            "seed": seed,
            "image_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "image_224_path": str(small_path.relative_to(ROOT)).replace("\\", "/"),
            "image_role": "paired_sd",
            "image_generator": "stable-diffusion-1.5",
        }
        pd.DataFrame([row]).to_csv(MANIFEST, mode="a", header=False, index=False)
        elapsed = time.perf_counter() - started
        print(
            f"saved {index}/{len(todo)} {record.subset} {stem} "
            f"{time.perf_counter() - image_started:.1f}s avg {elapsed / index:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

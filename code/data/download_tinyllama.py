"""Resume the TinyLlama weight file, then save a local folder the generator can load."""

from __future__ import annotations

import shutil
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "hotel_ai" / "tinyllama"
BLOB = (
    Path.home()
    / ".cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/blobs"
    / "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933.incomplete"
)
URL = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"
TOTAL = 2_200_119_864
SMALL = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def resume() -> None:
    have = BLOB.stat().st_size if BLOB.exists() else 0
    print(f"start {have}/{TOTAL}", flush=True)
    if have > TOTAL:
        raise SystemExit(f"local file is larger than the remote file: {have}")
    if have == TOTAL:
        print("weights already complete", flush=True)
        return
    with requests.get(URL, headers={"Range": f"bytes={have}-"}, stream=True, timeout=120) as resp:
        print("status", resp.status_code, resp.headers.get("content-range"), flush=True)
        if resp.status_code != 206:
            raise SystemExit(2)
        BLOB.parent.mkdir(parents=True, exist_ok=True)
        with BLOB.open("ab") as handle:
            got = 0
            for chunk in resp.iter_content(1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                got += len(chunk)
                if got // (50 * 1024 * 1024) != (got - len(chunk)) // (50 * 1024 * 1024):
                    print(f"have {have + got}", flush=True)
    size = BLOB.stat().st_size
    print(f"weights {size}", flush=True)
    if size != TOTAL:
        raise SystemExit(f"download stopped at {size}")


def save_local() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/{name}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
        print("saved", name, target.stat().st_size, flush=True)
    weights = DEST / "model.safetensors"
    if not weights.exists() or weights.stat().st_size != TOTAL:
        shutil.copyfile(BLOB, weights)
    print("TINY_OK", DEST, flush=True)


if __name__ == "__main__":
    resume()
    save_local()

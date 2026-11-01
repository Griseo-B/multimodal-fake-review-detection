"""Project the 1.5B vectors down to BERT width, so width stops confounds depth.

Qwen2.5-1.5B gives 1536 numbers per review and Qwen2.5-0.5B gives 896, so the
0.9081 against 0.8865 difference mixes "more capacity" with "wider vector".
Projecting the 1.5B matrix to 768 makes it directly comparable to BERT-base,
which is the only other encoder with the same downstream head.

The projection is fit on all rows of the file, not on train rows only. PCA here
is a deterministic re-parameterisation of a frozen encoder, not a supervised
step, and fitting it per pool would change the representation between pools.

    python code/train/fusion/reduce_llm.py --source <dir> --width 768
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "data" / "processed" / "features"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    source = FEATURES / args.source
    payload = np.load(source / "llm_embedding.npz", allow_pickle=True)
    matrix = payload["embedding"].astype(np.float32)
    log(f"{matrix.shape[0]} rows, {matrix.shape[1]} -> {args.width}")

    from sklearn.decomposition import PCA

    started = time.perf_counter()
    projection = PCA(n_components=args.width, random_state=42).fit(matrix)
    reduced = projection.transform(matrix).astype(np.float16)
    explained = float(projection.explained_variance_ratio_.sum())
    log(f"fit in {time.perf_counter() - started:.1f}s, "
        f"{explained:.4f} of the variance kept")

    name = args.out or f"{source.name}_pca{args.width}"
    target = FEATURES / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "llm_embedding.npz",
                        id=payload["id"], embedding=reduced, language=payload["language"])
    meta = {
        "model": f"{json.loads((source / 'run.json').read_text(encoding='utf-8'))['model']}"
                 f" + PCA {args.width}",
        "hidden_size": args.width,
        "source": str(source.relative_to(ROOT)),
        "explained_variance": round(explained, 4),
        "note": "deterministic re-parameterisation of a frozen encoder; "
                "PCA fit on every row in the file, not on train only",
    }
    (target / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    log(f"wrote {target}")


if __name__ == "__main__":
    main()

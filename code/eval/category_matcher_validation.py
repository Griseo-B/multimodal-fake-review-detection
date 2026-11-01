"""How accurate is the category matcher used to re-pair fake reviews?

`repair_web_image_pairing.py` labels a fake review's cuisine / product category
from keywords, then gives it a photo of that category. If the labeler is wrong
the "correspondence" is fictional, so it has to be measured rather than
asserted.

The 125 Dianping shops we collected come with the search query we used to find
them, which *is* the cuisine label, and their reviews are real, human-written
text about that cuisine. That is a free held-out test set: run the same
classifier on those review texts and compare with the true query.

    python code/eval/category_matcher_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLLECTED = ROOT / "data" / "raw" / "supplement" / "collected_20261029"
OUT = ROOT / "experiments" / "20261101_web_image_pair"


def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pairing", ROOT / "code" / "data" / "repair_web_image_pairing.py"
    )
    pairing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairing)

    rows = []
    for folder in sorted(COLLECTED.glob("dianping_auto*")):
        meta = folder / "dianping_reviews.json"
        if not meta.is_file():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        for shop in data.get("shops", []):
            truth = shop.get("query") or ""
            if truth not in pairing.CUISINE_KEYWORDS:
                continue
            for review in shop.get("reviews", []):
                text = review.get("text") or ""
                if len(text) < 10:
                    continue
                pred, hits = pairing.classify(text, pairing.CUISINE_KEYWORDS)
                rows.append({"truth": truth, "pred": pred or "", "hits": hits, "text": text})

    table = pd.DataFrame(rows)
    called = table[table["pred"] != ""]
    correct = int((called["pred"] == called["truth"]).sum())

    per_class = {}
    for cuisine in sorted(table["truth"].unique()):
        sub = table[table["truth"] == cuisine]
        called_sub = sub[sub["pred"] != ""]
        per_class[cuisine] = {
            "reviews": int(len(sub)),
            "called": int(len(called_sub)),
            "coverage": round(len(called_sub) / max(len(sub), 1), 3),
            "precision": round(float((called_sub["pred"] == cuisine).mean()), 3) if len(called_sub) else None,
        }

    confusion = (
        called.groupby(["truth", "pred"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).head(25)
    )
    report = {
        "held_out_reviews": int(len(table)),
        "called": int(len(called)),
        "coverage": round(len(called) / max(len(table), 1), 3),
        "precision": round(correct / max(len(called), 1), 3),
        "per_class": per_class,
        "top_confusions": confusion.to_dict(orient="records"),
        "note": (
            "Ground truth is the Dianping search query of each shop we crawled, i.e. the "
            "cuisine the shop actually serves. Reviews are genuine user text, not fake ones, "
            "so this measures the labeler, not the dataset."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "category_matcher_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"reviews {report['held_out_reviews']} called {report['called']} "
          f"coverage {report['coverage']} precision {report['precision']}")
    for cuisine, info in per_class.items():
        print(f"  {cuisine:8s} n={info['reviews']:4d} called={info['called']:4d} "
              f"cov={info['coverage']:.2f} prec={info['precision']}")


if __name__ == "__main__":
    main()

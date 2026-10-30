"""Twenty template explanations from the frozen TF-IDF model.

Image-text similarity is left blank until CLIP features exist.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def top_terms(vectorizer, model, text: str, k: int = 5) -> str:
    matrix = vectorizer.transform([text])
    coef = model.coef_[0]
    contribution = matrix.multiply(coef).tocsr()
    rows, cols = contribution.nonzero()
    pairs = sorted(((contribution[rows[i], cols[i]], cols[i]) for i in range(len(cols))), reverse=True)
    names = vectorizer.get_feature_names_out()
    positive_class = model.classes_[1]
    negative_class = model.classes_[0]
    positive = [f"{names[index]} ({value:.3f})" for value, index in pairs[:k]]
    negative = [f"{names[index]} ({value:.3f})" for value, index in pairs[-k:]]
    return f"推向 {positive_class}: " + ", ".join(positive) + f" | 推向 {negative_class}: " + ", ".join(negative)


def main() -> None:
    bundle = joblib.load(ROOT / "experiments" / "20261026_aigen_tfidf" / "model_tfidf.joblib")
    frame = pd.read_csv(ROOT / "data" / "frozen" / "aigen_food.csv")
    test = frame.loc[
        frame["usable"].astype(str).isin(["True", "true", "1"]) & (frame["split"] == "test")
    ].copy()
    pred = bundle["model"].predict(bundle["vectorizer"].transform(test["text"]))
    test["prediction"] = pred
    wrong = test.loc[test["prediction"] != test["label"]]
    correct = test.loc[test["prediction"] == test["label"]].sample(20, random_state=42)
    chosen = pd.concat([wrong, correct], ignore_index=True).head(20)
    rewrite = pd.read_csv(ROOT / "experiments" / "20261026_rewrite_tfidf" / "rewritten_200.csv")
    changed = rewrite.loc[rewrite["pred_original"] != rewrite["pred_rewritten"]]
    rows = []
    for row in chosen.itertuples(index=False):
        rows.append(
            {
                "source_id": row.source_id,
                "label": row.label,
                "prediction": row.prediction,
                "kind": "原句",
                "top_features": top_terms(bundle["vectorizer"], bundle["model"], row.text),
                "image_width": row.image_width,
                "image_height": row.image_height,
                "image_text_similarity": "尚未抽取 CLIP，不填写匹配分",
                "text_preview": str(row.text).replace("\n", " ")[:160],
            }
        )
    for row in changed.head(max(0, 20 - len(rows))).itertuples(index=False):
        rows.append(
            {
                "source_id": row.source_id,
                "label": row.label,
                "prediction": row.pred_rewritten,
                "kind": "改写后",
                "top_features": top_terms(bundle["vectorizer"], bundle["model"], row.rewritten),
                "image_width": "",
                "image_height": "",
                "image_text_similarity": "尚未抽取 CLIP，不填写匹配分",
                "text_preview": str(row.rewritten).replace("\n", " ")[:160],
            }
        )
    out = ROOT / "experiments" / "20261026_explanations"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).head(20).to_csv(out / "explanations_20.csv", index=False)
    print("wrote", len(rows), "wrong_in_test", len(wrong))


if __name__ == "__main__":
    main()

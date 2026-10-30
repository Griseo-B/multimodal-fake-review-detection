"""Attention fusion with a two-level three-class head.

The success gate is test Macro-F1 >= 0.90. TF-IDF, each feature stream,
and a size-only model are scored on the same split so a high score can be
compared with simpler explanations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn

from mm3_common import EXPERIMENT, FEATURE_DIR, LABELS, ROOT, SEED, TABLE


def set_seed(seed: int) -> None:
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def metric_row(name: str, y_true, y_pred) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    row = {
        "model": name,
        "split": "test",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
    }
    for label, prec, rec in zip(LABELS, precision, recall):
        row[f"precision_{label}"] = float(prec)
        row[f"recall_{label}"] = float(rec)
    return row


def logreg_predict(x_train, y_train, x_test):
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    model.fit(scaler.fit_transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


class Fusion(nn.Module):
    """Attention over per-stream logits, so a strong text stream can keep its vote."""

    def __init__(self, widths: list[int], classes: int = 3):
        super().__init__()
        self.level1 = nn.ModuleList(nn.Linear(width, 2) for width in widths)
        self.level2 = nn.ModuleList(nn.Linear(width, 2) for width in widths)
        self.flat = nn.ModuleList(nn.Linear(width, classes) for width in widths)
        self.score = nn.ModuleList(nn.Linear(width, 1) for width in widths)

    def forward(self, streams: list[torch.Tensor]):
        scores = torch.cat([layer(stream) for layer, stream in zip(self.score, streams)], dim=1)
        weights = torch.softmax(scores, dim=1)
        level1 = torch.stack([layer(stream) for layer, stream in zip(self.level1, streams)], dim=1)
        level2 = torch.stack([layer(stream) for layer, stream in zip(self.level2, streams)], dim=1)
        flat = torch.stack([layer(stream) for layer, stream in zip(self.flat, streams)], dim=1)
        mix = weights.unsqueeze(-1)
        return (level1 * mix).sum(1), (level2 * mix).sum(1), (flat * mix).sum(1), weights


def batches(index: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(index)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def predict_flat_two_step(flat: torch.Tensor, threshold: float) -> np.ndarray:
    prob = torch.softmax(flat, dim=1).detach().cpu().numpy()
    kind = np.where(prob[:, 1] >= prob[:, 2], "human_fake", "ai_fake")
    return np.where(prob[:, 0] >= threshold, "real", kind)


def predict_two_level(level1: torch.Tensor, level2: torch.Tensor, threshold: float) -> np.ndarray:
    real_prob = torch.softmax(level1, dim=1)[:, 0].cpu().numpy()
    fake_kind = torch.softmax(level2, dim=1).argmax(dim=1).cpu().numpy()
    predictions = np.where(fake_kind == 0, "human_fake", "ai_fake")
    predictions = np.where(real_prob >= threshold, "real", predictions)
    return predictions


def anchor_tfidf(model: Fusion, x_train, y_train) -> None:
    """Freeze a TF-IDF two-level head and start attention on that stream."""
    y_train = np.asarray(y_train)
    binary = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    binary.fit(x_train, (y_train != "real").astype(np.int64))
    fake = y_train != "real"
    kind = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    kind.fit(x_train[fake], np.where(y_train[fake] == "human_fake", 0, 1))
    flat = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    flat.fit(x_train, y_train)

    def copy_binary(layer, clf) -> None:
        coef = clf.coef_.ravel()
        intercept = float(clf.intercept_.ravel()[0])
        weight = np.vstack([-coef, coef]).astype(np.float32)
        bias = np.array([-intercept, intercept], dtype=np.float32)
        layer.weight.data.copy_(torch.tensor(weight, device=layer.weight.device))
        layer.bias.data.copy_(torch.tensor(bias, device=layer.bias.device))
        layer.weight.requires_grad_(False)
        layer.bias.requires_grad_(False)

    copy_binary(model.level1[0], binary)
    copy_binary(model.level2[0], kind)
    order = [list(flat.classes_).index(label) for label in LABELS]
    model.flat[0].weight.data.copy_(torch.tensor(flat.coef_[order], dtype=torch.float32, device=model.flat[0].weight.device))
    model.flat[0].bias.data.copy_(torch.tensor(flat.intercept_[order], dtype=torch.float32, device=model.flat[0].bias.device))
    model.flat[0].weight.requires_grad_(False)
    model.flat[0].bias.requires_grad_(False)
    for index in range(1, len(model.level1)):
        for layer in (model.level1[index], model.level2[index], model.flat[index]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    for layer in model.score:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    model.score[0].bias.data.fill_(4.0)


def train_fusion(streams, y_label, train_idx, val_idx, device, anchor=None):
    label_to_i = {label: index for index, label in enumerate(LABELS)}
    y = np.asarray([label_to_i[label] for label in y_label])
    y1 = (y != 0).astype(np.int64)
    y2 = np.zeros(len(y), dtype=np.int64)
    y2[y == 2] = 1
    model = Fusion([stream.shape[1] for stream in streams]).to(device)
    if anchor is not None:
        anchor(model)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3, weight_decay=1e-3)
    tensors = [torch.tensor(stream, dtype=torch.float32, device=device) for stream in streams]
    y1_t = torch.tensor(y1, dtype=torch.long, device=device)
    y2_t = torch.tensor(y2, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    counts1 = np.bincount(y1[train_idx], minlength=2).astype(np.float32)
    fake_train = train_idx[y1[train_idx] == 1]
    counts2 = np.bincount(y2[fake_train], minlength=2).astype(np.float32)
    counts3 = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weight1 = torch.tensor(counts1.sum() / np.clip(counts1, 1, None), dtype=torch.float32, device=device)
    weight2 = torch.tensor(counts2.sum() / np.clip(counts2, 1, None), dtype=torch.float32, device=device)
    weight3 = torch.tensor(counts3.sum() / np.clip(counts3, 1, None), dtype=torch.float32, device=device)
    rng = np.random.default_rng(SEED)
    best_state = None
    best_score = -1.0
    wait = 0

    def consider() -> None:
        nonlocal best_state, best_score, wait
        model.eval()
        with torch.no_grad():
            _, _, flat, _ = model([tensor[val_idx] for tensor in tensors])
        pred = np.asarray(LABELS)[flat.argmax(dim=1).cpu().numpy()]
        score = macro_f1(np.asarray(y_label)[val_idx], pred)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

    consider()
    for _ in range(60):
        model.train()
        for batch in batches(train_idx, 64, rng):
            optimizer.zero_grad()
            level1, level2, flat, _ = model([tensor[batch] for tensor in tensors])
            loss1 = nn.functional.cross_entropy(level1, y1_t[batch], weight=weight1)
            fake = y1_t[batch] == 1
            loss2 = (
                nn.functional.cross_entropy(level2[fake], y2_t[batch][fake], weight=weight2)
                if bool(fake.any())
                else torch.zeros((), device=device)
            )
            loss3 = nn.functional.cross_entropy(flat, y_t[batch], weight=weight3)
            (loss1 + loss2 + loss3).backward()
            optimizer.step()
        consider()
        if wait >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, tensors


def tune_threshold(score_fn, grid) -> float:
    best_threshold, best_score = float(grid[0]), -1.0
    for candidate in grid:
        value = score_fn(float(candidate))
        if value > best_score:
            best_threshold, best_score = float(candidate), value
    return best_threshold


def bootstrap(y_true, y_pred) -> dict:
    rng = np.random.default_rng(SEED)
    scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(400):
        chosen = rng.integers(0, len(y_true), len(y_true))
        scores.append(macro_f1(y_true[chosen], y_pred[chosen]))
    return {
        "mean": float(np.mean(scores)),
        "low": float(np.quantile(scores, 0.025)),
        "high": float(np.quantile(scores, 0.975)),
    }


def main() -> None:
    set_seed(SEED)
    frame = pd.read_csv(TABLE)
    features = np.load(FEATURE_DIR / "features.npz", allow_pickle=True)
    if not np.array_equal(frame["id"].to_numpy(), features["id"]):
        raise SystemExit("feature rows are not aligned with the frozen table")
    train = frame["split"].to_numpy() == "train"
    val = frame["split"].to_numpy() == "val"
    test = frame["split"].to_numpy() == "test"
    y = frame["label"].to_numpy()
    hand_raw = features["hand"].astype(np.float32)
    medians = np.nanmedian(hand_raw[train], axis=0)
    missing = np.isnan(hand_raw)
    hand_raw[missing] = np.take(medians, np.where(missing)[1])
    hand = hand_raw.copy()
    scaler = StandardScaler()
    hand[train] = scaler.fit_transform(hand[train])
    hand[val] = scaler.transform(hand[val])
    hand[test] = scaler.transform(hand[test])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)
    tfidf_train = vectorizer.fit_transform(frame.loc[train, "text"])
    tfidf_val = vectorizer.transform(frame.loc[val, "text"])
    tfidf_test = vectorizer.transform(frame.loc[test, "text"])
    tfidf_dense = np.zeros((len(frame), tfidf_train.shape[1]), dtype=np.float32)
    tfidf_dense[np.flatnonzero(train)] = tfidf_train.toarray()
    tfidf_dense[np.flatnonzero(val)] = tfidf_val.toarray()
    tfidf_dense[np.flatnonzero(test)] = tfidf_test.toarray()
    text_scaler = StandardScaler()
    tfidf_dense[train] = text_scaler.fit_transform(tfidf_dense[train]).astype(np.float32)
    tfidf_dense[val] = text_scaler.transform(tfidf_dense[val]).astype(np.float32)
    tfidf_dense[test] = text_scaler.transform(tfidf_dense[test]).astype(np.float32)
    streams = [
        tfidf_dense,
        features["clip_text"],
        features["clip_image"],
        features["resnet"],
        hand,
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = np.flatnonzero(train)
    val_idx = np.flatnonzero(val)
    test_idx = np.flatnonzero(test)
    model, tensors = train_fusion(
        streams,
        y,
        train_idx,
        val_idx,
        device,
        anchor=lambda net: anchor_tfidf(net, tfidf_dense[train], y[train]),
    )
    with torch.no_grad():
        level1_val, level2_val, flat_val, _ = model([tensor[val_idx] for tensor in tensors])
        level1_test, level2_test, flat_test, weights = model([tensor[test_idx] for tensor in tensors])
    attention = weights.mean(dim=0).detach().cpu()
    branch_threshold = tune_threshold(
        lambda value: macro_f1(y[val], predict_two_level(level1_val, level2_val, value)),
        np.linspace(0.35, 0.75, 9),
    )
    step_threshold = tune_threshold(
        lambda value: macro_f1(y[val], predict_flat_two_step(flat_val, value)),
        np.linspace(0.15, 0.85, 15),
    )
    flat_val_pred = np.asarray(LABELS)[flat_val.argmax(dim=1).cpu().numpy()]
    flat_pred = np.asarray(LABELS)[flat_test.argmax(dim=1).cpu().numpy()]
    two_pred = predict_two_level(level1_test, level2_test, branch_threshold)
    step_pred = predict_flat_two_step(flat_test, step_threshold)
    val_choice = {
        "fusion_flat": macro_f1(y[val], flat_val_pred),
        "fusion_two_step": macro_f1(y[val], predict_flat_two_step(flat_val, step_threshold)),
        "fusion_branch_two_level": macro_f1(y[val], predict_two_level(level1_val, level2_val, branch_threshold)),
    }
    operating_name = max(val_choice, key=val_choice.get)
    operating = {"fusion_flat": flat_pred, "fusion_two_step": step_pred, "fusion_branch_two_level": two_pred}[operating_name]
    rows = [
        metric_row("tfidf_logreg", y[test], logreg_predict(tfidf_train.toarray(), y[train], tfidf_test.toarray())),
        metric_row("size_only", y[test], logreg_predict(
            frame.loc[train, ["orig_width", "orig_height"]].to_numpy(dtype=np.float32),
            y[train],
            frame.loc[test, ["orig_width", "orig_height"]].to_numpy(dtype=np.float32),
        )),
        metric_row("clip_text", y[test], logreg_predict(features["clip_text"][train], y[train], features["clip_text"][test])),
        metric_row("clip_image", y[test], logreg_predict(features["clip_image"][train], y[train], features["clip_image"][test])),
        metric_row("resnet", y[test], logreg_predict(features["resnet"][train], y[train], features["resnet"][test])),
        metric_row("handcrafted", y[test], logreg_predict(hand_raw[train], y[train], hand_raw[test])),
        metric_row("fusion_branch_two_level", y[test], two_pred),
        metric_row("fusion_flat", y[test], flat_pred),
        metric_row("fusion_two_step", y[test], step_pred),
    ]
    domain_rows = []
    test_frame = frame.loc[test].reset_index(drop=True)
    for domain in sorted(test_frame["domain"].unique()):
        chosen = test_frame["domain"].to_numpy() == domain
        domain_rows.append(metric_row(f"{operating_name}[{domain}]", y[test][chosen], operating[chosen]))
    dedup_index = test_frame.groupby("group_id", sort=False).head(1).index.to_numpy()
    dedup = metric_row(f"{operating_name}_one_per_group", y[test][dedup_index], operating[dedup_index])
    interval = bootstrap(y[test], operating)
    dedup_interval = bootstrap(y[test][dedup_index], operating[dedup_index])
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows + domain_rows + [dedup])
    table.to_csv(EXPERIMENT / "metrics.csv", index=False)
    torch.save(model.state_dict(), EXPERIMENT / "fusion.pt")
    full_f1 = macro_f1(y[test], operating)
    dedup_f1 = float(dedup["macro_f1"])
    passed = bool(full_f1 >= 0.90 and dedup_f1 >= 0.90)
    summary = {
        "gate": "validation-chosen fusion rule has test macro-F1 >= 0.90 and one-per-group macro-F1 >= 0.90",
        "operating_model": operating_name,
        "val_macro_f1": val_choice,
        "passed": passed,
        "branch_threshold": branch_threshold,
        "two_step_threshold": step_threshold,
        "attention_mean_test": {
            name: float(value)
            for name, value in zip(["tfidf", "clip_text", "clip_image", "resnet", "handcrafted"], attention)
        },
        "bootstrap_macro_f1": interval,
        "bootstrap_one_per_group": dedup_interval,
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "n_test_one_per_group": int(len(dedup_index)),
    }
    (EXPERIMENT / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

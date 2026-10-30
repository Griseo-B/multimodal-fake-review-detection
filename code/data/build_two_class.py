"""Build one real / human-written-fake table from data already on disk.

Li 2014 restaurant and doctor files are not included. Caught paid reviews,
rule labels, and model-generated text are not included. The output does not
replace mm_v2.csv.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "frozen" / "two_class.csv"
AUDIT = ROOT / "data" / "frozen" / "two_class_audit.json"

COLUMNS = ["id", "text", "label", "language", "source", "subtype", "polarity", "item"]
PRIORITY = ["ott", "blt", "soldner", "four_city", "clips", "derev"]


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def add(rows: list[dict], **fields: str) -> None:
    text = normalize(fields["text"])
    if len(text) < 20:
        return
    fields["text"] = text
    rows.append(fields)


def load_ott(rows: list[dict]) -> None:
    root = RAW / "ott" / "op_spam_v1.4"
    for path in root.rglob("*.txt"):
        parts = path.parts
        polarity = "pos" if any("positive_polarity" in part for part in parts) else "neg"
        label = "human_fake" if any("deceptive" in part for part in parts) else "real"
        hotel = path.stem.split("_")[1]
        add(
            rows,
            text=path.read_text(encoding="utf-8", errors="replace"),
            label=label,
            language="en",
            source="ott",
            subtype="mturk" if label == "human_fake" else "published",
            polarity=polarity,
            item=hotel,
        )


def load_four_city(rows: list[dict]) -> None:
    folder = RAW / "human_written_fake" / "four_city"
    with zipfile.ZipFile(folder / "fake.zip") as archive:
        for name in archive.namelist():
            if not name.endswith(".txt") or name.startswith("__MACOSX") or "/." in name:
                continue
            city = Path(name).stem
            for line in archive.read(name).decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                payload = json.loads(line)
                add(
                    rows,
                    text=payload["Answer"]["review_text"],
                    label="human_fake",
                    language="en",
                    source="four_city",
                    subtype="mturk",
                    polarity="pos",
                    item=city,
                )
    with zipfile.ZipFile(folder / "true.zip") as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            city = Path(name).stem.split("_")[0]
            for line in archive.read(name).decode("utf-8", "replace").splitlines():
                add(
                    rows,
                    text=line,
                    label="real",
                    language="en",
                    source="four_city",
                    subtype="published",
                    polarity="pos",
                    item=city,
                )


def load_soldner(rows: list[dict]) -> None:
    path = RAW / "human_written_fake" / "soldner2022" / "review_data.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rating = int(float(record["review_rating"]))
            if rating >= 4:
                polarity = "pos"
            elif rating <= 2:
                polarity = "neg"
            else:
                polarity = "neu"
            source_name = record["data_source_label"]
            veracity = record["veracity_label"]
            ownership = record["ownership_label"]
            if source_name == "prolific" and veracity == "truthful" and ownership == "owner":
                label, subtype = "real", "owner_truthful"
            elif source_name == "amazon" and veracity == "truthful":
                label, subtype = "real", "verified_purchase"
            elif source_name == "prolific" and veracity == "deceptive" and ownership == "owner":
                label, subtype = "human_fake", "owner_opposite"
            elif source_name == "prolific" and veracity == "deceptive" and ownership == "non_owner":
                label, subtype = "human_fake", "non_owner"
            else:
                continue
            add(
                rows,
                text=record["review_text"],
                label=label,
                language="en",
                source="soldner",
                subtype=subtype,
                polarity=polarity,
                item=record["brand"],
            )


def load_blt(rows: list[dict]) -> None:
    raw = (RAW / "human_written_fake" / "original" / "blt-c-2016.tar").read_bytes()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
        member = archive.extractfile("blt-c/data/BLT-C_Boulder_Lies_and_Truths_Corpus.csv")
        text = member.read().decode("utf-8", "replace")
    for record in csv.DictReader(io.StringIO(text)):
        if record["Rejected"] != "-":
            continue
        if "PLAGIARIZED" in record["Plagiarized"]:
            continue
        truth = record["Truth Value"]
        if truth == "T":
            label, subtype = "real", "knows_item"
        elif truth == "F":
            label, subtype = "human_fake", "known_opposite"
        elif truth == "D":
            label, subtype = "human_fake", "unknown_item"
        else:
            continue
        polarity = record["Sentiment Polarity"]
        if polarity not in {"pos", "neg"}:
            continue
        domain = record["Domain"]
        if domain not in {"Electronics", "Hotels"}:
            continue
        add(
            rows,
            text=record["Review"],
            label=label,
            language="en",
            source="blt",
            subtype=subtype,
            polarity=polarity,
            item=domain.lower(),
        )


def load_clips(rows: list[dict]) -> None:
    raw = (RAW / "human_written_fake" / "original" / "csi-2014.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            if "__MACOSX" in name or "/reviews/" not in name or not name.endswith(".txt"):
                continue
            stem = Path(name).stem
            if "Truth" in stem:
                label = "real"
            elif "Decep" in stem:
                label = "human_fake"
            else:
                continue
            polarity = "pos" if stem.endswith("_Pos") else "neg" if stem.endswith("_Neg") else ""
            if not polarity:
                continue
            body = archive.read(name).decode("utf-8", "replace")
            add(
                rows,
                text=body,
                label=label,
                language="nl",
                source="clips",
                subtype="task",
                polarity=polarity,
                item="",
            )


ATTR = re.compile(r'\b(author|gold2016)="([^"]*)"')
BODY = re.compile(r"<body>(.*)</body>", re.DOTALL)
OBJECT = re.compile(r"<object>(.*)</object>", re.DOTALL)


def load_derev(rows: list[dict]) -> None:
    path = RAW / "human_written_fake" / "original" / "derev-2020.zip"
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            head = archive.read(name).decode("latin-1", "replace")
            found = dict(ATTR.findall(head[:800]))
            author = found.get("author", "")
            gold = found.get("gold2016", "")
            if author == "TURKER":
                label, subtype = "human_fake", "turker"
            elif gold == "1":
                label, subtype = "real", "innocent_book"
            else:
                continue
            body = BODY.search(head)
            obj = OBJECT.search(head)
            if body is None:
                continue
            add(
                rows,
                text=body.group(1),
                label=label,
                language="en",
                source="derev",
                subtype=subtype,
                polarity="",
                item=normalize(obj.group(1)) if obj else "",
            )


def dedupe(rows: list[dict]) -> tuple[list[dict], dict]:
    rank = {name: index for index, name in enumerate(PRIORITY)}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(normalize(row["text"]).casefold(), []).append(row)
    kept = []
    conflict = 0
    duplicate = 0
    for group in grouped.values():
        labels = {row["label"] for row in group}
        if len(labels) > 1:
            conflict += len(group)
            continue
        group.sort(key=lambda row: rank[row["source"]])
        duplicate += len(group) - 1
        kept.append(group[0])
    kept.sort(key=lambda row: (rank[row["source"]], row["label"], row["text"][:80]))
    for index, row in enumerate(kept, start=1):
        row["id"] = f"{row['source']}-{index:05d}"
    stats = {"conflict_rows": conflict, "duplicate_rows_removed": duplicate, "kept": len(kept)}
    return kept, stats


def main() -> None:
    rows: list[dict] = []
    load_ott(rows)
    load_four_city(rows)
    load_soldner(rows)
    load_blt(rows)
    load_clips(rows)
    load_derev(rows)
    before = Counter((row["source"], row["label"], row["subtype"]) for row in rows)
    kept, dedupe_stats = dedupe(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    audit = {
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "loaded": {f"{source}|{label}|{subtype}": count for (source, label, subtype), count in sorted(before.items())},
        "dedupe": dedupe_stats,
        "kept_by_source_label": dict(Counter(f"{row['source']}|{row['label']}" for row in kept)),
        "kept_by_language_label": dict(Counter(f"{row['language']}|{row['label']}" for row in kept)),
        "excluded": [
            "DeRev gold2016=0 caught paid reviews",
            "BLT rejected and plagiarized rows",
            "Li 2014 restaurant and doctor files, author page did not open",
            "rule labels, platform filters, and model-generated text",
        ],
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

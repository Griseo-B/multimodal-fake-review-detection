"""Mark which two-class rows were written inside one experiment.

same_experiment: the real reviews and the fake reviews were both produced
by the study's writing task.
different_origin: fake reviews were tasked, but the real reviews were taken
from a public website. Those labels stay, and they are not mixed into the
same-experiment training pool.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "frozen" / "two_class.csv"
OUT = ROOT / "data" / "frozen" / "two_class_protocol.csv"
AUDIT = ROOT / "data" / "frozen" / "two_class_protocol_audit.json"

SAME = {
    ("blt", "knows_item"),
    ("blt", "known_opposite"),
    ("blt", "unknown_item"),
    ("clips", "task"),
    ("soldner", "owner_truthful"),
    ("soldner", "owner_opposite"),
    ("soldner", "non_owner"),
}


def protocol_for(row: dict) -> str:
    if (row["source"], row["subtype"]) in SAME:
        return "same_experiment"
    return "different_origin"


def main() -> None:
    with SRC.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0]) + ["protocol"]
    for row in rows:
        row["protocol"] = protocol_for(row)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["protocol"], row["language"], row["label"]) for row in rows)
    audit = {
        "rows": len(rows),
        "counts": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(counts.items())},
        "note": "different_origin rows keep their labels and stay out of same-experiment training.",
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

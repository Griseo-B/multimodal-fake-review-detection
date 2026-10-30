"""Shared constants for the three-domain, three-class multimodal set."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
PER_CELL = 120
LABELS = ["real", "human_fake", "ai_fake"]
DOMAINS = ["food", "hotel", "ecommerce"]

TABLE = ROOT / "data" / "frozen" / "mm3.csv"
IMAGE_DIR = ROOT / "data" / "processed" / "mm3" / "images"
FEATURE_DIR = ROOT / "data" / "processed" / "mm3"
HOTEL_AI_CSV = ROOT / "data" / "raw" / "hotel_ai" / "reviews.csv"
EXPERIMENT = ROOT / "experiments" / "20261027_mm3"

"""P3C constants. No system retune."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p3c"
REPORTS = ROOT / "reports" / "P3C"
BENCH = ROOT / "benchmarks" / "p3c"
CACHE = RESULTS / "cache"

SEED = 20260826
PRIMARY_MODEL = "qwen3:8b"
PRIMARY_DIGEST = "500a1f067a9f"
PRIMARY_PROMPT = "v2"
GEMMA_GEN = "gemma3:12b"

HIST_RAW_EXACT = 0.779
HIST_CANONICAL = 0.841

# Unused HARTH subjects from the P3 freeze (never evaluated).
HARTH_CLOSURE_SUBJECTS = ("S009", "S010", "S013")
HARTH_P3_EVAL_SUBJECTS = ("S006", "S008", "S012", "S015", "S020", "S021")

MARGIN_BANDS = (
    ("exact", 0.0),
    ("pct_0_1", 0.001),
    ("pct_0_25", 0.0025),
    ("pct_0_5", 0.005),
    ("pct_1", 0.01),
    ("pct_2", 0.02),
    ("pct_5", 0.05),
    ("pct_10", 0.10),
)

# Grouped for the required terminal bands.
MARGIN_GROUP = {
    "extremely_near": ("exact", "pct_0_1", "pct_0_25"),
    "near": ("pct_0_5", "pct_1"),
    "moderate": ("pct_2",),
    "clear": ("pct_5", "pct_10"),
}

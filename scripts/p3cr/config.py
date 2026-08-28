"""P3C-R constants. No HARTH/EC/holdout. No prior-blind tuning."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p3cr"
REPORTS = ROOT / "reports" / "P3CR"
BENCH = ROOT / "benchmarks" / "p3cr"
CACHE = RESULTS / "cache"

SEED = 20260827
PRIMARY_MODEL = "qwen3:8b"
PRIMARY_DIGEST = "500a1f067a9f"
HISTORICAL_PROMPT = "v2"
GEMMA_GEN = "gemma3:12b"
THIRD_GEN = "llama3.1:8b"

OPS = (
    "dominant_frequency",
    "rms_amplitude",
    "peak_amplitude",
    "signal_range",
    "trend_ratio",
    "periodicity_strength",
    "spectral_energy_ratio_low",
    "cross_channel_lag_ms",
)

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

MARGIN_GROUP = {
    "extremely_near": ("pct_0_1", "pct_0_25"),
    "near": ("pct_0_5", "pct_1"),
    "moderate": ("pct_2",),
    "clear": ("pct_5", "pct_10"),
}

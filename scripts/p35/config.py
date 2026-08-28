"""P3.5 constants. No prior-blind tuning. No holdout."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p35"
REPORTS = ROOT / "reports" / "P3_5_IR"
BENCH = ROOT / "benchmarks" / "p35"
CACHE = RESULTS / "cache"

SEED = 20260828
PRIMARY_MODEL = "qwen3:8b"
PRIMARY_DIGEST = "500a1f067a9f"
BASELINE_PROMPT = "v2"
IR_PROMPT_ID = "ir_interface"

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

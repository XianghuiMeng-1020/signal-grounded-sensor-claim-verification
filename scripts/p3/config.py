"""P3 constants. Frozen before P3 test metrics are inspected."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p3"
REPORTS = ROOT / "reports" / "P3"
BENCH_P3 = ROOT / "benchmarks" / "p3"
CACHE = RESULTS / "cache"
SEED = 20270823
PRIMARY_MODEL = "qwen3:8b"
PRIMARY_DIGEST = "500a1f067a9f"
PRIMARY_PROMPT = "v2"
SECONDARY_MODEL = "gemma3:12b"
RAW_V3_EXACT = 0.9426229508196722

FORBIDDEN_SPLITS = ("final_sealed_holdout",)
HOLDOUT_MARKERS = ("holdout", "final_sealed")

# Measurement output domains (mathematical).
# vs_value asserted_value must lie in the output domain to be executable.
# vs_threshold thresholds may be any real number.
MEASUREMENT_DOMAIN = {
    "dominant_frequency": (0.0, None),  # [0, Nyquist]; Nyquist checked with fs
    "rms_amplitude": (0.0, None),
    "peak_amplitude": (0.0, None),
    "signal_range": (0.0, None),
    "trend_ratio": (0.0, None),
    "cross_channel_lag_ms": (None, None),  # signed; magnitude limited by ±30 samples
    "periodicity_strength": (0.0, 1.0),
    "spectral_energy_ratio_low": (0.0, 1.0),
}

# M1 tolerance: DEVELOPMENT-only candidate. Selected before P3 test inspection.
# Multiplier on production tolerance_for(op, actual).
M1_CANDIDATES = (0.25, 0.5, 1.0, 1.5)
MARGIN_BANDS = (
    ("extremely_near", 0.005),
    ("near", 0.01),
    ("moderate", 0.02),
    ("clear_5", 0.05),
    ("clear_10", 0.10),
)

# External dataset freeze (written before PRIMARY on that set).
EXTERNAL_DATASET = "HARTH"
EXTERNAL_FS = 50.0
EXTERNAL_CHANNELS = ("back_accel", "thigh_accel")
EXTERNAL_SUBJECT_GROUPS = {
    "p3_external_eval": ("S006", "S008", "S012", "S015", "S020", "S021"),
    "p3_external_unused": ("S009", "S010", "S013"),
}

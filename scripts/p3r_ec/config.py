"""P3R-EC constants. Contract numbers come from the written P2R spec, not P3 1040."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p3r_ec"
REPORTS = ROOT / "reports" / "P3R_EC"
BENCH = ROOT / "benchmarks" / "p3r_ec"

SEED_DEV = 20260824
SEED_BLIND = 20260825

# Written P2R contract table (reports/P2R_REPAIR/02_EVIDENCE_CONTRACTS.md).
# Duplicated here so the independent oracle does not import production check_contract.
ORACLE_SPEC = {
    "rms_amplitude": {"min_n": 1, "required_fs": False, "n_channels": 1, "need_var": False, "output": "nonneg"},
    "peak_amplitude": {"min_n": 1, "required_fs": False, "n_channels": 1, "need_var": False, "output": "nonneg"},
    "signal_range": {"min_n": 1, "required_fs": False, "n_channels": 1, "need_var": False, "output": "nonneg"},
    "trend_ratio": {"min_n": 4, "required_fs": False, "n_channels": 1, "need_var": False, "output": "nonneg"},
    "dominant_frequency": {"min_n": 4, "required_fs": True, "n_channels": 1, "need_var": False, "output": "hz"},
    "spectral_energy_ratio_low": {"min_n": 4, "required_fs": True, "n_channels": 1, "need_var": False, "output": "unit"},
    "periodicity_strength": {"min_n": 8, "required_fs": False, "n_channels": 1, "need_var": False, "output": "unit"},
    "cross_channel_lag_ms": {"min_n": 61, "required_fs": True, "n_channels": 2, "need_var": True, "output": "lag"},
}

LAG_MAX_SAMPLES = 30
OPS = tuple(ORACLE_SPEC.keys())

HARD_INVALID = (
    "missing_required_channel",
    "required_all_nan",
    "empty_channel",
    "insufficient_n",
    "invalid_fs",
    "missing_fs",
)
MISSINGNESS = (
    "sparse_nan_required",
    "contiguous_gap_required",
    "multiple_gaps_required",
    "partial_required_dropout",
    "async_pair_support",
)
VALID_CTRL = (
    "clean",
    "mild_noise",
    "quantize_8bit",
    "scale_x2",
    "resample_x2_meta_ok",
    "unused_channel_nan",
    "unused_channel_dropout",
    "truncate_above_min",
)

FORBIDDEN_P3_STRESS = "perturbation_cases_head.json"

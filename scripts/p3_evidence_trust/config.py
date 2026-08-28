"""Phase 3 Evidence Trust — pre-registered constants. Not tuned on outcomes."""
from __future__ import annotations

from pathlib import Path

from p2.config import VALIDATION_TOL, dominant_freq_abs_tol
from p2_phase2.config import MAX_PER_DATASET, MAX_WINDOWS, OPS, UNITS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p3_evidence_trust"
REPORTS = ROOT / "reports" / "P3_EVIDENCE_TRUST"

SEED = 20260825
EXPERIMENT_IDS = {
    "m3": "p3e3_kernel_numerical_v1",
    "m1": "p3e1_contract_necessity_v1",
    "m2": "p3e2_decision_margin_v1",
}

MARGIN_K = (0.25, 0.5, 1.0, 2.0, 4.0)
E1_AWGN10_SUPPORTED = 614
E1_N = 768

ABLATIONS = (
    "drop_nonfinite",
    "drop_min_n",
    "drop_fs",
    "drop_second_channel",
    "drop_equal_length",
    "drop_variance",
    "drop_output_domain",
)

# Primitive → applicable one-clause ablations (frozen contract fields).
ABLATION_APPLIES = {
    "drop_nonfinite": OPS,
    "drop_min_n": (
        "trend_ratio",
        "dominant_frequency",
        "spectral_energy_ratio_low",
        "periodicity_strength",
        "cross_channel_lag_ms",
    ),
    "drop_fs": (
        "dominant_frequency",
        "spectral_energy_ratio_low",
        "cross_channel_lag_ms",
    ),
    "drop_second_channel": ("cross_channel_lag_ms",),
    "drop_equal_length": ("cross_channel_lag_ms",),
    "drop_variance": ("cross_channel_lag_ms",),
    "drop_output_domain": OPS,
}

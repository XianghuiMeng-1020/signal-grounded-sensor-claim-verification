"""Phase 2 Experiment 1 — pre-registered constants.

Frozen before any degraded verdict is inspected. Not tuned on outcomes.
ClaimProgram, evidence contracts, DSP kernels, and claim thresholds are not
edited by this experiment. Only the waveform x[n] is perturbed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p2_phase2"
REPORTS = ROOT / "reports" / "P2_PHASE2"

SEED = 20260825
EXPERIMENT_ID = "p2e1_waveform_degradation_v1"

# Window pool: later-offset unused windows; no holdout; no prior scored cells.
MAX_WINDOWS = 96
MAX_PER_DATASET = 32

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

# Clean-SUPPORTED construction: gt, threshold = v − 1·tol(v). Not optimized.
THRESHOLD_TOL_MULT = 1.0

# Waveform operators. Names and levels frozen here.
SNR_DB = (20, 10, 0)
DROPOUT_P = 0.10
CLIP_FRAC = 0.60

UNITS = {
    "dominant_frequency": "Hz",
    "rms_amplitude": "raw",
    "peak_amplitude": "raw",
    "signal_range": "raw",
    "trend_ratio": "ratio",
    "periodicity_strength": "score_0_1",
    "spectral_energy_ratio_low": "fraction",
    "cross_channel_lag_ms": "ms",
}

PERTURBATIONS = (
    "awgn_snr20",
    "awgn_snr10",
    "awgn_snr0",
    "dropout_10pct",
    "clip_0p60",
)

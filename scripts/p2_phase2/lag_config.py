"""Phase 2 Experiment 2 — pre-registered lag time-base constants.

Frozen before any Mode A / Mode B verdict is inspected. Not tuned on outcomes.
Kernels, contracts, and production Kleene stay frozen. Only an evaluation mode
rewrites how a lag numeric threshold is interpreted (samples vs milliseconds).
"""
from __future__ import annotations

from pathlib import Path

from p2.config import LAG_MAX_SAMPLES
from p2r.pipeline import (
    EVAL_MODE_LAG_PHYSICAL,
    EVAL_MODE_LAG_SAMPLE,
    EVAL_MODE_PRODUCTION,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p2_phase2"
REPORTS = ROOT / "reports" / "P2_PHASE2"

SEED = 20260825
EXPERIMENT_ID = "p2e2_lag_timebase_v1"

MAX_WINDOWS = 96
MAX_PER_DATASET = 32

# Native rates of the three locked corpora, and the evaluation rate grid.
FS_GRID = (20.0, 50.0, 100.0)

# Mode A: fixed ±30 samples (production search radius), interpreted as samples.
SAMPLE_BOUND = int(LAG_MAX_SAMPLES)
# Mode B: the same radius converted at 100 Hz, the highest native rate.
# 30 samples / 100 Hz = 300 ms. Not taken from a scored outcome.
PHYSICAL_BOUND_MS = float(SAMPLE_BOUND) * 1000.0 / 100.0

EVAL_PRODUCTION = EVAL_MODE_PRODUCTION
EVAL_SAMPLE = EVAL_MODE_LAG_SAMPLE
EVAL_PHYSICAL = EVAL_MODE_LAG_PHYSICAL

# Physical delays that are integer samples at every rate in FS_GRID.
# In-box vs out-of-box is a consequence of ±30 samples, not a tuned ladder.
DELAYS_MS = (0, 100, 200, 300, 400, 600, 800, 1500, 1600)

# Independent-gold lag sample tolerance (frozen P2), used only as a diagnostic
# for measurement faithfulness. Not a Kleene threshold.
FAITHFUL_ABS_SAMPLES = 0.51

# Production lag contract: n >= 2*30+1.
MIN_N_LAG = 2 * SAMPLE_BOUND + 1


def delay_samples(delay_ms: float, fs: float) -> int:
    return int(round(float(delay_ms) * float(fs) / 1000.0))


def sample_bound_ms(fs: float) -> float:
    return float(SAMPLE_BOUND) * 1000.0 / float(fs)


def in_sample_box(delay_ms: float, fs: float) -> bool:
    return abs(delay_samples(delay_ms, fs)) <= SAMPLE_BOUND


def faithful_tol_ms(fs: float) -> float:
    return FAITHFUL_ABS_SAMPLES / float(fs) * 1000.0

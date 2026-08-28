"""P4 HAR→claim baseline — pre-registered constants.

Frozen before any HAR test verdict or verifier comparison is inspected.
Does not edit kernels, contracts, Kleene, prompts, or V3/HARTH numbers.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "p4_har_claim"
REPORTS = ROOT / "reports" / "P4_HAR_CLAIM"

SEED = 20260825
EXPERIMENT_ID = "p4_har_claim_v1"

# Same cap as the unused-window loader. Not a new windowing protocol.
MAX_PER_SUBJECT = 24

# Standard RF. Not searched.
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 2

INVALIDATIONS = (
    "missing_channel",
    "invalid_fs",
    "dropout_10pct",
)

# Extra scored-cell exclusions so this cell does not recycle Phase-2 windows.
PRIOR_RESULT_GLOBS = (
    "results/p2_phase2/*.json",
    "results/p3_evidence_trust/*.json",
)

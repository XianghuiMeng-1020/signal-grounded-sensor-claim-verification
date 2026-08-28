"""Pre-registered P2 constants.

These values are frozen before primitive validation and before any
challenge-set metric is inspected. Do not edit after seeing failures.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
RESULTS_P2 = ROOT / "results" / "p2"
BENCH_P2 = ROOT / "benchmarks" / "p2"
BENCH_P2_V3 = ROOT / "benchmarks" / "p2_v3"
BENCHMARK_VERSION_V3 = "independent_gold_v3_selfcontained"
REPORTS = ROOT / "reports" / "P2_SCIENTIFIC_ESCALATION"

SEED = 20270823
WINDOW = 256

# ---------------------------------------------------------------------------
# Primitive mathematical definitions (resolved ambiguities)
# ---------------------------------------------------------------------------
# RMS includes DC (raw second-moment), not AC RMS.
# peak_amplitude is max |x-mean(x)|.
# trend_ratio uses AC-RMS of halves split at n//2.
# cross_channel_lag support is physical: |ℓ| ≤ L(fs)=floor(T_max * fs)
#   with T_max = 0.300 s. This is the 30-sample box at 100 Hz converted to
#   time (scripts/p2_phase2/lag_config.py). Not taken from challenge/sealed.
# periodicity_strength is max |Rxx[ℓ]|/Rxx[0] on lags [3, n/2) (dimensionless).
# spectral_energy_ratio_low uses rectangular-window rFFT power, f < 3 Hz,
#   after mean removal. DC is theoretically ~0 after mean removal.
# dominant_frequency is k̂*fs/N Hz: non-DC rFFT magnitude peak (boxcar, no Welch).

PRIMITIVE_NAMES = (
    "dominant_frequency",
    "rms_amplitude",
    "peak_amplitude",
    "signal_range",
    "trend_ratio",
    "cross_channel_lag_ms",
    "periodicity_strength",
    "spectral_energy_ratio_low",
)

# Claim-interpretation tolerances (published pilot convention, re-implemented
# independently; not copied from production source at runtime).
TOLERANCE_ABS = {
    "dominant_frequency": 0.4,
    "trend_ratio": 0.25,
    "cross_channel_lag_ms": 15.0,
    "periodicity_strength": 0.12,
    "spectral_energy_ratio_low": 0.12,
}
RELATIVE_OPS = ("rms_amplitude", "peak_amplitude", "signal_range")
RELATIVE_FRAC = 0.15
RELATIVE_FLOOR = 0.3
SIMILAR_REL_FRAC = 0.25
LAG_T_MAX_S = 0.300
LAG_MAX_SAMPLES = 30  # L(100 Hz); historical alias, not a rate-independent box
SPECTRAL_CUTOFF_HZ = 3.0
PERIODICITY_MIN_LAG = 3


def lag_max_samples(fs: float, t_max: float = LAG_T_MAX_S) -> int:
    """Physical search radius in samples: floor(T_max * fs)."""
    rate = float(fs)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("invalid_sampling_rate")
    return int(math.floor(t_max * rate + 1e-12))

# ---------------------------------------------------------------------------
# Phase 2: pre-registered validation tolerances (FROZEN before seeing results)
# ---------------------------------------------------------------------------
# Analytical / same-definition independent implementation vs production.
# Frequency: within one rFFT bin of the known tone, and production vs
# independent-canonical within one bin of each other.
# Amplitude-like: tight absolute/relative because the definition is unique.
# Lag: within 0.51 samples in milliseconds at the test fs.
# Periodicity / spectral ratio: tight on constructed signals.

P2_VALIDATION_CASES_VERSION = "p2_prim_val_v1"

def dominant_freq_abs_tol(n: int, fs: float) -> float:
    return float(fs / n) + 1e-9  # one DFT bin


VALIDATION_TOL = {
    "rms_amplitude": {"abs": 1e-8, "rel": 1e-8},
    "peak_amplitude": {"abs": 1e-8, "rel": 1e-8},
    "signal_range": {"abs": 1e-8, "rel": 1e-8},
    "trend_ratio": {"abs": 1e-6, "rel": 1e-6},
    "periodicity_strength": {"abs": 5e-3, "rel": None},
    "spectral_energy_ratio_low": {"abs": 5e-3, "rel": None},
    "cross_channel_lag_ms": {"abs_samples": 0.51},
    "dominant_frequency": {"bins": 1.0},
}

# Alternate estimator (periodogram/Welch) is DIAGNOSTIC only and is not a
# pass/fail criterion for Gate C. Gate C uses analytical + canonical scipy
# implementations of the resolved definitions above.

# ---------------------------------------------------------------------------
# Phase 4: pre-registered subject groups (explicit, not result-tuned)
# ---------------------------------------------------------------------------
# Sorted names, then residue class. Guarantees each dataset appears in all
# three splits. Frozen before claim generation.
PAMAP2_SPLIT = {
    "development": ("subject101", "subject104", "subject107"),
    "challenge": ("subject102", "subject105", "subject108"),
    "final_sealed_holdout": ("subject103", "subject106", "subject109"),
}
MHEALTH_SPLIT = {
    "development": ("mHealth_subject1", "mHealth_subject4", "mHealth_subject7"),
    "challenge": ("mHealth_subject2", "mHealth_subject5", "mHealth_subject8", "mHealth_subject10"),
    "final_sealed_holdout": ("mHealth_subject3", "mHealth_subject6", "mHealth_subject9"),
}
# WISDM: numeric subject id from filename data_<id>_accel_watch.txt
# id % 3 == 0 development, == 1 challenge, == 2 holdout.

MAX_WINDOWS_PER_SUBJECT = 4
MAX_WINDOWS_PER_SPLIT = 42  # stratified cap after load

# ---------------------------------------------------------------------------
# Phase 5–7 generation caps (pre-registered)
# ---------------------------------------------------------------------------
SURFACES_PER_PROGRAM = {
    "development": 3,
    "challenge": 8,
    "final_sealed_holdout": 4,
}
MARGIN_BANDS = (
    ("very_close", 0.0, 0.01),
    ("close", 0.01, 0.05),
    ("moderate", 0.05, 0.10),
    ("clear", 0.10, 1e9),
)
# Abstention-margin candidates; choose ONLY on development, then freeze.
ABSTENTION_CANDIDATES = (0.0, 0.01, 0.03, 0.05)
UNVERIFIABLE_TARGET_FCR = 0.05

# ---------------------------------------------------------------------------
# Phase 6 unverifiable families (pre-registered)
# ---------------------------------------------------------------------------
UNVERIFIABLE_FAMILIES = (
    "unsupported_measurement_type",
    "unavailable_channel",
    "missing_channel",
    "insufficient_length",
    "invalid_sampling_rate",
    "ambiguous_channel_reference",
    "ambiguous_comparator",
    "unsupported_physiological_proxy",
    "qualitative_no_executable_definition",
    "unsupported_logical_nesting",
    "too_many_predicates",
    "corrupted_or_missing_evidence",
)

# ---------------------------------------------------------------------------
# Paths to historical pilot (do not overwrite)
# ---------------------------------------------------------------------------
PILOT_V1_BENCH = RESULTS / "f_round8_compose_bench.json"
PILOT_V1_RAW = RESULTS / "f_round8_g9_pilot_raw.json"
PILOT_R6_BENCH = RESULTS / "f_round6_bench.json"
PILOT_R6_RAW = RESULTS / "f_round6_pilot_raw.json"
PILOT_R6_SCORE = RESULTS / "f_round6_score_summary.json"
PILOT_R7_BENCH = RESULTS / "f_round7_bench.json"
PILOT_R7_RAW = RESULTS / "f_round7_kill_pilot_raw.json"

DATA_PAMAP2 = ROOT / "data" / "pamap2_inner" / "PAMAP2_Dataset" / "Protocol"
DATA_WISDM = ROOT / "data" / "wisdm_inner" / "wisdm-dataset" / "raw" / "watch" / "accel"
DATA_MHEALTH = ROOT / "data" / "mhealth_inner" / "MHEALTHDATASET"

FS = {"PAMAP2": 100.0, "WISDM": 20.0, "MHEALTH": 50.0}

HOLDOUT_NAME = "final_sealed_holdout"
EVALUABLE_SPLITS = ("development", "challenge")

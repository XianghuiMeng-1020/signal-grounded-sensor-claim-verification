"""Pre-registered perturbation theory + independent post-perturbation gold."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p2r.pipeline import run_pipeline
from p2r.validator import from_legacy

from .config import SEED

# primitive × perturbation → expected class. Frozen before results.
# INVARIANT | EQUIVARIANT | TRUTH_MAY_CHANGE | EVIDENCE_INVALIDATED | NOT_APPLICABLE
AMP = {"rms_amplitude", "peak_amplitude", "signal_range"}
SPEC = {"dominant_frequency", "spectral_energy_ratio_low", "periodicity_strength"}
ALLP = AMP | SPEC | {"trend_ratio", "cross_channel_lag_ms"}

THEORY = {
    "noise_snr20": {p: "TRUTH_MAY_CHANGE" for p in ALLP},
    "noise_snr10": {p: "TRUTH_MAY_CHANGE" for p in ALLP},
    "noise_snr0": {p: "TRUTH_MAY_CHANGE" for p in ALLP},
    "scale_x2": {**{p: "EQUIVARIANT" for p in AMP}, **{p: "INVARIANT" for p in SPEC}, "trend_ratio": "INVARIANT", "cross_channel_lag_ms": "INVARIANT"},
    "scale_x0.5": {**{p: "EQUIVARIANT" for p in AMP}, **{p: "INVARIANT" for p in SPEC}, "trend_ratio": "INVARIANT", "cross_channel_lag_ms": "INVARIANT"},
    "quantize_8bit": {p: "TRUTH_MAY_CHANGE" for p in ALLP},
    "common_shift": {**{p: "INVARIANT" for p in ALLP - {"trend_ratio", "cross_channel_lag_ms"}}, "trend_ratio": "TRUTH_MAY_CHANGE", "cross_channel_lag_ms": "INVARIANT"},
    "boundary_shift": {p: "TRUTH_MAY_CHANGE" for p in ALLP},
    "resample_x2_meta_ok": {**{p: "INVARIANT" for p in SPEC | {"cross_channel_lag_ms"}}, **{p: "TRUTH_MAY_CHANGE" for p in AMP}, "trend_ratio": "INVARIANT"},
    "sparse_missing_5pct": {p: "EVIDENCE_INVALIDATED" for p in ALLP},
    "contiguous_gap": {p: "EVIDENCE_INVALIDATED" for p in ALLP},
    "channel_dropout": {p: "EVIDENCE_INVALIDATED" for p in ALLP},
    "corrupt_fs_x2": {**{p: "TRUTH_MAY_CHANGE" for p in SPEC | {"cross_channel_lag_ms"}}, **{p: "INVARIANT" for p in AMP | {"trend_ratio"}}},
}


def _rng(key: str):
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng((SEED + h) % 2**32)


def apply_perturbation(name: str, channels: dict, fs: float, key: str) -> tuple[dict, float, bool]:
    """Return (channels', fs', evidence_invalid)."""
    rng = _rng(key + name)
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in channels.items()}
    ev_bad = False
    nfs = fs
    if name.startswith("noise_snr"):
        snr = {"noise_snr20": 20, "noise_snr10": 10, "noise_snr0": 0}[name]
        for k, x in out.items():
            p = np.mean(x ** 2) + 1e-12
            sigma = np.sqrt(p / (10 ** (snr / 10)))
            out[k] = x + rng.normal(0, sigma, size=x.shape)
    elif name.startswith("scale_"):
        g = 2.0 if name.endswith("x2") else 0.5
        out = {k: v * g for k, v in out.items()}
    elif name == "quantize_8bit":
        for k, x in out.items():
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi <= lo:
                continue
            q = np.round((x - lo) / (hi - lo) * 255.0) / 255.0 * (hi - lo) + lo
            out[k] = q
    elif name == "common_shift":
        out = {k: np.roll(v, 8) for k, v in out.items()}
    elif name == "boundary_shift":
        out = {k: np.roll(v, 16) for k, v in out.items()}
    elif name == "resample_x2_meta_ok":
        nfs = fs * 2
        out = {k: np.repeat(v, 2)[: v.size * 2] for k, v in out.items()}
    elif name == "sparse_missing_5pct":
        for k, x in out.items():
            m = rng.random(x.size) < 0.05
            x = x.astype(float)
            x[m] = np.nan
            out[k] = x
        ev_bad = True
    elif name == "contiguous_gap":
        for k, x in out.items():
            x = x.astype(float)
            x[40:80] = np.nan
            out[k] = x
        ev_bad = True
    elif name == "channel_dropout":
        first = next(iter(out))
        out[first] = None
        ev_bad = True
    elif name == "corrupt_fs_x2":
        nfs = fs * 2.0
    return out, nfs, ev_bad


def theory_class(pert: str, measurement: str) -> str:
    return THEORY.get(pert, {}).get(measurement, "NOT_APPLICABLE")

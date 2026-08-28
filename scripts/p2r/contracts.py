"""Evidence contracts for the eight frozen primitives.

Contracts are derived from the *resolved mathematical definitions* already
used by the independent reference layer (scripts/p2/independent_dsp.py) and
the production kernels (scripts/f_round6_operators.py). They are not tuned
on robustness scores.

Preprocessing permitted (definitional only):
  mean removal, z-score, rFFT / autocorrelation as specified.
Not permitted: interpolation, imputation, channel substitution, fs invention.

Any non-finite sample is a contract failure: the frozen estimators are defined
on a complete finite sequence. Dropping NaNs would be a different estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from p2.config import LAG_T_MAX_S, lag_max_samples

from .schema import MEASUREMENTS

OK = "OK"
VALID = OK  # protocol alias; Kleene still keys on OK
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
UNSUPPORTED = "UNSUPPORTED"
INVALID_METADATA = "INVALID_METADATA"
MISSING_CHANNEL = "MISSING_CHANNEL"
INVALID_MEASUREMENT = "INVALID_MEASUREMENT"

# Physical lag support: L(fs)=floor(T_max fs). 30 is L(100 Hz) only.
LAG_MAX_SAMPLES = 30
PERIODICITY_MIN_LAG = 3


@dataclass
class MeasurementResult:
    status: str
    value: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _finite_1d(x) -> tuple[Optional[np.ndarray], Optional[str]]:
    if x is None:
        return None, "missing_channel"
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable_channel:{exc}"
    if arr.size == 0:
        return None, "empty_channel"
    if np.any(~np.isfinite(arr)):
        return None, "nonfinite_samples"
    return arr, None


def contract_spec(measurement: str) -> dict[str, Any]:
    """Declarative contract. min_n comes from the definition, not from a score."""
    base = {
        "required_fs": False,
        "required_channels": 1,
        "min_finite_n": 1,
        "require_positive_variance": False,
        "equal_length_if_two_channels": False,
        "missing_samples_tolerated": False,
        "permitted_preprocessing": ["cast_float64"],
        "units": "raw",
        "output_domain": "real",
        "refuse_if": [],
    }
    if measurement == "rms_amplitude":
        return {
            **base,
            "justification": "RMS is ||x||_2/sqrt(N) on a complete finite vector. N>=1.",
            "units": "raw",
            "output_domain": "[0, +inf)",
            "permitted_preprocessing": ["cast_float64"],
            "refuse_if": ["missing_channel", "empty_channel", "nonfinite_samples"],
        }
    if measurement == "peak_amplitude":
        return {
            **base,
            "justification": "max|x-mean(x)| requires a defined sample mean; N>=1 finite.",
            "permitted_preprocessing": ["cast_float64", "mean_remove"],
            "output_domain": "[0, +inf)",
            "refuse_if": ["missing_channel", "empty_channel", "nonfinite_samples"],
        }
    if measurement == "signal_range":
        return {
            **base,
            "justification": "max-min on a complete finite vector. N>=1 (range of a singleton is 0).",
            "output_domain": "[0, +inf)",
            "refuse_if": ["missing_channel", "empty_channel", "nonfinite_samples"],
        }
    if measurement == "trend_ratio":
        return {
            **base,
            "min_finite_n": 4,
            "justification": (
                "Split at n//2; each half uses AC-RMS. A 1-sample half has AC-RMS=0 "
                "and is degenerate, so each half needs >=2 samples ⇒ n>=4."
            ),
            "permitted_preprocessing": ["cast_float64", "split_halves", "mean_remove_per_half"],
            "units": "ratio",
            "output_domain": "(0, +inf)",
            "refuse_if": ["missing_channel", "empty_channel", "nonfinite_samples", "n<4"],
        }
    if measurement == "dominant_frequency":
        return {
            **base,
            "required_fs": True,
            "min_finite_n": 4,
            "justification": (
                "Non-DC rFFT peak. Needs valid fs>0 to place bins in Hz, and n>=4 so "
                "bins besides DC exist. No extra duration law: the object is the DFT peak, "
                "not a continuous-time frequency estimator."
            ),
            "permitted_preprocessing": ["cast_float64", "mean_remove", "rfft_boxcar"],
            "units": "Hz",
            "output_domain": "[0, fs/2]",
            "refuse_if": ["missing_channel", "nonfinite_samples", "n<4", "invalid_or_missing_fs"],
        }
    if measurement == "spectral_energy_ratio_low":
        return {
            **base,
            "required_fs": True,
            "min_finite_n": 4,
            "justification": (
                "Fraction of rFFT power with f<3 Hz. Requires fs to label bins. n>=4."
            ),
            "permitted_preprocessing": ["cast_float64", "mean_remove", "rfft_boxcar"],
            "units": "fraction",
            "output_domain": "[0, 1]",
            "refuse_if": ["missing_channel", "nonfinite_samples", "n<4", "invalid_or_missing_fs"],
        }
    if measurement == "periodicity_strength":
        return {
            **base,
            "min_finite_n": 8,
            "justification": (
                "Normalized autocorrelation search on lags [3, n/2). Non-empty search "
                "requires n/2 > 3 ⇒ n>=8. Frozen min-lag is 3 samples, not a Hz law."
            ),
            "permitted_preprocessing": ["cast_float64", "mean_remove", "autocorrelate"],
            "units": "score_0_1",
            "output_domain": "[0, 1]",
            "refuse_if": ["missing_channel", "nonfinite_samples", "n<8"],
        }
    if measurement == "cross_channel_lag_ms":
        return {
            **base,
            "required_fs": True,
            "required_channels": 2,
            "min_finite_n": None,
            "require_positive_variance": True,
            "equal_length_if_two_channels": True,
            "justification": (
                "Estimator searches ±L(fs) samples, L=floor(0.300 fs). "
                "Need n >= 2L+1. Zero-variance channels make z-score / argmax undefined."
            ),
            "permitted_preprocessing": ["cast_float64", "zscore", "fft_correlate"],
            "units": "ms",
            "output_domain": "[-T_max*1000, T_max*1000] ms",
            "refuse_if": [
                "missing_channel",
                "nonfinite_samples",
                "length_mismatch",
                "n<2L+1",
                "invalid_or_missing_fs",
                "degenerate_zero_variance",
            ],
        }
    return {
        **base,
        "justification": "unknown measurement is unsupported by the frozen ontology",
        "refuse_if": ["unknown_measurement"],
    }


def check_contract(measurement: str, channels: dict, fs) -> MeasurementResult:
    """Validate every named series in `channels`. Callers must pass required names only.

    A None / absent required name is MISSING_CHANNEL. Dict-order slicing is not used.
    """
    if measurement not in MEASUREMENTS:
        return MeasurementResult(UNSUPPORTED, None, {"reason": "unknown_measurement", "measurement": measurement})
    spec = contract_spec(measurement)
    needed = spec["required_channels"]
    if not channels:
        return MeasurementResult(MISSING_CHANNEL, None, {"reason": "no_named_channels"})

    arrays = []
    used = []
    for name, raw in channels.items():
        if raw is None:
            return MeasurementResult(MISSING_CHANNEL, None, {"reason": "missing_channel", "channel": name})
        arr, err = _finite_1d(raw)
        if err == "missing_channel":
            return MeasurementResult(MISSING_CHANNEL, None, {"reason": err, "channel": name})
        if err:
            return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": err, "channel": name})
        arrays.append(arr)
        used.append(name)

    if len(used) < needed:
        return MeasurementResult(MISSING_CHANNEL, None, {"reason": "missing_channel", "have": used})

    if spec["equal_length_if_two_channels"] and len(arrays) >= 2 and arrays[0].size != arrays[1].size:
        return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": "channel_length_mismatch"})

    n = min(a.size for a in arrays)

    if spec["required_fs"]:
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return MeasurementResult(INVALID_METADATA, None, {"reason": "invalid_or_missing_fs", "fs": fs})

    min_n = spec["min_finite_n"]
    if measurement == "cross_channel_lag_ms":
        min_n = 2 * lag_max_samples(float(fs), LAG_T_MAX_S) + 1
    if min_n is not None and n < min_n:
        return MeasurementResult(
            INSUFFICIENT_EVIDENCE,
            None,
            {"reason": "insufficient_length", "n": n, "min_n": min_n},
        )

    if spec["require_positive_variance"]:
        for i, arr in enumerate(arrays):
            if float(np.std(arr)) < 1e-15:
                return MeasurementResult(
                    INSUFFICIENT_EVIDENCE, None, {"reason": "degenerate_channel", "which": i}
                )

    return MeasurementResult(OK, None, {"n": n, "fs": fs, "channels": used, "spec": spec["units"]})


def check_output_domain(measurement: str, value, fs) -> MeasurementResult:
    """Post-execution domain from the written contract. No new cutoffs."""
    if value is None:
        return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "null_output"})
    try:
        v = float(value)
    except Exception:
        return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "non_numeric_output"})
    if not np.isfinite(v):
        return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "nonfinite_output", "value": v})
    spec = contract_spec(measurement)
    domain = spec.get("output_domain", "real")
    if domain in ("[0, +inf)", "(0, +inf)"):
        if v < 0.0:
            return MeasurementResult(INVALID_MEASUREMENT, v, {"reason": "negative_amplitude_or_ratio"})
    elif domain == "[0, 1]":
        if v < 0.0 or v > 1.0:
            return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "ratio_outside_unit_interval", "value": v})
    elif domain == "[0, fs/2]":
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return MeasurementResult(INVALID_METADATA, None, {"reason": "invalid_or_missing_fs", "fs": fs})
        if v < -1e-9 or v > float(fs) / 2.0 + 1e-9:
            return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "frequency_outside_nyquist", "value": v})
    elif measurement == "cross_channel_lag_ms":
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return MeasurementResult(INVALID_METADATA, None, {"reason": "invalid_or_missing_fs", "fs": fs})
        box_ms = LAG_T_MAX_S * 1000.0
        if abs(v) > box_ms + 1e-6:
            return MeasurementResult(INVALID_MEASUREMENT, None, {"reason": "lag_outside_search", "value": v})
    return MeasurementResult(OK, v, {"domain": domain})


def license(measurement: str, channels: dict, fs) -> bool:
    """L(r, p) ∈ {0,1}. True only if the record licenses the typed measurement."""
    return check_contract(measurement, channels, fs).status == OK

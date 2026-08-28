"""Independent evidence-validity oracle.

Interprets the *written* P2R contract table (copied in config.ORACLE_SPEC).
Does not import production `check_contract` or `execute_predicate_measurement`.
Does not import production kernels.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from p2.independent_dsp import MeasurementError, measure, tolerance_for

from .config import LAG_MAX_SAMPLES, ORACLE_SPEC

VALID = "VALID"
MISSING_CHANNEL = "MISSING_CHANNEL"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
INVALID_METADATA = "INVALID_METADATA"
UNSUPPORTED = "UNSUPPORTED"


def _arr(x) -> tuple[Optional[np.ndarray], Optional[str]]:
    if x is None:
        return None, "missing"
    try:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
    except Exception:
        return None, "unreadable"
    if a.size == 0:
        return None, "empty"
    if np.any(~np.isfinite(a)):
        return None, "nonfinite"
    return a, None


def gold_status(measurement: str, named: dict, fs) -> dict[str, Any]:
    """named maps required channel names -> array|None only."""
    if measurement not in ORACLE_SPEC:
        return {"status": UNSUPPORTED, "reason": "unknown_measurement"}
    spec = ORACLE_SPEC[measurement]
    if not named:
        return {"status": MISSING_CHANNEL, "reason": "no_named_channels"}
    arrays = []
    for name, raw in named.items():
        arr, err = _arr(raw)
        if err == "missing" or err == "unreadable":
            return {"status": MISSING_CHANNEL, "reason": err, "channel": name}
        if err == "empty":
            return {"status": INSUFFICIENT_EVIDENCE, "reason": "empty_channel", "channel": name}
        if err == "nonfinite":
            return {"status": INSUFFICIENT_EVIDENCE, "reason": "nonfinite_samples", "channel": name}
        arrays.append(arr)
    if len(arrays) < spec["n_channels"]:
        return {"status": MISSING_CHANNEL, "reason": "too_few_channels", "have": len(arrays)}
    if spec["n_channels"] == 2 and arrays[0].size != arrays[1].size:
        return {"status": INSUFFICIENT_EVIDENCE, "reason": "channel_length_mismatch"}
    n = arrays[0].size
    if n < spec["min_n"]:
        return {"status": INSUFFICIENT_EVIDENCE, "reason": "insufficient_length", "n": n, "min_n": spec["min_n"]}
    if spec["required_fs"]:
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return {"status": INVALID_METADATA, "reason": "invalid_or_missing_fs", "fs": fs}
    if spec["need_var"]:
        for i, a in enumerate(arrays):
            if float(np.std(a)) < 1e-15:
                return {"status": INSUFFICIENT_EVIDENCE, "reason": "degenerate_channel", "which": i}
    return {"status": VALID, "reason": None, "n": n, "fs": fs}


def gold_verdict(measurement: str, named: dict, fs, asserted: float) -> str:
    st = gold_status(measurement, named, fs)
    if st["status"] != VALID:
        return "UNVERIFIABLE"
    try:
        actual = float(measure(measurement, named, fs))
    except MeasurementError:
        return "UNVERIFIABLE"
    tol = tolerance_for(measurement, actual)
    return "SUPPORTED" if abs(actual - float(asserted)) <= tol else "CONTRADICTED"


def output_domain_ok(measurement: str, value: float, fs) -> bool:
    spec = ORACLE_SPEC.get(measurement)
    if spec is None or value is None:
        return False
    try:
        v = float(value)
    except Exception:
        return False
    if not np.isfinite(v):
        return False
    kind = spec["output"]
    if kind == "nonneg":
        return v >= 0.0
    if kind == "unit":
        return 0.0 <= v <= 1.0
    if kind == "hz":
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return False
        return -1e-9 <= v <= float(fs) / 2.0 + 1e-9
    if kind == "lag":
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return False
        box = LAG_MAX_SAMPLES * 1000.0 / float(fs)
        return abs(v) <= box + 1e-6
    return False


def required_names(measurement: str, channel_a: str, channel_b: Optional[str]) -> list[str]:
    spec = ORACLE_SPEC[measurement]
    if spec["n_channels"] == 2:
        return [channel_a, channel_b]
    return [channel_a]

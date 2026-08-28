"""Evaluation-only one-clause shadows. Production contracts are not imported for write."""
from __future__ import annotations

from typing import Optional

import numpy as np

from f_round6_operators import compute as prod_compute
from f_round6_operators import tolerance_for as prod_tol
from p2r.contracts import (
    INSUFFICIENT_EVIDENCE,
    INVALID_METADATA,
    MISSING_CHANNEL,
    OK,
    UNSUPPORTED,
    MeasurementResult,
    check_contract,
    check_output_domain,
    contract_spec,
)
from p2r.kleene import FALSE, TRUE, UNKNOWN, compose, verdict_from_tv
from p2r.pipeline import run_oracle
from p2r.schema import MEASUREMENTS, Predicate


def _as_array(raw, allow_nonfinite: bool) -> tuple[Optional[np.ndarray], Optional[str]]:
    if raw is None:
        return None, "missing_channel"
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None, "empty_channel"
    if not allow_nonfinite and np.any(~np.isfinite(arr)):
        return None, "nonfinite_samples"
    return arr, None


def shadow_check_contract(measurement: str, channels: dict, fs, ablation: str) -> MeasurementResult:
    if measurement not in MEASUREMENTS:
        return MeasurementResult(UNSUPPORTED, None, {"reason": "unknown_measurement"})
    spec = contract_spec(measurement)
    needed = spec["required_channels"]
    if ablation == "drop_second_channel":
        needed = 1
        channels = {k: v for k, v in channels.items() if v is not None}
    if not channels:
        return MeasurementResult(MISSING_CHANNEL, None, {"reason": "no_named_channels"})

    allow_nf = ablation == "drop_nonfinite"
    arrays = []
    used = []
    for name, raw in channels.items():
        arr, err = _as_array(raw, allow_nf)
        if err == "missing_channel":
            return MeasurementResult(MISSING_CHANNEL, None, {"reason": err, "channel": name})
        if err:
            return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": err, "channel": name})
        arrays.append(arr)
        used.append(name)

    if len(used) < needed:
        return MeasurementResult(MISSING_CHANNEL, None, {"reason": "missing_channel", "have": used})

    if (
        ablation != "drop_equal_length"
        and spec["equal_length_if_two_channels"]
        and len(arrays) >= 2
        and arrays[0].size != arrays[1].size
    ):
        return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": "channel_length_mismatch"})

    n = min(a.size for a in arrays)
    if ablation != "drop_min_n" and n < spec["min_finite_n"]:
        return MeasurementResult(
            INSUFFICIENT_EVIDENCE,
            None,
            {"reason": "insufficient_length", "n": n, "min_n": spec["min_finite_n"]},
        )

    if ablation != "drop_fs" and spec["required_fs"]:
        if fs is None or not np.isfinite(fs) or float(fs) <= 0:
            return MeasurementResult(INVALID_METADATA, None, {"reason": "invalid_or_missing_fs", "fs": fs})

    if ablation != "drop_variance" and spec["require_positive_variance"]:
        for i, arr in enumerate(arrays):
            finite = arr[np.isfinite(arr)] if ablation == "drop_nonfinite" else arr
            if finite.size == 0 or float(np.std(finite)) < 1e-15:
                return MeasurementResult(
                    INSUFFICIENT_EVIDENCE, None, {"reason": "degenerate_channel", "which": i}
                )

    return MeasurementResult(OK, None, {"n": n, "fs": fs, "channels": used, "shadow": ablation})


def _kleene_compare(pred: Predicate, actual: float) -> str:
    if not np.isfinite(actual):
        return UNKNOWN
    if pred.comparator == "eq":
        if pred.reference_value is None:
            return UNKNOWN
        tol = prod_tol(pred.measurement, actual)
        return TRUE if abs(actual - float(pred.reference_value)) <= tol else FALSE
    if pred.comparator in ("gt", "lt") and pred.reference_value is not None:
        thr = float(pred.reference_value)
        truth = actual > thr if pred.comparator == "gt" else actual < thr
        return TRUE if truth else FALSE
    return UNKNOWN


def shadow_verdict(program, available, fs, channels, ablation: str) -> dict:
    pred = program.predicates[0]
    if pred.measurement == "cross_channel_lag_ms":
        named = {pred.channel_a: channels.get(pred.channel_a), pred.channel_b: channels.get(pred.channel_b)}
    else:
        named = {pred.channel_a: channels.get(pred.channel_a)}

    if ablation == "output_domain" or ablation == "drop_output_domain":
        gate = check_contract(pred.measurement, named, fs)
        if gate.status != OK:
            return {"verdict": "UNVERIFIABLE", "reason": gate.diagnostics.get("reason"), "kernel_exception": False}
        cmap = {k: v for k, v in named.items() if v is not None}
        try:
            val = float(prod_compute(pred.measurement, cmap, fs))
        except Exception as exc:  # noqa: BLE001
            return {"verdict": "UNVERIFIABLE", "reason": f"kernel_exception:{exc}", "kernel_exception": True}
        tv = _kleene_compare(pred, val)
        return {
            "verdict": verdict_from_tv(compose(program.connective, [tv])),
            "reason": "shadow_skip_output_domain",
            "kernel_exception": False,
            "value": val if np.isfinite(val) else None,
        }

    gate = shadow_check_contract(pred.measurement, named, fs, ablation)
    if gate.status != OK:
        return {"verdict": "UNVERIFIABLE", "reason": gate.diagnostics.get("reason"), "kernel_exception": False}
    cmap = {k: v for k, v in named.items() if v is not None}
    try:
        val = float(prod_compute(pred.measurement, cmap, fs))
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "UNVERIFIABLE", "reason": f"kernel_exception:{exc}", "kernel_exception": True}
    post = check_output_domain(pred.measurement, val, fs)
    if post.status != OK:
        return {
            "verdict": "UNVERIFIABLE",
            "reason": post.diagnostics.get("reason"),
            "kernel_exception": False,
            "leftover_clause": True,
        }
    tv = _kleene_compare(pred, val)
    return {
        "verdict": verdict_from_tv(compose(program.connective, [tv])),
        "reason": "shadow_commit",
        "kernel_exception": False,
        "value": val,
    }


def production_verdict(program, available, fs, channels) -> dict:
    rec = run_oracle(program, available, fs, channels)
    return {"verdict": rec["verdict"], "reason": rec.get("reason")}

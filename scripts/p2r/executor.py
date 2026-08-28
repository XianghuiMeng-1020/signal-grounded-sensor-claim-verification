"""Contract-gated DSP execution.

Calls production kernels ONLY after the evidence contract returns OK.
Never inspects raw language. Never invents a numeric value on failure.
Does not modify scripts/f_round6_operators.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .contracts import (
    INSUFFICIENT_EVIDENCE,
    INVALID_MEASUREMENT,
    INVALID_METADATA,
    MISSING_CHANNEL,
    OK,
    UNSUPPORTED,
    MeasurementResult,
    check_contract,
    check_output_domain,
)
from .schema import Predicate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from f_round6_operators import compute as prod_compute  # noqa: E402
from f_round6_operators import tolerance_for as prod_tol  # noqa: E402

from .kleene import FALSE, TRUE, UNKNOWN  # noqa: E402

# Tests assert the kernel is not entered when the pre-gate fails.
kernel_call_count = 0


def reset_kernel_counter() -> None:
    global kernel_call_count
    kernel_call_count = 0


def _kernel(op: str, chmap: dict, fs):
    global kernel_call_count
    kernel_call_count += 1
    return prod_compute(op, chmap, fs)


def execute_predicate_measurement(pred: Predicate, channels: dict, fs) -> MeasurementResult:
    """channels must be a name->array dict. No sentence field allowed."""
    if "surface_text" in channels or "sentence" in channels:
        raise TypeError("DSP executor must not receive raw language")
    # For vs_channel, each named series must independently satisfy the single-channel contract.
    if pred.reference_channel and pred.measurement != "cross_channel_lag_ms":
        gate_a = check_contract(pred.measurement, {pred.channel_a: channels.get(pred.channel_a)}, fs)
        gate_b = check_contract(pred.measurement, {pred.reference_channel: channels.get(pred.reference_channel)}, fs)
        if gate_a.status != OK:
            return gate_a
        if gate_b.status != OK:
            return gate_b
        try:
            va = float(_kernel(pred.measurement, {pred.channel_a: channels[pred.channel_a]}, fs))
            vb = float(_kernel(pred.measurement, {pred.reference_channel: channels[pred.reference_channel]}, fs))
        except Exception as exc:  # noqa: BLE001
            return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": "production_exception", "detail": str(exc)})
        pa = check_output_domain(pred.measurement, va, fs)
        pb = check_output_domain(pred.measurement, vb, fs)
        if pa.status != OK:
            return pa
        if pb.status != OK:
            return pb
        return MeasurementResult(OK, va, {"b": vb, "pair": True})
    if pred.measurement == "cross_channel_lag_ms":
        named = {pred.channel_a: channels.get(pred.channel_a), pred.channel_b: channels.get(pred.channel_b)}
    else:
        named = {pred.channel_a: channels.get(pred.channel_a)}
    gate = check_contract(pred.measurement, named, fs)
    if gate.status != OK:
        return gate
    try:
        val = float(_kernel(pred.measurement, named, fs))
    except Exception as exc:  # noqa: BLE001
        return MeasurementResult(INSUFFICIENT_EVIDENCE, None, {"reason": "production_exception", "detail": str(exc)})
    post = check_output_domain(pred.measurement, val, fs)
    if post.status != OK:
        return post
    return MeasurementResult(OK, val, {})


def _finite(x) -> bool:
    try:
        return x is not None and float(x) == float(x) and abs(float(x)) != float("inf")
    except Exception:
        return False


def predicate_truth(pred: Predicate, channels: dict, fs) -> tuple[str, dict]:
    """Return (TRUE|FALSE|UNKNOWN, evidence). UNKNOWN if contract fails."""
    res = execute_predicate_measurement(pred, channels, fs)
    if res.status != OK:
        return UNKNOWN, {"measurement": res}
    actual = res.value
    if pred.comparator == "eq":
        if pred.reference_value is None:
            return UNKNOWN, {"reason": "missing_reference_value"}
        tol = prod_tol(pred.measurement, actual)
        return (TRUE if abs(actual - float(pred.reference_value)) <= tol else FALSE), {
            "actual": actual, "reference": pred.reference_value, "tolerance": tol
        }
    if pred.comparator in ("gt", "lt") and pred.reference_channel:
        b = res.diagnostics.get("b")
        if b is None:
            return UNKNOWN, {"reason": "missing_pair_value"}
        truth = actual > b if pred.comparator == "gt" else actual < b
        return (TRUE if truth else FALSE), {"a": actual, "b": b}
    if pred.comparator in ("similar", "different"):
        b = res.diagnostics.get("b")
        if b is None:
            return UNKNOWN, {"reason": "missing_pair_value"}
        scale = 0.25 * max(abs(actual), abs(b), 1e-9)
        sim = abs(actual - b) < scale
        truth = sim if pred.comparator == "similar" else (not sim)
        return (TRUE if truth else FALSE), {"a": actual, "b": b}
    if pred.comparator in ("gt", "lt") and pred.reference_value is not None:
        thr = float(pred.reference_value)
        truth = actual > thr if pred.comparator == "gt" else actual < thr
        return (TRUE if truth else FALSE), {"actual": actual, "threshold": thr}
    return UNKNOWN, {"reason": "malformed_comparator"}

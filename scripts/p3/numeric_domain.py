"""Frozen numeric-domain rules for executable vs_value claims."""
from __future__ import annotations

from typing import Optional

from p2r.schema import ClaimProgram, Predicate

from .config import MEASUREMENT_DOMAIN


def vs_value_in_domain(pred: Predicate, fs: Optional[float] = None) -> tuple[bool, Optional[str]]:
    """A vs_value assertion must lie in the measurement output domain.

    vs_threshold (gt/lt + reference_value, no reference_channel) may be any real.
    """
    if pred.comparator != "eq" or pred.reference_value is None:
        return True, None
    lo, hi = MEASUREMENT_DOMAIN[pred.measurement]
    v = float(pred.reference_value)
    if pred.measurement == "dominant_frequency" and fs:
        hi = float(fs) / 2.0
    if lo is not None and v < lo - 1e-12:
        return False, "asserted_value_below_measurement_domain"
    if hi is not None and v > hi + 1e-12:
        return False, "asserted_value_above_measurement_domain"
    if pred.measurement == "spectral_energy_ratio_low" and pred.unit == "percent" and abs(v) > 1.0 + 1e-12:
        # percent should already be converted to fraction by the validator
        pass
    return True, None


def program_domain_ok(program: ClaimProgram, fs: Optional[float] = None) -> tuple[bool, Optional[str]]:
    if program.parse_status != "OK":
        return True, None
    for pred in program.predicates:
        ok, reason = vs_value_in_domain(pred, fs)
        if not ok:
            return False, reason
    return True, None


def classify_numeric_role(pred: dict) -> str:
    """A = threshold comparison; B = purported measurement value/domain."""
    mode = pred.get("mode")
    if mode == "vs_threshold" or (pred.get("comparator") in ("gt", "lt") and pred.get("reference_channel") is None and pred.get("threshold") is not None):
        return "A_THRESHOLD"
    if mode == "vs_value" or pred.get("comparator") == "eq" or pred.get("asserted_value") is not None:
        return "B_PURPORTED_MEASUREMENT_VALUE"
    return "OTHER"

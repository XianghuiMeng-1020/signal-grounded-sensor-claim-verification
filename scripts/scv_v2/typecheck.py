"""Static type compatibility for SCV V2. No DSP. No LLM."""
from __future__ import annotations

from typing import Optional

from p2r.schema import COMPARATORS, CONNECTIVES, MEASUREMENTS, ClaimProgram

from .ontology import FS_REQUIRED, MEASUREMENT_UNITS, TWO_CHANNEL, UNSUPPORTED_PRIMITIVES


def type_status(program: ClaimProgram, fs: Optional[float], source_text: str = "") -> tuple[str, str]:
    """Return (VALID|INCOMPATIBLE|UNSUPPORTED, reason)."""
    low = (source_text or "").lower()
    for bad in UNSUPPORTED_PRIMITIVES:
        if re_word(bad, low):
            return "UNSUPPORTED", f"unsupported_primitive:{bad}"
    if program.parse_status != "OK":
        return "INCOMPATIBLE", program.parse_reason or "v1_not_ok"
    if program.connective not in CONNECTIVES:
        return "INCOMPATIBLE", "bad_connective"
    n = len(program.predicates or [])
    if program.connective == "SINGLE" and n != 1:
        return "INCOMPATIBLE", "single_arity"
    if program.connective in ("AND", "OR") and n < 2:
        return "INCOMPATIBLE", "compound_arity"
    if program.connective == "IF_THEN" and n != 2:
        return "INCOMPATIBLE", "if_then_arity"
    for pred in program.predicates:
        if pred.measurement not in MEASUREMENTS:
            return "UNSUPPORTED", f"unknown_measurement:{pred.measurement}"
        if pred.comparator not in COMPARATORS:
            return "INCOMPATIBLE", f"bad_comparator:{pred.comparator}"
        if pred.measurement in TWO_CHANNEL and not pred.channel_b:
            return "INCOMPATIBLE", "lag_needs_two_channels"
        if pred.measurement not in TWO_CHANNEL and pred.channel_b:
            return "INCOMPATIBLE", "unexpected_second_channel"
        allowed = MEASUREMENT_UNITS[pred.measurement]
        if pred.unit and pred.unit not in allowed:
            return "INCOMPATIBLE", f"wrong_unit:{pred.unit}"
        if pred.comparator in ("gt", "lt", "eq") and pred.reference_channel is None:
            if pred.reference_value is None:
                return "INCOMPATIBLE", "missing_threshold"
        if pred.measurement in FS_REQUIRED:
            try:
                ok_fs = fs is not None and float(fs) > 0
            except (TypeError, ValueError):
                ok_fs = False
            if not ok_fs:
                return "INCOMPATIBLE", "missing_fs"
    return "VALID", "ok"


def re_word(token: str, text: str) -> bool:
    import re

    return re.search(r"(?<![\w])" + re.escape(token) + r"(?![\w])", text) is not None

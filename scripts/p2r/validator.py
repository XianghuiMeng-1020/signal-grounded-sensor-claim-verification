"""Deterministic schema validator / canonicalizer.

Allowed: case fold, known channel aliases from sensor metadata, unit conversion,
comparator synonyms, whitespace. Forbidden: reading gold/split/family IDs.
Ambiguity → parse_status AMBIGUOUS, not a guessed program.
"""
from __future__ import annotations

from typing import Any, Optional

from .schema import (
    COMPARATORS,
    CONNECTIVES,
    FORBIDDEN_INFERENCE_KEYS,
    MEASUREMENTS,
    UNITS,
    ClaimProgram,
    Predicate,
    assert_no_leakage,
)

# Aliases come only from dataset channel naming / ordinary sensor vocabulary.
# They never map a vague phrase onto a unique channel when two channels exist.
CHANNEL_ALIASES = {
    "hand": "hand_accel",
    "hand_accel": "hand_accel",
    "hand-accel": "hand_accel",
    "chest": "chest_accel",
    "chest_accel": "chest_accel",
    "chest-accel": "chest_accel",
    "ankle": "ankle_accel",
    "ankle_accel": "ankle_accel",
    "ankle-accel": "ankle_accel",
    "x": "x_accel",
    "x_accel": "x_accel",
    "x-axis": "x_accel",
    "x axis": "x_accel",
    "y": "y_accel",
    "y_accel": "y_accel",
    "y-axis": "y_accel",
    "y axis": "y_accel",
    "back": "back_accel",
    "back_accel": "back_accel",
    "lower back": "back_accel",
    "thigh": "thigh_accel",
    "thigh_accel": "thigh_accel",
}

COMPARATOR_ALIASES = {
    "eq": "eq",
    "equal": "eq",
    "approximately": "eq",
    "vs_value": "eq",
    "gt": "gt",
    "greater": "gt",
    "higher": "gt",
    "above": "gt",
    "lt": "lt",
    "less": "lt",
    "lower": "lt",
    "below": "lt",
    "similar": "similar",
    "approx_equal": "similar",
    "different": "different",
    "dissimilar": "different",
}

UNIT_ALIASES = {
    "hz": "Hz",
    "hertz": "Hz",
    "cycles per second": "Hz",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "seconds": "s",
    "raw": "raw",
    "raw units": "raw",
    "ratio": "ratio",
    "x": "ratio",
    "fraction": "fraction",
    "percent": "percent",
    "%": "percent",
    "score_0_1": "score_0_1",
}

MEASUREMENT_ALIASES = {m: m for m in MEASUREMENTS}
MEASUREMENT_ALIASES.update({
    "rms": "rms_amplitude",
    "root_mean_square": "rms_amplitude",
    "peak": "peak_amplitude",
    "range": "signal_range",
    "peak_to_peak": "signal_range",
    "lag": "cross_channel_lag_ms",
    "periodicity": "periodicity_strength",
    "low_band_ratio": "spectral_energy_ratio_low",
})


def resolve_channel(name: Optional[str], available: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (resolved, error). error='ambiguous' if not uniquely determined."""
    if not name:
        return None, "missing_channel_name"
    raw = str(name).strip().lower().replace(" channel", "").replace(" sensor stream", "")
    avail_l = {a.lower(): a for a in available}
    if raw in avail_l:
        return avail_l[raw], None
    alias = CHANNEL_ALIASES.get(raw)
    if alias and alias in available:
        return alias, None
    # prefix match against available names only
    hits = [a for a in available if a.lower().startswith(raw) or raw == a.split("_")[0].lower()]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, "ambiguous_channel"
    return None, "unresolved_channel"


def canonicalize_unit(unit: Optional[str], value: Optional[float]) -> tuple[Optional[str], Optional[float]]:
    if unit is None:
        return None, value
    u = UNIT_ALIASES.get(str(unit).strip().lower(), str(unit))
    if u not in UNITS:
        return None, value
    if u == "percent" and value is not None:
        return "fraction", float(value) / 100.0 if abs(value) > 1.0 or True and abs(value) > 1 else float(value) / 100.0
    if u == "s" and value is not None:
        return "ms", float(value) * 1000.0
    return u, value


def _canon_unit(unit: Optional[str], value: Optional[float]) -> tuple[Optional[str], Optional[float]]:
    if unit is None:
        return None, value
    key = str(unit).strip().lower()
    u = UNIT_ALIASES.get(key, str(unit) if str(unit) in UNITS else None)
    if u is None:
        return None, value
    if u == "percent" and value is not None:
        return "fraction", float(value) / 100.0
    if u == "s" and value is not None:
        return "ms", float(value) * 1000.0
    return u, value


def from_legacy(structure: dict, available: list[str]) -> ClaimProgram:
    """Map P2 gt_structure / extracted dicts onto ClaimProgram. No gold fields used."""
    leak = {k: structure[k] for k in structure if k in FORBIDDEN_INFERENCE_KEYS}
    if leak:
        raise ValueError(f"legacy structure carried forbidden keys {list(leak)}")
    if structure.get("unverifiable") or not structure.get("predicates"):
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="empty_or_flagged")
    conn = structure.get("connective", "SINGLE")
    if conn not in CONNECTIVES:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="bad_connective")
    preds = []
    for raw in structure.get("predicates") or []:
        p, err = _legacy_pred(raw, available)
        if err:
            return ClaimProgram(conn, [], parse_status="AMBIGUOUS", parse_reason=err)
        preds.append(p)
    return ClaimProgram(conn, preds, parse_status="OK")


def _legacy_pred(raw: dict, available: list[str]) -> tuple[Optional[Predicate], Optional[str]]:
    meas = raw.get("op") or raw.get("measurement")
    if meas not in MEASUREMENTS:
        return None, f"unknown_measurement:{meas}"
    chs = list(raw.get("channels") or [])
    if raw.get("channel_a"):
        chs = [raw["channel_a"]] + ([raw["channel_b"]] if raw.get("channel_b") else [])
    if not chs:
        return None, "missing_channel"
    a, e = resolve_channel(chs[0], available)
    if e:
        return None, e
    b = None
    if meas == "cross_channel_lag_ms":
        if len(chs) < 2:
            return None, "lag_needs_two_channels"
        b, e = resolve_channel(chs[1], available)
        if e:
            return None, e
    mode = raw.get("mode")
    rel = raw.get("relation")
    if mode == "vs_value" or raw.get("comparator") == "eq":
        comp = "eq"
        ref_v = raw.get("asserted_value", raw.get("reference_value"))
        ref_c = None
        unit = raw.get("unit")
    elif mode == "vs_channel" or raw.get("reference_channel") or raw.get("compare_channel"):
        comp = rel or raw.get("comparator")
        if comp not in COMPARATORS:
            return None, f"bad_comparator:{comp}"
        ref_v = None
        rc, e = resolve_channel(raw.get("compare_channel") or raw.get("reference_channel"), available)
        if e:
            return None, e
        ref_c = rc
        unit = raw.get("unit")
    elif mode == "vs_threshold":
        comp = rel or raw.get("comparator")
        if comp not in ("gt", "lt"):
            return None, f"bad_threshold_comparator:{comp}"
        ref_v = raw.get("threshold", raw.get("reference_value"))
        ref_c = None
        unit = raw.get("unit")
    else:
        comp = raw.get("comparator")
        if comp not in COMPARATORS:
            return None, f"unknown_mode:{mode}"
        ref_v = raw.get("reference_value", raw.get("asserted_value", raw.get("threshold")))
        ref_c = raw.get("reference_channel")
        unit = raw.get("unit")
    unit, ref_v = _canon_unit(unit, None if ref_v is None else float(ref_v))
    return Predicate(meas, a, comp, channel_b=b, reference_value=ref_v, reference_channel=ref_c, unit=unit), None


def validate_program(program: ClaimProgram, available: list[str]) -> ClaimProgram:
    if program.parse_status != "OK":
        return program
    if program.connective not in CONNECTIVES:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="bad_connective")
    if program.connective == "IF_THEN" and len(program.predicates) != 2:
        return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="if_then_arity")
    if program.connective == "SINGLE" and len(program.predicates) != 1:
        if len(program.predicates) == 0:
            return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="empty")
        # do not silently drop extras
        return ClaimProgram(program.connective, program.predicates, parse_status="AMBIGUOUS", parse_reason="single_arity")
    out = []
    for pred in program.predicates:
        if pred.measurement not in MEASUREMENTS:
            return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="unknown_measurement")
        a, e = resolve_channel(pred.channel_a, available)
        if e:
            return ClaimProgram(program.connective, [], parse_status="AMBIGUOUS" if e.startswith("ambiguous") else "UNPARSEABLE", parse_reason=e)
        b = pred.channel_b
        if pred.measurement == "cross_channel_lag_ms":
            if not b:
                others = [c for c in available if c != a]
                if len(others) == 1:
                    b = others[0]
                else:
                    return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="lag_needs_two_channels")
            b, e = resolve_channel(b, available)
            if e:
                return ClaimProgram(program.connective, [], parse_status="AMBIGUOUS" if "ambiguous" in e else "UNPARSEABLE", parse_reason=e)
        rc = pred.reference_channel
        if rc:
            rc, e = resolve_channel(rc, available)
            if e:
                return ClaimProgram(program.connective, [], parse_status="AMBIGUOUS" if "ambiguous" in e else "UNPARSEABLE", parse_reason=e)
        if pred.comparator not in COMPARATORS:
            return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="bad_comparator")
        if pred.comparator == "eq" and pred.reference_value is None:
            return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="eq_needs_value")
        if pred.comparator in ("similar", "different") and not rc:
            return ClaimProgram(program.connective, [], parse_status="UNPARSEABLE", parse_reason="similar_needs_channel")
        unit, val = _canon_unit(pred.unit, pred.reference_value)
        out.append(Predicate(pred.measurement, a, pred.comparator, b, val, rc, unit))
    return ClaimProgram(program.connective, out, parse_status="OK")


def inference_view(surface_text: str, available_channels: list[str], fs: float) -> dict[str, Any]:
    payload = {
        "surface_text": surface_text,
        "available_channels": list(available_channels),
        "fs": fs,
    }
    assert_no_leakage(payload)
    return payload

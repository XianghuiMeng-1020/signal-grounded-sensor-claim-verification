"""Independent logical adjudication for gold labels.

Re-implements SINGLE/AND/OR/IF_THEN and the three predicate modes from the
published semantics, without importing f_round8_composer. Uses the
independent DSP reference for measurements.
"""
from __future__ import annotations

from typing import Any, Optional

from .config import SIMILAR_REL_FRAC
from .independent_dsp import MeasurementError, measure, tolerance_for


def _ch(rep: dict, name: str):
    return rep["channels"].get(name)


def predicate_truth(rep: dict, predicate: dict) -> tuple[Optional[bool], dict]:
    op = predicate.get("op")
    channels = list(predicate.get("channels") or [])
    mode = predicate.get("mode")
    try:
        if mode == "vs_value":
            if op == "cross_channel_lag_ms":
                if len(channels) < 2:
                    return None, {"reason": "missing_channels"}
                chmap = {channels[0]: _ch(rep, channels[0]), channels[1]: _ch(rep, channels[1])}
                if any(v is None for v in chmap.values()):
                    return None, {"reason": "missing_channels"}
                actual = measure(op, chmap, rep.get("fs"))
            else:
                if not channels:
                    return None, {"reason": "missing_channels"}
                if _ch(rep, channels[0]) is None:
                    return None, {"reason": "missing_channels"}
                actual = measure(op, {channels[0]: _ch(rep, channels[0])}, rep.get("fs"))
            asserted = predicate.get("asserted_value")
            if asserted is None:
                return None, {"reason": "missing_asserted_value"}
            tol = predicate.get("tolerance_override")
            if tol is None:
                tol = tolerance_for(op, actual)
            truth = abs(float(asserted) - actual) <= float(tol)
            return truth, {"actual": actual, "asserted": asserted, "tolerance": float(tol)}

        if mode == "vs_channel":
            a_name = channels[0] if channels else None
            b_name = predicate.get("compare_channel")
            if a_name is None or b_name is None or _ch(rep, a_name) is None or _ch(rep, b_name) is None:
                return None, {"reason": "missing_channels"}
            a = measure(op, {a_name: _ch(rep, a_name)}, rep.get("fs"))
            b = measure(op, {b_name: _ch(rep, b_name)}, rep.get("fs"))
            rel = predicate.get("relation")
            scale = SIMILAR_REL_FRAC * max(abs(a), abs(b), 1e-9)
            if rel == "gt":
                truth = a > b
            elif rel == "lt":
                truth = a < b
            elif rel == "similar":
                truth = abs(a - b) < scale
            elif rel == "different":
                truth = abs(a - b) >= scale
            else:
                return None, {"reason": "unknown_relation", "relation": rel}
            return truth, {"a": a, "b": b, "relation": rel}

        if mode == "vs_threshold":
            thr = predicate.get("threshold", predicate.get("reference_value"))
            rel = predicate.get("relation") or predicate.get("comparator")
            if rel in ("gt", "greater", ">"):
                rel = "gt"
            elif rel in ("lt", "less", "<"):
                rel = "lt"
            if thr is None or rel not in ("gt", "lt"):
                return None, {"reason": "malformed_threshold_predicate"}
            if op == "cross_channel_lag_ms":
                if len(channels) < 2 or _ch(rep, channels[0]) is None or _ch(rep, channels[1]) is None:
                    return None, {"reason": "missing_channels"}
                chmap = {channels[0]: _ch(rep, channels[0]), channels[1]: _ch(rep, channels[1])}
            else:
                if not channels or _ch(rep, channels[0]) is None:
                    return None, {"reason": "missing_channels"}
                chmap = {channels[0]: _ch(rep, channels[0])}
            actual = measure(op, chmap, rep.get("fs"))
            # Frozen method has no measurement-uncertainty band. Equality is FALSE for
            # strict gt/lt (same as production executor). Not UNVERIFIABLE.
            truth = (actual > float(thr)) if rel == "gt" else (actual < float(thr))
            return truth, {"actual": actual, "threshold": float(thr), "relation": rel, "equal": abs(actual - float(thr)) < 1e-15}

        return None, {"reason": "unknown_mode", "mode": mode}
    except MeasurementError as exc:
        return None, {"reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 — gold must abstain, never guess
        return None, {"reason": "exception", "detail": str(exc)}


def adjudicate(rep: dict, structure: dict) -> dict[str, Any]:
    predicates = list(structure.get("predicates") or [])
    connective = structure.get("connective", "SINGLE")
    truths, evidences = [], []
    for pred in predicates:
        t, ev = predicate_truth(rep, pred)
        truths.append(t)
        evidences.append(ev)
    if not truths or any(t is None for t in truths):
        return {"verdict": "UNVERIFIABLE", "predicate_truths": truths, "evidence": evidences}
    if connective == "SINGLE":
        combined = bool(truths[0])
    elif connective == "AND":
        combined = all(truths)
    elif connective == "OR":
        combined = any(truths)
    elif connective == "IF_THEN":
        if len(truths) != 2:
            return {"verdict": "UNVERIFIABLE", "predicate_truths": truths, "evidence": evidences}
        combined = (not truths[0]) or truths[1]
    else:
        return {"verdict": "UNVERIFIABLE", "predicate_truths": truths, "evidence": evidences}
    return {
        "verdict": "SUPPORTED" if combined else "CONTRADICTED",
        "predicate_truths": truths,
        "evidence": evidences,
    }


def normalized_margin(rep: dict, structure: dict) -> Optional[float]:
    """Smallest normalized distance-to-boundary over predicates. None if unverifiable."""
    margins = []
    for pred, ev in zip(structure.get("predicates") or [], adjudicate(rep, structure)["evidence"]):
        mode = pred.get("mode")
        if mode == "vs_value" and "actual" in ev and "asserted" in ev:
            denom = max(abs(ev["actual"]), abs(float(ev["asserted"])), 1e-9)
            margins.append(abs(ev["actual"] - float(ev["asserted"])) / denom)
        elif mode == "vs_threshold" and "actual" in ev and "threshold" in ev:
            denom = max(abs(ev["actual"]), abs(float(ev["threshold"])), 1e-9)
            margins.append(abs(ev["actual"] - float(ev["threshold"])) / denom)
        elif mode == "vs_channel" and "a" in ev and "b" in ev:
            denom = max(abs(ev["a"]), abs(ev["b"]), 1e-9)
            margins.append(abs(ev["a"] - ev["b"]) / denom)
        else:
            return None
    if not margins:
        return None
    return float(min(margins))

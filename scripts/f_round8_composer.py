"""
Round 8, Part 1A -- GENERIC compositional claim-adjudication engine for F.

This is the missing piece identified by Round-7's G9 attack: claim EXTRACTION already
generalizes to composed claims (100% correct operator identification), but the
Round-6/7 verifier stopped at recomputing sub-quantities (`COMPOSED_SUBRESULTS`) and
never combined them into a final SUPPORTED/CONTRADICTED verdict.

This module represents ANY claim -- single or composed, AND/OR/IF-THEN, 2 or 3
predicates -- as:

    claim -> atomic predicates -> executable measurements -> predicate truth values
          -> logical composition -> SUPPORTED / CONTRADICTED / UNVERIFIABLE

No connective (AND/OR/IF_THEN) or arity (1/2/3 predicates) is hard-coded per test
sentence. The SAME `adjudicate()` function below handles every composition structure
in `f_round8_compose_bench.py`, including the ones held out from development.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f_round6_operators import OPERATORS, compute, tolerance_for

SIMILAR_REL_FRAC = 0.25  # matches the Round-7 compose-bench's own "similar" definition


def _actual_for_channel(rep, op, channel):
    return compute(op, {channel: rep["channels"][channel]}, rep["fs"])


def compute_predicate_truth(rep, predicate):
    """Returns (truth: bool|None, evidence: dict). truth=None means UNVERIFIABLE
    (missing operator/channel, or malformed predicate) -- never guessed."""
    op = predicate.get("op")
    channels = predicate.get("channels") or []
    mode = predicate.get("mode")
    if op not in OPERATORS:
        return None, dict(reason="unknown_operator", op=op)
    valid = [c for c in channels if c in rep["channels"]]

    try:
        if mode == "vs_value":
            if op == "cross_channel_lag_ms":
                if len(valid) < 2:
                    return None, dict(reason="missing_channels")
                actual = compute(op, {valid[0]: rep["channels"][valid[0]], valid[1]: rep["channels"][valid[1]]}, rep["fs"])
            else:
                if len(valid) < 1:
                    return None, dict(reason="missing_channels")
                actual = _actual_for_channel(rep, op, valid[0])
            asserted = predicate.get("asserted_value")
            if asserted is None:
                return None, dict(reason="missing_asserted_value")
            tol = predicate.get("tolerance_override") or tolerance_for(op, actual)
            truth = abs(float(asserted) - actual) <= tol
            return truth, dict(actual=actual, asserted=asserted, tolerance=tol)

        elif mode == "vs_channel":
            ch_a = predicate.get("channels", [None])[0]
            ch_b = predicate.get("compare_channel")
            if ch_a not in rep["channels"] or ch_b not in rep["channels"]:
                return None, dict(reason="missing_channels")
            a = _actual_for_channel(rep, op, ch_a)
            b = _actual_for_channel(rep, op, ch_b)
            rel = predicate.get("relation")
            if rel == "gt":
                truth = a > b
            elif rel == "lt":
                truth = a < b
            elif rel == "similar":
                truth = abs(a - b) < SIMILAR_REL_FRAC * max(abs(a), abs(b), 1e-9)
            elif rel == "different":
                truth = abs(a - b) >= SIMILAR_REL_FRAC * max(abs(a), abs(b), 1e-9)
            else:
                return None, dict(reason="unknown_relation", relation=rel)
            return truth, dict(a=a, b=b, relation=rel)

        elif mode == "vs_threshold":
            if len(valid) < 1:
                return None, dict(reason="missing_channels")
            actual = _actual_for_channel(rep, op, valid[0])
            thr = predicate.get("threshold")
            rel = predicate.get("relation")
            if thr is None or rel not in ("gt", "lt"):
                return None, dict(reason="malformed_threshold_predicate")
            truth = (actual > thr) if rel == "gt" else (actual < thr)
            return truth, dict(actual=actual, threshold=thr, relation=rel)

        else:
            return None, dict(reason="unknown_mode", mode=mode)
    except Exception as e:
        return None, dict(reason="exception", detail=str(e))


def adjudicate(rep, structure):
    """structure = {"predicates": [...], "connective": "SINGLE"|"AND"|"OR"|"IF_THEN"}
    Returns dict(verdict=SUPPORTED|CONTRADICTED|UNVERIFIABLE, predicate_truths=[...], evidence=[...]).
    Generic across ANY connective/arity -- no per-test branch."""
    predicates = structure.get("predicates") or []
    connective = structure.get("connective", "SINGLE")
    truths, evidences = [], []
    for p in predicates:
        t, ev = compute_predicate_truth(rep, p)
        truths.append(t)
        evidences.append(ev)

    if any(t is None for t in truths) or not truths:
        return dict(verdict="UNVERIFIABLE", predicate_truths=truths, evidence=evidences)

    if connective == "SINGLE":
        combined = truths[0]
    elif connective == "AND":
        combined = all(truths)
    elif connective == "OR":
        combined = any(truths)
    elif connective == "IF_THEN":
        if len(truths) != 2:
            return dict(verdict="UNVERIFIABLE", predicate_truths=truths, evidence=evidences)
        premise, consequent = truths
        combined = (not premise) or consequent  # material implication
    else:
        return dict(verdict="UNVERIFIABLE", predicate_truths=truths, evidence=evidences)

    verdict = "SUPPORTED" if combined else "CONTRADICTED"
    return dict(verdict=verdict, predicate_truths=truths, evidence=evidences)

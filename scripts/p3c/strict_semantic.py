"""Independently justified program equivalence. Not verdict-on-one-waveform."""
from __future__ import annotations

from typing import Optional

from p2r.schema import ClaimProgram, Predicate
from p2r.validator import validate_program
from p3.semantic_canon import canonical_program, programs_canonically_equal


def _vals_close(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(b)))


def _pred_key(p: Predicate) -> tuple:
    return (
        p.measurement,
        p.channel_a,
        p.channel_b,
        p.comparator,
        p.reference_channel,
        None if p.reference_value is None else round(float(p.reference_value), 6),
    )


def _preds_eq(a: Predicate, b: Predicate) -> bool:
    return (
        a.measurement == b.measurement
        and a.channel_a == b.channel_a
        and a.channel_b == b.channel_b
        and a.comparator == b.comparator
        and a.reference_channel == b.reference_channel
        and _vals_close(a.reference_value, b.reference_value)
    )


def normalize(program: ClaimProgram, available: list[str]) -> ClaimProgram:
    """Alias resolution (deployment-visible names) + lag canon + drop redundant lag ref."""
    v = validate_program(program, available)
    return canonical_program(v)


def programs_strictly_equivalent(pred: ClaimProgram, gold: ClaimProgram, available: list[str]) -> bool:
    """Rules (all-signal / definitional, not per-window verdict):

    1. Frozen channel aliases resolve to the same deployment-visible name.
    2. Lag order/sign canon + redundant reference_channel drop (P3 lag write-up).
    3. AND / OR are commutative (Kleene tables), so predicate order does not matter.
    4. IF_THEN / SINGLE keep order.

    Not used: same verdict on one waveform; invented numeric slop beyond 1e-3 relative.
    """
    a, b = normalize(pred, available), normalize(gold, available)
    a_ok = a.parse_status == "OK" and bool(a.predicates)
    b_ok = b.parse_status == "OK" and bool(b.predicates)
    if a_ok != b_ok:
        return False
    if not a_ok:
        return True
    if a.connective != b.connective:
        return False
    if len(a.predicates) != len(b.predicates):
        return False
    if a.connective in ("AND", "OR"):
        sa = sorted(a.predicates, key=_pred_key)
        sb = sorted(b.predicates, key=_pred_key)
        return all(_preds_eq(x, y) for x, y in zip(sa, sb))
    return all(_preds_eq(x, y) for x, y in zip(a.predicates, b.predicates))


def previous_canonical(pred: ClaimProgram, gold: ClaimProgram) -> bool:
    return programs_canonically_equal(pred, gold)


def classify_mismatch(pred: ClaimProgram, gold: ClaimProgram, available: list[str]) -> str:
    """Taxonomy for non-raw-exact pairs. Equivalence classes first."""
    if programs_strictly_equivalent(pred, gold, available):
        pn = normalize(pred, available)
        gn = normalize(gold, available)
        if previous_canonical(pred, gold):
            return "schema_redundant_or_lag_canon"
        if pred.connective in ("AND", "OR") and [ _pred_key(p) for p in pn.predicates] != [ _pred_key(p) for p in pred.predicates]:
            return "logically_equivalent_composition"
        # alias resolution changed a name
        raw_a = [(p.channel_a, p.channel_b) for p in pred.predicates]
        nor_a = [(p.channel_a, p.channel_b) for p in pn.predicates]
        if raw_a != nor_a:
            return "channel_alias_canonical_name"
        return "mathematically_equivalent_representation"
    g_ok = gold.parse_status == "OK" and bool(gold.predicates)
    p_ok = pred.parse_status == "OK" and bool(pred.predicates)
    if g_ok != p_ok:
        return "unsupported_ambiguity_disagreement"
    if not g_ok:
        return "unsupported_ambiguity_disagreement"
    if pred.connective != gold.connective:
        return "connective_error"
    if len(pred.predicates) != len(gold.predicates):
        return "missing_or_extra_predicate"
    n = min(len(pred.predicates), len(gold.predicates))
    gp, pp = gold.predicates[:n], pred.predicates[:n]
    if any(a.measurement != b.measurement for a, b in zip(gp, pp)):
        return "primitive_selection_error"
    pn, gn = normalize(pred, available), normalize(gold, available)
    if any(a.channel_a != b.channel_a or a.channel_b != b.channel_b for a, b in zip(pn.predicates, gn.predicates)):
        return "channel_semantic_error"
    if any(a.comparator != b.comparator for a, b in zip(pn.predicates, gn.predicates)):
        return "comparator_direction_error"
    if any(not _vals_close(a.reference_value, b.reference_value) for a, b in zip(pn.predicates, gn.predicates)):
        return "numeric_value_error"
    return "other_genuine_semantic_error"


EQUIV_LABELS = {
    "schema_redundant_or_lag_canon",
    "logically_equivalent_composition",
    "channel_alias_canonical_name",
    "mathematically_equivalent_representation",
}

GENUINE_FIELD = {
    "primitive_selection_error": "primitive",
    "channel_semantic_error": "channel",
    "comparator_direction_error": "comparator/value",
    "numeric_value_error": "comparator/value",
    "connective_error": "connective",
    "missing_or_extra_predicate": "structure",
    "unsupported_ambiguity_disagreement": "structure",
    "other_genuine_semantic_error": "other",
}

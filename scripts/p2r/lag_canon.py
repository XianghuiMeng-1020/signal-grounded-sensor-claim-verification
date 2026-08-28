"""Frozen lag canonicalization.

Production / independent definition (identical structure):

    xc = correlate(zscore(A), zscore(B), mode='full')
    lag_samples = argmax_|xc| inside ±MAX_LAG around lag 0

Cross-correlation reversal:

    correlate(B, A)[center + k] = correlate(A, B)[center - k]

Therefore, when the peak is unique,

    lag(A, B) = k  iff  lag(B, A) = -k

Comparator transport under channel reversal:

    eq  k   <->  eq  (-k)
    gt  k   <->  lt  (-k)
    lt  k   <->  gt  (-k)

This is derived from the primitive, not from LLM outputs or verdicts.
Canonical representation: lexicographic (channel_a, channel_b).
"""
from __future__ import annotations

from typing import Optional

from .schema import ClaimProgram, Predicate

_FLIP = {"gt": "lt", "lt": "gt", "eq": "eq", "similar": "similar", "different": "different"}


def reverse_lag_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return -float(value)


def canonicalize_lag_predicate(pred: Predicate) -> Predicate:
    if pred.measurement != "cross_channel_lag_ms":
        return pred
    a, b = pred.channel_a, pred.channel_b
    if not a or not b or a <= b:
        return pred
    return Predicate(
        measurement=pred.measurement,
        channel_a=b,
        comparator=_FLIP.get(pred.comparator, pred.comparator),
        channel_b=a,
        reference_value=reverse_lag_value(pred.reference_value),
        reference_channel=pred.reference_channel,
        unit=pred.unit,
    )


def canonicalize_program(program: ClaimProgram) -> ClaimProgram:
    return ClaimProgram(
        program.connective,
        [canonicalize_lag_predicate(p) for p in program.predicates],
        parse_status=program.parse_status,
        parse_reason=program.parse_reason,
    )

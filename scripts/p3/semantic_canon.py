"""Canonical semantic form for ClaimProgram.

Derived from the lag operator and schema field roles, not from LLM verdicts.

Lag operator (independent / production, identical structure):

    xc = correlate(zscore(A), zscore(B), mode='full')
    lag_samples = argmax_|xc| inside ±MAX_LAG of lag 0
    lag_ms = lag_samples / fs * 1000

Reversal: lag(A,B)=k  iff  lag(B,A)=-k when the peak is unique.
This holds for ALL finite equal-length nondegenerate signals, not only
the windows that happened to keep a verdict.

Schema roles (frozen):

    channel_a, channel_b : the two operands of cross_channel_lag_ms
    reference_channel    : compare-channel for a *unary* measurement
                           (gt/lt/similar/different vs another series)

The executor calls compute(lag, {channel_a, channel_b}) and never uses
reference_channel as a third lag operand. Therefore two lag predicates
that agree after order/sign canon and differ only in reference_channel
are mathematically equivalent for all signals.

Canonical lag predicate:

    lexicographic (channel_a, channel_b)
    value/comparator transported under swap
    reference_channel := None
"""
from __future__ import annotations

from dataclasses import replace

from p2r.lag_canon import canonicalize_program as order_canon
from p2r.schema import ClaimProgram, Predicate


def drop_redundant_lag_ref(pred: Predicate) -> Predicate:
    if pred.measurement != "cross_channel_lag_ms":
        return pred
    if pred.reference_channel is None:
        return pred
    return replace(pred, reference_channel=None)


def canonical_program(program: ClaimProgram) -> ClaimProgram:
    ordered = order_canon(program)
    return ClaimProgram(
        ordered.connective,
        [drop_redundant_lag_ref(p) for p in ordered.predicates],
        parse_status=ordered.parse_status,
        parse_reason=ordered.parse_reason,
    )


def _vals_close(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(b)))


def _pred_eq(a: Predicate, b: Predicate) -> bool:
    return (
        a.measurement == b.measurement
        and a.channel_a == b.channel_a
        and a.channel_b == b.channel_b
        and a.comparator == b.comparator
        and a.reference_channel == b.reference_channel
        and _vals_close(a.reference_value, b.reference_value)
    )


def programs_canonically_equal(pred: ClaimProgram, gold: ClaimProgram) -> bool:
    if (gold.parse_status == "OK" and bool(gold.predicates)) != (pred.parse_status == "OK" and bool(pred.predicates)):
        return False
    if not (gold.parse_status == "OK" and gold.predicates):
        return not (pred.parse_status == "OK" and pred.predicates)
    a, b = canonical_program(pred), canonical_program(gold)
    if a.connective != b.connective or len(a.predicates) != len(b.predicates):
        return False
    return all(_pred_eq(x, y) for x, y in zip(a.predicates, b.predicates))

from scripts.p2r.schema import ClaimProgram, Predicate
from scripts.p3.semantic_canon import canonical_program, programs_canonically_equal


def test_lag_reference_channel_is_dropped():
    gold = ClaimProgram("SINGLE", [Predicate(
        "cross_channel_lag_ms", "chest_accel", "eq",
        channel_b="ankle_accel", reference_value=10.0, unit="ms",
    )])
    pred = ClaimProgram("SINGLE", [Predicate(
        "cross_channel_lag_ms", "chest_accel", "eq",
        channel_b="ankle_accel", reference_value=10.0,
        reference_channel="ankle_accel", unit="ms",
    )])
    assert programs_canonically_equal(pred, gold)
    assert canonical_program(pred).predicates[0].reference_channel is None


def test_order_and_sign_still_equivalent():
    gold = ClaimProgram("SINGLE", [Predicate(
        "cross_channel_lag_ms", "ankle_accel", "eq",
        channel_b="chest_accel", reference_value=-10.0, unit="ms",
    )])
    pred = ClaimProgram("SINGLE", [Predicate(
        "cross_channel_lag_ms", "chest_accel", "eq",
        channel_b="ankle_accel", reference_value=10.0, unit="ms",
    )])
    assert programs_canonically_equal(pred, gold)


def test_non_lag_reference_channel_is_kept():
    a = ClaimProgram("SINGLE", [Predicate("rms_amplitude", "hand_accel", "gt", reference_channel="chest_accel")])
    b = ClaimProgram("SINGLE", [Predicate("rms_amplitude", "hand_accel", "gt", reference_channel=None)])
    assert not programs_canonically_equal(a, b)

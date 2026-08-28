import numpy as np

from scripts.p2.independent_dsp import cross_channel_lag_ms
from scripts.p2r.lag_canon import canonicalize_lag_predicate, canonicalize_program
from scripts.p2r.schema import ClaimProgram, Predicate


def test_lag_sign_reverses_when_channels_swap():
    fs = 100.0
    t = np.arange(256) / fs
    a = np.sin(2 * np.pi * 4.0 * t)
    delay = 5
    b = np.roll(a, delay)
    lab = cross_channel_lag_ms(a, b, fs)
    lba = cross_channel_lag_ms(b, a, fs)
    tol = 0.51 / fs * 1000.0 + 1e-9
    assert abs(lab + lba) < tol
    assert abs(abs(lab) - delay / fs * 1000.0) < tol


def test_canonical_order_is_lexicographic_and_negates():
    p = Predicate(
        "cross_channel_lag_ms",
        "y_accel",
        "eq",
        channel_b="x_accel",
        reference_value=50.0,
        unit="ms",
    )
    c = canonicalize_lag_predicate(p)
    assert c.channel_a == "x_accel"
    assert c.channel_b == "y_accel"
    assert c.reference_value == -50.0
    assert c.comparator == "eq"


def test_gt_becomes_lt_under_swap():
    p = Predicate(
        "cross_channel_lag_ms",
        "chest_accel",
        "gt",
        channel_b="ankle_accel",
        reference_value=10.0,
    )
    c = canonicalize_lag_predicate(p)
    assert c.channel_a == "ankle_accel"
    assert c.channel_b == "chest_accel"
    assert c.comparator == "lt"
    assert c.reference_value == -10.0


def test_already_sorted_unchanged():
    p = Predicate("cross_channel_lag_ms", "ankle_accel", "eq", channel_b="chest_accel", reference_value=3.0)
    c = canonicalize_lag_predicate(p)
    assert c.channel_a == "ankle_accel"
    assert c.reference_value == 3.0


def test_program_canon_does_not_touch_non_lag():
    prog = ClaimProgram(
        "SINGLE",
        [Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0)],
    )
    out = canonicalize_program(prog)
    assert out.predicates[0].measurement == "rms_amplitude"
    assert out.predicates[0].reference_value == 1.0

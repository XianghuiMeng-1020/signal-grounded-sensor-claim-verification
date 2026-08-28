"""P4 lag representation audit. No kernel, sign, or threshold change."""
from __future__ import annotations

import numpy as np

from p2r.executor import execute_predicate_measurement
from p2r.schema import Predicate
from p3.semantic_canon import drop_redundant_lag_ref


def scientific_lag_slots(pred: Predicate) -> tuple[str, str]:
    """Frozen executable fields interpreted as source/target. Schema names unchanged."""
    assert pred.measurement == "cross_channel_lag_ms"
    return pred.channel_a, pred.channel_b


def test_source_target_are_channel_a_channel_b():
    pred = Predicate(
        "cross_channel_lag_ms",
        "hand_accel",
        "gt",
        channel_b="chest_accel",
        reference_value=10.0,
        reference_channel=None,
    )
    source, target = scientific_lag_slots(pred)
    assert source == "hand_accel"
    assert target == "chest_accel"


def test_kernel_uses_only_the_two_lag_operands():
    rng = np.random.default_rng(0)
    x = rng.normal(size=256)
    y = np.roll(x, 8)
    pred = Predicate(
        "cross_channel_lag_ms",
        "hand_accel",
        "gt",
        channel_b="chest_accel",
        reference_value=10.0,
        reference_channel=None,
    )
    res = execute_predicate_measurement(
        pred, {"hand_accel": x, "chest_accel": y}, 100.0
    )
    assert res.status == "OK"
    assert res.value is not None


def test_redundant_reference_is_not_a_third_lag_operand_in_canon():
    pred = Predicate(
        "cross_channel_lag_ms",
        "hand_accel",
        "gt",
        channel_b="chest_accel",
        reference_value=10.0,
        reference_channel="chest_accel",
    )
    cleaned = drop_redundant_lag_ref(pred)
    assert cleaned.reference_channel is None
    assert cleaned.channel_a == pred.channel_a
    assert cleaned.channel_b == pred.channel_b

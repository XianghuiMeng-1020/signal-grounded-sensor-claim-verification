"""Property tests for production contract enforcement. Not copied from EC-DEV rows."""
import inspect

import numpy as np
import pytest

from scripts.p2.independent_dsp import measure as ref_measure
from scripts.p2r import executor
from scripts.p2r.contracts import (
    INSUFFICIENT_EVIDENCE,
    INVALID_MEASUREMENT,
    INVALID_METADATA,
    MISSING_CHANNEL,
    OK,
    check_contract,
    check_output_domain,
)
from scripts.p2r.kleene import FALSE, TRUE, UNKNOWN, compose, verdict_from_tv
from scripts.p2r.schema import Predicate
from scripts.p3r_ec import independent_oracle as oracle


def _sine(n=128, fs=100.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 3.0 * t)


def test_p1_absent_required_channel_does_not_call_kernel():
    executor.reset_kernel_counter()
    pred = Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0)
    res = executor.execute_predicate_measurement(pred, {"chest_accel": _sine()}, 100.0)
    assert res.status == MISSING_CHANNEL
    assert res.value is None
    assert executor.kernel_call_count == 0


def test_p2_invalid_fs_does_not_call_sampling_kernel():
    executor.reset_kernel_counter()
    pred = Predicate("dominant_frequency", "hand_accel", "eq", reference_value=3.0)
    res = executor.execute_predicate_measurement(pred, {"hand_accel": _sine()}, 0.0)
    assert res.status == INVALID_METADATA
    assert res.value is None
    assert executor.kernel_call_count == 0


def test_p3_insufficient_support_does_not_call_kernel():
    executor.reset_kernel_counter()
    pred = Predicate("periodicity_strength", "hand_accel", "eq", reference_value=0.2)
    res = executor.execute_predicate_measurement(pred, {"hand_accel": _sine(n=5)}, 100.0)
    assert res.status == INSUFFICIENT_EVIDENCE
    assert executor.kernel_call_count == 0


def test_p4_invalid_contract_has_no_numeric_value():
    x = _sine()
    x[3] = np.nan
    r = check_contract("rms_amplitude", {"hand_accel": x}, 100.0)
    assert r.status == INSUFFICIENT_EVIDENCE
    assert r.value is None


@pytest.mark.parametrize("conn", ["SINGLE", "AND", "OR", "IF_THEN"])
def test_p5_unknown_propagates_every_connective(conn):
    if conn == "SINGLE":
        assert compose(conn, [UNKNOWN]) == UNKNOWN
    elif conn == "IF_THEN":
        # Material implication: TRUE→UNKNOWN is UNKNOWN. Tables are not changed.
        assert compose(conn, [TRUE, UNKNOWN]) == UNKNOWN
    else:
        assert compose(conn, [TRUE, UNKNOWN]) in (TRUE, UNKNOWN)
        assert verdict_from_tv(compose(conn, [UNKNOWN, UNKNOWN])) == "UNVERIFIABLE"


def test_p6_valid_old_contract_inputs_still_ok():
    x = _sine(n=256)
    for op in (
        "rms_amplitude",
        "peak_amplitude",
        "signal_range",
        "trend_ratio",
        "dominant_frequency",
        "periodicity_strength",
        "spectral_energy_ratio_low",
    ):
        assert check_contract(op, {"hand_accel": x}, 100.0).status == OK
    y = np.roll(x, 4)
    assert check_contract("cross_channel_lag_ms", {"hand_accel": x, "chest_accel": y}, 100.0).status == OK


def test_p7_valid_production_matches_independent_reference():
    x = _sine(n=256)
    pred = Predicate("rms_amplitude", "hand_accel", "eq", reference_value=0.7)
    res = executor.execute_predicate_measurement(pred, {"hand_accel": x}, 100.0)
    ref = float(ref_measure("rms_amplitude", {"hand_accel": x}, 100.0))
    assert res.status == OK
    assert abs(res.value - ref) < 1e-9


def test_p8_nonfinite_output_cannot_become_true_false():
    post = check_output_domain("rms_amplitude", float("nan"), 100.0)
    assert post.status == INVALID_MEASUREMENT
    assert post.value is None
    pred = Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0)
    tv, ev = executor.predicate_truth(pred, {"hand_accel": np.array([np.nan, 1.0])}, 100.0)
    assert tv == UNKNOWN
    assert ev["measurement"].value is None


def test_oracle_module_does_not_import_production_validator():
    assert "p2r.contracts" not in oracle.__dict__
    assert "check_contract" not in oracle.__dict__
    assert "f_round6_operators" not in inspect.getsource(oracle)


def test_nan_on_required_channel_unknown():
    executor.reset_kernel_counter()
    x = _sine(64)
    x[10] = np.nan
    pred = Predicate("signal_range", "hand_accel", "gt", reference_value=0.1)
    tv, ev = executor.predicate_truth(pred, {"hand_accel": x}, 50.0)
    assert tv == UNKNOWN
    assert executor.kernel_call_count == 0


def test_unused_channel_nan_does_not_invalidate():
    x = _sine(64)
    y = _sine(64)
    y[2] = np.nan
    pred = Predicate("rms_amplitude", "hand_accel", "gt", reference_value=0.0)
    res = executor.execute_predicate_measurement(pred, {"hand_accel": x, "chest_accel": y}, 100.0)
    assert res.status == OK
    assert res.value is not None

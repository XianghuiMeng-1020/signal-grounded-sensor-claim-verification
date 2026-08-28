import numpy as np

from scripts.p2r.contracts import (
    INSUFFICIENT_EVIDENCE,
    INVALID_METADATA,
    MISSING_CHANNEL,
    OK,
    UNSUPPORTED,
    check_contract,
)


def _sig(n=256):
    return {"hand_accel": np.sin(np.linspace(0, 20, n))}


def test_rms_ok_on_finite():
    assert check_contract("rms_amplitude", _sig(), 100.0).status == OK


def test_nan_is_insufficient_not_ok():
    x = np.ones(32)
    x[3] = np.nan
    r = check_contract("rms_amplitude", {"hand_accel": x}, 100.0)
    assert r.status == INSUFFICIENT_EVIDENCE
    assert r.value is None


def test_empty_channel_insufficient():
    r = check_contract("rms_amplitude", {"hand_accel": []}, 100.0)
    assert r.status == INSUFFICIENT_EVIDENCE
    assert r.value is None


def test_missing_fs_for_frequency():
    r = check_contract("dominant_frequency", _sig(), None)
    assert r.status == INVALID_METADATA
    assert r.value is None


def test_bad_fs():
    r = check_contract("spectral_energy_ratio_low", _sig(), 0)
    assert r.status == INVALID_METADATA


def test_unknown_measurement_unsupported():
    r = check_contract("heart_rate_estimate", _sig(), 100)
    assert r.status == UNSUPPORTED
    assert r.value is None


def test_periodicity_short_window():
    r = check_contract("periodicity_strength", {"hand_accel": np.ones(5)}, 100)
    assert r.status == INSUFFICIENT_EVIDENCE


def test_lag_needs_two_and_length():
    a = np.sin(np.linspace(0, 10, 256))
    r = check_contract("cross_channel_lag_ms", {"hand_accel": a}, 100)
    assert r.status == MISSING_CHANNEL
    r2 = check_contract(
        "cross_channel_lag_ms",
        {"hand_accel": a, "chest_accel": np.roll(a, 3)},
        100,
    )
    assert r2.status == OK


def test_valid_noise_still_ok():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 20, 256)) + 0.01 * rng.normal(size=256)
    assert check_contract("rms_amplitude", {"hand_accel": x}, 100).status == OK

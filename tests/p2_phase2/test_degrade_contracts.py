"""Property checks: dropout must refuse; AWGN/clip must remain finite."""
import numpy as np

from scripts.p2_phase2.degrade import apply, awgn, clip_amp, dropout
from scripts.p2r.contracts import OK, check_contract


def test_dropout_is_nonfinite_and_fails_contract():
    x = np.ones(256)
    y = dropout(x, 0.1, "unit:dropout")
    assert np.any(~np.isfinite(y))
    gate = check_contract("rms_amplitude", {"ch": y}, 50.0)
    assert gate.status != OK


def test_awgn_and_clip_stay_finite():
    x = np.linspace(-1, 1, 256)
    a = awgn(x, 10.0, "unit:awgn")
    c = clip_amp(x, 0.6)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(c))
    assert check_contract("rms_amplitude", {"ch": a}, 50.0).status == OK
    assert check_contract("rms_amplitude", {"ch": c}, 50.0).status == OK


def test_apply_touches_only_named_channel():
    ch = {"a": np.ones(32), "b": np.full(32, 2.0)}
    out = apply("dropout_10pct", ch, ["a"], "unit:named")
    assert np.any(~np.isfinite(out["a"]))
    assert np.all(out["b"] == 2.0)

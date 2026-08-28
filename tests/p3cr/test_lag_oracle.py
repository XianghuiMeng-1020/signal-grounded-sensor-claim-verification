"""Analytic tests for the independent lag threshold oracle.

Production verifier is not imported and does not label gold.
"""
from __future__ import annotations

import numpy as np
import pytest

from p2.config import LAG_MAX_SAMPLES, lag_max_samples
from p2.independent_adjudicator import adjudicate, predicate_truth
from p2.independent_dsp import cross_channel_lag_ms, measure


def _pair(delay_samples: int, n: int = 256, fs: float = 100.0, seed: int = 0):
    rng = np.random.default_rng(seed + 17)
    x = rng.normal(size=n)
    y = np.roll(x, delay_samples)
    return {"channels": {"a": x, "b": y}, "fs": fs}


def test_sign_and_milliseconds_several_rates():
    cases = []
    for fs in (20.0, 50.0, 100.0, 200.0):
        for delay in (-12, -3, 0, 5, 18):
            if abs(delay) >= lag_max_samples(fs):
                continue
            rep = _pair(delay, n=256, fs=fs, seed=int(fs) + delay)
            got = cross_channel_lag_ms(rep["channels"]["a"], rep["channels"]["b"], fs)
            expect_abs = abs(delay) / fs * 1000.0
            tol = 0.51 / fs * 1000.0 + 1e-6
            assert abs(abs(got) - expect_abs) < tol
            swapped = cross_channel_lag_ms(rep["channels"]["b"], rep["channels"]["a"], fs)
            assert abs(got + swapped) < tol
            cases.append(True)
    assert len(cases) >= 12


def test_measure_uses_first_two_names_as_order():
    rep = _pair(8, fs=100.0)
    lab = measure("cross_channel_lag_ms", {"a": rep["channels"]["a"], "b": rep["channels"]["b"]}, 100.0)
    lba = measure("cross_channel_lag_ms", {"b": rep["channels"]["b"], "a": rep["channels"]["a"]}, 100.0)
    assert abs(lab + lba) < 0.51 / 100.0 * 1000.0 + 1e-6


def test_threshold_truth_and_equality_is_not_unverifiable():
    rep = _pair(10, fs=100.0)
    actual = measure("cross_channel_lag_ms", {"ch0": rep["channels"]["a"], "ch1": rep["channels"]["b"]}, 100.0)
    pred_gt = {
        "op": "cross_channel_lag_ms",
        "channels": ["ch0", "ch1"],
        "mode": "vs_threshold",
        "threshold": actual,
        "relation": "gt",
    }
    # remap names into rep
    rep2 = {"channels": {"ch0": rep["channels"]["a"], "ch1": rep["channels"]["b"]}, "fs": 100.0}
    t, ev = predicate_truth(rep2, pred_gt)
    assert t is False
    assert ev.get("equal") is True
    gold = adjudicate(rep2, {"connective": "SINGLE", "predicates": [pred_gt]})
    assert gold["verdict"] == "CONTRADICTED"

    pred_lt = dict(pred_gt, relation="lt", threshold=actual - 5.0)
    gold2 = adjudicate(rep2, {"connective": "SINGLE", "predicates": [pred_lt]})
    assert gold2["verdict"] == "CONTRADICTED"

    pred_gt_hi = dict(pred_gt, relation="gt", threshold=actual - 5.0)
    gold3 = adjudicate(rep2, {"connective": "SINGLE", "predicates": [pred_gt_hi]})
    assert gold3["verdict"] == "SUPPORTED"


def test_search_box_limit():
    assert LAG_MAX_SAMPLES == 30
    rep = _pair(8, n=128, fs=50.0)
    val = cross_channel_lag_ms(rep["channels"]["a"], rep["channels"]["b"], 50.0)
    assert np.isfinite(val)


def test_production_not_used_for_gold():
    import inspect
    src = inspect.getsource(predicate_truth)
    assert "p2r.executor" not in src
    assert "run_pipeline" not in src

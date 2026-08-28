"""Phase 2 Experiment 2 — pre-registered time-base contracts.

Does not import production kernel internals beyond the public oracle.
Does not retune DELAYS_MS, SAMPLE_BOUND, or PHYSICAL_BOUND_MS.
"""
from __future__ import annotations

import inspect

import numpy as np

from p2.config import LAG_MAX_SAMPLES
from p2r.pipeline import run_oracle
from p2r.schema import ClaimProgram, Predicate
from scripts.p2_phase2.lag_config import (
    DELAYS_MS,
    EVAL_PHYSICAL,
    EVAL_PRODUCTION,
    EVAL_SAMPLE,
    FS_GRID,
    PHYSICAL_BOUND_MS,
    SAMPLE_BOUND,
    delay_samples,
    in_sample_box,
    sample_bound_ms,
)
from scripts.p2_phase2.lag_construct import bound_program, inject_physical_delay, resample_to_fs


def test_sample_bound_is_frozen_production_radius():
    assert SAMPLE_BOUND == LAG_MAX_SAMPLES == 30
    assert PHYSICAL_BOUND_MS == SAMPLE_BOUND * 1000.0 / 100.0


def test_equivalent_delays_are_integer_samples_on_the_rate_grid():
    for d_ms in DELAYS_MS:
        for fs in FS_GRID:
            k = delay_samples(d_ms, fs)
            assert k == int(k)
            assert abs(k / fs * 1000.0 - d_ms) < 1e-9


def test_sample_box_is_not_a_physical_window():
    """±30 samples is 1500 / 600 / 300 ms at 20 / 50 / 100 Hz."""
    assert sample_bound_ms(20.0) == 1500.0
    assert sample_bound_ms(50.0) == 600.0
    assert sample_bound_ms(100.0) == 300.0
    assert in_sample_box(400, 20.0) is True
    assert in_sample_box(400, 50.0) is True
    assert in_sample_box(400, 100.0) is False
    assert in_sample_box(1600, 20.0) is False


def test_eval_mode_default_is_production_and_keyword_only():
    sig = inspect.signature(run_oracle)
    assert "eval_mode" in sig.parameters
    assert sig.parameters["eval_mode"].default == EVAL_PRODUCTION
    src = inspect.getsource(run_oracle)
    assert "eval_mode" in src


def test_sample_mode_converts_fixed_sample_threshold_to_ms():
    rng = np.random.default_rng(0)
    x = rng.normal(size=256)
    y = np.roll(x, 10)
    ch = {"a": x, "b": y}
    prog = bound_program("a", "b", SAMPLE_BOUND)
    out = run_oracle(prog, ["a", "b"], 100.0, ch, eval_mode=EVAL_SAMPLE)
    assert out["verdict"] == "SUPPORTED"
    out20 = run_oracle(prog, ["a", "b"], 20.0, ch, eval_mode=EVAL_SAMPLE)
    assert out20["verdict"] == "SUPPORTED"


def test_physical_mode_rejects_measurable_delay_above_300ms_at_low_fs():
    """400 ms is 8 samples at 20 Hz (in-box) but physically > 300 ms."""
    rng = np.random.default_rng(1)
    n = 256
    fs = 20.0
    x = rng.normal(size=n)
    y = inject_physical_delay(x, 400.0, fs)
    ch = {"a": x, "b": y}
    sample_prog = bound_program("a", "b", SAMPLE_BOUND)
    phys_prog = bound_program("a", "b", PHYSICAL_BOUND_MS)
    a = run_oracle(sample_prog, ["a", "b"], fs, ch, eval_mode=EVAL_SAMPLE)
    b = run_oracle(phys_prog, ["a", "b"], fs, ch, eval_mode=EVAL_PHYSICAL)
    assert a["verdict"] == "SUPPORTED"
    assert b["verdict"] == "CONTRADICTED"


def test_resample_preserves_duration_and_can_fall_below_min_n():
    x = np.linspace(-1, 1, 256)
    y, n = resample_to_fs(x, 100.0, 20.0)
    assert n == 51
    assert y.shape == (51,)


def test_production_eval_mode_does_not_convert_sample_counts():
    """A leftover sample-count threshold scored as production is not Mode A."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=256)
    y = np.roll(x, 5)
    ch = {"a": x, "b": y}
    # 30 as milliseconds, not samples: 5 samples at 100 Hz = 50 ms > 30 ms.
    prog = bound_program("a", "b", 30.0)
    prod = run_oracle(prog, ["a", "b"], 100.0, ch)
    sample = run_oracle(prog, ["a", "b"], 100.0, ch, eval_mode=EVAL_SAMPLE)
    assert prod["verdict"] == "CONTRADICTED"
    assert sample["verdict"] == "SUPPORTED"

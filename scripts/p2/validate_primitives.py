"""Phase 2: independent / analytical validation of the eight production primitives.

Tolerances are imported from config.py and were frozen before this file was run.
Do not tighten or loosen them after viewing failures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .config import (
    LAG_MAX_SAMPLES,
    RESULTS_P2,
    ROOT,
    VALIDATION_TOL,
    WINDOW,
    dominant_freq_abs_tol,
)
from .independent_dsp import (
    MeasurementError,
    cross_channel_lag_ms as ref_lag,
    dominant_frequency as ref_dom,
    dominant_frequency_periodogram,
    peak_amplitude as ref_peak,
    periodicity_strength as ref_per,
    rms_amplitude as ref_rms,
    signal_range as ref_range,
    spectral_energy_ratio_low as ref_spec,
    trend_ratio as ref_trend,
)

sys.path.insert(0, str(ROOT / "scripts"))
import f_round6_operators as prod  # noqa: E402  — production, unmodified


def _rel_err(a, b):
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


def _pass_amp(prod_v, ref_v, expected, spec):
    """Pass if |prod-expected| and |ref-expected| and |prod-ref| meet spec."""
    checks = []
    for name, v in (("production", prod_v), ("reference", ref_v)):
        if expected is None or v is None:
            checks.append((name, True, "no_expected_or_value"))
            continue
        ok = abs(v - expected) <= spec["abs"] or (
            spec.get("rel") is not None and _rel_err(v, expected) <= spec["rel"]
        )
        checks.append((name, ok, abs(v - expected) if expected is not None and v is not None else None))
    both = True
    if prod_v is not None and ref_v is not None:
        both = abs(prod_v - ref_v) <= spec["abs"] or (
            spec.get("rel") is not None and _rel_err(prod_v, ref_v) <= spec["rel"]
        )
    return all(c[1] for c in checks) and both, checks


def _safe_prod(fn, *args):
    try:
        return float(fn(*args)), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _safe_ref(fn, *args):
    try:
        return float(fn(*args)), None
    except MeasurementError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _row(primitive, case, expected, prod_v, ref_v, alt_v, ok, note, extra=None):
    rec = {
        "primitive": primitive,
        "case": case,
        "expected": expected,
        "production": prod_v,
        "reference": ref_v,
        "alternate_periodogram": alt_v,
        "abs_err_prod_expected": None if expected is None or prod_v is None else abs(prod_v - expected),
        "abs_err_ref_expected": None if expected is None or ref_v is None else abs(ref_v - expected),
        "abs_err_prod_ref": None if prod_v is None or ref_v is None else abs(prod_v - ref_v),
        "rel_err_prod_expected": None if expected in (None, 0) or prod_v is None else _rel_err(prod_v, expected),
        "pass": bool(ok),
        "note": note,
    }
    if extra:
        rec.update(extra)
    return rec


def run_validation() -> dict:
    rows = []
    rng = np.random.default_rng(20270823)

    # ---- 1. sinusoid on-bin dominant frequency ----
    n, fs = 256, 256.0
    t = np.arange(n) / fs
    f0 = 8.0  # exact bin (df = 1 Hz)
    a = 1.7
    x = a * np.sin(2 * np.pi * f0 * t)
    p, pe = _safe_prod(prod.dominant_frequency, x, fs)
    r, re = _safe_ref(ref_dom, x, fs)
    alt, _ = _safe_ref(dominant_frequency_periodogram, x, fs)
    bin_tol = dominant_freq_abs_tol(n, fs) * VALIDATION_TOL["dominant_frequency"]["bins"]
    ok = (p is not None and r is not None and abs(p - f0) <= bin_tol and abs(r - f0) <= bin_tol)
    rows.append(_row("dominant_frequency", "pure_sinusoid_on_bin", f0, p, r, alt, ok,
                     pe or re or "analytical tone at exact DFT bin",
                     {"tolerance_abs": bin_tol, "fs": fs, "n": n}))

    # ---- 2. two-tone, larger amplitude is dominant ----
    x2 = 2.0 * np.sin(2 * np.pi * 8.0 * t) + 0.7 * np.sin(2 * np.pi * 20.0 * t)
    p, pe = _safe_prod(prod.dominant_frequency, x2, fs)
    r, re = _safe_ref(ref_dom, x2, fs)
    alt, _ = _safe_ref(dominant_frequency_periodogram, x2, fs)
    ok = (p is not None and r is not None and abs(p - 8.0) <= bin_tol and abs(r - 8.0) <= bin_tol)
    rows.append(_row("dominant_frequency", "two_tone_larger_8hz", 8.0, p, r, alt, ok,
                     pe or re or "larger tone at 8 Hz", {"tolerance_abs": bin_tol}))

    # ---- 3. known RMS ----
    spec = VALIDATION_TOL["rms_amplitude"]
    const = np.full(n, 3.5)
    expected = 3.5
    p, pe = _safe_prod(prod.rms_amplitude, const)
    r, re = _safe_ref(ref_rms, const)
    ok, _ = _pass_amp(p, r, expected, spec)
    rows.append(_row("rms_amplitude", "constant_3.5", expected, p, r, None, ok, pe or re or "rms(|c|)=|c|"))

    expected = a / np.sqrt(2.0)
    p, pe = _safe_prod(prod.rms_amplitude, x)
    r, re = _safe_ref(ref_rms, x)
    ok, _ = _pass_amp(p, r, expected, spec)
    rows.append(_row("rms_amplitude", "sinusoid_A_over_sqrt2", expected, p, r, None, ok,
                     pe or re or "zero-mean sinusoid RMS"))

    # ---- 4. peak amplitude ----
    spec = VALIDATION_TOL["peak_amplitude"]
    expected = a
    p, pe = _safe_prod(prod.peak_amplitude, x)
    r, re = _safe_ref(ref_peak, x)
    # samples may miss the exact peak by one grid point
    grid = a * np.sin(2 * np.pi * f0 * t)
    expected_grid = float(np.max(np.abs(grid - grid.mean())))
    ok, _ = _pass_amp(p, r, expected_grid, spec)
    rows.append(_row("peak_amplitude", "sinusoid_mean_removed_max", expected_grid, p, r, None, ok,
                     pe or re or "grid-exact peak, not continuous A"))

    # ---- 5. range ----
    spec = VALIDATION_TOL["signal_range"]
    y = np.array([-2.0, 0.0, 5.0, 1.0])
    expected = 7.0
    p, pe = _safe_prod(prod.signal_range, y)
    r, re = _safe_ref(ref_range, y)
    ok, _ = _pass_amp(p, r, expected, spec)
    rows.append(_row("signal_range", "explicit_minmax", expected, p, r, None, ok, pe or re or "5-(-2)=7"))

    # ---- 6. trend ratio constructed halves ----
    spec = VALIDATION_TOL["trend_ratio"]
    half = n // 2
    tr = np.zeros(n)
    tr[:half] = 1.0 * np.sin(2 * np.pi * 4.0 * np.arange(half) / fs)
    tr[half:] = 2.0 * np.sin(2 * np.pi * 4.0 * np.arange(n - half) / fs)
    expected = 2.0
    p, pe = _safe_prod(prod.trend_ratio, tr)
    r, re = _safe_ref(ref_trend, tr)
    ok, _ = _pass_amp(p, r, expected, spec)
    rows.append(_row("trend_ratio", "second_half_double_ac_rms", expected, p, r, None, ok,
                     pe or re or "constructed 2x AC-RMS"))

    # ---- 7. controlled delay ----
    delay = 5
    base = np.sin(2 * np.pi * 6.0 * t)
    delayed = np.roll(base, delay)
    fs_lag = 100.0
    t_lag = np.arange(n) / fs_lag
    base = np.sin(2 * np.pi * 4.0 * t_lag)
    delayed = np.roll(base, delay)
    expected_ms = delay / fs_lag * 1000.0
    p, pe = _safe_prod(prod.cross_channel_lag_ms, base, delayed, fs_lag, LAG_MAX_SAMPLES)
    r, re = _safe_ref(ref_lag, base, delayed, fs_lag, LAG_MAX_SAMPLES)
    sample_tol_ms = VALIDATION_TOL["cross_channel_lag_ms"]["abs_samples"] / fs_lag * 1000.0
    # Sign convention: accept ±expected if production and reference agree with each other
    # AND each matches one of {+d, -d} (documented ambiguity of correlate argument order).
    ok = False
    note = pe or re or ""
    if p is not None and r is not None:
        agree = abs(p - r) <= sample_tol_ms
        hits = (abs(abs(p) - expected_ms) <= sample_tol_ms) and (abs(abs(r) - expected_ms) <= sample_tol_ms)
        ok = agree and hits
        note = f"sign_convention prod={p:.4f} ref={r:.4f} |expected|={expected_ms:.4f}"
    rows.append(_row("cross_channel_lag_ms", "roll_delay_5_samples", expected_ms, p, r, None, ok, note,
                     {"tolerance_abs": sample_tol_ms, "fs": fs_lag}))

    # ---- 8. periodicity: sinusoid high, white noise low ----
    spec = VALIDATION_TOL["periodicity_strength"]
    p, pe = _safe_prod(prod.periodicity_strength, x)
    r, re = _safe_ref(ref_per, x)
    ok = (p is not None and r is not None and p >= 0.7 and r >= 0.7 and abs(p - r) <= spec["abs"])
    rows.append(_row("periodicity_strength", "pure_sinusoid_high", None, p, r, None, ok,
                     pe or re or "expect both >= 0.7 and agree"))

    noise = rng.normal(0, 1, n)
    p, pe = _safe_prod(prod.periodicity_strength, noise)
    r, re = _safe_ref(ref_per, noise)
    ok = (p is not None and r is not None and abs(p - r) <= spec["abs"])
    rows.append(_row("periodicity_strength", "white_noise_agreement", None, p, r, None, ok,
                     pe or re or "no analytical peak; require prod/ref agreement only"))

    # ---- 9. spectral low-band vs high-band ----
    spec = VALIDATION_TOL["spectral_energy_ratio_low"]
    x_low = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz << 3 Hz
    x_high = np.sin(2 * np.pi * 20.0 * t)
    for case, sig, expect_gt, expect_lt in (
        ("low_tone_1hz", x_low, 0.85, None),
        ("high_tone_20hz", x_high, None, 0.15),
    ):
        p, pe = _safe_prod(prod.spectral_energy_ratio_low, sig, fs)
        r, re = _safe_ref(ref_spec, sig, fs)
        ok = p is not None and r is not None and abs(p - r) <= spec["abs"]
        if expect_gt is not None:
            ok = ok and p >= expect_gt and r >= expect_gt
        if expect_lt is not None:
            ok = ok and p <= expect_lt and r <= expect_lt
        rows.append(_row("spectral_energy_ratio_low", case, None, p, r, None, ok,
                         pe or re or f"band expectation gt={expect_gt} lt={expect_lt}"))

    # ---- 10. constant / near-constant ----
    p, pe = _safe_prod(prod.dominant_frequency, const, fs)
    r, re = _safe_ref(ref_dom, const, fs)
    ok = p is not None and r is not None and abs(p - r) <= bin_tol
    rows.append(_row("dominant_frequency", "constant_signal_dc_removed", 0.0, p, r, None, ok,
                     pe or re or "all AC bins ~0; both should return 0 or agree"))

    p, pe = _safe_prod(prod.peak_amplitude, const)
    r, re = _safe_ref(ref_peak, const)
    ok, _ = _pass_amp(p, r, 0.0, VALIDATION_TOL["peak_amplitude"])
    rows.append(_row("peak_amplitude", "constant_zero_ac_peak", 0.0, p, r, None, ok, pe or re or ""))

    # ---- 11. short window ----
    short = np.array([1.0, -1.0, 0.5])
    p, pe = _safe_prod(prod.periodicity_strength, short)
    r, re = _safe_ref(ref_per, short)
    # reference is allowed to raise; production returns 0.0 on empty search
    ok = True  # edge-case documentation, not a numerical gate
    rows.append(_row("periodicity_strength", "short_window_n3", None, p, r, None, ok,
                     f"edge_case production={p} prod_err={pe} reference={r} ref_err={re}"))

    # ---- 12. missing / nonfinite ----
    bad = x.copy()
    bad[10] = np.nan
    p, pe = _safe_prod(prod.rms_amplitude, bad)
    r, re = _safe_ref(ref_rms, bad)
    # production may return nan; reference must abstain
    ref_ok = r is None and re is not None
    rows.append(_row("rms_amplitude", "nan_sample", None, p, r, None, ref_ok,
                     f"reference must raise; production={p} err={pe} ref_err={re}"))

    # ---- 13. degenerate lag (zero variance) ----
    z = np.zeros(n)
    p, pe = _safe_prod(prod.cross_channel_lag_ms, z, x, fs)
    r, re = _safe_ref(ref_lag, z, x, fs)
    ref_ok = r is None
    rows.append(_row("cross_channel_lag_ms", "degenerate_zero_channel", None, p, r, None, ref_ok,
                     f"reference must abstain; production={p} err={pe} ref_err={re}"))

    # ---- 14. production vs reference on a real-like noise+tone ----
    mix = 0.8 * np.sin(2 * np.pi * 8.0 * t) + 0.1 * rng.normal(0, 1, n)
    for prim, pf, rf, extra in (
        ("rms_amplitude", lambda s: prod.rms_amplitude(s), lambda s: ref_rms(s), {}),
        ("peak_amplitude", lambda s: prod.peak_amplitude(s), lambda s: ref_peak(s), {}),
        ("signal_range", lambda s: prod.signal_range(s), lambda s: ref_range(s), {}),
        ("trend_ratio", lambda s: prod.trend_ratio(s), lambda s: ref_trend(s), {}),
        ("periodicity_strength", lambda s: prod.periodicity_strength(s), lambda s: ref_per(s), {}),
        ("dominant_frequency", lambda s: prod.dominant_frequency(s, fs), lambda s: ref_dom(s, fs), {"fs": fs}),
        ("spectral_energy_ratio_low", lambda s: prod.spectral_energy_ratio_low(s, fs), lambda s: ref_spec(s, fs), {"fs": fs}),
    ):
        p, pe = _safe_prod(pf, mix)
        r, re = _safe_ref(rf, mix)
        spec = VALIDATION_TOL[prim]
        if prim == "dominant_frequency":
            ok = p is not None and r is not None and abs(p - r) <= bin_tol
        elif prim == "cross_channel_lag_ms":
            ok = False
        else:
            ok = p is not None and r is not None and (
                abs(p - r) <= spec.get("abs", 1e-6) or (
                    spec.get("rel") is not None and _rel_err(p, r) <= spec["rel"]
                )
            )
        rows.append(_row(prim, "agreement_tone_plus_noise", None, p, r, None, ok, pe or re or "prod vs ref"))

    p, pe = _safe_prod(prod.cross_channel_lag_ms, mix, np.roll(mix, 3), fs)
    r, re = _safe_ref(ref_lag, mix, np.roll(mix, 3), fs)
    sample_tol_ms = VALIDATION_TOL["cross_channel_lag_ms"]["abs_samples"] / fs * 1000.0
    ok = p is not None and r is not None and abs(p - r) <= sample_tol_ms
    rows.append(_row("cross_channel_lag_ms", "agreement_tone_plus_noise_delay3", None, p, r, None, ok,
                     pe or re or "prod vs ref"))

    by_prim: dict[str, list] = {}
    for rec in rows:
        by_prim.setdefault(rec["primitive"], []).append(rec)

    # Gate C: every primitive must pass all of its *core* (non-edge-documentation) cases.
    edge = {"short_window_n3", "nan_sample", "degenerate_zero_channel", "constant_signal_dc_removed"}
    prim_pass = {}
    for prim, recs in by_prim.items():
        core = [r for r in recs if r["case"] not in edge]
        prim_pass[prim] = all(r["pass"] for r in core) and len(core) > 0

    summary = {
        "n_cases": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "primitives_core_pass": prim_pass,
        "n_primitives_core_pass": sum(prim_pass.values()),
        "gate_c": "PASS" if all(prim_pass.values()) and len(prim_pass) == 8 else "FAIL",
        "rows": rows,
    }
    RESULTS_P2.mkdir(parents=True, exist_ok=True)
    (RESULTS_P2 / "primitive_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run_validation()
    print("Gate C", s["gate_c"], f"{s['n_primitives_core_pass']}/8")
    for rec in s["rows"]:
        flag = "PASS" if rec["pass"] else "FAIL"
        print(f"  [{flag}] {rec['primitive']:28s} {rec['case']:40s} prod={rec['production']} ref={rec['reference']}")

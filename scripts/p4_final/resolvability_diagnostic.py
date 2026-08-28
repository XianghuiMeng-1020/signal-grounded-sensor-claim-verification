"""Post-hoc measurement-resolvability diagnostic on the frozen invariance run.

Does not retune T_max, thresholds, rates, or subsets.
Bins were fixed before inspecting per-bin rates:
  rho = |v_ref - theta| / Delta,  Delta = sample interval (ms) or bin width (Hz).
  partitions: rho < 0.5;  0.5 <= rho < 1;  1 <= rho < 2;  rho >= 2.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import correlate, decimate

ROOT_BOOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_BOOT / "scripts"))

from p2.config import ROOT
from p35.windows_ir import load_unused_windows

T_MAX_S = 0.300


def L_of_fs(fs: float) -> int:
    return int(np.floor(T_MAX_S * float(fs)))

OUT = ROOT / "results" / "p4_final"
REP = ROOT / "reports" / "ICASSP_FINAL_10OF10"
RUN = OUT / "physical_invariance_run.json"
BINS = (
    ("rho<0.5", lambda r: r < 0.5),
    ("0.5<=rho<1", lambda r: 0.5 <= r < 1.0),
    ("1<=rho<2", lambda r: 1.0 <= r < 2.0),
    ("rho>=2", lambda r: r >= 2.0),
)
N_NOM = 256.0


def _bin(rho: float) -> str:
    for name, fn in BINS:
        if fn(rho):
            return name
    return "rho>=2"


def _rate(xs: list[bool]) -> dict:
    n = len(xs)
    k = sum(1 for x in xs if x)
    return {"n": n, "k": k, "p": (k / n) if n else None}


def _peak_gap(a: np.ndarray, b: np.ndarray, fs: float) -> dict:
    za = (a - a.mean()) / (a.std() + 1e-12)
    zb = (b - b.mean()) / (b.std() + 1e-12)
    xc = correlate(za, zb, mode="full")
    mid = len(a) - 1
    L = int(L_of_fs(fs))
    lo, hi = mid - L, mid + L
    sl = xc[lo : hi + 1]
    absv = np.abs(sl)
    order = np.argsort(absv)[::-1]
    rmax = float(absv[order[0]])
    r2 = float(absv[order[1]]) if order.size > 1 else 0.0
    return {"r_max": rmax, "r_2nd": r2, "g": rmax - r2, "g_norm": (rmax - r2) / max(rmax, 1e-12)}


def main() -> None:
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    rows = blob["rows"]
    # --- H1/H2 lag ---
    lag_rows = []
    for r in rows:
        if r["op"] != "lag_physical" or not r.get("licensed") or r.get("v") is None:
            continue
        fs = float(r["fs"])
        dt = 1000.0 / fs
        m = abs(float(r["v_ref"]) - float(r["threshold"]))
        rho = m / dt
        err_ms = abs(float(r["v"]) - float(r["v_ref"]))
        lag_rows.append({
            **{k: r[k] for k in ("window_id", "fs", "v_ref", "threshold", "v", "verdict_match", "e_abs", "e_margin")},
            "dt_ms": dt,
            "m_tau_ms": m,
            "rho_tau": rho,
            "rho_bin": _bin(rho),
            "err_ms": err_ms,
            "err_samples": err_ms / dt,
            "L_fs": int(L_of_fs(fs)),
        })

    lag_by = defaultdict(lambda: defaultdict(list))
    for r in lag_rows:
        lag_by[int(r["fs"])][r["rho_bin"]].append(r["verdict_match"])

    lag_err = {}
    for fs in (50, 20):
        sub = [r for r in lag_rows if int(r["fs"]) == fs]
        lag_err[str(fs)] = {
            "n": len(sub),
            "median_err_ms": float(np.median([r["err_ms"] for r in sub])) if sub else None,
            "median_err_samples": float(np.median([r["err_samples"] for r in sub])) if sub else None,
            "median_rho": float(np.median([r["rho_tau"] for r in sub])) if sub else None,
            "verdict": _rate([r["verdict_match"] for r in sub]),
            "by_rho_bin": {name: _rate(lag_by[fs][name]) for name, _ in BINS},
        }

    # Peak-gap diagnostic on the same 32 unused PAMAP2 windows (not a new set).
    wins = [w for w in load_unused_windows() if w.get("dataset") == "PAMAP2"]
    wins = sorted(wins, key=lambda w: w["window_id"])[:32]
    peak = []
    for w in wins:
        names = list(w["channels"])
        if len(names) < 2:
            continue
        a100 = np.asarray(w["channels"][names[0]], dtype=float)
        b100 = np.asarray(w["channels"][names[1]], dtype=float)
        series = {
            100.0: (a100, b100),
            50.0: (decimate(a100, 2, n=8, ftype="iir", zero_phase=True),
                    decimate(b100, 2, n=8, ftype="iir", zero_phase=True)),
            20.0: (decimate(a100, 5, n=8, ftype="iir", zero_phase=True),
                    decimate(b100, 5, n=8, ftype="iir", zero_phase=True)),
        }
        for fs, (aa, bb) in series.items():
            g = _peak_gap(aa, bb, fs)
            g.update({"window_id": w["window_id"], "fs": fs, "n": int(aa.size)})
            peak.append(g)
    peak_med = {}
    for fs in (100, 50, 20):
        sub = [p for p in peak if int(p["fs"]) == fs]
        peak_med[str(fs)] = {
            "median_g": float(np.median([p["g"] for p in sub])) if sub else None,
            "median_g_norm": float(np.median([p["g_norm"] for p in sub])) if sub else None,
        }

    # Join peak gap at target fs to lag rows
    gap_map = {(p["window_id"], int(p["fs"])): p for p in peak}
    for r in lag_rows:
        p = gap_map.get((r["window_id"], int(r["fs"])))
        if p:
            r["g"] = p["g"]
            r["g_norm"] = p["g_norm"]
    disagree = [r for r in lag_rows if not r["verdict_match"]]
    agree = [r for r in lag_rows if r["verdict_match"]]

    # --- I frequency ---
    freq_rows = []
    for r in rows:
        if r["op"] != "dominant_frequency" or not r.get("licensed") or r.get("v") is None:
            continue
        fs = float(r["fs"])
        # decimated length: 256/q
        q = {50.0: 2, 20.0: 5}[fs]
        n_fs = math.floor(N_NOM / q)
        df = fs / n_fs
        m = abs(float(r["v_ref"]) - float(r["threshold"]))
        rho = m / df
        nyq = fs / 2.0
        freq_rows.append({
            **{k: r[k] for k in ("window_id", "fs", "v_ref", "threshold", "v", "verdict_match", "e_abs")},
            "n_fs": n_fs,
            "df_hz": df,
            "m_f_hz": m,
            "rho_f": rho,
            "rho_bin": _bin(rho),
            "nyquist": nyq,
            "near_nyquist": float(r["v_ref"]) > 0.8 * nyq or float(r["v"]) > 0.8 * nyq,
        })
    freq_by = defaultdict(lambda: defaultdict(list))
    for r in freq_rows:
        freq_by[int(r["fs"])][r["rho_bin"]].append(r["verdict_match"])
    freq_sum = {}
    for fs in (50, 20):
        sub = [r for r in freq_rows if int(r["fs"]) == fs]
        freq_sum[str(fs)] = {
            "n": len(sub),
            "df_hz": fs / math.floor(N_NOM / {50: 2, 20: 5}[fs]),
            "median_rho": float(np.median([r["rho_f"] for r in sub])) if sub else None,
            "verdict": _rate([r["verdict_match"] for r in sub]),
            "disagree_n": sum(1 for r in sub if not r["verdict_match"]),
            "disagree_near_nyquist": sum(1 for r in sub if (not r["verdict_match"]) and r["near_nyquist"]),
            "by_rho_bin": {name: _rate(freq_by[fs][name]) for name, _ in BINS},
        }

    # --- J low-band signed boundary ---
    lb_rows = []
    for r in rows:
        if r["op"] != "spectral_energy_ratio_low" or not r.get("licensed") or r.get("v") is None:
            continue
        d_ref = float(r["v_ref"]) - float(r["threshold"])
        d_fs = float(r["v"]) - float(r["threshold"])
        lb_rows.append({
            **{k: r[k] for k in ("window_id", "fs", "v_ref", "threshold", "v", "verdict_match", "e_abs", "e_margin")},
            "d_ref": d_ref,
            "d_fs": d_fs,
            "same_side": (d_ref > 0) == (d_fs > 0),
            "emargin_gt1": float(r["e_margin"]) > 1.0,
        })
    lb_sum = {}
    for fs in (50, 20):
        sub = [r for r in lb_rows if int(r["fs"]) == fs]
        lb_sum[str(fs)] = {
            "n": len(sub),
            "verdict": _rate([r["verdict_match"] for r in sub]),
            "same_side": _rate([r["same_side"] for r in sub]),
            "emargin_gt1_n": sum(1 for r in sub if r["emargin_gt1"]),
            "emargin_gt1_and_same_side": sum(1 for r in sub if r["emargin_gt1"] and r["same_side"]),
            "median_e_margin": float(np.median([r["e_margin"] for r in sub])) if sub else None,
            "median_abs_d_fs": float(np.median([abs(r["d_fs"]) for r in sub])) if sub else None,
        }

    # Does resolvability explain mixed lag?
    # Compare low-rho vs high-rho verdict at 20 Hz (bins predeclared).
    low = _rate(lag_by[20]["rho<0.5"] + lag_by[20]["0.5<=rho<1"])
    high = _rate(lag_by[20]["1<=rho<2"] + lag_by[20]["rho>=2"])
    # concentration: fraction of disagreements in rho<1
    dis20 = [r for r in lag_rows if int(r["fs"]) == 20 and not r["verdict_match"]]
    all20 = [r for r in lag_rows if int(r["fs"]) == 20]
    frac_dis_lowrho = (
        sum(1 for r in dis20 if r["rho_tau"] < 1.0) / len(dis20) if dis20 else None
    )
    explains = "NO"
    if dis20 and frac_dis_lowrho is not None and frac_dis_lowrho >= 0.75 and (low["p"] or 0) < (high["p"] or 1):
        explains = "YES"
    elif dis20 and (low["p"] is not None) and (high["p"] is not None) and low["p"] < high["p"]:
        explains = "PARTLY"

    out = {
        "analysis": "post-hoc mechanism diagnostic",
        "retune": False,
        "T_max_s": T_MAX_S,
        "rho_definition": "rho = |v_ref - theta| / representation_quantum; quantum = 1000/fs ms (lag) or fs/N Hz (frequency)",
        "bins": [b[0] for b in BINS],
        "lag": lag_err,
        "lag_disagree_n": {"50": sum(1 for r in lag_rows if int(r["fs"]) == 50 and not r["verdict_match"]),
                           "20": len(dis20)},
        "lag_disagree_median_rho_20": float(np.median([r["rho_tau"] for r in dis20])) if dis20 else None,
        "lag_agree_median_rho_20": float(np.median([r["rho_tau"] for r in all20 if r["verdict_match"]])) if all20 else None,
        "frac_20hz_disagreements_with_rho_lt_1": frac_dis_lowrho,
        "lag_20_rho_lt_1": low,
        "lag_20_rho_ge_1": high,
        "peak_gap_median": peak_med,
        "disagree_median_g_20": float(np.median([r.get("g") or math.nan for r in dis20])) if dis20 else None,
        "agree_median_g_20": float(np.median([r.get("g") or math.nan for r in all20 if r["verdict_match"]])) if all20 else None,
        "frequency": freq_sum,
        "lowband": lb_sum,
        "resolvability_explains_mixed": explains,
        "any_result_removed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resolvability_diagnostic.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (OUT / "resolvability_lag_rows.json").write_text(json.dumps(lag_rows, indent=2), encoding="utf-8")

    md = [
        "# Representation resolvability diagnostic (post-hoc)",
        "",
        "This analysis is **not** a pre-registered confirmatory discovery.",
        "It uses only the frozen physical-invariance outputs. No threshold, rate, or T_max was changed.",
        "",
        "Definition (diagnostic, not a theorem):",
        "",
        r"$\rho = |v_{\mathrm{ref}}-\theta|/\Delta$, where $\Delta=1000/f_s$ ms for lag and $\Delta=f_s/N$ Hz for dominant frequency.",
        "",
        "Bins fixed a priori: $\\rho<0.5$, $[0.5,1)$, $[1,2)$, $\\rho\\ge 2$.",
        "",
        f"- T_max remains {T_MAX_S} s",
        f"- Resolvability explains mixed 20 Hz lag? **{explains}**",
        f"- Any unfavorable result removed? **NO**",
        "",
        "## Lag",
        "",
        json.dumps(lag_err, indent=2),
        "",
        f"- 20 Hz disagreements in $\\rho<1$: {frac_dis_lowrho}",
        f"- 20 Hz verdict $\\rho<1$: {low}",
        f"- 20 Hz verdict $\\rho\\ge 1$: {high}",
        "",
        "## Dominant frequency",
        "",
        json.dumps(freq_sum, indent=2),
        "",
        "## Low-band signed boundary",
        "",
        json.dumps(lb_sum, indent=2),
        "",
        "Interpretation note: $e_{\\mathrm{margin}}=|v-v_{\\mathrm{ref}}|/|v_{\\mathrm{ref}}-\\theta|$ is unsigned.",
        "Verdict stability depends on $\\mathrm{sign}(v-\\theta)$, not on whether absolute drift exceeds the scalar margin.",
        "",
        "## Peak-gap (same 32 windows; not a new metric learned from errors)",
        "",
        json.dumps(peak_med, indent=2),
    ]
    (REP / "RESOLVABILITY_DIAGNOSTIC.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "explains": explains,
        "lag": lag_err,
        "freq": {k: {"verdict": v["verdict"], "bins": v["by_rho_bin"], "disagree": v["disagree_n"]} for k, v in freq_sum.items()},
        "lowband": lb_sum,
        "frac_dis_lowrho": frac_dis_lowrho,
    }, indent=2))


if __name__ == "__main__":
    main()

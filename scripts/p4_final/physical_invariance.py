"""Physical-unit / resampling invariance. Protocol frozen before this run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from f_round6_operators import compute as prod_compute
from f_round6_operators import cross_channel_lag_ms as prod_lag
from p2.config import ROOT
from p2.independent_dsp import MeasurementError, measure, tolerance_for
from p2r.contracts import license
from p35.windows_ir import load_unused_windows

OUT = ROOT / "results" / "p4_final"
REP = ROOT / "reports" / "ICASSP_FINAL_10OF10"
SEED = 20260826
MAX_PAMAP = 32
OPS_UNARY = ("dominant_frequency", "spectral_energy_ratio_low")
RATES = (100.0, 50.0, 20.0)
Q = {50.0: 2, 20.0: 5}


def _boot_median_ci(x: list[float], rng: np.random.Generator, n=2000) -> tuple[float, float, float]:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    meds = []
    for _ in range(n):
        samp = rng.choice(arr, size=arr.size, replace=True)
        meds.append(float(np.median(samp)))
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return float(np.median(arr)), float(lo), float(hi)


def _boot_rate_ci(hits: list[int], rng: np.random.Generator, n=2000) -> tuple[float, float, float]:
    arr = np.asarray(hits, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rates = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(n)]
    lo, hi = np.quantile(rates, [0.025, 0.975])
    return float(np.mean(arr)), float(lo), float(hi)


def _decimate_win(ch: dict, q: int) -> dict:
    return {k: decimate(np.asarray(v, dtype=float), q, n=8, ftype="iir", zero_phase=True) for k, v in ch.items()}


def _gt_verdict(v: float, thr: float) -> str:
    return "SUPPORTED" if v > thr else "CONTRADICTED"


def main() -> None:
    rng = np.random.default_rng(SEED)
    wins = [w for w in load_unused_windows() if w.get("dataset") == "PAMAP2"]
    wins = sorted(wins, key=lambda w: w["window_id"])[:MAX_PAMAP]
    rows = []
    for w in wins:
        ch100 = {k: np.asarray(v, dtype=float) for k, v in w["channels"].items()}
        names = list(ch100)
        series = {
            100.0: ch100,
            50.0: _decimate_win(ch100, 2),
            20.0: _decimate_win(ch100, 5),
        }
        for op in OPS_UNARY:
            a = names[0]
            try:
                vref = float(measure(op, {a: series[100.0][a]}, 100.0))
            except MeasurementError:
                continue
            thr = vref - tolerance_for(op, vref)
            vref_ver = _gt_verdict(vref, thr)
            for fs in (50.0, 20.0):
                cmap = {a: series[fs][a]}
                licensed = license(op, cmap, fs)
                rec = {
                    "window_id": w["window_id"],
                    "op": op,
                    "fs": fs,
                    "licensed": licensed,
                    "v_ref": vref,
                    "threshold": thr,
                }
                if not licensed:
                    rec.update({"v": None, "e_abs": None, "e_margin": None, "verdict_match": None})
                    rows.append(rec)
                    continue
                v = float(prod_compute(op, cmap, fs))
                e_abs = abs(v - vref)
                e_m = e_abs / max(abs(vref - thr), 1e-6)
                rec.update({
                    "v": v,
                    "e_abs": e_abs,
                    "e_margin": e_m,
                    "verdict_match": _gt_verdict(v, thr) == vref_ver,
                })
                rows.append(rec)
        if len(names) >= 2:
            a, b = names[0], names[1]
            try:
                vref = float(measure("cross_channel_lag_ms", {a: series[100.0][a], b: series[100.0][b]}, 100.0))
            except MeasurementError:
                vref = None
            if vref is not None:
                thr = vref - tolerance_for("cross_channel_lag_ms", vref)
                vref_ver = _gt_verdict(vref, thr)
                for fs in (50.0, 20.0):
                    cmap = {a: series[fs][a], b: series[fs][b]}
                    for mode, fn in (
                        ("physical", lambda c, f: float(prod_lag(c[a], c[b], f))),
                        ("sample30", lambda c, f: float(prod_lag(c[a], c[b], f, max_lag=30))),
                    ):
                        if mode == "sample30" and cmap[a].size <= 61:
                            licensed = False
                        else:
                            licensed = license("cross_channel_lag_ms", cmap, fs) if mode == "physical" else cmap[a].size > 61
                        rec = {
                            "window_id": w["window_id"],
                            "op": f"lag_{mode}",
                            "fs": fs,
                            "licensed": licensed,
                            "v_ref": vref,
                            "threshold": thr,
                        }
                        if not licensed:
                            rec.update({"v": None, "e_abs": None, "e_margin": None, "verdict_match": None})
                            rows.append(rec)
                            continue
                        v = fn(cmap, fs)
                        e_abs = abs(v - vref)
                        rec.update({
                            "v": v,
                            "e_abs": e_abs,
                            "e_margin": e_abs / max(abs(vref - thr), 1e-6),
                            "verdict_match": _gt_verdict(v, thr) == vref_ver,
                        })
                        rows.append(rec)

    summary = {"n_windows": len(wins), "n_rows": len(rows), "cells": {}}
    for op in ("dominant_frequency", "spectral_energy_ratio_low", "lag_physical", "lag_sample30"):
        for fs in (50.0, 20.0):
            sub = [r for r in rows if r["op"] == op and r["fs"] == fs and r["licensed"] and r["e_abs"] is not None]
            abs_list = [float(r["e_abs"]) for r in sub]
            mar_list = [float(r["e_margin"]) for r in sub]
            hits = [1 if r["verdict_match"] else 0 for r in sub]
            med, lo, hi = _boot_median_ci(abs_list, rng)
            mmed, mlo, mhi = _boot_median_ci(mar_list, rng)
            rate, rlo, rhi = _boot_rate_ci(hits, rng)
            summary["cells"][f"{op}@{int(fs)}"] = {
                "n_licensed": len(sub),
                "median_abs": [med, lo, hi],
                "median_e_margin": [mmed, mlo, mhi],
                "verdict_agree": [rate, rlo, rhi],
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physical_invariance_run.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "physical_invariance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Physical invariance results", "", f"Windows: {len(wins)} PAMAP2 100 Hz unused.", ""]
    md.append("| Cell | n | median |e| | median e_margin | verdict agree |")
    md.append("|---|---:|---|---|---|")
    for k, c in summary["cells"].items():
        md.append(
            f"| {k} | {c['n_licensed']} | "
            f"{c['median_abs'][0]:.4g} [{c['median_abs'][1]:.4g},{c['median_abs'][2]:.4g}] | "
            f"{c['median_e_margin'][0]:.3g} | "
            f"{100*c['verdict_agree'][0]:.1f}% [{100*c['verdict_agree'][1]:.1f},{100*c['verdict_agree'][2]:.1f}] |"
        )
    (REP / "PHYSICAL_INVARIANCE_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

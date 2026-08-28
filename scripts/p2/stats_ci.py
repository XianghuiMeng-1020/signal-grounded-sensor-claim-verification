"""Uncertainty helpers. Wilson intervals; McNemar for paired binary outcomes."""
from __future__ import annotations

import math
from typing import Iterable

from scipy.stats import binomtest, chi2


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> dict:
    if n <= 0:
        return {"n": 0, "k": 0, "p": None, "lo": None, "hi": None, "method": "wilson"}
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z_from_alpha(alpha)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return {
        "n": n,
        "k": k,
        "p": phat,
        "lo": max(0.0, centre - half),
        "hi": min(1.0, centre + half),
        "method": "wilson",
        "alpha": alpha,
    }


def _z_from_alpha(alpha: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(1.0 - alpha / 2.0))


def proportion(k: int, n: int) -> dict:
    out = wilson_interval(k, n)
    out["exact_pvalue_vs_half"] = None if n == 0 else float(binomtest(k, n, 0.5).pvalue)
    return out


def mcnemar_paired(a_correct: Iterable[bool], b_correct: Iterable[bool]) -> dict:
    """McNemar exact/chi-square on paired correctness. Exploratory unless pre-registered."""
    a = list(a_correct)
    b = list(b_correct)
    if len(a) != len(b):
        raise ValueError("paired series length mismatch")
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "pvalue": 1.0, "note": "no_discordant"}
    # Exact binomial McNemar
    p_exact = float(binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
    p_chi = float(1.0 - chi2.cdf(stat, 1))
    return {
        "n01": n01,
        "n10": n10,
        "statistic": stat,
        "pvalue_exact": p_exact,
        "pvalue_chi2_cc": p_chi,
        "note": "exploratory_unless_pre_registered",
    }


def fmt_ci(d: dict) -> str:
    if not d or d.get("p") is None:
        return "NA"
    return f"{100 * d['p']:.1f}% [{100 * d['lo']:.1f}%, {100 * d['hi']:.1f}%] (n={d['n']})"

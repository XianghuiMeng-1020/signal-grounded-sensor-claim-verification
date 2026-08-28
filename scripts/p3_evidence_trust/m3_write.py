"""Write Module 3 numerical-identity report."""
from __future__ import annotations

from p2.config import VALIDATION_TOL

from .config import EXPERIMENT_IDS, OPS, REPORTS, SEED


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def write(payload: dict) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "03_NUMERICAL_IDENTITY.md"
    s = payload["summary"]
    rows = [
        "| Kernel | n | max |P-R| | max rel | mean |P-R| | within VALIDATION_TOL | worst window |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for op in OPS:
        c = s["by_op"][op]
        if not c.get("n"):
            rows.append(f"| `{op}` | 0 | — | — | — | n/a | — |")
            continue
        w = c["worst"]
        rows.append(
            f"| `{op}` | {c['n']} | {c['max_abs_error']:.6g} | {c['max_rel_error']:.6g} | "
            f"{c['mean_abs_error']:.6g} | {_pct(c['frac_within_tol'])} "
            f"({c['n_within_tol']}/{c['n']}) | `{w['window_id']}` |"
        )
    tols = [
        "| Kernel | Frozen VALIDATION_TOL |",
        "|---|---|",
    ]
    for op in OPS:
        tols.append(f"| `{op}` | `{VALIDATION_TOL[op]}` |")

    body = f"""# Phase 3 Module 3 — Numerical identity of signal measurement operators

Experiment id: `{EXPERIMENT_IDS["m3"]}`  
Seed: `{SEED}`  
Run once. Evaluation only. Definitions not changed. Welch remains closed.

## Reviewer objection

Are the DSP measurements themselves trustworthy, or are the eight names arbitrary labels?

## Protocol

Paired identical inputs on unused later-offset windows (same pool as Phase 2 Exp 1).  
Production: `f_round6_operators.compute`.  
Reference: `p2.independent_dsp.measure` (same frozen equations; SciPy path).  
No alternative estimator. No threshold edit. No production kernel edit.

n paired legal measurements = {s["n_paired"]} (skipped non-measurable / degenerate = {s["n_skipped"]}).

This is **numerical consistency** between production measurement operators and independent analytical reference implementations. It is not a unit-test score and not a definition contest.

## Frozen tolerances (not fitted on this arm)

{chr(10).join(tols)}

## Results

Decision: **{s["decision"]}**

{chr(10).join(rows)}

Gate C analytic restatement (already frozen, not a new discovery): 25/25 cases, 8/8 primitives, see `reports/P2_SCIENTIFIC_ESCALATION/02_PRIMITIVE_REFERENCE_VALIDATION.md`. This arm does not change that claim.

## Reading

A PASS means that on unused IMU windows the two realizations of each frozen equation agree inside pre-registered numerical tolerances. The ontology names are measurements.

A STOP means at least one paired error exceeded `VALIDATION_TOL`. Phase 3 does not continue. Production kernels are not edited in this phase.

## What was not done

No Welch/periodogram pass/fail. No manuscript edit. No V3 rescore. No tolerance retune.
"""
    path.write_text(body, encoding="utf-8")
    return str(path)

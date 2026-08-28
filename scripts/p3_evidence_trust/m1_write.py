"""Write Module 1 contract-necessity report."""
from __future__ import annotations

from .config import ABLATIONS, EXPERIMENT_IDS, REPORTS, SEED


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def write(payload: dict) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "01_CONTRACT_NECESSITY.md"
    s = payload["summary"]
    rows = [
        "| Ablation | n invalid | prod FCR | weak FCR | S inflation | C inflation | U retention | leftover | kernel exc. | control match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ab in ABLATIONS:
        c = s["by_ablation"][ab]
        rows.append(
            f"| `{ab}` | {c['n_invalid']} | {_pct(c['production_fcr'])} "
            f"({c['n_production_commit_on_invalid']}/{c['n_invalid'] or 0}) | "
            f"{_pct(c['weak_fcr'])} ({c['n_weak_commit']}/{c['n_invalid'] or 0}) | "
            f"{_pct(c['supported_inflation'])} ({c['n_supported_inflation']}) | "
            f"{_pct(c['contradicted_inflation'])} ({c['n_contradicted_inflation']}) | "
            f"{_pct(c['unknown_retention'])} ({c['n_unknown_retention']}) | "
            f"{c['n_leftover_clause']} | {c['n_kernel_exception']} | "
            f"{_pct(c['control_match_rate'])} |"
        )

    body = f"""# Phase 3 Module 1 — Evidence-contract necessity

Experiment id: `{EXPERIMENT_IDS["m1"]}`  
Seed: `{SEED}`  
Run once. Evaluation-only shadows. Production contracts not edited.

## Reviewer objection

Are evidence contracts arbitrary refuse-if rules, or are they necessary physical requirements?

## Protocol

Carrier programs: Phase 2 Exp 1 construction (768 unused-window oracle SINGLE `gt` programs).  
For each applicable primitive × one-clause ablation, build a **constructed invalid probe** and score:

- **Full:** production `run_oracle` (frozen contracts + kernels + Kleene)
- **Weak:** the same kernel and Kleene, with exactly one validity clause omitted

Ablations: nonfinite, min length, sampling frequency, second channel, equal length, variance, output domain.

The goal is not a higher score. The goal is to measure **unsafe commitment** introduced by a weakened requirement.

n carriers = {s["n_carriers"]}. n scores = {s["n_scores"]}. n invalid-by-construction = {s["n_invalid"]}.

## Critical gate

Production FCR on invalid probes = {_pct(s["production_fcr_invalid"])} ({s["n_production_commit_on_invalid"]}/{s["n_invalid"]}).

Decision: **{s["decision"]}**

Legal-record control (shadow on unperturbed carriers): fail count = {s["control_fail"]} / {s["control_n"]}.

## Results

{chr(10).join(rows)}

## Reading

Full contracts must send invalid evidence to UNVERIFIABLE (FCR 0). That gate is the same fail-closed fact as Phase 2 dropout / EC-BLIND, restated at clause-matched probes.

A clause is **necessary** when removing it converts those UNVERIFIABLE rows into SUPPORTED or CONTRADICTED (commitment inflation). That is unsafe behavior of the *shadow*, not an improved verifier.

UNKNOWN retention after a single-clause removal means another leftover clause or a kernel exception still refused. That clause is then **not separately identifiable** on this probe. It is not a reason to delete the clause.

`drop_output_domain` has no invalid-by-construction probe: production kernels of the frozen equations do not emit out-of-domain values when the pre-gate passes. Necessity of the post-domain check is therefore not identifiable here.

## What was not done

No contract redesign. No manuscript edit. No V3 rescore. No search over which clauses to drop.
"""
    path.write_text(body, encoding="utf-8")
    return str(path)

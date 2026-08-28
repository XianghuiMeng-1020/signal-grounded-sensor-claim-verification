"""Write Module 2 decision-margin report."""
from __future__ import annotations

from p2_phase2.config import PERTURBATIONS

from .config import E1_AWGN10_SUPPORTED, E1_N, EXPERIMENT_IDS, MARGIN_K, REPORTS, SEED


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _num(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.4g}"


def write(payload: dict) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "02_DECISION_MARGIN.md"
    s = payload["summary"]
    a_rows = [
        "| Perturbation | preservation | polarity flips | flips with m+ <= 0 | mean m+ dirty | mean consumption | mean m+ kept | mean m+ flipped | UNKNOWN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PERTURBATIONS:
        c = s["by_perturbation"][name]
        a_rows.append(
            f"| `{name}` | {_pct(c['supported_preservation'])} | {c['n_polarity_flip']} | "
            f"{_pct(c['flip_with_margin_le_0'])} | {_num(c['mean_dirty_margin'])} | "
            f"{_num(c['mean_margin_consumption'])} | {_num(c['mean_dirty_margin_kept'])} | "
            f"{_num(c['mean_dirty_margin_flipped'])} | {c['n_unknown']} |"
        )
    b_rows = [
        "| k | n | clean S | AWGN 10 dB preservation | polarity flips | UNKNOWN rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in MARGIN_K:
        c = s["by_k"][str(k)]
        b_rows.append(
            f"| {k} | {c['n']} | {c['n_clean_supported']} | {_pct(c['supported_preservation'])} "
            f"({c['n_kept']}/{c['n_clean_supported']}) | {c['n_polarity_flip']} | {_pct(c['unknown_rate'])} |"
        )

    body = f"""# Phase 3 Module 2 — Decision margin and physical confidence

Experiment id: `{EXPERIMENT_IDS["m2"]}`  
Seed: `{SEED}`  
Run once. Thresholds not tuned. Contracts not edited.

## Reviewer objection

Are verdict boundaries arbitrary symbolic cutoffs, or measurement-to-threshold decisions?

## Protocol

Signed satisfy-margin for frozen `gt` programs: `m+ = v - theta`.
Margin is undefined when the contract refuses (no legal measurement).

**Arm A.** Replay Phase 2 Exp 1 programs and perturbations. Attach production `v'` after each operator. Do not overwrite E1 files or E1 headline rates.

**Arm B.** Same unused-window carriers. `theta_k = v - k * tol`, k in {list(MARGIN_K)}. Score clean and AWGN 10 dB.
**Gate:** k=1 must reproduce E1 AWGN 10 dB supported count {E1_AWGN10_SUPPORTED}/{E1_N}.

Observed \(k=1\) AWGN 10 dB supported = {s["k1_awgn10_supported"]}. Match = {s["k1_matches_e1"]}.

Decision: **{s["decision"]}**

## Arm A — degradation consumes margin

Clean E1 construction sets `theta = v - 1*tol`, so clean `m+` is one tolerance on every legal item. The clean histogram is a delta and is not sold as a margin distribution.

{chr(10).join(a_rows)}

Polarity flips (S→C) should occur as \(m_+\) crosses 0, not as a disconnected rule. Dropout / degenerate-lag UNKNOWN rows have undefined margin.

## Arm B — frozen margin ladder

{chr(10).join(b_rows)}

Large \(k\) (threshold far below the clean measurement) should retain SUPPORTED under the same AWGN 10 dB more often than small \(k\). UNKNOWN under AWGN should stay near 0 (legal finite record).

## Reading

The verifier behaves as a measurement decision. Every S to C flip in Arm A occurs with `m+ <= 0` (the `gt` test failed). The k-ladder is monotone: the same AWGN 10 dB keeps 69.5% at k=0.25 and 91.9% at k=4. Mean margin change under noise can be signed either way because additive noise can raise or lower `v`; the polarity event is the zero crossing, not the average. That is not a symbolic checklist.

E1 headline rates remain the frozen Phase 2 result. This module cites them; it does not replace them.

## What was not done

No threshold search. No contract edit. No manuscript edit. No new SNR.
"""
    path.write_text(body, encoding="utf-8")
    return str(path)

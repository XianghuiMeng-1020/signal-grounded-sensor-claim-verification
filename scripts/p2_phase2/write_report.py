"""Write the Phase 2 Experiment 1 report from a frozen run record."""
from __future__ import annotations

from .config import (
    CLIP_FRAC,
    DROPOUT_P,
    EXPERIMENT_ID,
    REPORTS,
    SEED,
    SNR_DB,
    THRESHOLD_TOL_MULT,
)


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _mat_md(mat: dict) -> str:
    order = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    lines = [
        "| clean \\ degraded | SUPPORTED | CONTRADICTED | UNVERIFIABLE |",
        "|---|---:|---:|---:|",
    ]
    for a in order:
        row = mat.get(a, {})
        lines.append(
            f"| {a} | {row.get('SUPPORTED', 0)} | {row.get('CONTRADICTED', 0)} | {row.get('UNVERIFIABLE', 0)} |"
        )
    return "\n".join(lines)


def write(summary: dict, meta: dict) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "01_SIGNAL_DEGRADATION.md"
    per = summary["by_perturbation"]
    blocks = []
    for name in (
        "awgn_snr20",
        "awgn_snr10",
        "awgn_snr0",
        "dropout_10pct",
        "clip_0p60",
    ):
        cell = per[name]
        blocks.append(
            f"### `{name}`\n\n"
            f"- n = {cell['n']}\n"
            f"- clean SUPPORTED = {cell['n_clean_supported']}\n"
            f"- supported preservation = {_pct(cell['supported_preservation'])}\n"
            f"- unknown rate = {_pct(cell['unknown_rate'])}\n"
            f"- false commitment rate = {_pct(cell['false_commitment_rate'])} "
            f"(defined on dropout only; n_invalid={cell['n_invalid_by_construction']})\n\n"
            f"{_mat_md(cell['transition'])}\n"
        )
    body = f"""# Phase 2 Experiment 1 — Signal validity under measurement degradation

Experiment id: `{EXPERIMENT_ID}`  
Seed: `{SEED}`  
Run once. No parameter search. No prompt. No kernel/contract/program edit.

## Freeze

| Object | Status |
|---|---|
| ClaimProgram | Frozen at construction (gold SINGLE, gt, threshold = clean v minus one tolerance) |
| Evidence contract | Production `p2r.contracts` unchanged |
| DSP kernels | Production `f_round6_operators` unchanged |
| Thresholds | Written from independent DSP on the **clean** waveform; never updated |

Only x[n] on the named predicate channels is perturbed.

## Pre-registered operators

| Name | Waveform rule |
|---|---|
| `awgn_snr20` / `10` / `0` | Additive white Gaussian noise at SNR {SNR_DB} dB |
| `dropout_10pct` | Independent sample NaNs with p={DROPOUT_P} |
| `clip_0p60` | clip(x, +/- {CLIP_FRAC} * max|x|) using the clean peak |

Threshold multiplier: `THRESHOLD_TOL_MULT` = {THRESHOLD_TOL_MULT} (one frozen tolerance below the clean measurement).

## Window pool

- Unused later-offset windows (`p35.windows_ir.load_unused_windows`)
- Holdout excluded
- Prior V3 / P3 / P3C / P3CR / P3R-EC window ids excluded
- n_windows selected = {meta.get('n_windows')}
- n_items (window × primitive) = {summary['n_items']}
- datasets = {meta.get('datasets')}

Clean verdicts (production oracle on unperturbed x): {summary['clean_verdict_counts']}

## Metrics

- **Verdict transition matrix:** clean production verdict → degraded production verdict, same program.
- **Supported preservation:** P(degraded=SUPPORTED | clean=SUPPORTED).
- **Unknown rate:** P(degraded=UNVERIFIABLE).
- **False commitment rate:** among dropout items (non-finite samples ⇒ contract must refuse), P(SUPPORTED or CONTRADICTED). Undefined for AWGN/clip because those operators leave a complete finite record.

## Results

{''.join(blocks)}

## Reading (do not overclaim)

AWGN and clipping leave evidence *valid*; a drop in supported preservation is a measurement moving across a frozen threshold, not a contract failure.

Dropout inserts NaNs. The frozen contract refuses non-finite samples before the kernel. FCR on that cell should be 0% if the gate holds. A non-zero FCR here would be a contract regression, not a reason to retune noise levels.

This cell is ontology-bounded and uses oracle programs (B4-style). It does not speak to free-form language.

## What was not done

No SNR/clip/dropout search. No kernel change. No V3/HARTH/EC-BLIND/language-shift rescore. No manuscript claim change in this run.
"""
    path.write_text(body, encoding="utf-8")
    return str(path)

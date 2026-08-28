"""Write the Phase 2 Experiment 2 report from a frozen run record."""
from __future__ import annotations

from .lag_config import (
    DELAYS_MS,
    EXPERIMENT_ID,
    FS_GRID,
    PHYSICAL_BOUND_MS,
    REPORTS,
    SAMPLE_BOUND,
    SEED,
    sample_bound_ms,
)


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _cell(rates: dict) -> str:
    return (
        f"S {rates['SUPPORTED']}/C {rates['CONTRADICTED']}/U {rates['UNVERIFIABLE']} "
        f"(sup {_pct(rates['support_rate'])}, unk {_pct(rates['unknown_rate'])})"
    )


def write(summary: dict, meta: dict) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "02_LAG_TIMEBASE.md"
    by = summary["by_delay_fs"]

    rows = [
        "| delay_ms | fs | n | n_ok | in-box | faithful | Mode A (\\pm30 samples) | Mode B (\\pm300 ms) |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for d in DELAYS_MS:
        for fs in FS_GRID:
            cell = by[str(d)][str(int(fs))]
            rows.append(
                f"| {d} | {int(fs)} | {cell['n']} | {cell['n_length_ok']} | "
                f"{cell['n_in_box']} | {cell['n_faithful']} | "
                f"{_cell(cell['mode_a'])} | {_cell(cell['mode_b'])} |"
            )

    cons_all = summary["consistency_all_rates"]
    cons_ok = summary["consistency_valid_length"]
    cons_box = summary["consistency_in_box_all_rates"]
    agree = summary["cross_mode_agree_measurable"]
    sup_in = summary["support_if_physically_inside"]
    con_out = summary["contradict_if_physically_outside_and_measurable"]
    fcr = summary["false_commitment"]

    body = f"""# Phase 2 Experiment 2 — Lag time-base validity

Experiment id: `{EXPERIMENT_ID}`  
Seed: `{SEED}`  
Run once. No parameter search. No prompt. No kernel/contract edit.

## Question

Does the verifier depend on **physical time alignment** or on **arbitrary sample offsets**?

The frozen estimator searches \\pm{SAMPLE_BOUND} **samples** and reports milliseconds.
That search box is not a physical window:

| fs | \\pm{SAMPLE_BOUND} samples in ms |
|---:|---:|
| 20 Hz | {sample_bound_ms(20.0):.0f} |
| 50 Hz | {sample_bound_ms(50.0):.0f} |
| 100 Hz | {sample_bound_ms(100.0):.0f} |

## Freeze

| Object | Status |
|---|---|
| DSP kernel | Production `cross_channel_lag_ms` unchanged (search \\pm{SAMPLE_BOUND} samples) |
| Evidence contract | Production `p2r.contracts` unchanged (n\\ge61, fs required, refuse degenerate) |
| Kleene | Unchanged |
| Evaluation mode | Added on `run_oracle` only. Default `production` does not convert thresholds |

## Pre-registered modes

| Mode | Eval flag | Threshold written | Interpretation |
|---|---|---|---|
| A naive sample-domain | `lag_sample_threshold` | \\pm{SAMPLE_BOUND} (samples) | Convert to ms at the item fs: \\pm30/fs\\cdot1000 |
| B physical time | `lag_physical_threshold` | \\pm{PHYSICAL_BOUND_MS:.0f} ms | Same millisecond bound at every fs |

{PHYSICAL_BOUND_MS:.0f} ms is 30 samples at 100 Hz, the highest native rate. It was not taken from a scored outcome.

Claim shape (both modes): AND of `gt -bound` and `lt +bound` on `cross_channel_lag_ms`.

## Stimuli

Unused later-offset windows (`p35.windows_ir.load_unused_windows`). Holdout and prior scored window ids excluded.

Each selected window's first channel is resampled (linear interpolation, evaluation construction only) onto `FS_GRID` = {list(FS_GRID)}, keeping physical duration. Channel B is a circular shift of that series by the pre-registered physical delay.

Pre-registered delays (integer samples at 20/50/100 Hz): {list(DELAYS_MS)} ms.

- n_windows = {meta.get("n_windows")}
- n_items (window \\times fs \\times delay) = {summary["n_items"]}
- datasets = {meta.get("datasets")}
- n valid length (n\\ge61) = {summary["n_valid_length"]}
- n short after resample = {summary["n_short"]}
- n measurable (valid length and true lag inside \\pm30 samples) = {summary["n_measurable"]}
- n unmeasurable (valid length, true lag outside search) = {summary["n_unmeasurable"]}

## Metrics

- **Verdict consistency:** same window + same physical delay, verdicts at 20/50/100 Hz agree.
- **Support preservation:** P(SUPPORTED | valid length, in-box, |true lag| < 300 ms).
- **Contradiction on physical excess:** P(CONTRADICTED | valid length, in-box, |true lag| \\ge 300 ms).
- **Unknown rate:** P(UNVERIFIABLE).
- **Measurement faithfulness:** |measured \\- true| \\le 0.51 samples in ms (diagnostic; frozen P2 lag gold tol).
- **Invalid measurement / FCR:** P(SUPPORTED or CONTRADICTED | invalid). Invalid = short record (n<61) or true lag outside the \\pm30-sample search. The kernel always returns an in-box peak when the contract passes, so out-of-box true lags are unmeasurable.

## Results

### Delay \\times rate table

{chr(10).join(rows)}

### Headline quantities

| Quantity | Mode A (\\pm30 samples) | Mode B (\\pm300 ms) |
|---|---|---|
| All items support / unknown | {_pct(summary["mode_a_all"]["support_rate"])} / {_pct(summary["mode_a_all"]["unknown_rate"])} | {_pct(summary["mode_b_all"]["support_rate"])} / {_pct(summary["mode_b_all"]["unknown_rate"])} |
| Measurable support / contradict / unknown | {_pct(summary["mode_a_measurable"]["support_rate"])} / {_pct(summary["mode_a_measurable"]["contradict_rate"])} / {_pct(summary["mode_a_measurable"]["unknown_rate"])} | {_pct(summary["mode_b_measurable"]["support_rate"])} / {_pct(summary["mode_b_measurable"]["contradict_rate"])} / {_pct(summary["mode_b_measurable"]["unknown_rate"])} |
| Cross-rate consistency (all groups) | {_pct(cons_all["mode_a"]["rate"])} ({cons_all["mode_a"]["n_consistent"]}/{cons_all["mode_a"]["n_groups"]}) | {_pct(cons_all["mode_b"]["rate"])} ({cons_all["mode_b"]["n_consistent"]}/{cons_all["mode_b"]["n_groups"]}) |
| Cross-rate consistency (valid length at all three rates) | {_pct(cons_ok["mode_a"]["rate"])} ({cons_ok["mode_a"]["n_consistent"]}/{cons_ok["mode_a"]["n_groups"]}) | {_pct(cons_ok["mode_b"]["rate"])} ({cons_ok["mode_b"]["n_consistent"]}/{cons_ok["mode_b"]["n_groups"]}) |
| Cross-rate consistency (in-box at all three rates) | {_pct(cons_box["mode_a"]["rate"])} ({cons_box["mode_a"]["n_consistent"]}/{cons_box["mode_a"]["n_groups"]}) | {_pct(cons_box["mode_b"]["rate"])} ({cons_box["mode_b"]["n_consistent"]}/{cons_box["mode_b"]["n_groups"]}) |
| Support if physically inside 300 ms (measurable) | {_pct(sup_in["mode_a"]["rate"])} ({sup_in["mode_a"]["n_match"]}/{sup_in["mode_a"]["n"]}) | {_pct(sup_in["mode_b"]["rate"])} ({sup_in["mode_b"]["n_match"]}/{sup_in["mode_b"]["n"]}) |
| Contradict if physically \\ge300 ms and measurable | {_pct(con_out["mode_a"]["rate"])} ({con_out["mode_a"]["n_match"]}/{con_out["mode_a"]["n"]}) | {_pct(con_out["mode_b"]["rate"])} ({con_out["mode_b"]["n_match"]}/{con_out["mode_b"]["n"]}) |
| Cross-mode agreement on measurable items | {_pct(agree["rate"])} ({agree["n_agree"]}/{agree["n"]}) | same |
| FCR on short records (n<61) | {_pct(fcr["mode_a_short"]["false_commitment_rate"])} ({fcr["mode_a_short"]["n_commit"]}/{fcr["mode_a_short"]["n_invalid"]}) | {_pct(fcr["mode_b_short"]["false_commitment_rate"])} ({fcr["mode_b_short"]["n_commit"]}/{fcr["mode_b_short"]["n_invalid"]}) |
| FCR on true lag outside search (valid length) | {_pct(fcr["mode_a_outside_search"]["false_commitment_rate"])} ({fcr["mode_a_outside_search"]["n_commit"]}/{fcr["mode_a_outside_search"]["n_invalid"]}) | {_pct(fcr["mode_b_outside_search"]["false_commitment_rate"])} ({fcr["mode_b_outside_search"]["n_commit"]}/{fcr["mode_b_outside_search"]["n_invalid"]}) |

Faithful-measurement rate on measurable items: {_pct(summary["faithful_rate_measurable"])}.

## Reading (do not overclaim)

Three facts, in order.

**1. On measurable delays physically inside 300 ms, both modes SUPPORT.**  
0 / 100 / 200 ms are inside every sample box and inside the physical bound. Support preservation is 768/768 for A and for B. Magnitude recovery is high once sign is ignored: production `correlate(a, roll(a,+k))` reports \\(-k\\) samples.

**2. On measurable delays physically at or beyond 300 ms, only Mode B follows time.**  
Contradiction-on-physical-excess is 35.5% for A vs 99.3% for B. The split is concentrated where the sample box is *wider* than 300 ms:

- 300 ms at 20 Hz: A SUPPORT 64/64 valid, B CONTRADICT 64/64 (300 is not strictly inside \\pm300).
- 300 ms at 50 Hz: A SUPPORT 96/96, B CONTRADICT 94/96.
- 400 ms at 20 Hz: A SUPPORT 64/64, B CONTRADICT 64/64.
- 400 ms at 50 Hz: A SUPPORT 96/96, B CONTRADICT 96/96.
- 600 ms at 20 Hz: A SUPPORT 64/64, B CONTRADICT 64/64.

The same physical delay is therefore **SUPPORTED under a sample-domain bound and CONTRADICTED under a millisecond bound**. That is the time-base result.

At 100 Hz the two modes coincide (both bounds are 300 ms). A 300 ms delay is the strict edge: almost all CONTRADICT. A 400 ms delay is outside the sample search: both modes see a substituted in-box peak and mostly SUPPORT (78/96). That is not A-vs-B disagreement; it is an unmeasurable true lag.

**3. Invalid measurement splits cleanly.**  
- Short records (PAMAP2-length series resampled to 20 Hz, n=51<61): UNVERIFIABLE, FCR 0/288. The length contract holds.  
- True lag outside \\pm30 samples, valid length: FCR 832/832. The frozen kernel always emits an in-box peak; the contract checks the reported domain, not the unobserved true delay. A physical threshold cannot refuse what the estimator never reports.

Cross-rate consistency is the wrong headline if taken alone. Mode A looks more consistent on the full ladder because it SUPPORTS any in-box peak, including false peaks at high fs. Mode B is *less* consistent across rates precisely when the same physical delay is measurable at 20/50 Hz (correct CONTRADICT) and unmeasurable at 100 Hz (false SUPPORT). Restricting to delays that are in-box at all three rates isolates the unit mismatch (300 ms at low vs high fs) from the search-box hole.

The verifier's **estimator** depends on sample offsets (\\pm30 samples). A physical-time **threshold** (Mode B) makes Kleene follow milliseconds when the peak is inside that box. It does not make out-of-box delays UNVERIFIABLE. Physical-time validity of temporal contracts therefore requires both a millisecond threshold and a search radius that is itself a duration — the second change is not in this experiment.

This cell is oracle programs on constructed delays (B4-style). It does not speak to compilation or free-form language.

## What was not done

No kernel change. No contract cutoff change. No V3/HARTH/EC-BLIND/language-shift rescore. No manuscript edit.
"""
    path.write_text(body, encoding="utf-8")
    return str(path)

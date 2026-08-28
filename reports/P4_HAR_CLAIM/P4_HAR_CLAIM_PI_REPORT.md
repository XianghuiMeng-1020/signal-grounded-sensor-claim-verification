# PHASE 2 PI REPORT — HAR → Claim Baseline

**Status.** Complete. Run once. No manuscript edit. No frozen-number change.

| Item | Value |
|---|---|
| Experiment | `p4_har_claim_v1` |
| Seed | `20260825` |
| Dictionary | `p4_har_claim_dictionary_v1` / `f72a5b1419ff7b5badf0ede5d264750d5ca75ba863972c5ebb23e73cc1d2d906` |
| Entry point | `python scripts/run_p4_har_claim.py` |
| Tests | `tests/p4_har_claim` — 12 passed |

---

## 1. Protocol

IMU window → Random Forest HAR → predicted activity → frozen family claim
→ HAR verdict (class match, never UNKNOWN).

Same posed claim → contracts + frozen DSP + Kleene (proposed).

Illegal copies of each challenge window: missing named channel, `fs=0`,
10% NaN dropout.

Dictionary written before scoring. No prompt search. No kernel change.

---

## 2. Dataset choice

Minimum clean subset: **PAMAP2 + WISDM + MHEALTH**.

**HARTH excluded** — no existing `p2.windows` loader; adding it would be new
preprocessing.

Pool: unused later-offset windows, existing subject splits, holdout refused,
Phase-2 window ids excluded.

| Dataset | Subjects (all / train / test) | Windows all | Train | Test | Activities in pool | \(f_s\) |
|---|---|---|---|---|---|---|
| WISDM | 34 / 17 / 17 | 751 | 374 | 377 | A walk, B jog | 20 Hz |
| MHEALTH | 7 / 3 / 4 | 109 | 45 | 64 | stand, sit, lie, stairs | 50 Hz |
| PAMAP2 | 3 / 1 / 2 | 12 | 9 | 3 | lying only | 100 Hz |

Features: per named channel mean, std, min, max, RMS, dominant frequency.

The unused-mappable pool is **WISDM-heavy**. That is reported, not hidden.

---

## 3. Baseline implementation

| Field | Frozen choice |
|---|---|
| Architecture | `RandomForestClassifier` (sklearn 1.8) |
| Trees / depth / leaf | 100 / 12 / 2 |
| Seed | 20260825 |
| Train | development unused labeled windows, \(n=428\) |
| Test | challenge unused labeled windows, \(n=444\) |
| Items | 1776 = 444 legal + 1332 illegal |

Activity→claim (manual, scale-free):

- static → `spectral_energy_ratio_low > 0.50`
- walk → `dominant_frequency > 0.8` Hz
- run → `dominant_frequency > 2.0` Hz

---

## 4. Results

| Cell | HAR → claim | Proposed stack |
|---|---|---|
| **FCR** (primary) | **1.0** (1332/1332) | **0.0** (0/1332) |
| UNKNOWN rate (all) | 0.0 | 0.75 (illegal slice) |
| UNKNOWN rate (illegal) | 0.0 | 1.0 |
| Legal verdict acc. | 0.394 | 1.0 (identity: gold := proposed) |
| Supported recall (legal) | 0.887 | 1.0 (identity) |
| Contradicted recall (legal) | 0.008 | 1.0 (identity) |
| HAR activity acc. (challenge) | **0.917** | — |

Legal gold: 195 SUPPORTED / 249 CONTRADICTED. The stereotyped activity claim
is often false on the waveform. HAR still emits SUPPORTED on 420/444 legal
rows.

---

## 5. R3 / R12

**R12 (HAR already does this): CLOSED.**

Recognition works (91.7%) and still cannot refuse invalid evidence (FCR 100%).
That is the required sentence: HAR is not verification.

**R3 (just an LLM wrapper): PARTIALLY CLOSED.**

This cell rules out a recognizer substitute. LLM-specific wrapper evidence
remains B2/B3/B4/B5.

---

Manuscript not updated. V3 / HARTH / Phase-2 numbers untouched.

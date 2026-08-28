"""Write the HAR→claim PI report. Does not edit the manuscript."""
from __future__ import annotations

import json

from .config import EXPERIMENT_ID, REPORTS, RESULTS, SEED
from .dictionary import DICTIONARY_ID, FAMILY_CLAIMS, dictionary_sha256


def write(meta: dict, metrics: dict, audit: dict, model: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    har = metrics["har"]
    prop = metrics["proposed"]
    md = f"""# P4 HAR → Claim Baseline

**Experiment.** `{EXPERIMENT_ID}`  
**Seed.** `{SEED}`  
**Dictionary.** `{DICTIONARY_ID}` sha256 `{dictionary_sha256()}`  
**Manuscript.** Not edited. Frozen V3 / HARTH / Phase-2 numbers not rescored.

## Question

Does activity recognition supply evidence-level verification?

Hypothesis: a standard HAR classifier may label locomotion, but it cannot refuse
invalid measurement claims or preserve UNKNOWN.

## Protocol

1. Unused, labeled, non-holdout windows from PAMAP2 / WISDM / MHEALTH.
2. Existing subject-grouped development / challenge splits.
3. Frozen activity→family→claim dictionary written before scoring.
4. Random Forest HAR trained on development unused windows (no search).
5. Posed claim is the dictionary claim of the **gold** activity.
6. HAR verdict = SUPPORTED iff predicted family matches posed family; else CONTRADICTED. Never UNKNOWN.
7. Proposed verdict = contracts + frozen DSP + Strong Kleene on the same claim.
8. Illegal slice: missing channel, `fs=0`, 10% NaN dropout.

HARTH is excluded: no existing `p2.windows` loader; adding it would be new preprocessing.

## Dataset pool

```
{json.dumps(audit, indent=2)}
```

## Model

```
{json.dumps(model, indent=2)}
```

Frozen family claims:

```
{json.dumps(FAMILY_CLAIMS, indent=2)}
```

## Results

Primary — False Commitment Rate (gold UNVERIFIABLE predicted SUPPORTED or CONTRADICTED):

| System | FCR |
|---|---|
| HAR → claim | {har.get("fcr")} |
| Proposed verification | {prop.get("fcr")} |

Legal records (answerable claims):

| System | n | Verdict acc. | Supported recall | Contradicted recall | UNKNOWN rate |
|---|---|---|---|---|---|
| HAR → claim | {har["legal"].get("n")} | {har["legal"].get("verdict_accuracy")} | {har["legal"].get("supported_recall")} | {har["legal"].get("contradicted_recall")} | {har["legal"].get("unknown_rate")} |
| Proposed | {prop["legal"].get("n")} | {prop["legal"].get("verdict_accuracy")} | {prop["legal"].get("supported_recall")} | {prop["legal"].get("contradicted_recall")} | {prop["legal"].get("unknown_rate")} |

Illegal records (constructed missing evidence):

| System | n | UNKNOWN rate | FCR on this slice |
|---|---|---|---|
| HAR → claim | {har["illegal"].get("n")} | {har["illegal"].get("unknown_rate")} | {har.get("fcr")} |
| Proposed | {prop["illegal"].get("n")} | {prop["illegal"].get("unknown_rate")} | {prop.get("fcr")} |

Recognition accuracy on legal challenge windows (diagnostic, not the headline): {meta.get("recognition_accuracy_challenge")}

## Interpretation

Do not read this as “our method beats HAR.”

HAR answers which activity label is likely. The posed object is a measurement
claim. Class match is not a licensed kernel, has no evidence contract, and has
no UNKNOWN. If FCR(HAR) is high while FCR(proposed) is 0 on the same illegal
rows, recognition failed to provide evidence-level verification.

## R3 / R12

R3 (LLM wrapper) and R12 (HAR already does this) are addressed only if the
illegal slice shows HAR committing and the proposed stack refusing.
"""
    (REPORTS / "01_HAR_CLAIM_BASELINE.md").write_text(md, encoding="utf-8")
    (RESULTS / "har_claim_summary.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics, "audit": audit, "model": model}, indent=2),
        encoding="utf-8",
    )

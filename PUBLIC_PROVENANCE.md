# Public release provenance

This public release is derived from the frozen scientific state
`7c4681eb2be52929793772a279a7a0fe4a3c85ee` (tag
`icassp-final-science-locked-before-writing`), certified for submission at
tag `icassp-10of10-final-certified`.
Item-level sealed confirmatory artifacts are intentionally excluded
to preserve holdout integrity.

The public commit SHA is **not** equal to that private research-tree commit.
Private research tags are not reused here.

Public-safe release tag on this history: `icassp2027-public-release-v2`.
The prior tag `icassp2027-public-release-v1` reflected an earlier phase of
this project (private freeze `e6cb44d475b47f3982e2fcfc25a3d29d4e61ff82`)
before the evidence-contract / H2 / H2B experiments existed; see
`PAPER_AGGREGATE_RESULTS.md` for the current headline numbers.

Included from the current freeze (source and non-item-level aggregates only):

- typed evidence contract: source/semantic grounding $G$, physical evidence
  licensing $L$, DSP measurement $M$, Strong-Kleene composition
  (`scripts/scv_v2/`, `scripts/p2r/`)
- physical-time lag: \(L(f_s)=\lfloor 0.300\,f_s\rfloor\), \(\tau=1000\,\hat\ell/f_s\) ms
- dominant frequency in Hz: \(\hat k f_s/N\)
- normalized periodicity: \(\max |R_{xx}[\ell]|/R_{xx}[0]\)
- contracts, Kleene, compiler adapter (`qwen3:8b` / prompt `v2` / seed `20270823`);
  the exact committed prompt template is at `prompts/qwen3_8b_prompt_v2.json`
- unused-window physical-invariance and resolvability *scripts* (need local PAMAP2)
- aggregate score stamps for E1/E2/E3c (`h4_score_stamp.json`, `h3_score_stamp.json`,
  `h2b_official_analysis_stamp.json`) and the headline hash manifest
  (`result_manifest.json`)

Intentionally withheld:

- `final_sealed_holdout*.jsonl` and any sealed gold/inference JSONL
- `h2_official_predictions.jsonl`, `h2b_v1_predictions.jsonl`, `h2b_v2_predictions.jsonl`
  and any other sealed sentence/record item files
- the submitted manuscript LaTeX/PDF (ships with the CMS submission package, not this repository)
- process/audit memoranda (planning notes, internal review gates); only code,
  prompts, environment locks, and non-item-level result aggregates are public
- model caches, raw corpora, credentials

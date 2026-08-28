"""Phase 0: lag semantic integrity + numeric-domain audit. No PRIMARY rerun."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from p2.config import BENCH_P2_V3  # noqa: E402
from p2.evaluate import load_split  # noqa: E402
from p2r.eval_p2r import _gold_program  # noqa: E402

from .config import RAW_V3_EXACT, RESULTS  # noqa: E402
from .io_util import program_from_dict, write_json  # noqa: E402
from .numeric_domain import classify_numeric_role, vs_value_in_domain  # noqa: E402
from .semantic_canon import programs_canonically_equal  # noqa: E402


def run_lag_integrity() -> dict:
    rows = load_split("challenge", bench_root=BENCH_P2_V3)
    recs = (RESULTS.parent / "p2r_lm1" / "v3_challenge_primary_records.json")
    records = __import__("json").loads(recs.read_text(encoding="utf-8"))
    by = {r["claim_id"]: r for r in records}
    n = len(records)
    raw_exact = sum(int(r["exact"]) for r in records) / n
    canon_hits = 0
    lag_n = 0
    lag_raw = 0
    lag_canon = 0
    only_ref_gap = 0
    for row in rows:
        rec = by[row["claim_id"]]
        gold = _gold_program(row)
        pred = program_from_dict(rec["pred_program"])
        is_lag = any(p.measurement == "cross_channel_lag_ms" for p in gold.predicates)
        if is_lag:
            lag_n += 1
            lag_raw += int(rec["exact"])
        eq = programs_canonically_equal(pred, gold)
        canon_hits += int(eq)
        if is_lag:
            lag_canon += int(eq)
            if eq and not rec["exact"]:
                only_ref_gap += 1
    return {
        "n": n,
        "RAW_EXACT_PROGRAM": raw_exact,
        "RAW_EXACT_PROGRAM_preregistered": RAW_V3_EXACT,
        "CANONICAL_SEMANTIC_PROGRAM_ACCURACY": canon_hits / n,
        "lag_n": lag_n,
        "lag_raw_exact": lag_raw / lag_n if lag_n else None,
        "lag_canonical_exact": lag_canon / lag_n if lag_n else None,
        "exact_misses_closed_by_dropping_lag_reference_channel": only_ref_gap,
        "inference_rerun": False,
        "equivalence_scope": "all_signals_via_executor_and_correlation_reversal",
    }


def run_numeric_audit() -> dict:
    rows = load_split("challenge", bench_root=BENCH_P2_V3)
    recs = __import__("json").loads((RESULTS.parent / "p2r_lm1" / "v3_challenge_primary_records.json").read_text(encoding="utf-8"))
    by = {r["claim_id"]: r for r in recs}
    fa = [r for r in recs if r["gold_verdict"] != "UNVERIFIABLE" and r["pred_verdict"] == "UNVERIFIABLE"]
    cases = []
    roles = Counter()
    for rec in fa:
        row = next(x for x in rows if x["claim_id"] == rec["claim_id"])
        gold = _gold_program(row)
        roles_here = []
        for gp, raw in zip(gold.predicates, (row.get("semantic_program") or {}).get("predicates") or [{}]):
            role = classify_numeric_role(raw if isinstance(raw, dict) else {})
            if gold.predicates:
                role = "B_PURPORTED_MEASUREMENT_VALUE" if gp.comparator == "eq" else (
                    "A_THRESHOLD" if gp.comparator in ("gt", "lt") and not gp.reference_channel else role
                )
            in_dom, reason = vs_value_in_domain(gp, row.get("fs")) if gold.predicates else (True, None)
            roles_here.append({"role": role, "in_domain": in_dom, "reason": reason, "measurement": gp.measurement, "value": gp.reference_value, "unit": gp.unit})
            roles[role] += 1
        cases.append({
            "claim_id": rec["claim_id"],
            "dataset": rec["dataset"],
            "pred_reason": rec["pred_program"].get("parse_reason"),
            "gold_verdict": rec["gold_verdict"],
            "predicates": roles_here,
            "surface_text": row["inference"]["surface_text"],
        })
    # generator defect: force_false vs_value can leave the output domain
    n_ood_gold = 0
    n_spec_ood = 0
    for row in rows:
        gold = _gold_program(row)
        for p in gold.predicates:
            ok, _ = vs_value_in_domain(p, row.get("fs"))
            if not ok:
                n_ood_gold += 1
                if p.measurement == "spectral_energy_ratio_low":
                    n_spec_ood += 1
    return {
        "n_false_abstention": len(fa),
        "threshold_cases_A": roles.get("A_THRESHOLD", 0),
        "invalid_measurement_cases_B": roles.get("B_PURPORTED_MEASUREMENT_VALUE", 0),
        "cases": cases,
        "v3_challenge_ood_vs_value_gold": n_ood_gold,
        "v3_challenge_ood_spectral_vs_value": n_spec_ood,
        "generator_mechanism": "_make_single_vs_value force_false = actual ± (3..6)*tol can exit [0,1] for spectral_energy_ratio_low",
        "holdout_opened": False,
        "v3_holdout_status_if_same_generator": "POTENTIALLY_SUPERSEDED_UNEVALUATED",
        "frozen_rule": "vs_value asserted_value must lie in measurement output domain; vs_threshold may be any real",
        "validator_action": "P3 rejects out-of-domain vs_value as UNPARSEABLE; generator for P3 stays inside domain",
    }


def main():
    lag = run_lag_integrity()
    num = run_numeric_audit()
    write_json("phase0_lag.json", lag)
    write_json("phase0_numeric.json", num)
    print("LAG canon", lag["CANONICAL_SEMANTIC_PROGRAM_ACCURACY"], "raw", lag["RAW_EXACT_PROGRAM"])
    print("FA", num["n_false_abstention"], "A", num["threshold_cases_A"], "B", num["invalid_measurement_cases_B"])
    return lag, num


if __name__ == "__main__":
    main()

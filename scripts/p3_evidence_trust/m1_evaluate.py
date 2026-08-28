"""Score full vs one-clause shadow on constructed invalid probes."""
from __future__ import annotations

from collections import Counter, defaultdict

from .config import ABLATIONS, EXPERIMENT_IDS, RESULTS, SEED
from .m1_probes import applicable, apply_probe
from .m1_shadow import production_verdict, shadow_verdict
from .windows import e1_items


def score() -> dict:
    items = e1_items()
    records = []
    for it in items:
        for ab in ABLATIONS:
            if not applicable(it["op"], ab):
                continue
            probe = apply_probe(it, ab)
            full = production_verdict(
                it["program"], it["available_channels"], probe["fs"], probe["channels"]
            )
            weak = shadow_verdict(
                it["program"],
                it["available_channels"],
                probe["fs"],
                probe["channels"],
                ab,
            )
            ctrl = shadow_verdict(
                it["program"],
                it["available_channels"],
                it["fs"],
                it["channels"],
                ab,
            )
            prod_clean = production_verdict(
                it["program"], it["available_channels"], it["fs"], it["channels"]
            )
            records.append(
                {
                    "item_id": it["item_id"],
                    "dataset": it["dataset"],
                    "op": it["op"],
                    "ablation": ab,
                    "invalid_by_construction": probe["invalid_by_construction"],
                    "full_verdict": full["verdict"],
                    "weak_verdict": weak["verdict"],
                    "weak_reason": weak.get("reason"),
                    "kernel_exception": bool(weak.get("kernel_exception")),
                    "leftover_clause": bool(weak.get("leftover_clause")),
                    "control_match": ctrl["verdict"] == prod_clean["verdict"],
                    "control_full": prod_clean["verdict"],
                    "control_weak": ctrl["verdict"],
                }
            )
    return _summarize(records, len(items))


def _summarize(records: list[dict], n_carriers: int) -> dict:
    by_ab = defaultdict(list)
    for r in records:
        by_ab[r["ablation"]].append(r)

    per = {}
    prod_commit_invalid = 0
    n_invalid = 0
    for ab in ABLATIONS:
        rows = by_ab.get(ab, [])
        inv = [r for r in rows if r["invalid_by_construction"]]
        n_inv = len(inv)
        n_invalid += n_inv
        n_prod_commit = sum(1 for r in inv if r["full_verdict"] in ("SUPPORTED", "CONTRADICTED"))
        prod_commit_invalid += n_prod_commit
        n_fu = sum(1 for r in inv if r["full_verdict"] == "UNVERIFIABLE")
        inf_s = sum(1 for r in inv if r["full_verdict"] == "UNVERIFIABLE" and r["weak_verdict"] == "SUPPORTED")
        inf_c = sum(1 for r in inv if r["full_verdict"] == "UNVERIFIABLE" and r["weak_verdict"] == "CONTRADICTED")
        unk_keep = sum(1 for r in inv if r["full_verdict"] == "UNVERIFIABLE" and r["weak_verdict"] == "UNVERIFIABLE")
        n_weak_commit = sum(1 for r in inv if r["weak_verdict"] in ("SUPPORTED", "CONTRADICTED"))
        n_exc = sum(1 for r in inv if r["kernel_exception"])
        n_left = sum(1 for r in inv if r["leftover_clause"])
        ctrl = [r for r in rows if r["control_full"] == "SUPPORTED"]
        n_ctrl_ok = sum(1 for r in ctrl if r["control_match"])
        per[ab] = {
            "n": len(rows),
            "n_invalid": n_inv,
            "production_fcr": n_prod_commit / n_inv if n_inv else None,
            "n_production_commit_on_invalid": n_prod_commit,
            "supported_inflation": inf_s / n_fu if n_fu else None,
            "contradicted_inflation": inf_c / n_fu if n_fu else None,
            "commitment_inflation": (inf_s + inf_c) / n_fu if n_fu else None,
            "unknown_retention": unk_keep / n_fu if n_fu else None,
            "n_full_unverifiable": n_fu,
            "n_supported_inflation": inf_s,
            "n_contradicted_inflation": inf_c,
            "n_unknown_retention": unk_keep,
            "weak_fcr": n_weak_commit / n_inv if n_inv else None,
            "n_weak_commit": n_weak_commit,
            "n_kernel_exception": n_exc,
            "n_leftover_clause": n_left,
            "control_n": len(ctrl),
            "control_match_rate": n_ctrl_ok / len(ctrl) if ctrl else None,
            "weak_counts": dict(Counter(r["weak_verdict"] for r in inv)),
            "full_counts": dict(Counter(r["full_verdict"] for r in inv)),
        }

    prod_fcr = prod_commit_invalid / n_invalid if n_invalid else None
    control_all = [r for r in records if r["control_full"] == "SUPPORTED"]
    decision = "PASS" if prod_fcr == 0.0 else "STOP"
    if control_all and any(not r["control_match"] for r in control_all):
        # Legal-record change is a shadow bug unless it is drop_output_domain on a
        # value that production would also accept (should still match).
        bad = [r for r in control_all if not r["control_match"]]
        # drop_min_n etc. on *clean* legal records must match. Flag STOP only if
        # a non-length shadow changes a legal clean verdict.
        serious = [r for r in bad if r["ablation"] not in ("drop_min_n",)]
        # actually all clean records are legal length; all ablations on clean should match
        if bad:
            decision = "STOP"
            control_fail = len(bad)
        else:
            control_fail = 0
    else:
        control_fail = 0

    summary = {
        "n_carriers": n_carriers,
        "n_scores": len(records),
        "n_invalid": n_invalid,
        "production_fcr_invalid": prod_fcr,
        "n_production_commit_on_invalid": prod_commit_invalid,
        "control_n": len(control_all),
        "control_fail": control_fail,
        "decision": decision,
        "by_ablation": per,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"experiment_id": EXPERIMENT_IDS["m1"], "seed": SEED, "run_count": 1},
        "summary": summary,
        "records": records,
    }
    import json

    (RESULTS / "contract_necessity_run.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload

"""Audit EXISTING P3 language-shift outputs. No PRIMARY rerun."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from p2.stats_ci import proportion
from p3.config import RESULTS as P3_RESULTS
from p3.eval_common import _gold
from p3.io_util import program_from_dict

from .config import HIST_CANONICAL, HIST_RAW_EXACT, RESULTS
from .strict_semantic import (
    EQUIV_LABELS,
    GENUINE_FIELD,
    classify_mismatch,
    previous_canonical,
    programs_strictly_equivalent,
)


def _raw_exact(pred, gold) -> bool:
    if gold.parse_status != pred.parse_status:
        return False
    if gold.parse_status != "OK":
        return True
    if gold.connective != pred.connective or len(gold.predicates) != len(pred.predicates):
        return False
    from p3.semantic_canon import _pred_eq
    return all(_pred_eq(a, b) for a, b in zip(pred.predicates, gold.predicates))


def run_audit() -> dict:
    recs = json.loads((P3_RESULTS / "language_shift_primary_records.json").read_text(encoding="utf-8"))
    rows = {r["claim_id"]: r for r in json.loads((P3_RESULTS / "language_shift_rows.json").read_text(encoding="utf-8"))}
    n = 0
    raw_k = canon_k = strict_k = 0
    labels = Counter()
    genuine_field = Counter()
    verdict_changing = 0
    benign_genuine = 0
    fcr = fa = 0
    n_unv = n_ans = 0
    details = []
    for rec in recs:
        row = rows.get(rec["claim_id"])
        if not row:
            continue
        n += 1
        gold = _gold(row)
        pred = program_from_dict(rec["pred_program"])
        avail = row["available_channels"]
        raw = bool(rec.get("exact")) or _raw_exact(pred, gold)
        canon = bool(rec.get("canonical")) or previous_canonical(pred, gold)
        strict = programs_strictly_equivalent(pred, gold, avail)
        raw_k += int(raw)
        canon_k += int(canon)
        strict_k += int(strict)
        gv, pv = rec["gold_verdict"], rec["pred_verdict"]
        if gv == "UNVERIFIABLE":
            n_unv += 1
            fcr += int(pv in ("SUPPORTED", "CONTRADICTED"))
        else:
            n_ans += 1
            fa += int(pv == "UNVERIFIABLE")
        if not raw:
            lab = classify_mismatch(pred, gold, avail)
            labels[lab] += 1
            if lab not in EQUIV_LABELS:
                genuine_field[GENUINE_FIELD.get(lab, "other")] += 1
                if gv != pv:
                    verdict_changing += 1
                else:
                    benign_genuine += 1
            details.append({
                "claim_id": rec["claim_id"],
                "label": lab,
                "gold_verdict": gv,
                "pred_verdict": pv,
                "source": rec.get("source") or row.get("source"),
            })
    out = {
        "n": n,
        "historical_raw_exact": HIST_RAW_EXACT,
        "historical_canonical": HIST_CANONICAL,
        "measured_raw_exact": raw_k / n if n else None,
        "measured_previous_canonical": canon_k / n if n else None,
        "strict_semantic_equivalence": strict_k / n if n else None,
        "strict_semantic_ci": proportion(strict_k, n) if n else None,
        "total_non_exact": n - raw_k,
        "equivalent_or_redundant": sum(labels[k] for k in EQUIV_LABELS),
        "genuine_semantic_errors": sum(v for k, v in labels.items() if k not in EQUIV_LABELS),
        "verdict_changing_semantic_errors": verdict_changing,
        "benign_genuine_errors": benign_genuine,
        "false_commitment": proportion(fcr, n_unv) if n_unv else {"p": 0, "n": 0, "k": 0},
        "false_abstention": proportion(fa, n_ans) if n_ans else None,
        "mismatch_labels": dict(labels),
        "genuine_by_field": dict(genuine_field),
        "primary_weakness": "actual_semantic_compilation" if sum(v for k, v in labels.items() if k not in EQUIV_LABELS) > sum(labels[k] for k in EQUIV_LABELS) else "scoring_representation",
        "PRIMARY_rerun": False,
        "details_head": details[:40],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "frozen_ls_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("AUDIT", {k: out[k] for k in out if k != "details_head"}, flush=True)
    return out

"""Baseline v2 vs IR→compiler. Checkpointed. No SEM-BLIND."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from p2.stats_ci import proportion
from p2r.ollama_adapter import extract_ollama
from p2r.pipeline import run_pipeline
from p3.eval_common import _gold, field_raw
from p3c.strict_semantic import programs_strictly_equivalent

from .compiler import compile_ir
from .config import PRIMARY_MODEL, RESULTS, BASELINE_PROMPT
from .ir_adapter import extract_ir
from .ir_schema import ATOMIC, COMPOSITE, CONDITIONAL, UNCERTAIN, irs_equal, validate_ir


def _atoms(node: dict) -> list[dict]:
    t = node.get("type")
    if t == ATOMIC:
        return [node]
    if t == COMPOSITE:
        out = []
        for c in node.get("children") or []:
            out.extend(_atoms(c))
        return out
    if t == CONDITIONAL:
        return _atoms(node.get("antecedent") or {}) + _atoms(node.get("consequent") or {})
    return []


def _structure_label(node: dict) -> str:
    t = node.get("type")
    if t in UNCERTAIN:
        return t
    if t == ATOMIC:
        return "SINGLE"
    if t == COMPOSITE:
        n = len(node.get("children") or [])
        return f"{node.get('operator')}_{n}"
    if t == CONDITIONAL:
        return "IF_THEN"
    return "OTHER"


def ir_fields(pred_ir: dict, gold_ir: dict) -> dict:
    g, _ = validate_ir(gold_ir)
    p, _ = validate_ir(pred_ir)
    if g is None:
        g = {"type": "UNSUPPORTED_LANGUAGE"}
    if p is None:
        p = {"type": "UNSUPPORTED_LANGUAGE", "reason": "invalid"}
    exact = irs_equal(p, g)
    gs, ps = _structure_label(g), _structure_label(p)
    g_unc = g.get("type") in UNCERTAIN
    p_unc = p.get("type") in UNCERTAIN
    ga, pa = _atoms(g), _atoms(p)
    if g_unc or p_unc:
        return {
            "ir_exact": exact,
            "primitive": g_unc and p_unc,
            "channel": g_unc and p_unc,
            "comparator": g_unc and p_unc,
            "connective": g.get("type") == p.get("type"),
            "structure": gs == ps,
        }
    prim = len(ga) == len(pa) and all(a.get("primitive") == b.get("primitive") for a, b in zip(ga, pa))
    if gs.startswith("AND") or gs.startswith("OR"):
        gp = sorted(x.get("primitive") or "" for x in ga)
        pp = sorted(x.get("primitive") or "" for x in pa)
        prim = gp == pp
    ch_ok = len(ga) == len(pa)
    if ch_ok:
        def chs(a):
            return tuple("" if a.get(k) is None else str(a.get(k)) for k in (
                "channel", "source_channel", "target_channel", "reference_channel",
            ))
        if gs.startswith("AND") or gs.startswith("OR"):
            ch_ok = sorted(map(chs, ga)) == sorted(map(chs, pa))
        else:
            ch_ok = all(chs(a) == chs(b) for a, b in zip(ga, pa))
    cmp_ok = len(ga) == len(pa)
    if cmp_ok:
        if gs.startswith("AND") or gs.startswith("OR"):
            cmp_ok = sorted(x.get("operator") or "" for x in ga) == sorted(x.get("operator") or "" for x in pa)
        else:
            cmp_ok = all((a.get("operator") == b.get("operator")) for a, b in zip(ga, pa))
    conn = (g.get("type"), g.get("operator")) == (p.get("type"), p.get("operator")) if g.get("type") == COMPOSITE else g.get("type") == p.get("type")
    return {
        "ir_exact": exact,
        "primitive": prim,
        "channel": ch_ok,
        "comparator": cmp_ok,
        "connective": conn,
        "structure": gs == ps,
    }


def evaluate_ir_path(rows, label: str, ckpt_name: str) -> dict:
    ckpt = RESULTS / ckpt_name
    done = {}
    if ckpt.exists():
        done = {r["claim_id"]: r for r in json.loads(ckpt.read_text(encoding="utf-8"))}
        print(f"  {label} resume {len(done)}", flush=True)
    recs = []
    field_hits = Counter()
    ir_hits = Counter()
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 25 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        if row["claim_id"] in done:
            rec = done[row["claim_id"]]
        else:
            gold = _gold(row)
            gold_ir = row.get("gold_ir") or {}
            pred_ir = extract_ir(row["surface_text"], row["available_channels"], row.get("fs"))
            pred = compile_ir(pred_ir, row["available_channels"])
            ch = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
            out = run_pipeline(row["surface_text"], row["available_channels"], row["fs"], ch, lambda t, c, f, _p=pred: _p)
            compiled_gold = compile_ir(gold_ir, row["available_channels"])
            fs = field_raw(pred, gold)
            irf = ir_fields(pred_ir, gold_ir)
            rec = {
                "claim_id": row["claim_id"],
                "source": row.get("source"),
                "family": row.get("family"),
                "primitive": row.get("primitive") or (gold.predicates[0].measurement if gold.predicates else None),
                "connective": row.get("connective") or gold.connective,
                "n_pred": row.get("n_pred") if row.get("n_pred") is not None else len(gold.predicates),
                "gold_verdict": row["gold_composed_verdict"],
                "pred_verdict": out["verdict"],
                "exact": bool(fs.get("exact")),
                "strict": programs_strictly_equivalent(pred, gold, row["available_channels"]),
                "compile_exact": programs_strictly_equivalent(pred, compiled_gold, row["available_channels"]),
                "pred_program": pred.to_dict(),
                "pred_ir": pred_ir,
                "field": {k: bool(fs.get(k)) for k in ("measurement", "channel", "comparator", "value", "connective")},
                "ir_field": irf,
                "path": "ir_compiler",
            }
        recs.append(rec)
        for k, v in (rec.get("field") or {}).items():
            field_hits[k] += int(v)
        for k, v in (rec.get("ir_field") or {}).items():
            ir_hits[k] += int(v)
        if i % 20 == 0 or i == len(rows):
            RESULTS.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return _summarize(label, recs, field_hits, ir_hits, "ir_compiler")


def evaluate_baseline(rows, label: str, ckpt_name: str) -> dict:
    ckpt = RESULTS / ckpt_name
    done = {}
    if ckpt.exists():
        done = {r["claim_id"]: r for r in json.loads(ckpt.read_text(encoding="utf-8"))}
        print(f"  {label} resume {len(done)}", flush=True)
    recs = []
    field_hits = Counter()
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 25 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        if row["claim_id"] in done:
            rec = done[row["claim_id"]]
        else:
            gold = _gold(row)
            pred, _ = extract_ollama(
                row["surface_text"], row["available_channels"], row.get("fs"),
                model=PRIMARY_MODEL, prompt_version=BASELINE_PROMPT,
            )
            ch = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
            out = run_pipeline(row["surface_text"], row["available_channels"], row["fs"], ch, lambda t, c, f, _p=pred: _p)
            fs = field_raw(pred, gold)
            rec = {
                "claim_id": row["claim_id"],
                "source": row.get("source"),
                "family": row.get("family"),
                "primitive": row.get("primitive") or (gold.predicates[0].measurement if gold.predicates else None),
                "connective": row.get("connective") or gold.connective,
                "n_pred": row.get("n_pred") if row.get("n_pred") is not None else len(gold.predicates),
                "gold_verdict": row["gold_composed_verdict"],
                "pred_verdict": out["verdict"],
                "exact": bool(fs.get("exact")),
                "strict": programs_strictly_equivalent(pred, gold, row["available_channels"]),
                "pred_program": pred.to_dict(),
                "field": {k: bool(fs.get(k)) for k in ("measurement", "channel", "comparator", "value", "connective")},
                "path": "baseline_v2",
            }
        recs.append(rec)
        for k, v in (rec.get("field") or {}).items():
            field_hits[k] += int(v)
        if i % 20 == 0 or i == len(rows):
            RESULTS.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return _summarize(label, recs, field_hits, Counter(), "baseline_v2")


def _ci(k, n):
    return proportion(k, n) if n else {"p": None, "n": 0, "k": 0}


def _summarize(label, recs, field_hits, ir_hits, path):
    n = len(recs)
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for r in recs if r["pred_verdict"] == lab and r["gold_verdict"] == lab)
        fp = sum(1 for r in recs if r["pred_verdict"] == lab and r["gold_verdict"] != lab)
        fn = sum(1 for r in recs if r["pred_verdict"] != lab and r["gold_verdict"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        reca = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * reca / (prec + reca) if prec + reca else 0.0
        per[lab] = {"precision": prec, "recall": reca, "f1": f1}
        f1s.append(f1)
    unv = [r for r in recs if r["gold_verdict"] == "UNVERIFIABLE"]
    ans = [r for r in recs if r["gold_verdict"] != "UNVERIFIABLE"]
    fcr = sum(1 for r in unv if r["pred_verdict"] in ("SUPPORTED", "CONTRADICTED"))
    fa = sum(1 for r in ans if r["pred_verdict"] == "UNVERIFIABLE")

    def _by(key):
        g = defaultdict(list)
        for r in recs:
            if r.get(key) is None:
                continue
            g[r[key]].append(r)
        out = {}
        for k, xs in g.items():
            a = [x for x in xs if x["gold_verdict"] != "UNVERIFIABLE"]
            u = [x for x in xs if x["gold_verdict"] == "UNVERIFIABLE"]
            irf = [x.get("ir_field") or {} for x in xs]
            out[str(k)] = {
                "n": len(xs),
                "exact": sum(x["exact"] for x in xs) / len(xs),
                "strict": sum(bool(x.get("strict")) for x in xs) / len(xs),
                "compile_exact": sum(bool(x.get("compile_exact")) for x in xs) / len(xs),
                "verdict": sum(x["pred_verdict"] == x["gold_verdict"] for x in xs) / len(xs),
                "primitive": sum((x.get("field") or {}).get("measurement", False) for x in xs) / len(xs),
                "ir_primitive": sum(bool(f.get("primitive")) for f in irf) / len(xs),
                "ir_structure": sum(bool(f.get("structure")) for f in irf) / len(xs),
                "fcr": _ci(sum(x["pred_verdict"] in ("SUPPORTED", "CONTRADICTED") for x in u), len(u)) if u else {"p": None, "n": 0, "k": 0},
                "fa": _ci(sum(x["pred_verdict"] == "UNVERIFIABLE" for x in a), len(a)) if a else None,
            }
        return out

    three = [r for r in recs if (r.get("n_pred") or 0) >= 3]
    return {
        "label": label,
        "path": path,
        "n": n,
        "exact_program": sum(r["exact"] for r in recs) / n if n else None,
        "strict_semantic": sum(bool(r.get("strict")) for r in recs) / n if n else None,
        "compile_exact": sum(bool(r.get("compile_exact")) for r in recs) / n if n else None,
        "primitive": field_hits["measurement"] / n if n else None,
        "channel": field_hits["channel"] / n if n else None,
        "comparator": field_hits["comparator"] / n if n else None,
        "connective": field_hits["connective"] / n if n else None,
        "ir_exact": ir_hits["ir_exact"] / n if n and ir_hits else None,
        "ir_primitive": ir_hits["primitive"] / n if n and ir_hits else None,
        "ir_channel": ir_hits["channel"] / n if n and ir_hits else None,
        "ir_comparator": ir_hits["comparator"] / n if n and ir_hits else None,
        "ir_connective": ir_hits["connective"] / n if n and ir_hits else None,
        "ir_structure": ir_hits["structure"] / n if n and ir_hits else None,
        "verdict_accuracy": _ci(sum(r["pred_verdict"] == r["gold_verdict"] for r in recs), n),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per,
        "false_commitment": _ci(fcr, len(unv)) if unv else {"p": 0.0, "n": 0, "k": 0},
        "false_abstention": _ci(fa, len(ans)) if ans else None,
        "by_source": _by("source"),
        "by_connective": _by("connective"),
        "by_n_pred": _by("n_pred"),
        "by_primitive": _by("primitive"),
        "by_family": _by("family"),
        "if_then": _by("connective").get("IF_THEN"),
        "three_predicate": {
            "n": len(three),
            "strict": sum(bool(x.get("strict")) for x in three) / len(three) if three else None,
            "ir_structure": sum(bool((x.get("ir_field") or {}).get("structure")) for x in three) / len(three) if three else None,
        },
        "records": recs,
    }


def slim(m):
    d = {k: v for k, v in m.items() if k != "records"}
    d["n_records"] = len(m.get("records") or [])
    return d


def ir_dev_gates(m: dict) -> dict:
    by_c = m.get("by_connective") or {}
    if_then = (by_c.get("IF_THEN") or {}).get("strict")
    three = (m.get("three_predicate") or {}).get("strict")
    # 3-predicate semantic: prefer IR structure if present else program strict
    checks = {
        "if_then_semantic_ge_80": (if_then is not None and if_then >= 0.80, if_then),
        "three_pred_semantic_ge_85": (three is not None and three >= 0.85, three),
        "overall_strict_ge_90": (m.get("strict_semantic") is not None and m["strict_semantic"] >= 0.90, m.get("strict_semantic")),
        "primitive_ge_95": (m.get("primitive") is not None and m["primitive"] >= 0.95, m.get("primitive")),
        "verdict_ge_92": ((m.get("verdict_accuracy") or {}).get("p") is not None and m["verdict_accuracy"]["p"] >= 0.92, (m.get("verdict_accuracy") or {}).get("p")),
        "fcr_le_5": ((m.get("false_commitment") or {}).get("p") is not None and m["false_commitment"]["p"] <= 0.05, (m.get("false_commitment") or {}).get("p")),
    }
    return {
        "pass": all(v[0] for v in checks.values()),
        "checks": {k: {"ok": ok, "value": val} for k, (ok, val) in checks.items()},
    }

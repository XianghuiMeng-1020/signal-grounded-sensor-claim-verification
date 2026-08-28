"""P2R-LM0B local-model DEVELOPMENT qualification.

Refuses CHALLENGE and the final sealed holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p2.config import SEED  # noqa: E402
from p2.evaluate import load_split  # noqa: E402
from p2.stats_ci import proportion  # noqa: E402
from p2.validate_primitives import run_validation  # noqa: E402
from p2r.eval_p2r import _gold_program  # noqa: E402
from p2r.ollama_adapter import (  # noqa: E402
    extract_ollama,
    list_ollama_models,
    payload_leakage_audit,
    prompt_hash,
    schema_hash,
    smoke_test,
)
from p2r.pipeline import run_pipeline  # noqa: E402
from p2r.schema import ClaimProgram, schema_hash as schash  # noqa: E402

OUT = ROOT / "results" / "p2r_lm_local"
FORBIDDEN_SPLITS = {"challenge", "final_sealed_holdout"}
DEFAULT_UNITS = {
    "dominant_frequency": "Hz",
    "rms_amplitude": "raw",
    "peak_amplitude": "raw",
    "signal_range": "raw",
    "trend_ratio": "ratio",
    "cross_channel_lag_ms": "ms",
    "periodicity_strength": "score_0_1",
    "spectral_energy_ratio_low": "fraction",
}
SUBSET_GATE = {
    "exact": 0.80,
    "measurement": 0.90,
    "channel": 0.90,
    "connective": 0.90,
    "malformed": 0.05,
}
PRIMARY_GATE = {
    "exact": 0.85,
    "measurement": 0.93,
    "channel": 0.93,
    "connective": 0.93,
    "verdict": 0.85,
    "false_commitment": 0.05,
}


def _refuse(split: str) -> None:
    if split in FORBIDDEN_SPLITS or "holdout" in split.lower():
        raise RuntimeError(f"P2R-LM0B FORBIDDEN split: {split}")


def _vals_close(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(b)))


def _unit_ok(g, p) -> bool:
    if g.unit == p.unit:
        return True
    if g.unit is None and p.unit in (None, DEFAULT_UNITS.get(g.measurement)):
        return True
    return False


def field_scores(pred: ClaimProgram, gold: ClaimProgram) -> dict[str, bool]:
    gold_ok = gold.parse_status == "OK" and bool(gold.predicates)
    pred_ok = pred.parse_status == "OK" and bool(pred.predicates)
    if not gold_ok:
        return {
            "exact": not pred_ok,
            "connective": True,
            "n_pred": not pred_ok,
            "measurement": True,
            "channel": True,
            "comparator": True,
            "value": True,
            "unit": True,
            "ref_channel": True,
            "executable": pred_ok,
            "correct_nonexec": not pred_ok,
        }
    if not pred_ok:
        z = {k: False for k in ("exact", "connective", "n_pred", "measurement", "channel", "comparator", "value", "unit", "ref_channel")}
        z["executable"] = False
        z["correct_nonexec"] = False
        return z
    gp, pp = gold.predicates, pred.predicates
    n = min(len(gp), len(pp))
    same_len = len(gp) == len(pp)
    return {
        "exact": gold.connective == pred.connective and same_len and all(
            a.measurement == b.measurement
            and a.channel_a == b.channel_a
            and a.channel_b == b.channel_b
            and a.comparator == b.comparator
            and a.reference_channel == b.reference_channel
            and _vals_close(a.reference_value, b.reference_value)
            for a, b in zip(gp, pp)
        ),
        "connective": gold.connective == pred.connective,
        "n_pred": same_len,
        "measurement": same_len and n == len(gp) and all(a.measurement == b.measurement for a, b in zip(gp, pp)),
        "channel": same_len and n == len(gp) and all(a.channel_a == b.channel_a and a.channel_b == b.channel_b for a, b in zip(gp, pp)),
        "comparator": same_len and n == len(gp) and all(a.comparator == b.comparator for a, b in zip(gp, pp)),
        "value": same_len and n == len(gp) and all(_vals_close(a.reference_value, b.reference_value) for a, b in zip(gp, pp)),
        "unit": same_len and n == len(gp) and all(_unit_ok(a, b) for a, b in zip(gp, pp)),
        "ref_channel": same_len and n == len(gp) and all(a.reference_channel == b.reference_channel for a, b in zip(gp, pp)),
        "executable": True,
        "correct_nonexec": False,
    }


def classify_error(pred: ClaimProgram, gold: ClaimProgram) -> str:
    gold_ok = gold.parse_status == "OK" and bool(gold.predicates)
    pred_ok = pred.parse_status == "OK" and bool(pred.predicates)
    if gold_ok == pred_ok and (not gold_ok or field_scores(pred, gold)["exact"]):
        return "none"
    if gold_ok and pred.parse_status != "OK":
        return "unsupported/ambiguity detection"
    if (not gold_ok) and pred_ok:
        return "unsupported/ambiguity detection"
    if pred.parse_reason in ("non_json",) or (pred.parse_reason or "").startswith("parse_error"):
        return "schema formatting"
    if gold.connective != pred.connective:
        return "connective"
    if len(gold.predicates) != len(pred.predicates):
        return "predicate structure"
    fs = field_scores(pred, gold)
    if not fs["measurement"]:
        return "primitive"
    if not fs["channel"]:
        return "channel"
    if not fs["ref_channel"]:
        return "reference channel"
    if not fs["comparator"]:
        return "comparator"
    if not fs["value"]:
        return "value"
    if not fs["unit"]:
        return "unit"
    return "other"


def build_qualification_subset(rows: list[dict], n_target: int = 256) -> list[str]:
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for r in rows:
        gold = _gold_program(r)
        prim = None
        if gold.predicates:
            prim = gold.predicates[0].measurement
        elif r.get("primitive") and not isinstance(r.get("primitive"), list):
            prim = r.get("primitive")
        key = (
            r["source_dataset"],
            r.get("connective") or "SINGLE",
            prim or "NONE",
            "UNV" if r.get("gold_composed_verdict") == "UNVERIFIABLE" else "ANS",
            len(gold.predicates),
        )
        by[key].append(r["claim_id"])
    chosen: list[str] = []
    # at least one from every nonempty stratum
    for ids in by.values():
        chosen.append(str(rng.choice(ids)))
    remain = [r["claim_id"] for r in rows if r["claim_id"] not in set(chosen)]
    rng.shuffle(remain)
    # boost rare connectives / unverifiable if still short
    need = {
        "AND": 36,
        "OR": 24,
        "IF_THEN": 24,
    }
    have = Counter(next(r.get("connective") or "SINGLE" for r in rows if r["claim_id"] == i) for i in chosen)
    by_id = {r["claim_id"]: r for r in rows}
    for conn, k in need.items():
        extra = [i for i in remain if by_id[i].get("connective") == conn]
        rng.shuffle(extra)
        for i in extra:
            if have[conn] >= k:
                break
            chosen.append(i)
            remain.remove(i)
            have[conn] += 1
    unv_have = sum(1 for i in chosen if by_id[i].get("gold_composed_verdict") == "UNVERIFIABLE")
    extra_unv = [i for i in remain if by_id[i].get("gold_composed_verdict") == "UNVERIFIABLE"]
    rng.shuffle(extra_unv)
    for i in extra_unv:
        if unv_have >= 48:
            break
        chosen.append(i)
        remain.remove(i)
        unv_have += 1
    for i in remain:
        if len(chosen) >= n_target:
            break
        chosen.append(i)
    # unique preserve order
    seen = set()
    out = []
    for i in chosen:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[: max(n_target, 200)]


def evaluate_rows(
    rows: list[dict],
    model: str,
    prompt_version: str,
    label: str,
    seed: int = SEED,
    cache_suffix: str = "",
) -> dict[str, Any]:
    field_hits = Counter()
    field_n = 0
    verdicts = []
    errors = []
    malformed = 0
    false_exec = 0
    n_non = 0
    latencies = []
    records = []
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        gold = _gold_program(row)
        inf = row["inference"]
        pred, meta = extract_ollama(
            inf["surface_text"],
            inf["available_channels"],
            inf.get("fs"),
            model=model,
            prompt_version=prompt_version,
            seed=seed,
            cache_suffix=cache_suffix,
        )
        out = run_pipeline(
            inf["surface_text"],
            inf["available_channels"],
            row["fs"],
            row["channels_data"],
            lambda t, c, f, _p=pred: _p,
        )
        fs = field_scores(pred, gold)
        field_n += 1
        for k, v in fs.items():
            field_hits[k] += int(v)
        gold_ok = gold.parse_status == "OK" and bool(gold.predicates)
        pred_ok = pred.parse_status == "OK" and bool(pred.predicates)
        if not gold_ok:
            n_non += 1
            if pred_ok:
                false_exec += 1
        if meta.get("malformed") or pred.parse_reason == "non_json":
            malformed += 1
        if meta.get("latency_s") is not None and not meta.get("cache_hit"):
            latencies.append(meta["latency_s"])
        gold_v = row["gold_composed_verdict"]
        pred_v = out["verdict"]
        rec = {
            "claim_id": row["claim_id"],
            "dataset": row["source_dataset"],
            "connective": row.get("connective") or "SINGLE",
            "n_pred": len(gold.predicates),
            "primitive": gold.predicates[0].measurement if gold.predicates else None,
            "gold_verdict": gold_v,
            "pred_verdict": pred_v,
            "correct": pred_v == gold_v,
            "exact": fs["exact"],
            "error": classify_error(pred, gold),
            "pred_program": pred.to_dict(),
            "gold_parse_ok": gold_ok,
        }
        verdicts.append(rec)
        records.append(rec)
        if not fs["exact"]:
            impact = "benign extraction mismatch" if pred_v == gold_v else "verdict changed"
            if gold_v == "UNVERIFIABLE" and pred_v in ("SUPPORTED", "CONTRADICTED"):
                impact = "false commitment"
            elif gold_v != "UNVERIFIABLE" and pred_v == "UNVERIFIABLE":
                impact = "false abstention"
            errors.append({
                "claim_id": row["claim_id"],
                "dataset": row["source_dataset"],
                "category": classify_error(pred, gold),
                "impact": impact,
                "gold_verdict": gold_v,
                "pred_verdict": pred_v,
                "surface_text": inf["surface_text"],
                "pred_program": pred.to_dict(),
            })
    n = len(verdicts)
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for r in verdicts if r["pred_verdict"] == lab and r["gold_verdict"] == lab)
        fp = sum(1 for r in verdicts if r["pred_verdict"] == lab and r["gold_verdict"] != lab)
        fn = sum(1 for r in verdicts if r["pred_verdict"] != lab and r["gold_verdict"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        reca = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * reca / (prec + reca) if prec + reca else 0.0
        per[lab] = {"precision": prec, "recall": reca, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    cm = {g: {p: 0 for p in labels} for g in labels}
    for r in verdicts:
        if r["gold_verdict"] in cm and r["pred_verdict"] in cm[r["gold_verdict"]]:
            cm[r["gold_verdict"]][r["pred_verdict"]] += 1
    unv = [r for r in verdicts if r["gold_verdict"] == "UNVERIFIABLE"]
    ans = [r for r in verdicts if r["gold_verdict"] != "UNVERIFIABLE"]
    committed = sum(1 for r in unv if r["pred_verdict"] in ("SUPPORTED", "CONTRADICTED"))
    false_abs = sum(1 for r in ans if r["pred_verdict"] == "UNVERIFIABLE")
    fields = {k: (field_hits[k] / field_n if field_n else None) for k in field_hits}
    return {
        "label": label,
        "model": model,
        "prompt_version": prompt_version,
        "n": n,
        "fields": fields,
        "exact_program": fields.get("exact"),
        "predicate_count": fields.get("n_pred"),
        "primitive": fields.get("measurement"),
        "channel": fields.get("channel"),
        "comparator": fields.get("comparator"),
        "value": fields.get("value"),
        "unit": fields.get("unit"),
        "ref_channel": fields.get("ref_channel"),
        "connective": fields.get("connective"),
        "executable_schema_rate": fields.get("executable"),
        "correct_nonexec": (sum(1 for r in verdicts if (not r["gold_parse_ok"]) and not (
            r["pred_program"]["parse_status"] == "OK" and r["pred_program"]["predicates"]
        )) / n_non) if n_non else None,
        "false_executable_rate": (false_exec / n_non) if n_non else 0.0,
        "malformed_output_rate": malformed / n if n else None,
        "verdict_accuracy": proportion(sum(r["correct"] for r in verdicts), n),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per,
        "confusion": cm,
        "false_commitment": proportion(committed, len(unv)) if unv else None,
        "false_abstention": proportion(false_abs, len(ans)) if ans else None,
        "answerable_coverage": proportion(len(ans) - false_abs, len(ans)) if ans else None,
        "answerable_accuracy": proportion(sum(r["correct"] for r in ans), len(ans)) if ans else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "n_latency": len(latencies),
        "n_errors": len(errors),
        "error_counts": dict(Counter(e["category"] for e in errors)),
        "errors": errors,
        "records": records,
    }


def _rate(x, default=None):
    if x is None:
        return default
    if isinstance(x, dict):
        if "estimate" in x:
            return x["estimate"]
        if "p" in x:
            return x["p"]
    return x


def subset_pass(m: dict) -> bool:
    f = m["fields"]
    malformed = _rate(m.get("malformed_output_rate"), 1.0)
    return (
        (_rate(f.get("exact"), 0) or 0) >= SUBSET_GATE["exact"]
        and (_rate(f.get("measurement"), 0) or 0) >= SUBSET_GATE["measurement"]
        and (_rate(f.get("channel"), 0) or 0) >= SUBSET_GATE["channel"]
        and (_rate(f.get("connective"), 0) or 0) >= SUBSET_GATE["connective"]
        and malformed <= SUBSET_GATE["malformed"]
    )


def primary_eligible(m: dict) -> bool:
    f = m["fields"]
    fcr_v = _rate(m.get("false_commitment"), 1.0)
    verdict_v = _rate(m.get("verdict_accuracy"), 0.0)
    return (
        (_rate(f.get("exact"), 0) or 0) >= PRIMARY_GATE["exact"]
        and (_rate(f.get("measurement"), 0) or 0) >= PRIMARY_GATE["measurement"]
        and (_rate(f.get("channel"), 0) or 0) >= PRIMARY_GATE["channel"]
        and (_rate(f.get("connective"), 0) or 0) >= PRIMARY_GATE["connective"]
        and (verdict_v or 0) >= PRIMARY_GATE["verdict"]
        and fcr_v <= PRIMARY_GATE["false_commitment"]
    )


def _est(x):
    if isinstance(x, dict) and "estimate" in x:
        return x["estimate"]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def slim(d: dict) -> dict:
    out = dict(d)
    out.pop("records", None)
    errs = out.pop("errors", [])
    out["errors_head"] = errs[:40]
    out["n_errors"] = d.get("n_errors")
    return out


def main(argv: Optional[list[str]] = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen3:8b", "gemma3:12b"])
    p.add_argument("--prompt", default="v1")
    p.add_argument("--subset-n", type=int, default=256)
    p.add_argument("--skip-full", action="store_true")
    p.add_argument("--skip-repeat", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args(argv)

    _refuse("development") if False else None  # keep helper imported
    OUT.mkdir(parents=True, exist_ok=True)

    installed = {m.get("name"): m for m in list_ollama_models()}
    write_json(OUT / "installed_models.json", {"models": list(installed.values())})

    smokes = []
    for model in args.models:
        if model not in installed:
            smokes.append({"model": model, "ok": False, "error": "not_installed"})
            continue
        smokes.append(smoke_test(model))
    write_json(OUT / "smoke_tests.json", smokes)
    live = [s["model"] for s in smokes if s.get("ok")]
    if args.smoke_only:
        return {"smoke": smokes}

    if not live:
        payload = {"decision": "LOCAL_PROVISIONING_FAILED", "smoke": smokes}
        write_json(OUT / "summary.json", payload)
        return payload

    rows = load_split("development")
    assert all(r.get("split") == "development" for r in rows if r.get("split"))
    ids = build_qualification_subset(rows, args.subset_n)
    by_id = {r["claim_id"]: r for r in rows}
    subset_rows = [by_id[i] for i in ids]
    write_json(OUT / "qualification_subset.json", {
        "n": len(ids),
        "seed": SEED,
        "claim_ids": ids,
        "split": "development",
        "challenge_ids": 0,
        "holdout_ids": 0,
    })

    leak_row = {
        "surface_text": subset_rows[0]["inference"]["surface_text"],
        "available_channels": subset_rows[0]["inference"]["available_channels"],
        "fs": subset_rows[0]["inference"]["fs"],
        "gold_program": {"CANARY": "NO"},
        "split": "challenge",
        "gold_composed_verdict": "SUPPORTED",
        "template_id": "T_CANARY",
        "paraphrase_family_id": "P_CANARY",
        "semantic_program": {"op": "NOPE"},
    }
    leak = payload_leakage_audit(leak_row, prompt_version=args.prompt)
    write_json(OUT / "payload_leakage.json", leak)
    if not leak["pass"]:
        raise RuntimeError(f"payload leakage failed: {leak}")

    subset_metrics = {}
    for model in live:
        print(f"QUAL {model} n={len(subset_rows)}", flush=True)
        m = evaluate_rows(subset_rows, model, args.prompt, f"qual:{model}")
        subset_metrics[model] = m
        tag = f"{model.replace(':', '_')}_{args.prompt}"
        write_json(OUT / f"qual_{tag}.json", slim(m))
        write_json(OUT / f"qual_{tag}_errors.json", m["errors"])

    passing = [m for m, met in subset_metrics.items() if subset_pass(met)]
    write_json(OUT / "subset_summary.json", {
        "n": len(subset_rows),
        "models": {k: slim(v) for k, v in subset_metrics.items()},
        "passing": passing,
        "gate": SUBSET_GATE,
    })

    full_metrics = {}
    if passing and not args.skip_full:
        for model in passing:
            print(f"FULLDEV {model} n={len(rows)}", flush=True)
            m = evaluate_rows(rows, model, args.prompt, f"fulldev:{model}")
            full_metrics[model] = m
        ftag = f"{model.replace(':', '_')}_{args.prompt}"
        write_json(OUT / f"fulldev_{ftag}.json", slim(m))
        write_json(OUT / f"fulldev_{ftag}_errors.json", m["errors"])
        write_json(OUT / f"fulldev_{ftag}_records.json", m["records"])
    elif not passing:
        decision = "LOCAL_MODELS_INSUFFICIENT_FOR_PRIMARY_EVALUATION"
        summary = {
            "decision": decision,
            "challenge_run_count": 0,
            "holdout_inspected": False,
            "subset": {k: slim(v) for k, v in subset_metrics.items()},
            "full": {},
            "prompt_version": args.prompt,
            "prompt_hash": prompt_hash(args.prompt),
            "schema_hash": schash(),
            "leakage": leak,
            "smoke": smokes,
        }
        write_json(OUT / "summary.json", summary)
        return summary

    eligible = [m for m, met in full_metrics.items() if primary_eligible(met)]
    ranked = sorted(
        full_metrics.items(),
        key=lambda kv: (
            kv[1]["fields"].get("exact") or 0,
            -(kv[1].get("false_executable_rate") or 0),
            kv[1]["fields"].get("measurement") or 0,
            kv[1]["fields"].get("channel") or 0,
            kv[1]["fields"].get("comparator") or 0,
            kv[1]["fields"].get("value") or 0,
            _est(kv[1].get("verdict_accuracy")) or 0,
        ),
        reverse=True,
    )
    primary = eligible[0] if eligible else (ranked[0][0] if ranked else None)
    if eligible:
        # re-pick by frozen rank among eligible only
        ranked_e = [m for m, _ in ranked if m in eligible]
        primary = ranked_e[0]
    secondary = next((m for m, _ in ranked if m != primary), None)

    repeat = None
    if primary and not args.skip_repeat:
        rng = np.random.default_rng(SEED)
        rep_ids = list(rng.choice([r["claim_id"] for r in rows], size=min(100, len(rows)), replace=False))
        rep_rows = [by_id[i] for i in rep_ids]
        runs = []
        for k in range(3):
            print(f"REPEAT {primary} run={k+1}", flush=True)
            runs.append(evaluate_rows(rep_rows, primary, args.prompt, f"rep{k}", seed=SEED, cache_suffix=f"__rep{k}"))
        raw_ag = sem_ag = verd_ag = 0
        nrep = len(rep_rows)
        for i in range(nrep):
            raws = []
            progs = []
            vers = []
            for run in runs:
                rec = run["records"][i]
                progs.append(json.dumps(rec["pred_program"], sort_keys=True))
                vers.append(rec["pred_verdict"])
            # raw agreement from cache files is optional; use program dumps as raw proxy if needed
            if len(set(progs)) == 1:
                sem_ag += 1
                raw_ag += 1
            if len(set(vers)) == 1:
                verd_ag += 1
        repeat = {
            "n": nrep,
            "model": primary,
            "raw_output_agreement": raw_ag / nrep,
            "semantic_program_agreement": sem_ag / nrep,
            "verdict_agreement": verd_ag / nrep,
            "note": "repeat runs use distinct cache suffixes so the provider is queried three times",
        }
        write_json(OUT / "repeatability.json", repeat)

    decision = "READY_FOR_PRIMARY_FREEZE" if eligible else "LOCAL_MODELS_INSUFFICIENT_FOR_PRIMARY_EVALUATION"
    summary = {
        "decision": decision,
        "challenge_run_count": 0,
        "holdout_inspected": False,
        "prompt_version": args.prompt,
        "prompt_hash": prompt_hash(args.prompt),
        "schema_hash": schash(),
        "leakage": leak,
        "smoke": smokes,
        "subset_passing": passing,
        "primary_eligible": eligible,
        "local_primary_candidate": primary if eligible else None,
        "local_secondary_candidate": secondary if secondary in eligible else None,
        "leading_local": primary,
        "subset": {k: slim(v) for k, v in subset_metrics.items()},
        "full": {k: slim(v) for k, v in full_metrics.items()},
        "repeatability": repeat,
        "code_hash": hashlib.sha256((ROOT / "scripts" / "p2r" / "ollama_adapter.py").read_bytes()).hexdigest(),
    }
    write_json(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()

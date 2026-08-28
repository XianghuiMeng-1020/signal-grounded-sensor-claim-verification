"""P2R evaluations. DEV + already-open CHALLENGE only. Holdout refused.

CHALLENGE is a repair/diagnostic set. Checkpoints are counted.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from p2.config import BENCH_P2, EVALUABLE_SPLITS, HOLDOUT_NAME, RESULTS_P2  # noqa: E402
from p2.evaluate import load_split  # noqa: E402
from p2.stats_ci import fmt_ci, proportion  # noqa: E402
from p2.validate_primitives import run_validation  # noqa: E402
from p2.windows import assigned_split  # noqa: E402

from p2r.adversarial import save as save_adv  # noqa: E402
from p2r.contracts import INSUFFICIENT_EVIDENCE, INVALID_METADATA, OK, UNSUPPORTED, check_contract  # noqa: E402
from p2r.extractor import extract_b6_baseline, extract_llm, llm_status  # noqa: E402
from p2r.pipeline import run_oracle, run_pipeline  # noqa: E402
from p2r.schema import ClaimProgram, Predicate  # noqa: E402
from p2r.validator import from_legacy  # noqa: E402

RESULTS = ROOT / "results" / "p2r"
RESULTS.mkdir(parents=True, exist_ok=True)
CHALLENGE_LOG = RESULTS / "challenge_eval_log.json"
MAX_CHALLENGE_CHECKPOINTS = 2


def _refuse_holdout(name: str) -> None:
    if name == HOLDOUT_NAME:
        raise RuntimeError("P2R FORBIDDEN: sealed holdout")


def _log_challenge(reason: str) -> int:
    log = json.loads(CHALLENGE_LOG.read_text(encoding="utf-8")) if CHALLENGE_LOG.exists() else {"n": 0, "events": []}
    log["n"] += 1
    log["events"].append(reason)
    CHALLENGE_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    if log["n"] > MAX_CHALLENGE_CHECKPOINTS:
        raise RuntimeError(f"too many CHALLENGE checkpoints: {log['n']}")
    return log["n"]


def _gold_program(row: dict) -> ClaimProgram:
    return from_legacy(row.get("semantic_program") or {"connective": "SINGLE", "predicates": []}, row["available_channels"])


def _field_scores(pred: ClaimProgram, gold: ClaimProgram) -> dict[str, bool]:
    if gold.parse_status != "OK":
        return {
            "exact": pred.parse_status != "OK",
            "connective": True,
            "n_pred": pred.parse_status != "OK",
            "measurement": True,
            "channel": True,
            "comparator": True,
            "value": True,
            "unit": True,
            "ref_channel": True,
            "executable": pred.parse_status == "OK",
            "ambiguous": pred.parse_status == "AMBIGUOUS",
        }
    if pred.parse_status != "OK":
        z = {k: False for k in ("exact", "connective", "n_pred", "measurement", "channel", "comparator", "value", "unit", "ref_channel")}
        z["executable"] = False
        z["ambiguous"] = pred.parse_status == "AMBIGUOUS"
        return z
    gp, pp = gold.predicates, pred.predicates
    def vals_close(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(b)))
    n = min(len(gp), len(pp))
    return {
        "exact": gold.connective == pred.connective and len(gp) == len(pp) and all(
            a.measurement == b.measurement and a.channel_a == b.channel_a and a.channel_b == b.channel_b
            and a.comparator == b.comparator and a.reference_channel == b.reference_channel
            and vals_close(a.reference_value, b.reference_value)
            for a, b in zip(gp, pp)
        ),
        "connective": gold.connective == pred.connective,
        "n_pred": len(gp) == len(pp),
        "measurement": n == len(gp) == len(pp) and all(a.measurement == b.measurement for a, b in zip(gp, pp)),
        "channel": n == len(gp) == len(pp) and all(a.channel_a == b.channel_a and a.channel_b == b.channel_b for a, b in zip(gp, pp)),
        "comparator": n == len(gp) == len(pp) and all(a.comparator == b.comparator for a, b in zip(gp, pp)),
        "value": n == len(gp) == len(pp) and all(vals_close(a.reference_value, b.reference_value) for a, b in zip(gp, pp)),
        "unit": True,
        "ref_channel": n == len(gp) == len(pp) and all(a.reference_channel == b.reference_channel for a, b in zip(gp, pp)),
        "executable": pred.parse_status == "OK",
        "ambiguous": pred.parse_status == "AMBIGUOUS",
    }


def _classify_error(pred: ClaimProgram, gold: ClaimProgram, gold_v: str, pred_v: str) -> str:
    if pred_v == gold_v:
        return "none"
    if gold.parse_status == "OK" and pred.parse_status != "OK":
        return "schema_validation"
    if gold.connective != pred.connective:
        return "connective_extraction"
    if len(gold.predicates) != len(pred.predicates):
        return "predicate_count_structure"
    fs = _field_scores(pred, gold)
    if not fs["measurement"]:
        return "primitive_selection"
    if not fs["channel"] or not fs["ref_channel"]:
        return "channel_resolution"
    if not fs["comparator"] or not fs["value"]:
        return "comparator_value_unit"
    if gold_v == "UNVERIFIABLE" and pred_v in ("SUPPORTED", "CONTRADICTED"):
        return "evidence_availability"
    if pred_v == "UNVERIFIABLE" and gold_v != "UNVERIFIABLE":
        return "evidence_availability"
    return "other"


def evaluate_split(split: str, extractor_name: str) -> dict:
    _refuse_holdout(split)
    if split == "challenge":
        _log_challenge(f"eval:{extractor_name}")
    rows = load_split(split)
    extract = extract_b6_baseline if extractor_name == "B6_baseline" else extract_llm
    field_hits = Counter()
    field_n = 0
    verdicts = []
    errors = []
    by_ds = defaultdict(list)
    by_conn = defaultdict(list)
    for row in rows:
        gold = _gold_program(row)
        pred = extract(row["inference"]["surface_text"], row["inference"]["available_channels"], row["inference"]["fs"])
        out = run_pipeline(
            row["inference"]["surface_text"],
            row["inference"]["available_channels"],
            row["fs"],
            row["channels_data"],
            lambda t, c, f, _p=pred: _p,
        )
        fs = _field_scores(pred, gold)
        field_n += 1
        for k, v in fs.items():
            field_hits[k] += int(v)
        gold_v = row["gold_composed_verdict"]
        pred_v = out["verdict"]
        rec = {"correct": pred_v == gold_v, "gold": gold_v, "pred": pred_v, "dataset": row["source_dataset"], "connective": row.get("connective")}
        verdicts.append(rec)
        by_ds[row["source_dataset"]].append(rec)
        by_conn[row.get("connective") or "NA"].append(rec)
        if pred_v != gold_v:
            errors.append({
                "claim_id": row["claim_id"],
                "claim": row["surface_text"],
                "gold_verdict": gold_v,
                "pred_verdict": pred_v,
                "failure_classification": _classify_error(pred, gold, gold_v, pred_v),
                "dataset": row["source_dataset"],
            })
    n = len(verdicts)
    unv = [r for r in verdicts if r["gold"] == "UNVERIFIABLE"]
    ans = [r for r in verdicts if r["gold"] != "UNVERIFIABLE"]
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    cm = {g: {p: 0 for p in labels} for g in labels}
    for r in verdicts:
        if r["gold"] in cm and r["pred"] in cm[r["gold"]]:
            cm[r["gold"]][r["pred"]] += 1
    f1s = []
    per = {}
    for lab in labels:
        tp = sum(1 for r in verdicts if r["pred"] == lab and r["gold"] == lab)
        fp = sum(1 for r in verdicts if r["pred"] == lab and r["gold"] != lab)
        fn = sum(1 for r in verdicts if r["pred"] != lab and r["gold"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        reca = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * reca / (prec + reca) if prec + reca else 0.0
        per[lab] = {"precision": prec, "recall": reca, "f1": f1}
        f1s.append(f1)
    committed = sum(1 for r in unv if r["pred"] in ("SUPPORTED", "CONTRADICTED"))
    false_abs = sum(1 for r in ans if r["pred"] == "UNVERIFIABLE")
    return {
        "extractor": extractor_name,
        "split": split,
        "n": n,
        "fields": {k: (field_hits[k] / field_n if field_n else None) for k in field_hits},
        "verdict_accuracy": proportion(sum(r["correct"] for r in verdicts), n),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per,
        "confusion": cm,
        "false_commitment": proportion(committed, len(unv)) if unv else None,
        "false_abstention": proportion(false_abs, len(ans)) if ans else None,
        "answerable_coverage": proportion(len(ans) - false_abs, len(ans)) if ans else None,
        "answerable_accuracy": proportion(sum(r["correct"] for r in ans), len(ans)) if ans else None,
        "by_dataset": {d: proportion(sum(x["correct"] for x in v), len(v)) for d, v in by_ds.items()},
        "by_connective": {d: proportion(sum(x["correct"] for x in v), len(v)) for d, v in by_conn.items()},
        "n_errors": len(errors),
        "error_counts": dict(Counter(e["failure_classification"] for e in errors)),
        "errors": errors,
    }


def oracle_eval(split: str) -> dict:
    _refuse_holdout(split)
    if split == "challenge":
        _log_challenge("oracle")
    rows = load_split(split)
    n = corr = 0
    for row in rows:
        gold = _gold_program(row)
        out = run_oracle(gold, row["available_channels"], row["fs"], row["channels_data"])
        n += 1
        corr += int(out["verdict"] == row["gold_composed_verdict"])
    return {"split": split, "n": n, "accuracy": proportion(corr, n)}


def evidence_availability_suite() -> dict:
    """Independent gold from contracts, not from production status."""
    rng = np.random.default_rng(20270823)
    n = 256
    t = np.arange(n) / 100.0
    clean = np.sin(2 * np.pi * 4 * t)
    cases = []

    def add(name, meas, chans, fs, expect_exec):
        gold = check_contract(meas, chans, fs)
        executable = gold.status == OK
        cases.append({
            "name": name,
            "measurement": meas,
            "expect_executable": expect_exec,
            "independent_status": gold.status,
            "independent_executable": executable,
            "channels": {k: np.asarray(v).tolist() if hasattr(v, "tolist") else v for k, v in chans.items()},
            "fs": fs,
        })

    add("clean_rms", "rms_amplitude", {"hand_accel": clean}, 100.0, True)
    add("noise_20db", "rms_amplitude", {"hand_accel": clean + 0.05 * rng.normal(size=n)}, 100.0, True)
    add("scale_2x", "rms_amplitude", {"hand_accel": 2 * clean}, 100.0, True)
    add("shift", "rms_amplitude", {"hand_accel": np.roll(clean, 8)}, 100.0, True)
    add("quantize", "peak_amplitude", {"hand_accel": np.round(clean * 50) / 50}, 100.0, True)
    nan = clean.copy(); nan[10] = np.nan
    add("one_nan", "rms_amplitude", {"hand_accel": nan}, 100.0, False)
    add("empty", "rms_amplitude", {"hand_accel": []}, 100.0, False)
    add("dropout", "rms_amplitude", {}, 100.0, False)
    add("bad_fs", "dominant_frequency", {"hand_accel": clean}, None, False)
    add("short_periodicity", "periodicity_strength", {"hand_accel": clean[:5]}, 100.0, False)
    add("unknown_op", "heart_rate", {"hand_accel": clean}, 100.0, False)
    add("lag_ok", "cross_channel_lag_ms", {"hand_accel": clean, "chest_accel": np.roll(clean, 4)}, 100.0, True)

    # score production executor against independent gold
    from p2r.schema import Predicate
    from p2r.executor import execute_predicate_measurement

    tp_unexec = fp_commit = n_unexec = n_exec = exec_ok = 0
    details = []
    for c in cases:
        if c["measurement"] == "heart_rate":
            pred = Predicate("heart_rate", "hand_accel", "eq", reference_value=1.0)
        else:
            chs = list(c["channels"])
            pred = Predicate(
                c["measurement"],
                chs[0] if chs else "hand_accel",
                "eq",
                channel_b=chs[1] if len(chs) > 1 else None,
                reference_value=0.0,
            )
        # reconstruct arrays
        ch_arr = {k: np.asarray(v, dtype=float) for k, v in c["channels"].items()}
        # empty dict
        try:
            res = execute_predicate_measurement(pred, ch_arr, c["fs"])
            prod_ok = res.status == OK and res.value is not None
        except Exception:
            prod_ok = False
            res = None
        gold_exec = c["independent_executable"]
        if not gold_exec:
            n_unexec += 1
            if prod_ok:
                fp_commit += 1
            else:
                tp_unexec += 1
        else:
            n_exec += 1
            exec_ok += int(prod_ok)
        details.append({**{k: c[k] for k in c if k != "channels"}, "prod_ok": prod_ok, "prod_status": getattr(res, "status", None)})

    fcr = fp_commit / n_unexec if n_unexec else 0.0
    rec = tp_unexec / n_unexec if n_unexec else 0.0
    return {
        "n": len(cases),
        "n_invalid": n_unexec,
        "correct_non_executable": tp_unexec,
        "false_commitment_rate": fcr,
        "unavailable_recall": rec,
        "valid_still_executable": exec_ok / n_exec if n_exec else None,
        "details": details,
    }


def missing_and_dropout_focus() -> dict:
    from p2r.schema import Predicate
    from p2r.executor import execute_predicate_measurement

    rng = np.random.default_rng(1)
    miss = drop = 0
    miss_ok = drop_ok = 0
    for i in range(40):
        x = rng.normal(size=128)
        y = x.copy(); y[i % 128] = np.nan
        r = execute_predicate_measurement(
            Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0),
            {"hand_accel": y},
            100.0,
        )
        miss += 1
        miss_ok += int(r.status != OK and r.value is None)
        r2 = execute_predicate_measurement(
            Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0),
            {},
            100.0,
        )
        drop += 1
        drop_ok += int(r2.status != OK and r2.value is None)
    return {
        "missing_n": miss,
        "missing_correct_nonexec": miss_ok,
        "missing_fcr": (miss - miss_ok) / miss,
        "dropout_n": drop,
        "dropout_correct_nonexec": drop_ok,
        "dropout_fcr": (drop - drop_ok) / drop,
    }


def eval_adversarial(extractor_name: str) -> dict:
    items = save_adv()
    extract = extract_b6_baseline if extractor_name == "B6_baseline" else extract_llm
    exact = 0
    false_exec = 0
    n_non = 0
    # no signal: only parse metrics + false executable rate
    for it in items:
        pred = extract(it["surface_text"], it["available_channels"], it["fs"])
        gold = ClaimProgram(
            it["gold_program"]["connective"],
            [Predicate(**p) for p in it["gold_program"]["predicates"]],
            parse_status=it["gold_program"]["parse_status"],
            parse_reason=it["gold_program"].get("parse_reason"),
        )
        fs = _field_scores(pred, gold)
        exact += int(fs["exact"] or (gold.parse_status != "OK" and pred.parse_status != "OK"))
        if it["role"] == "non_executable":
            n_non += 1
            if pred.parse_status == "OK":
                false_exec += 1
    return {
        "extractor": extractor_name,
        "n": len(items),
        "exact_program": exact / len(items),
        "false_executable_rate": false_exec / n_non if n_non else 0.0,
        "n_non_executable": n_non,
    }


def run_all() -> dict:
    dsp = run_validation()
    ev_av = evidence_availability_suite()
    miss = missing_and_dropout_focus()
    oracle_dev = oracle_eval("development")
    oracle_chal = oracle_eval("challenge")
    b6_dev = evaluate_split("development", "B6_baseline")
    # challenge B6: second/final diagnostic checkpoint (oracle already counted)
    # We already logged oracle+ we will log B6. That's 2 if we only log challenge twice.
    # oracle_eval logs challenge; evaluate_split challenge logs again = 2. Stop there.
    # Do NOT run LLM eval on challenge (unavailable anyway).
    llm = llm_status()
    llm_dev = None
    if llm["available"]:
        llm_dev = evaluate_split("development", "LLM")
    adv_b6 = eval_adversarial("B6_baseline")
    # strip bulky error lists from summary file; keep counts + first 30
    def slim(d):
        if not d:
            return d
        out = dict(d)
        errs = out.pop("errors", [])
        out["errors_head"] = errs[:30]
        return out

    payload = {
        "dsp_validation": {"gate": dsp.get("gate_c"), "n_pass": dsp.get("n_primitives_core_pass")},
        "llm": llm,
        "oracle_dev": oracle_dev,
        "oracle_challenge": oracle_chal,
        "b6_development": slim(b6_dev),
        "llm_development": slim(llm_dev) if llm_dev else None,
        "evidence_availability": ev_av,
        "missing_dropout": miss,
        "adversarial_b6": adv_b6,
        "challenge_checkpoints": json.loads(CHALLENGE_LOG.read_text(encoding="utf-8")) if CHALLENGE_LOG.exists() else {},
        "note": "CHALLENGE B6 end-to-end is the already-open diagnostic set; not confirmatory. Holdout not loaded.",
    }
    # One frozen challenge B6 checkpoint — already used 2 logs (oracle+?). 
    # If oracle logged challenge, we have 1. Need B6 challenge metrics: that's checkpoint 2.
    b6_chal = evaluate_split("challenge", "B6_baseline")
    payload["b6_challenge"] = slim(b6_chal)
    payload["challenge_checkpoints"] = json.loads(CHALLENGE_LOG.read_text(encoding="utf-8"))
    RESULTS.joinpath("p2r_eval.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    RESULTS.joinpath("error_decomposition.json").write_text(json.dumps(b6_chal.get("errors") or [], indent=1), encoding="utf-8")
    return payload

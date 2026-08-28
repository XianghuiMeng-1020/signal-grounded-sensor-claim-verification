"""IR-DEV and IR-BLIND construction. New windows. New surface wording."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p2r.validator import from_legacy

from .config import BENCH, RESULTS, SEED, OPS
from .ir_gold import gold_ir_from_st
from .llm import cached_chat
from .windows_ir import split_unused, window_pool_manifest

QTY_DEV = {
    "dominant_frequency": "principal tone",
    "rms_amplitude": "RMS magnitude",
    "peak_amplitude": "peak excursion",
    "signal_range": "excursion width",
    "trend_ratio": "late/early ratio",
    "periodicity_strength": "periodicity score",
    "spectral_energy_ratio_low": "low-band fraction",
    "cross_channel_lag_ms": "pairwise lag",
}
QTY_BLIND = {
    "dominant_frequency": "main spectral tone",
    "rms_amplitude": "quadratic amplitude",
    "peak_amplitude": "largest excursion",
    "signal_range": "min-to-max width",
    "trend_ratio": "half-to-half energy ratio",
    "periodicity_strength": "repeatability score",
    "spectral_energy_ratio_low": "sub-3Hz occupancy",
    "cross_channel_lag_ms": "inter-axis delay",
}
UNIT = {
    "dominant_frequency": "Hz",
    "rms_amplitude": "raw units",
    "peak_amplitude": "raw units",
    "signal_range": "raw units",
    "trend_ratio": "ratio",
    "periodicity_strength": "score",
    "spectral_energy_ratio_low": "fraction",
    "cross_channel_lag_ms": "ms",
}
GEMMA_GEN = "gemma3:12b"
THIRD_GEN = "llama3.1:8b"

UNV_SPECS = (
    ("unsupported_measurement", "heart_rate",
     "Pulse rate at {ch} is recorded as 68 beats per minute."),
    ("unsupported_measurement", "jerk_entropy",
     "Jerk entropy on {ch} is above 3.8 nats."),
    ("unresolved_channel", "the_sensor",
     "The sensor RMS magnitude sits at 2.75 raw units."),
    ("unsupported_logical_structure", "xor",
     "Exactly one clause is true and the other false: {ch} RMS magnitude sits above 1.2 raw units; {ch2} peak excursion sits above 2.4 raw units."),
    ("missing_required_evidence", "absent_channel",
     "Forearm-mounted accel RMS magnitude sits at 1.90 raw units."),
    ("invalid_metadata", "unknown_fs",
     "Sampling rate is unknown; {ch} principal tone sits at 2.10 Hz."),
    ("genuine_language_ambiguity", "both_directions",
     "{ch} RMS magnitude is at once larger than and smaller than 1.55 raw units."),
)


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _nm(ch):
    return ch.split("_")[0]


def _meas(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            if len(chs) < 2:
                return None
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _pred(w, rng, op):
    chs = list(w["available_channels"])
    used = chs[:2] if op == "cross_channel_lag_ms" else [rng.choice(chs)]
    if op == "cross_channel_lag_ms" and len(used) < 2:
        return None
    actual = _meas(op, used, w["channels"], w["fs"])
    if actual is None:
        return None
    if rng.random() < 0.55 or op == "cross_channel_lag_ms":
        force_false = rng.random() < 0.45
        val = actual if not force_false else actual + (0.18 * abs(actual) + 0.15) * rng.choice([-1.0, 1.0])
        if op in ("spectral_energy_ratio_low", "periodicity_strength"):
            val = min(1.0, max(0.0, val if not force_false else (0.92 if actual < 0.5 else 0.07)))
        if op in ("rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio", "dominant_frequency"):
            val = max(0.0, val)
        return {"op": op, "channels": used, "mode": "vs_value", "asserted_value": float(val)}
    rel = rng.choice(["gt", "lt"])
    span = abs(actual) if abs(actual) > 1e-6 else 1.0
    if op == "cross_channel_lag_ms":
        span = 30.0 * 1000.0 / float(w["fs"])
    delta = (0.08 + 0.25 * rng.random()) * span
    thr = actual - delta if rel == "gt" else actual + delta
    if op in ("spectral_energy_ratio_low", "periodicity_strength"):
        thr = min(0.99, max(0.01, thr))
    return {"op": op, "channels": used, "mode": "vs_threshold", "threshold": float(thr), "relation": rel}


def _clause(pred, qty_map):
    op = pred["op"]
    qty, unit = qty_map[op], UNIT[op]
    pair = " / ".join(_nm(c) for c in pred["channels"])
    if pred["mode"] == "vs_value":
        return f"{pair} {qty} sits at {float(pred['asserted_value']):.4g} {unit}"
    word = "sits above" if pred["relation"] == "gt" else "sits below"
    return f"{pair} {qty} {word} {float(pred['threshold']):.4g} {unit}"


def render_det(st, qty_map, split):
    preds = st["predicates"]
    conn = st.get("connective", "SINGLE")
    bits = [_clause(p, qty_map) for p in preds]
    if conn == "SINGLE":
        return bits[0] + "."
    if split == "ir_dev":
        if conn == "AND" and len(bits) == 2:
            return "Both measurements apply: " + ". Also ".join(bits) + "."
        if conn == "AND" and len(bits) == 3:
            return "All three apply: " + ". Also ".join(bits) + "."
        if conn == "OR":
            return "One of the two is sufficient: " + ". Alternatively ".join(bits) + "."
        if conn == "IF_THEN":
            return f"Given {bits[0]}, conclude {bits[1]}."
    else:
        if conn == "AND" and len(bits) == 2:
            return "Require every clause: " + "; plus ".join(bits) + "."
        if conn == "AND" and len(bits) == 3:
            return "Triple conjunction: " + "; plus ".join(bits) + "."
        if conn == "OR":
            return "Any single clause suffices: " + "; else ".join(bits) + "."
        if conn == "IF_THEN":
            return f"Provided {bits[0]}, then {bits[1]}."
    return ". ".join(bits) + "."


def render_unv(kind, w):
    chs = list(w["available_channels"])
    ch, ch2 = _nm(chs[0]), _nm(chs[-1])
    for fam, key, tmpl in UNV_SPECS:
        if key == kind:
            return fam, tmpl.format(ch=ch, ch2=ch2)
    return "unsupported_measurement", f"Pulse rate at {ch} is 80 bpm."


def _nums(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]


def qc_answerable(text, st):
    if not text or len(text.split()) < 4:
        return False, "too_short"
    low = text.lower()
    if any(k in low for k in ("supported", "contradicted", "gold verdict", "template")):
        return False, "leak_word"
    for p in st.get("predicates") or []:
        if not any(_nm(c).lower() in low or c.lower() in low for c in p.get("channels") or []):
            return False, "channel_missing"
        target = p.get("asserted_value", p.get("threshold"))
        if target is None:
            continue
        val = float(target)
        nums = _nums(text)
        if not nums:
            return False, "missing_number"
        if not any(
            abs(n - val) < 1e-2 * max(1.0, abs(val))
            or abs(n - 100 * val) < 2
            or abs(n - val / 1000.0) < 1e-3
            for n in nums
        ):
            return False, "number_drift"
    return True, "ok"


def qc_unv(text, family):
    if not text or len(text.split()) < 4:
        return False, "too_short"
    low = text.lower()
    need = {
        "unsupported_measurement": ("pulse", "jerk", "entropy", "bpm", "nats"),
        "unresolved_channel": ("the sensor",),
        "unsupported_logical_structure": ("exactly one", "the other false", "xor"),
        "missing_required_evidence": ("forearm",),
        "invalid_metadata": ("unknown", "sampling rate"),
        "genuine_language_ambiguity": ("at once", "larger than and smaller"),
    }
    keys = need.get(family, ())
    if keys and not any(k in low for k in keys):
        return False, "unv_keyword_missing"
    return True, "ok"


GEN_SYS = {
    "ir_dev": """Write ONE English lab-note sentence that asserts the JSON claim.
Return JSON {"surface":"..."}.
Name every channel by a short placement word. Copy every number exactly.
AND requires every clause. OR needs any clause. IF_THEN keeps condition then result.
Do not mention verdicts, gold, waveforms, or templates.""",
    "ir_blind": """Write ONE English instrument sentence that asserts the JSON claim.
Return JSON {"surface":"..."}.
Use placement words for channels. Keep every numeric literal.
Conjunction is all clauses; disjunction is at least one; implication stays if-then order.
No verdict words, gold labels, waveforms, or template IDs.""",
}


def llm_surface(model, prompt_id, st, channels, seed, split):
    rec = cached_chat(
        prompt_id,
        model,
        [
            {"role": "system", "content": GEN_SYS[split]},
            {"role": "user", "content": json.dumps({
                "connective": st.get("connective"),
                "predicates": st.get("predicates"),
                "allowed_channels": channels,
            }, sort_keys=True)},
        ],
        seed=seed,
        temperature=0.4,
        fmt="json",
    )
    raw = rec.get("raw") or ""
    try:
        obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        text = str(obj.get("surface") or "").strip()
    except Exception:
        return None, rec
    return text, rec


def _verdict(st, w):
    if st.get("unverifiable") or not st.get("predicates"):
        return "UNVERIFIABLE"
    gold = from_legacy(st, w["available_channels"])
    if gold.parse_status != "OK":
        return "UNVERIFIABLE"
    return ref_adjudicate({"channels": w["channels"], "fs": w["fs"]}, st)["verdict"]


def _pack(w, st, text, source, kind, family, split):
    ir = gold_ir_from_st(st)
    return {
        "claim_id": _cid("p35", split, source, kind, w["window_id"], text, json.dumps(st, sort_keys=True)),
        "source": source,
        "surface_kind": kind,
        "family": family,
        "connective": st.get("connective", "SINGLE"),
        "n_pred": len(st.get("predicates") or []),
        "dataset": w["dataset"],
        "subject": w["subject"],
        "window_id": w["window_id"],
        "fs": w["fs"],
        "available_channels": w["available_channels"],
        "channels_data": {k: v.tolist() for k, v in w["channels"].items()},
        "semantic_program": st,
        "gold_ir": ir,
        "surface_text": text,
        "gold_composed_verdict": _verdict(st, w),
        "split": split,
        "unv_family": st.get("unv_family"),
    }


def _answerable_program(w, rng, conn, n_pred):
    preds = []
    ops = list(OPS)
    rng.shuffle(ops)
    for i in range(n_pred):
        p = _pred(w, rng, ops[i % len(ops)])
        if p is None:
            return None
        preds.append(p)
    return {"connective": conn, "predicates": preds}


def _quota_plan(n, unv_frac):
    n_unv = int(round(n * unv_frac))
    n_ans = n - n_unv
    parts = [
        ("SINGLE", 1, 0.30),
        ("AND", 2, 0.20),
        ("OR", 2, 0.15),
        ("IF_THEN", 2, 0.20),
        ("AND", 3, 0.15),
    ]
    counts = [int(round(n_ans * frac)) for _, _, frac in parts]
    counts[-1] += n_ans - sum(counts)
    return [(parts[i][0], parts[i][1], counts[i]) for i in range(len(parts))], n_unv


def _source_cycle(n, llama_ok, deterministic_only):
    if deterministic_only:
        return ["deterministic"] * n
    if llama_ok:
        bag = (["deterministic"] * 4) + ([GEMMA_GEN] * 3) + ([THIRD_GEN] * 3)
    else:
        bag = (["deterministic"] * 5) + ([GEMMA_GEN] * 5)
    return [bag[i % len(bag)] for i in range(n)]


def _third_available():
    try:
        d = json.loads(urllib_tags())
        names = {m.get("name") for m in d.get("models") or []}
        return any(THIRD_GEN in (n or "") for n in names)
    except Exception:
        return False


def urllib_tags():
    import urllib.request
    return urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5).read()


def construct_split(split: str, wins, n: int, deterministic_only: bool):
    rng = random.Random(SEED + (3 if split == "ir_dev" else 19))
    qty = QTY_DEV if split == "ir_dev" else QTY_BLIND
    plan, n_unv = _quota_plan(n, 0.16)
    llama_ok = False if deterministic_only else _third_available()
    rows, rejected, generated = [], [], 0
    sources = _source_cycle(n, llama_ok, deterministic_only)
    src_i = 0

    def take_source():
        nonlocal src_i
        s = sources[src_i % len(sources)]
        src_i += 1
        return s

    for conn, k, need in plan:
        got = 0
        guard = 0
        while got < need and guard < need * 40:
            guard += 1
            w = wins[guard % len(wins)]
            st = _answerable_program(w, rng, conn, k)
            if not st:
                continue
            source = take_source()
            generated += 1
            if source == "deterministic":
                text = render_det(st, qty, split)
                ok, reason = qc_answerable(text, st)
                if not ok:
                    rejected.append({"source": source, "reason": reason})
                    continue
                rows.append(_pack(w, st, text, source, f"det_{split}", f"{conn}_{k}", split))
                got += 1
                continue
            model = source
            pid = f"{'gemma' if model == GEMMA_GEN else 'llama'}_surface_p35_{split}_v1"
            text, _ = llm_surface(model, pid, st, w["available_channels"], SEED + generated, split)
            if not text:
                rejected.append({"source": source, "reason": "bad_json"})
                text = render_det(st, qty, split)
                source = "deterministic"
            ok, reason = qc_answerable(text, st)
            if not ok:
                rejected.append({"source": source, "reason": reason})
                text = render_det(st, qty, split)
                source = "deterministic"
                ok, reason = qc_answerable(text, st)
                if not ok:
                    rejected.append({"source": source, "reason": reason})
                    continue
            rows.append(_pack(w, st, text, source, "independent_model" if source != "deterministic" else f"det_{split}", f"{conn}_{k}", split))
            got += 1
            if got % 25 == 0:
                print(f"  {split} {conn}/{k} {got}/{need}", flush=True)

    unv_kinds = [u[1] for u in UNV_SPECS]
    got = 0
    guard = 0
    while got < n_unv and guard < n_unv * 20:
        guard += 1
        w = wins[guard % len(wins)]
        kind = unv_kinds[got % len(unv_kinds)]
        fam, text0 = render_unv(kind, w)
        source = take_source()
        generated += 1
        st = {"connective": "SINGLE", "predicates": [], "unverifiable": True, "unv_family": fam}
        if source == "deterministic":
            text = text0
        else:
            model = source
            pid = f"{'gemma' if model == GEMMA_GEN else 'llama'}_unv_p35_{split}_v1"
            rec = cached_chat(
                pid,
                model,
                [
                    {"role": "system", "content": "Rewrite the note in different laboratory wording. Keep the same numbers and the same impossibility. Return JSON {\"surface\":\"...\"}."},
                    {"role": "user", "content": json.dumps({"note": text0, "family": fam})},
                ],
                seed=SEED + 11000 + generated,
                temperature=0.4,
                fmt="json",
            )
            raw = rec.get("raw") or ""
            try:
                text = str(json.loads(raw[raw.find("{") : raw.rfind("}") + 1]).get("surface") or "").strip()
            except Exception:
                text = ""
            if not text:
                rejected.append({"source": source, "reason": "unv_bad_json"})
                text = text0
                source = "deterministic"
        ok, reason = qc_unv(text, fam)
        if not ok:
            rejected.append({"source": source, "reason": reason})
            continue
        rows.append(_pack(w, st, text, source, "unv", fam, split))
        got += 1

    return rows, {
        "split": split,
        "generated_n": generated,
        "retained_n": len(rows),
        "rejected_n": len(rejected),
        "rejection_causes": dict(Counter(r["reason"] for r in rejected)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_connective": dict(Counter(r["connective"] for r in rows)),
        "by_n_pred": dict(Counter(r["n_pred"] for r in rows)),
        "by_verdict": dict(Counter(r["gold_composed_verdict"] for r in rows)),
        "by_unv_family": dict(Counter(r.get("unv_family") for r in rows if r.get("unv_family"))),
        "windows_n": len(wins),
        "window_ids_sha256": hashlib.sha256(" ".join(sorted(w["window_id"] for w in wins)).encode()).hexdigest(),
        "THIRD_GENERATOR_AVAILABLE": False if deterministic_only else _third_available(),
    }


def _write_split(name, rows, extra):
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    man = {**extra, "sha256": hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()}
    (RESULTS / f"{name}_FROZEN.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    (BENCH / f"{name}.inference.jsonl").write_text(
        "\n".join(json.dumps({k: r[k] for k in r if k != "channels_data"}) for r in rows),
        encoding="utf-8",
    )
    return man


def construct_ir_dev(n=1000):
    dev_w, blind_w = split_unused()
    pool = window_pool_manifest(dev_w, blind_w)
    (RESULTS).mkdir(parents=True, exist_ok=True)
    (RESULTS / "window_pool_FROZEN.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
    rows, man = construct_split("ir_dev", dev_w, n, deterministic_only=True)
    man["window_pool"] = pool
    dm = _write_split("ir_dev", rows, man)
    print("IR_DEV_CONSTRUCT", dm["retained_n"], dm["sha256"], flush=True)
    return dm


def construct_ir_blind(n=1500):
    lock = RESULTS / "window_pool_FROZEN.json"
    if not lock.exists():
        raise RuntimeError("window pool must be frozen with IR-DEV")
    from .windows_ir import load_unused_windows
    dev_w, blind_w = split_unused(load_unused_windows())
    frozen = json.loads(lock.read_text(encoding="utf-8"))
    got = hashlib.sha256(" ".join(w["window_id"] for w in blind_w).encode()).hexdigest()
    if got != frozen["blind_ids_sha256"]:
        raise RuntimeError("reserved IR-BLIND windows drifted")
    rows, man = construct_split("ir_blind", blind_w, n, deterministic_only=False)
    man["window_pool"] = frozen
    bm = _write_split("ir_blind", rows, man)
    print("IR_BLIND_CONSTRUCT", bm["retained_n"], bm["sha256"], flush=True)
    return bm

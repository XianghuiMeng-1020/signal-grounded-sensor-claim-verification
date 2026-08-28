"""SEM-DEV and SEM-BLIND. Built and frozen before any prompt-v3 edit."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p2r.validator import from_legacy
from p3.llm_chat import cached_chat
from p3.windows_p3 import unique_windows

from .config import BENCH, GEMMA_GEN, RESULTS, SEED, THIRD_GEN, OPS

QTY = {
    "dev": {
        "dominant_frequency": "spectral peak frequency",
        "rms_amplitude": "root-mean-square level",
        "peak_amplitude": "maximum excursion",
        "signal_range": "full-scale span",
        "trend_ratio": "late-to-early energy quotient",
        "periodicity_strength": "cyclic regularity index",
        "spectral_energy_ratio_low": "sub-3-hertz power share",
        "cross_channel_lag_ms": "inter-channel delay",
    },
    "blind": {
        "dominant_frequency": "leading Fourier peak",
        "rms_amplitude": "quadratic-mean amplitude",
        "peak_amplitude": "largest absolute deviation",
        "signal_range": "peak-to-trough width",
        "trend_ratio": "second-half versus first-half energy",
        "periodicity_strength": "repeatability coefficient",
        "spectral_energy_ratio_low": "low-band energy occupancy",
        "cross_channel_lag_ms": "cross-sensor offset",
    },
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


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _nm(ch):
    return ch.split("_")[0]


def _split_windows(wins):
    dev, blind = [], []
    for w in wins:
        h = int(hashlib.sha256(w["window_id"].encode()).hexdigest(), 16)
        (dev if h % 5 < 2 else blind).append(w)
    return dev, blind


def _meas(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            if len(chs) < 2:
                return None
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _pred(w, rng, op, mode):
    chs = list(w["available_channels"])
    used = chs[:2] if op == "cross_channel_lag_ms" else [rng.choice(chs)]
    if op == "cross_channel_lag_ms" and len(used) < 2:
        return None
    actual = _meas(op, used, w["channels"], w["fs"])
    if actual is None:
        return None
    if mode == "vs_value":
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


def _clause(pred, qty_map, style):
    op = pred["op"]
    qty, unit = qty_map[op], UNIT[op]
    pair = " and ".join(_nm(c) for c in pred["channels"])
    if pred["mode"] == "vs_value":
        val = float(pred["asserted_value"])
        table = {
            "colon": f"{qty} on {pair}: {val:.4g} {unit}.",
            "reads": f"{pair} reads a {qty} of {val:.4g} {unit}.",
            "logged": f"Logged {qty} for {pair} equals {val:.4g} {unit}.",
            "reports": f"The note reports {qty} = {val:.4g} {unit} at {pair}.",
            "holds": f"{qty} holds at {val:.4g} {unit} for {pair}.",
            "listed": f"{pair} listed {qty} {val:.4g} {unit}.",
        }
        return table.get(style, table["colon"])
    rel, thr = pred["relation"], float(pred["threshold"])
    word = "is greater than" if rel == "gt" else "is less than"
    alt = "exceeds" if rel == "gt" else "falls below"
    table = {
        "colon": f"{qty} on {pair} {word} {thr:.4g} {unit}.",
        "reads": f"{pair} {qty} {alt} {thr:.4g} {unit}.",
        "logged": f"Logged {qty} for {pair} {word} {thr:.4g} {unit}.",
        "reports": f"The note reports {qty} on {pair} {alt} {thr:.4g} {unit}.",
        "holds": f"{qty} {word} {thr:.4g} {unit} for {pair}.",
        "listed": f"{pair} listed {qty} {alt} {thr:.4g} {unit}.",
    }
    return table.get(style, table["colon"])


def render_det(st, qty_map, style, split):
    preds = st["predicates"]
    conn = st.get("connective", "SINGLE")
    bits = [_clause(p, qty_map, style) for p in preds]
    if conn == "SINGLE":
        return bits[0]
    if split == "dev":
        if conn == "AND" and len(bits) == 2:
            return "Joint requirement. " + " In addition, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "AND" and len(bits) == 3:
            return "Three concurrent claims. " + " Next, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "OR":
            return "Either clause is enough. " + " Failing that, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "IF_THEN" and len(bits) == 2:
            return f"Whenever {bits[0].rstrip('.')}, it follows that {bits[1]}"
    else:
        if conn == "AND" and len(bits) == 2:
            return "Both of the following are claimed. " + " Likewise, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "AND" and len(bits) == 3:
            return "A three-part conjunction. " + " Then, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "OR":
            return "Satisfy at least one. " + " Otherwise, ".join(b.rstrip(".") for b in bits) + "."
        if conn == "IF_THEN" and len(bits) == 2:
            return f"Antecedent: {bits[0].rstrip('.')}. Consequent: {bits[1]}"
    return " ".join(bits)


UNV_SPECS = (
    ("unsupported_measurement", "heart_rate",
     "Heart-rate on {ch} is stated as 72 beats per minute."),
    ("unsupported_measurement", "jerk_entropy",
     "Jerk entropy of {ch} exceeds 4.1 nats."),
    ("unresolved_channel", "the_sensor",
     "The sensor root-mean-square level equals 3.40 raw units."),
    ("unsupported_logical_structure", "xor",
     "Exactly one of these holds, never both: {ch} RMS exceeds 1.0 raw units; {ch2} peak-to-trough width exceeds 2.0 raw units."),
    ("missing_required_evidence", "absent_channel",
     "Wrist-mounted accel RMS equals 2.10 raw units."),
    ("invalid_metadata", "unknown_fs",
     "Sampling rate is unknown; {ch} spectral peak frequency is 2.50 Hz."),
    ("genuine_language_ambiguity", "both_directions",
     "{ch} quadratic-mean amplitude is simultaneously greater than and less than 1.80 raw units."),
)


def render_unv(kind, w):
    chs = list(w["available_channels"])
    ch, ch2 = _nm(chs[0]), _nm(chs[-1])
    for fam, key, tmpl in UNV_SPECS:
        if key == kind:
            return fam, tmpl.format(ch=ch, ch2=ch2)
    return "unsupported_measurement", f"Heart-rate on {ch} is 80 bpm."


def _nums(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]


def qc_answerable(text, st):
    if not text or len(text.split()) < 5:
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
    if not text or len(text.split()) < 5:
        return False, "too_short"
    low = text.lower()
    need = {
        "unsupported_measurement": ("heart", "jerk", "entropy", "bpm", "nats"),
        "unresolved_channel": ("the sensor",),
        "unsupported_logical_structure": ("exactly one", "never both", "xor"),
        "missing_required_evidence": ("wrist",),
        "invalid_metadata": ("unknown", "sampling rate"),
        "genuine_language_ambiguity": ("simultaneously", "greater than and less than"),
    }
    keys = need.get(family, ())
    if keys and not any(k in low for k in keys):
        return False, "unv_keyword_missing"
    return True, "ok"


GEN_SYS = {
    "dev": """Write ONE English instrument-log sentence that asserts the JSON claim.
Return JSON {"surface":"..."}.
Name every channel with a short placement word.
Copy every numeric literal exactly.
AND = every clause is required. OR = any clause suffices. IF_THEN = condition then consequence, same order.
Mention every predicate. Do not mention verdicts, gold, waveforms, or templates.""",
    "blind": """Write ONE English bench-notebook sentence that asserts the JSON claim.
Return JSON {"surface":"..."}.
Use placement words for every named channel.
Reproduce each number unchanged.
Conjunction means all clauses; disjunction means at least one; implication is if-then in listed order.
Include every predicate. No verdict words, no gold, no waveform talk, no template IDs.""",
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
    return {
        "claim_id": _cid("p3cr", split, source, kind, w["window_id"], text, json.dumps(st, sort_keys=True)),
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
        op = ops[i % len(ops)]
        mode = "vs_value" if rng.random() < 0.55 or op == "cross_channel_lag_ms" and rng.random() < 0.5 else "vs_threshold"
        if n_pred > 1 and i == 0:
            mode = "vs_value"
        p = _pred(w, rng, op, mode if op != "cross_channel_lag_ms" or mode == "vs_value" or n_pred == 1 else "vs_value")
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


def _source_cycle(n, llama_ok):
    if llama_ok:
        bag = (["deterministic"] * 4) + ([GEMMA_GEN] * 3) + ([THIRD_GEN] * 3)
    else:
        bag = (["deterministic"] * 5) + ([GEMMA_GEN] * 5)
    out = []
    i = 0
    while len(out) < n:
        out.append(bag[i % len(bag)])
        i += 1
    return out


def _third_available():
    try:
        import json as _j
        import urllib.request
        d = _j.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5).read())
        names = {m.get("name") for m in d.get("models") or []}
        return any(THIRD_GEN in (n or "") for n in names)
    except Exception:
        return False


def construct_split(split: str, wins, n: int, llama_ok: bool):
    rng = random.Random(SEED + (1 if split == "dev" else 17))
    qty = QTY[split]
    styles = ("colon", "reads", "logged") if split == "dev" else ("reports", "holds", "listed")
    plan, n_unv = _quota_plan(n, 0.18)
    rows, rejected, generated = [], [], 0
    sources = _source_cycle(n, llama_ok)
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
                text = render_det(st, qty, rng.choice(styles), split)
                ok, reason = qc_answerable(text, st)
                if not ok:
                    rejected.append({"source": source, "reason": reason})
                    continue
                rows.append(_pack(w, st, text, source, f"det_{split}_{styles[0]}", f"{conn}_{k}", split))
                got += 1
                continue
            model = source
            pid = f"{'gemma' if model == GEMMA_GEN else 'llama'}_surface_p3cr_{split}_v1"
            text, meta = llm_surface(model, pid, st, w["available_channels"], SEED + generated, split)
            if not text:
                rejected.append({"source": source, "reason": "bad_json"})
                text = render_det(st, qty, rng.choice(styles), split)
                source = "deterministic"
            ok, reason = qc_answerable(text, st)
            if not ok:
                rejected.append({"source": source, "reason": reason})
                text = render_det(st, qty, rng.choice(styles), split)
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
            pid = f"{'gemma' if model == GEMMA_GEN else 'llama'}_unv_p3cr_{split}_v1"
            rec = cached_chat(
                pid,
                model,
                [
                    {"role": "system", "content": "Rewrite the note in different laboratory wording. Keep the same numbers and the same impossibility. Return JSON {\"surface\":\"...\"}."},
                    {"role": "user", "content": json.dumps({"note": text0, "family": fam})},
                ],
                seed=SEED + 9000 + generated,
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

    causes = Counter(r["reason"] for r in rejected)
    return rows, {
        "split": split,
        "generated_n": generated,
        "retained_n": len(rows),
        "rejected_n": len(rejected),
        "rejection_causes": dict(causes),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_connective": dict(Counter(r["connective"] for r in rows)),
        "by_n_pred": dict(Counter(r["n_pred"] for r in rows)),
        "by_verdict": dict(Counter(r["gold_composed_verdict"] for r in rows)),
        "by_unv_family": dict(Counter(r.get("unv_family") for r in rows if r.get("unv_family"))),
        "windows_n": len(wins),
        "window_ids_sha256": hashlib.sha256(" ".join(sorted(w["window_id"] for w in wins)).encode()).hexdigest(),
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


def construct(n_dev=1080, n_blind=1620):
    wins = unique_windows()
    dev_w, blind_w = _split_windows(wins)
    assert not {w["window_id"] for w in dev_w} & {w["window_id"] for w in blind_w}
    llama_ok = _third_available()
    print("P3CR windows", len(dev_w), len(blind_w), "llama", llama_ok, flush=True)
    drows, dman = construct_split("dev", dev_w, n_dev, llama_ok)
    brows, bman = construct_split("blind", blind_w, n_blind, llama_ok)
    dman["THIRD_GENERATOR_AVAILABLE"] = llama_ok
    bman["THIRD_GENERATOR_AVAILABLE"] = llama_ok
    dman["third_generator"] = THIRD_GEN if llama_ok else "THIRD_GENERATOR_NOT_AVAILABLE"
    bman["third_generator"] = THIRD_GEN if llama_ok else "THIRD_GENERATOR_NOT_AVAILABLE"
    dm = _write_split("sem_dev", drows, dman)
    bm = _write_split("sem_blind", brows, bman)
    overlap = {r["claim_id"] for r in drows} & {r["claim_id"] for r in brows}
    texts = {r["surface_text"] for r in drows} & {r["surface_text"] for r in brows}
    lock = {
        "dev_sha256": dm["sha256"],
        "blind_sha256": bm["sha256"],
        "claim_id_overlap": len(overlap),
        "surface_text_overlap": len(texts),
        "window_overlap": 0,
        "SEM_BLIND_SEALED_BEFORE_PROMPT_REPAIR": True,
    }
    (RESULTS / "sem_construction_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("SEM_CONSTRUCT", {"dev": dm["retained_n"], "blind": bm["retained_n"], "lock": lock}, flush=True)
    return dm, bm

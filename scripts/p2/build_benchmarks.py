"""Phase 3–7 benchmark construction.

Preserves pilot_v1. Writes independent_gold_v2 with grouped splits.
Inference payloads contain no gold / family / template IDs.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    BENCH_P2,
    EVALUABLE_SPLITS,
    HOLDOUT_NAME,
    MAX_WINDOWS_PER_SPLIT,
    MARGIN_BANDS,
    PILOT_V1_BENCH,
    PRIMITIVE_NAMES,
    RESULTS_P2,
    SEED,
    UNVERIFIABLE_FAMILIES,
)
from .independent_adjudicator import adjudicate, normalized_margin
from .independent_dsp import MeasurementError, measure, tolerance_for
from .language_realizations import realize
from .windows import audit_leakage, dump_window_manifest, load_all_windows


def _rid(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def _rep(window: dict) -> dict:
    return {"channels": window["channels"], "fs": window["fs"]}


def _val(window: dict, op: str, ch: str) -> float:
    return measure(op, {ch: window["channels"][ch]}, window["fs"])


def _chs(window: dict) -> list[str]:
    return list(window["channels"].keys())


def _inference_view(claim: dict) -> dict:
    return {
        "claim_id": claim["claim_id"],
        "surface_text": claim["surface_text"],
        "available_channels": claim["available_channels"],
        "fs": claim["fs"],
    }


def _gold_view(claim: dict) -> dict:
    keys = (
        "claim_id", "benchmark_version", "split", "source_dataset", "source_window_id",
        "subject", "activity", "session", "window_index", "fs", "available_channels",
        "semantic_program", "surface_text", "surface_style", "primitive", "channels",
        "threshold_or_value", "reference_measurement", "gold_predicate_truth",
        "gold_composed_verdict", "provenance", "generation_family",
        "paraphrase_family_id", "margin", "margin_band", "connective",
        "unverifiable_family",
    )
    return {k: claim.get(k) for k in keys}


def relabel_pilot_v1() -> dict:
    """Recompute independent gold on preserved R8 claims. Do not overwrite the file."""
    bench = json.loads(PILOT_V1_BENCH.read_text(encoding="utf-8"))
    rows = []
    flips = 0
    for item in bench:
        rep = {"channels": item["channels"], "fs": item["fs"]}
        structure = item["gt_structure"]
        ref = adjudicate(rep, structure)
        prod_label = item.get("gt_verdict")
        flip = ref["verdict"] != prod_label
        flips += int(flip)
        rows.append({
            "id": item["id"],
            "split": item["split"],
            "dataset": item["dataset"],
            "subject": item["subject"],
            "production_gold": prod_label,
            "independent_gold": ref["verdict"],
            "flip": flip,
            "predicate_truths": ref["predicate_truths"],
            "evidence": ref["evidence"],
            "structure": structure,
            "sentence": item["sentence"],
        })
    summary = {
        "n": len(rows),
        "n_flips": flips,
        "flip_rate": flips / len(rows) if rows else None,
        "by_split": {},
    }
    for split in ("DEV", "HELDOUT", "ADVERSARIAL_CORRECT"):
        sub = [r for r in rows if r["split"] == split]
        f = sum(r["flip"] for r in sub)
        summary["by_split"][split] = {"n": len(sub), "n_flips": f}
    BENCH_P2.mkdir(parents=True, exist_ok=True)
    (BENCH_P2 / "pilot_v1_independent_relabel.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    (RESULTS_P2 / "pilot_v1_relabel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _pack_claim(window, structure, family, surface, idx, extra=None) -> dict:
    ref = adjudicate(_rep(window), structure)
    preds = structure["predicates"]
    prims = [p.get("op") for p in preds]
    chans = []
    for p in preds:
        chans.extend(p.get("channels") or [])
        if p.get("compare_channel"):
            chans.append(p["compare_channel"])
    values = []
    for p in preds:
        if "asserted_value" in p:
            values.append(p.get("asserted_value"))
        if "threshold" in p:
            values.append(p.get("threshold"))
    measurements = []
    for ev in ref["evidence"]:
        if "actual" in ev:
            measurements.append(ev["actual"])
        if "a" in ev:
            measurements.append({"a": ev["a"], "b": ev.get("b")})
    family_id = _rid(json.dumps(structure, sort_keys=True), window["window_id"])
    claim_id = _rid(family_id, surface["paraphrase_index"], surface["surface_style"], idx)
    margin = normalized_margin(_rep(window), structure)
    band = None
    if margin is not None:
        for name, lo, hi in MARGIN_BANDS:
            if lo <= margin < hi:
                band = name
                break
    rec = {
        "claim_id": claim_id,
        "benchmark_version": "independent_gold_v2",
        "split": window["split"],
        "source_dataset": window["dataset"],
        "source_window_id": window["window_id"],
        "subject": window["subject"],
        "activity": window.get("activity"),
        "session": window["session"],
        "window_index": window["window_index"],
        "fs": window["fs"],
        "available_channels": _chs(window),
        "channels_data": window["channels"],
        "semantic_program": structure,
        "surface_text": surface["surface_text"],
        "surface_style": surface["surface_style"],
        "primitive": prims[0] if len(prims) == 1 else prims,
        "channels": list(dict.fromkeys(chans)),
        "threshold_or_value": values,
        "reference_measurement": measurements,
        "gold_predicate_truth": ref["predicate_truths"],
        "gold_composed_verdict": ref["verdict"],
        "provenance": "independent_reference_dsp+independent_adjudicator",
        "generation_family": family,
        "paraphrase_family_id": family_id,
        "margin": margin,
        "margin_band": band,
        "connective": structure.get("connective"),
        "unverifiable_family": None,
    }
    if extra:
        rec.update(extra)
    return rec


def _make_single_vs_value(window, op, force_false, rng) -> dict | None:
    chs = _chs(window)
    try:
        if op == "cross_channel_lag_ms":
            if len(chs) < 2:
                return None
            actual = measure(op, {chs[0]: window["channels"][chs[0]], chs[1]: window["channels"][chs[1]]}, window["fs"])
            used = [chs[0], chs[1]]
        else:
            ch = rng.choice(chs)
            actual = _val(window, op, ch)
            used = [ch]
    except MeasurementError:
        return None
    tol = tolerance_for(op, actual)
    if force_false:
        claimed = actual + rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0) * tol
    else:
        claimed = actual + rng.uniform(-0.55, 0.55) * tol
    structure = {
        "connective": "SINGLE",
        "predicates": [{"op": op, "channels": used, "mode": "vs_value", "asserted_value": float(claimed)}],
    }
    return structure


def _make_vs_channel(window, op, rng) -> dict | None:
    chs = _chs(window)
    if len(chs) < 2:
        return None
    a, b = chs[0], chs[1]
    try:
        va, vb = _val(window, op, a), _val(window, op, b)
    except MeasurementError:
        return None
    if op == "rms_amplitude" and rng.random() < 0.4:
        rel = "similar" if abs(va - vb) < 0.25 * max(abs(va), abs(vb), 1e-9) else "different"
    else:
        rel = "gt" if va > vb else "lt"
    return {
        "connective": "SINGLE",
        "predicates": [{
            "op": op, "channels": [a], "mode": "vs_channel",
            "compare_channel": b, "relation": rel,
        }],
    }


def _make_vs_threshold(window, op, thr, rng) -> dict | None:
    ch = _chs(window)[0]
    try:
        va = _val(window, op, ch)
    except MeasurementError:
        return None
    rel = "gt" if va > thr else "lt"
    return {
        "connective": "SINGLE",
        "predicates": [{
            "op": op, "channels": [ch], "mode": "vs_threshold",
            "threshold": float(thr), "relation": rel,
        }],
    }


def _compose(window, ops, connective, rng, force_false=False) -> dict | None:
    preds = []
    for op in ops:
        st = _make_vs_channel(window, op, rng)
        if st is None:
            return None
        preds.append(st["predicates"][0])
    if force_false and connective in ("AND", "OR", "IF_THEN"):
        # flip one asserted relation so composition can be false
        preds[0] = dict(preds[0])
        opp = {"gt": "lt", "lt": "gt", "similar": "different", "different": "similar"}
        if preds[0]["relation"] in opp:
            preds[0]["relation"] = opp[preds[0]["relation"]]
    return {"connective": connective, "predicates": preds}


def _margin_claim(window, op, band_name, lo, hi, rng) -> dict | None:
    """Construct a vs_threshold claim whose normalized |m-thr| falls in [lo, hi)."""
    ch = rng.choice(_chs(window))
    try:
        actual = _val(window, op, ch)
    except MeasurementError:
        return None
    denom = max(abs(actual), 1e-3)
    mid = 0.5 * (lo + (hi if hi < 10 else min(hi, 0.25)))
    if band_name == "clear":
        mid = 0.18
    # place threshold on either side
    sign = rng.choice([-1.0, 1.0])
    thr = actual - sign * mid * denom
    # relation chosen from true comparison
    rel = "gt" if actual > thr else "lt"
    # optionally flip for CONTRADICTED items
    if rng.random() < 0.4:
        rel = "lt" if rel == "gt" else "gt"
    return {
        "connective": "SINGLE",
        "predicates": [{
            "op": op, "channels": [ch], "mode": "vs_threshold",
            "threshold": float(thr), "relation": rel,
        }],
    }


def _unverifiable_claims(window, rng) -> list[dict]:
    chs = _chs(window)
    out = []

    def _nm(c):
        return c.split("_")[0]

    a0 = _nm(chs[0])
    family_specs = {
        "unsupported_measurement_type":
            "Jerk entropy of the recording exceeds 4.2 nats.",
        "unavailable_channel":
            "The gyroscope z-axis has RMS amplitude of approximately 1.20 raw units.",
        "missing_channel":
            "The missing magnetometer channel is more periodic than the accelerometer.",
        "insufficient_length":
            "Dominant frequency cannot be trusted because the window has only three samples.",
        "invalid_sampling_rate":
            "The dominant frequency is 4.0 Hz but the sampling rate metadata is unknown.",
        "ambiguous_channel_reference":
            "The sensor has higher RMS than the other sensor.",
        "ambiguous_comparator":
            f"The {a0} channel RMS is either greater or smaller than 2.0.",
        "unsupported_physiological_proxy":
            "Based on this window, heart rate was likely above 130 bpm.",
        "qualitative_no_executable_definition":
            "The movement looks vigorous and athletic.",
        "unsupported_logical_nesting":
            f"If the {a0} channel is periodic, then (the RMS is high and (the peak is large or the range is small)).",
        "too_many_predicates":
            f"The {a0} channel has high RMS, large range, rising energy, and strong periodicity, and also a 5 Hz dominant frequency.",
        "corrupted_or_missing_evidence":
            "The file is corrupt; nonetheless the chest RMS is 9.8 raw units.",
    }

    for fam, text in family_specs.items():
        structure = {"connective": "SINGLE", "predicates": []}
        # gold is UNVERIFIABLE by construction (unsupported evidence), not by production DSP
        family_id = _rid("unv", fam, window["window_id"])
        claim_id = _rid(family_id, text)
        out.append({
            "claim_id": claim_id,
            "benchmark_version": "independent_gold_v2",
            "split": window["split"],
            "source_dataset": window["dataset"],
            "source_window_id": window["window_id"],
            "subject": window["subject"],
            "activity": window.get("activity"),
            "session": window["session"],
            "window_index": window["window_index"],
            "fs": window["fs"],
            "available_channels": chs,
            "channels_data": window["channels"],
            "semantic_program": structure,
            "surface_text": text,
            "surface_style": "unverifiable",
            "primitive": None,
            "channels": [],
            "threshold_or_value": [],
            "reference_measurement": None,
            "gold_predicate_truth": [],
            "gold_composed_verdict": "UNVERIFIABLE",
            "provenance": "construction_rule:required_evidence_not_available",
            "generation_family": "UNVERIFIABLE",
            "paraphrase_family_id": family_id,
            "margin": None,
            "margin_band": None,
            "connective": "SINGLE",
            "unverifiable_family": fam,
        })
    return out


def _cap_windows(windows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for w in windows:
        by[w["split"]].append(w)
    capped = []
    rng = random.Random(SEED)
    for split, rows in by.items():
        by_ds = defaultdict(list)
        for w in rows:
            by_ds[w["dataset"]].append(w)
        chosen = []
        # round-robin datasets
        queues = {ds: rng.sample(v, k=len(v)) for ds, v in by_ds.items()}
        while len(chosen) < MAX_WINDOWS_PER_SPLIT and any(queues.values()):
            for ds in ("PAMAP2", "WISDM", "MHEALTH"):
                if queues.get(ds):
                    chosen.append(queues[ds].pop())
                if len(chosen) >= MAX_WINDOWS_PER_SPLIT:
                    break
        capped.extend(chosen)
    return capped


def build_independent_gold_v2() -> dict:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    windows = load_all_windows()
    leak = audit_leakage(windows)
    BENCH_P2.mkdir(parents=True, exist_ok=True)
    dump_window_manifest(windows, BENCH_P2 / "window_manifest_all.json")
    (RESULTS_P2 / "split_leakage_audit.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    windows = _cap_windows(windows)
    leak_capped = audit_leakage(windows)
    (RESULTS_P2 / "split_leakage_audit_capped.json").write_text(json.dumps(leak_capped, indent=2), encoding="utf-8")

    claims: list[dict] = []
    composed_pairs = [
        ("AND", ("rms_amplitude", "dominant_frequency")),
        ("OR", ("signal_range", "peak_amplitude")),
        ("IF_THEN", ("periodicity_strength", "spectral_energy_ratio_low")),
        ("AND", ("spectral_energy_ratio_low", "rms_amplitude")),
    ]
    idx = 0
    for w in windows:
        # singles cycling primitives
        for k, op in enumerate(PRIMITIVE_NAMES):
            if k % 2 != (w["window_index"] % 2):
                continue
            st = _make_single_vs_value(w, op, force_false=(idx % 3 == 0), rng=rng)
            if not st:
                continue
            for surf in realize(st, w["split"]):
                claims.append(_pack_claim(w, st, "SINGLE_VS_VALUE", surf, idx))
                idx += 1
        # one vs_channel
        op = PRIMITIVE_NAMES[(w["window_index"] + 1) % 5]
        st = _make_vs_channel(w, op, rng)
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_claim(w, st, "SINGLE_VS_CHANNEL", surf, idx))
                idx += 1
        # one vs_threshold
        st = _make_vs_threshold(w, "trend_ratio", 1.0, rng)
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_claim(w, st, "SINGLE_VS_THRESHOLD", surf, idx))
                idx += 1
        # compositions
        conn, ops = composed_pairs[w["window_index"] % len(composed_pairs)]
        st = _compose(w, ops, conn, rng, force_false=(idx % 2 == 0))
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_claim(w, st, f"COMPOSE_{conn}", surf, idx))
                idx += 1
        # margin strata (answerable)
        if w["split"] in EVALUABLE_SPLITS or w["split"] == HOLDOUT_NAME:
            band_name, lo, hi = MARGIN_BANDS[w["window_index"] % len(MARGIN_BANDS)]
            st = _margin_claim(w, "rms_amplitude", band_name, lo, hi, rng)
            if st:
                for surf in realize(st, w["split"])[: max(1, 2 if w["split"] == "challenge" else 1)]:
                    rec = _pack_claim(w, st, "MARGIN", surf, idx, extra={"intended_margin_band": band_name})
                    claims.append(rec)
                    idx += 1
        # unverifiable: one family per window, cycle families (plus full set on a few)
        fams = list(UNVERIFIABLE_FAMILIES)
        chosen_unv = _unverifiable_claims(w, rng)
        pick = chosen_unv[w["window_index"] % len(chosen_unv)]
        claims.append(pick)
        idx += 1
        if w["window_index"] == 0:
            claims.extend(chosen_unv)

    # write per-split files
    by_split = defaultdict(list)
    for c in claims:
        by_split[c["split"]].append(c)

    manifests = {}
    for split, rows in by_split.items():
        split_dir = BENCH_P2 / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        gold_path = split_dir / f"{split}.gold.jsonl"
        inf_path = split_dir / f"{split}.inference.jsonl"
        with gold_path.open("w", encoding="utf-8") as g, inf_path.open("w", encoding="utf-8") as inf:
            for rec in rows:
                g.write(json.dumps(_gold_record_with_signal(rec), ensure_ascii=False) + "\n")
                inf.write(json.dumps(_inference_view(rec), ensure_ascii=False) + "\n")
        digest = _file_sha256(gold_path)
        manifests[split] = {
            "n": len(rows),
            "gold_path": str(gold_path.relative_to(BENCH_P2.parent.parent)),
            "inference_path": str(inf_path.relative_to(BENCH_P2.parent.parent)),
            "sha256_gold": digest,
            "n_unverifiable": sum(1 for r in rows if r["gold_composed_verdict"] == "UNVERIFIABLE"),
            "n_supported": sum(1 for r in rows if r["gold_composed_verdict"] == "SUPPORTED"),
            "n_contradicted": sum(1 for r in rows if r["gold_composed_verdict"] == "CONTRADICTED"),
            "datasets": sorted({r["source_dataset"] for r in rows}),
            "connectives": sorted({r["connective"] for r in rows if r["connective"]}),
        }

    hold = manifests.get(HOLDOUT_NAME, {})
    holdout_note = {
        "status": "SEALED",
        "evaluated_in_p2": False,
        "sha256_gold": hold.get("sha256_gold"),
        "n": hold.get("n"),
        "instruction": "Do not open for metric computation during P2.",
    }
    (BENCH_P2 / "splits" / "final_sealed_holdout.MANIFEST.json").write_text(
        json.dumps(holdout_note, indent=2), encoding="utf-8"
    )
    (BENCH_P2 / "splits" / "final_sealed_holdout.sha256").write_text(
        (hold.get("sha256_gold") or "") + "\n", encoding="utf-8"
    )
    (BENCH_P2 / "SPLIT_MANIFEST.json").write_text(json.dumps({
        "seed": SEED,
        "leakage_all_windows": leak,
        "leakage_capped": leak_capped,
        "splits": manifests,
        "holdout_sealed": True,
    }, indent=2), encoding="utf-8")

    # also a compact stats file
    stats = {s: manifests[s] for s in manifests}
    (RESULTS_P2 / "benchmark_sizes.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"n_claims": len(claims), "splits": stats, "leakage": leak_capped}


def _gold_record_with_signal(rec: dict) -> dict:
    g = _gold_view(rec)
    g["channels_data"] = rec["channels_data"]
    if "intended_margin_band" in rec:
        g["intended_margin_band"] = rec["intended_margin_band"]
    return g


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

"""independent_gold_v3_selfcontained. Does not overwrite v2."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .build_benchmarks import (
    _cap_windows,
    _chs,
    _compose,
    _file_sha256,
    _gold_record_with_signal,
    _gold_view,
    _inference_view,
    _make_single_vs_value,
    _make_vs_channel,
    _make_vs_threshold,
    _margin_claim,
    _pack_claim,
    _rid,
    _unverifiable_claims,
)
from .config import (
    BENCH_P2,
    BENCH_P2_V3,
    BENCHMARK_VERSION_V3,
    EVALUABLE_SPLITS,
    HOLDOUT_NAME,
    MARGIN_BANDS,
    PRIMITIVE_NAMES,
    SEED,
)
from .language_realizations_v3 import realize
from .windows import audit_leakage, dump_window_manifest, load_all_windows


def _pack_v3(window, structure, family, surface, idx, extra=None) -> dict:
    rec = _pack_claim(window, structure, family, surface, idx, extra=extra)
    rec["benchmark_version"] = BENCHMARK_VERSION_V3
    return rec


def _unverifiable_v3(window, rng) -> list[dict]:
    rows = _unverifiable_claims(window, rng)
    for r in rows:
        r["benchmark_version"] = BENCHMARK_VERSION_V3
    return rows


def mark_old_holdout_superseded() -> None:
    """Do not read old holdout gold/inference bodies."""
    note = {
        "status": "SUPERSEDED_UNEVALUATED_DUE_TO_SELF_CONTAINMENT_DEFECT",
        "evaluated": False,
        "legacy_manifest": "benchmarks/p2/splits/final_sealed_holdout.MANIFEST.json",
        "legacy_sha256_file": "benchmarks/p2/splits/final_sealed_holdout.sha256",
        "legacy_n": 1368,
        "legacy_sha256_gold": "bd52870f5a58c07b56800c65cc884bb32745725b3674710757155e75ba772a2a",
        "instruction": "Do not open. Do not evaluate. Preserved in place.",
    }
    path = BENCH_P2 / "splits" / "final_sealed_holdout.SUPERSEDED.json"
    path.write_text(json.dumps(note, indent=2), encoding="utf-8")


def build_independent_gold_v3() -> dict:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    windows = load_all_windows()
    leak = audit_leakage(windows)
    BENCH_P2_V3.mkdir(parents=True, exist_ok=True)
    dump_window_manifest(windows, BENCH_P2_V3 / "window_manifest_all.json")
    windows = _cap_windows(windows)
    leak_capped = audit_leakage(windows)

    claims: list[dict] = []
    composed_pairs = [
        ("AND", ("rms_amplitude", "dominant_frequency")),
        ("OR", ("signal_range", "peak_amplitude")),
        ("IF_THEN", ("periodicity_strength", "spectral_energy_ratio_low")),
        ("AND", ("spectral_energy_ratio_low", "rms_amplitude")),
    ]
    idx = 0
    for w in windows:
        for k, op in enumerate(PRIMITIVE_NAMES):
            if k % 2 != (w["window_index"] % 2):
                continue
            st = _make_single_vs_value(w, op, force_false=(idx % 3 == 0), rng=rng)
            if not st:
                continue
            for surf in realize(st, w["split"]):
                claims.append(_pack_v3(w, st, "SINGLE_VS_VALUE", surf, idx))
                idx += 1
        op = PRIMITIVE_NAMES[(w["window_index"] + 1) % 5]
        st = _make_vs_channel(w, op, rng)
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_v3(w, st, "SINGLE_VS_CHANNEL", surf, idx))
                idx += 1
        st = _make_vs_threshold(w, "trend_ratio", 1.0, rng)
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_v3(w, st, "SINGLE_VS_THRESHOLD", surf, idx))
                idx += 1
        conn, ops = composed_pairs[w["window_index"] % len(composed_pairs)]
        st = _compose(w, ops, conn, rng, force_false=(idx % 2 == 0))
        if st:
            for surf in realize(st, w["split"]):
                claims.append(_pack_v3(w, st, f"COMPOSE_{conn}", surf, idx))
                idx += 1
        if w["split"] in EVALUABLE_SPLITS or w["split"] == HOLDOUT_NAME:
            band_name, lo, hi = MARGIN_BANDS[w["window_index"] % len(MARGIN_BANDS)]
            st = _margin_claim(w, "rms_amplitude", band_name, lo, hi, rng)
            if st:
                for surf in realize(st, w["split"])[: max(1, 2 if w["split"] == "challenge" else 1)]:
                    rec = _pack_v3(w, st, "MARGIN", surf, idx, extra={"intended_margin_band": band_name})
                    claims.append(rec)
                    idx += 1
        chosen_unv = _unverifiable_v3(w, rng)
        pick = chosen_unv[w["window_index"] % len(chosen_unv)]
        claims.append(pick)
        idx += 1
        if w["window_index"] == 0:
            claims.extend(chosen_unv)

    by_split = defaultdict(list)
    for c in claims:
        by_split[c["split"]].append(c)

    manifests = {}
    split_dir = BENCH_P2_V3 / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        gold_path = split_dir / f"{split}.gold.jsonl"
        inf_path = split_dir / f"{split}.inference.jsonl"
        with gold_path.open("w", encoding="utf-8") as g, inf_path.open("w", encoding="utf-8") as inf:
            for rec in rows:
                g.write(json.dumps(_gold_record_with_signal(rec), ensure_ascii=False) + "\n")
                inf.write(json.dumps(_inference_view(rec), ensure_ascii=False) + "\n")
        digest = _file_sha256(gold_path)
        manifests[split] = {
            "n": len(rows),
            "gold_path": str(gold_path.relative_to(BENCH_P2_V3.parent.parent)),
            "inference_path": str(inf_path.relative_to(BENCH_P2_V3.parent.parent)),
            "sha256_gold": digest,
            "n_unverifiable": sum(1 for r in rows if r["gold_composed_verdict"] == "UNVERIFIABLE"),
            "n_supported": sum(1 for r in rows if r["gold_composed_verdict"] == "SUPPORTED"),
            "n_contradicted": sum(1 for r in rows if r["gold_composed_verdict"] == "CONTRADICTED"),
            "datasets": sorted({r["source_dataset"] for r in rows}),
            "connectives": sorted({r.get("connective") for r in rows if r.get("connective")}),
        }

    hold = manifests.get(HOLDOUT_NAME, {})
    holdout_note = {
        "status": "SEALED",
        "benchmark_version": BENCHMARK_VERSION_V3,
        "evaluated_in_p2r_lm1": False,
        "sha256_gold": hold.get("sha256_gold"),
        "n": hold.get("n"),
        "instruction": "Do not open for metric computation during P2R-LM1.",
    }
    (split_dir / "final_sealed_holdout.MANIFEST.json").write_text(
        json.dumps(holdout_note, indent=2), encoding="utf-8"
    )
    (split_dir / "final_sealed_holdout.sha256").write_text((hold.get("sha256_gold") or "") + "\n", encoding="utf-8")
    (BENCH_P2_V3 / "SPLIT_MANIFEST.json").write_text(json.dumps({
        "seed": SEED,
        "benchmark_version": BENCHMARK_VERSION_V3,
        "leakage_all_windows": leak,
        "leakage_capped": leak_capped,
        "splits": manifests,
        "holdout_sealed": True,
    }, indent=2), encoding="utf-8")
    mark_old_holdout_superseded()
    return {"n_claims": len(claims), "splits": manifests, "leakage": leak_capped}


if __name__ == "__main__":
    print(json.dumps(build_independent_gold_v3(), indent=2, default=str))

"""Module 2 — signed margin replay and frozen k-ladder. Thresholds not tuned."""
from __future__ import annotations

import json
import numpy as np

from f_round6_operators import compute as prod_compute
from p2r.pipeline import run_oracle
from p2_phase2.config import PERTURBATIONS
from p2_phase2.construct import _program
from p2_phase2.degrade import apply

from .config import E1_AWGN10_SUPPORTED, E1_N, EXPERIMENT_IDS, MARGIN_K, RESULTS, SEED
from .windows import e1_items


def _verdict(prog, item, channels) -> str:
    return run_oracle(prog, item["available_channels"], item["fs"], channels)["verdict"]


def _measure(item, channels) -> float | None:
    named = {n: channels[n] for n in item["named_channels"]}
    try:
        v = float(prod_compute(item["op"], named, item["fs"]))
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _replay_e1(items: list[dict]) -> list[dict]:
    recs = []
    for it in items:
        clean_v = _measure(it, it["channels"])
        theta = float(it["threshold"])
        clean_verdict = _verdict(it["program"], it, it["channels"])
        row = {
            "item_id": it["item_id"],
            "dataset": it["dataset"],
            "op": it["op"],
            "threshold": theta,
            "tolerance": float(it["tolerance"]),
            "clean_v": clean_v,
            "clean_margin": None if clean_v is None else clean_v - theta,
            "clean_verdict": clean_verdict,
            "perturbed": {},
        }
        for name in PERTURBATIONS:
            dirty = apply(name, it["channels"], it["named_channels"], it["item_id"])
            v = _measure(it, dirty)
            verd = _verdict(it["program"], it, dirty)
            undefined = v is None or verd == "UNVERIFIABLE"
            row["perturbed"][name] = {
                "v": v,
                "verdict": verd,
                "margin": None if undefined else v - theta,
                "margin_undefined": undefined,
                "flipped": clean_verdict == "SUPPORTED" and verd != "SUPPORTED",
                "polarity_flip": clean_verdict == "SUPPORTED" and verd == "CONTRADICTED",
            }
        recs.append(row)
    return recs


def _ladder(items: list[dict]) -> list[dict]:
    recs = []
    for it in items:
        v0 = float(it["clean_value"])
        tol = float(it["tolerance"])
        dirty = apply("awgn_snr10", it["channels"], it["named_channels"], it["item_id"])
        for k in MARGIN_K:
            theta = v0 - float(k) * tol
            prog = _program(it["op"], it["named_channels"], theta)
            clean_verd = _verdict(prog, it, it["channels"])
            dirty_verd = _verdict(prog, it, dirty)
            recs.append(
                {
                    "item_id": it["item_id"],
                    "op": it["op"],
                    "k": float(k),
                    "threshold": theta,
                    "clean_verdict": clean_verd,
                    "dirty_verdict": dirty_verd,
                }
            )
    return recs


def _summarize(replay: list[dict], ladder: list[dict]) -> dict:
    by_pert = {}
    for name in PERTURBATIONS:
        rows = [r["perturbed"][name] for r in replay]
        legal = [x for x in rows if not x["margin_undefined"]]
        flips = [x for x in legal if x["polarity_flip"]]
        kept = [x for x in legal if x["verdict"] == "SUPPORTED"]
        n_s = sum(1 for r in replay if r["clean_verdict"] == "SUPPORTED")
        n_keep = sum(1 for r in replay if r["clean_verdict"] == "SUPPORTED" and r["perturbed"][name]["verdict"] == "SUPPORTED")
        cross = 0
        for r, x in zip(replay, rows):
            if x["polarity_flip"] and x["margin"] is not None and x["margin"] <= 0:
                cross += 1
        n_flip = sum(1 for x in rows if x["polarity_flip"])
        dirty_m = [x["margin"] for x in legal]
        clean_m = [r["clean_margin"] for r in replay if r["clean_margin"] is not None and not r["perturbed"][name]["margin_undefined"]]
        consumption = []
        for r in replay:
            x = r["perturbed"][name]
            if r["clean_margin"] is not None and x["margin"] is not None:
                consumption.append(r["clean_margin"] - x["margin"])
        by_pert[name] = {
            "n": len(rows),
            "n_legal": len(legal),
            "n_unknown": sum(1 for x in rows if x["verdict"] == "UNVERIFIABLE"),
            "supported_preservation": n_keep / n_s if n_s else None,
            "n_polarity_flip": n_flip,
            "flip_with_margin_le_0": cross / n_flip if n_flip else None,
            "mean_dirty_margin": float(np.mean(dirty_m)) if dirty_m else None,
            "mean_margin_consumption": float(np.mean(consumption)) if consumption else None,
            "mean_clean_margin": float(np.mean(clean_m)) if clean_m else None,
            "mean_dirty_margin_kept": float(np.mean([x["margin"] for x in kept])) if kept else None,
            "mean_dirty_margin_flipped": float(np.mean([x["margin"] for x in flips])) if flips else None,
        }

    by_k = {}
    k1_awgn_s = 0
    for k in MARGIN_K:
        sub = [r for r in ladder if r["k"] == float(k)]
        n = len(sub)
        n_cs = sum(1 for r in sub if r["clean_verdict"] == "SUPPORTED")
        n_keep = sum(1 for r in sub if r["clean_verdict"] == "SUPPORTED" and r["dirty_verdict"] == "SUPPORTED")
        n_flip = sum(1 for r in sub if r["clean_verdict"] == "SUPPORTED" and r["dirty_verdict"] == "CONTRADICTED")
        n_u = sum(1 for r in sub if r["dirty_verdict"] == "UNVERIFIABLE")
        if float(k) == 1.0:
            k1_awgn_s = n_keep
        by_k[str(k)] = {
            "n": n,
            "n_clean_supported": n_cs,
            "supported_preservation": n_keep / n_cs if n_cs else None,
            "n_kept": n_keep,
            "n_polarity_flip": n_flip,
            "unknown_rate": n_u / n if n else None,
        }

    k1_ok = k1_awgn_s == E1_AWGN10_SUPPORTED and len(replay) == E1_N
    return {
        "n_items": len(replay),
        "by_perturbation": by_pert,
        "by_k": by_k,
        "k1_awgn10_supported": k1_awgn_s,
        "k1_matches_e1": k1_ok,
        "decision": "PASS" if k1_ok else "STOP",
    }


def run() -> dict:
    items = e1_items()
    replay = _replay_e1(items)
    ladder = _ladder(items)
    summary = _summarize(replay, ladder)
    RESULTS.mkdir(parents=True, exist_ok=True)
    slim_replay = []
    for r in replay:
        slim_replay.append(
            {
                "item_id": r["item_id"],
                "op": r["op"],
                "threshold": r["threshold"],
                "clean_v": r["clean_v"],
                "clean_margin": r["clean_margin"],
                "clean_verdict": r["clean_verdict"],
                "perturbed": {
                    k: {
                        "verdict": v["verdict"],
                        "margin": v["margin"],
                        "polarity_flip": v["polarity_flip"],
                    }
                    for k, v in r["perturbed"].items()
                },
            }
        )
    payload = {
        "meta": {"experiment_id": EXPERIMENT_IDS["m2"], "seed": SEED, "run_count": 1},
        "summary": summary,
        "replay": slim_replay,
        "ladder": ladder,
    }
    (RESULTS / "decision_margin_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

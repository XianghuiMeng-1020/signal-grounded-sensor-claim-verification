"""Unused, labeled, non-holdout windows and frozen dictionary claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p2.windows import load_mhealth, load_pamap2, load_wisdm
from p2_phase2.degrade import dropout
from p35.windows_ir import used_window_keys

from .config import INVALIDATIONS, MAX_PER_SUBJECT, PRIOR_RESULT_GLOBS, ROOT
from .dictionary import family_of, mappable


def _prior_used_ids() -> set[str]:
    extra: set[str] = set()
    for pat in PRIOR_RESULT_GLOBS:
        for path in ROOT.glob(pat):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            recs = obj.get("records") if isinstance(obj, dict) else obj
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict):
                    continue
                iid = r.get("item_id") or r.get("window_id") or ""
                extra.add(str(iid).split(":")[0])
    return extra


def load_labeled_unused(max_per_subject: int = MAX_PER_SUBJECT) -> list[dict]:
    used_ids, used_hashes = used_window_keys()
    used_ids |= _prior_used_ids()
    raw = load_pamap2(max_per_subject) + load_wisdm(max_per_subject) + load_mhealth(max_per_subject)
    out = []
    seen = set()
    for w in raw:
        if w.get("split") == "final_sealed_holdout":
            continue
        if w["window_id"] in used_ids or w["content_hash"] in used_hashes:
            continue
        if w["window_id"] in seen:
            continue
        if not mappable(w["dataset"], w.get("activity")):
            continue
        seen.add(w["window_id"])
        ch = {k: np.asarray(v, dtype=np.float64) for k, v in w["channels"].items()}
        rec = dict(w)
        rec["channels"] = ch
        rec["available_channels"] = list(ch.keys())
        rec["family"] = family_of(w["dataset"], w["activity"])
        out.append(rec)
    out.sort(key=lambda x: x["window_id"])
    return out


def assert_no_holdout(windows: list[dict]) -> None:
    leaked = [w["window_id"] for w in windows if w.get("split") == "final_sealed_holdout"]
    if leaked:
        raise RuntimeError(f"holdout windows leaked: {leaked[:3]}")


def _channel_a(w: dict) -> str:
    return list(w["available_channels"])[0]


def _legal_item(w: dict) -> dict:
    ch_a = _channel_a(w)
    return {
        "item_id": f"{w['window_id']}:legal",
        "window_id": w["window_id"],
        "dataset": w["dataset"],
        "subject": w["subject"],
        "split": w["split"],
        "fs": float(w["fs"]),
        "gold_activity": w["activity"],
        "posed_family": w["family"],
        "channel_a": ch_a,
        "available_channels": list(w["available_channels"]),
        "channels": dict(w["channels"]),
        "slice": "legal",
        "invalidation": None,
    }


def _illegal_item(w: dict, kind: str) -> dict:
    ch_a = _channel_a(w)
    channels = {k: np.asarray(v, dtype=np.float64).copy() for k, v in w["channels"].items()}
    available = list(w["available_channels"])
    fs = float(w["fs"])
    if kind == "missing_channel":
        channels.pop(ch_a, None)
        available = [c for c in available if c != ch_a]
    elif kind == "invalid_fs":
        fs = 0.0
    elif kind == "dropout_10pct":
        channels[ch_a] = dropout(channels[ch_a], 0.10, f"{w['window_id']}:dropout")
    else:
        raise ValueError(kind)
    return {
        "item_id": f"{w['window_id']}:{kind}",
        "window_id": w["window_id"],
        "dataset": w["dataset"],
        "subject": w["subject"],
        "split": w["split"],
        "fs": fs,
        "gold_activity": w["activity"],
        "posed_family": w["family"],
        "channel_a": ch_a,
        "available_channels": available,
        "channels": channels,
        "slice": "illegal",
        "invalidation": kind,
    }


def split_windows(windows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [w for w in windows if w["split"] == "development"]
    test = [w for w in windows if w["split"] == "challenge"]
    return train, test


def build_items(test_windows: list[dict]) -> list[dict]:
    items = []
    for w in test_windows:
        items.append(_legal_item(w))
        for kind in INVALIDATIONS:
            items.append(_illegal_item(w, kind))
    return items


def pool_audit(windows: list[dict]) -> dict:
    by_ds: dict[str, list] = {}
    for w in windows:
        by_ds.setdefault(w["dataset"], []).append(w)
    out = {}
    for ds, rows in sorted(by_ds.items()):
        acts = sorted({str(r["activity"]) for r in rows})
        out[ds] = {
            "subjects": sorted({r["subject"] for r in rows}),
            "n_subjects": len({r["subject"] for r in rows}),
            "n_windows": len(rows),
            "activities": acts,
            "sampling_rate_hz": sorted({float(r["fs"]) for r in rows}),
            "splits": {s: sum(1 for r in rows if r["split"] == s) for s in ("development", "challenge")},
        }
    return out

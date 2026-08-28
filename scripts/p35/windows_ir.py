"""Unused later-offset windows. No holdout. No prior-blind window reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from p2.windows import load_mhealth, load_pamap2, load_wisdm, window_content_hash
from p3.windows_p3 import unique_windows

from .config import ROOT, RESULTS, SEED

MAX_PER_SUBJECT = 24
USED_SCAN_GLOBS = (
    "results/p3cr/*_rows.json",
    "results/p3c/*_rows.json",
    "results/p3/*_rows.json",
    "results/p3r_ec/*_rows.json",
    "benchmarks/p3cr/*.inference.jsonl",
    "benchmarks/p3c/*.inference.jsonl",
)


def _as_win(w: dict) -> dict:
    ch_raw = w.get("channels") or w.get("channels_data") or {}
    ch = {k: np.asarray(v, dtype=np.float64) for k, v in ch_raw.items() if v is not None}
    wid = w.get("window_id") or w.get("source_window_id")
    return {
        "window_id": wid,
        "content_hash": w.get("content_hash") or window_content_hash({k: v.tolist() for k, v in ch.items()}),
        "dataset": w.get("dataset") or w.get("source_dataset"),
        "subject": w.get("subject"),
        "split_source": w.get("split") or w.get("split_source"),
        "fs": float(w["fs"]),
        "available_channels": list(ch.keys()),
        "channels": ch,
        "window_index": w.get("window_index"),
    }


def used_window_keys() -> tuple[set[str], set[str]]:
    ids, hashes = set(), set()
    for w in unique_windows():
        ww = _as_win(w)
        if ww["window_id"]:
            ids.add(ww["window_id"])
        if ww["content_hash"]:
            hashes.add(ww["content_hash"])
    for pat in USED_SCAN_GLOBS:
        for path in ROOT.glob(pat):
            try:
                if path.suffix == ".jsonl":
                    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    rows = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                wid = r.get("window_id") or r.get("source_window_id")
                if wid:
                    ids.add(wid)
                if r.get("content_hash"):
                    hashes.add(r["content_hash"])
    return ids, hashes


def load_unused_windows(max_per_subject: int = MAX_PER_SUBJECT) -> list[dict]:
    used_ids, used_hashes = used_window_keys()
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
        seen.add(w["window_id"])
        out.append(_as_win(w))
    out.sort(key=lambda x: x["window_id"])
    return out


def split_unused(wins: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    wins = list(wins if wins is not None else load_unused_windows())
    if not wins:
        raise RuntimeError("no unused non-holdout windows")
    n_dev = max(1, int(round(len(wins) * 0.4)))
    return wins[:n_dev], wins[n_dev:]


def window_pool_manifest(dev_w, blind_w) -> dict:
    return {
        "seed": SEED,
        "max_per_subject": MAX_PER_SUBJECT,
        "n_dev_windows": len(dev_w),
        "n_reserved_blind_windows": len(blind_w),
        "dev_ids_sha256": hashlib.sha256(" ".join(w["window_id"] for w in dev_w).encode()).hexdigest(),
        "blind_ids_sha256": hashlib.sha256(" ".join(w["window_id"] for w in blind_w).encode()).hexdigest(),
        "overlap": len({w["window_id"] for w in dev_w} & {w["window_id"] for w in blind_w}),
        "holdout_included": False,
    }

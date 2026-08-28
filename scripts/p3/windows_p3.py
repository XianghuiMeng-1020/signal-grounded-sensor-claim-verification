"""Non-holdout windows from already-open V3 DEVELOPMENT and CHALLENGE gold."""
from __future__ import annotations

import numpy as np

from p2.config import BENCH_P2_V3
from p2.evaluate import load_split

from .guard import refuse_holdout


def load_open_rows():
    rows = []
    for split in ("development", "challenge"):
        refuse_holdout(split)
        rows.extend(load_split(split, bench_root=BENCH_P2_V3))
    return rows


def unique_windows(rows=None):
    rows = rows or load_open_rows()
    seen = {}
    for r in rows:
        wid = r.get("source_window_id") or r["claim_id"]
        if wid in seen:
            continue
        ch = {k: np.asarray(v, dtype=np.float64) for k, v in r["channels_data"].items()}
        seen[wid] = {
            "window_id": wid,
            "dataset": r["source_dataset"],
            "subject": r.get("subject"),
            "split_source": r.get("split"),
            "fs": float(r["fs"]),
            "available_channels": list(r["available_channels"]),
            "channels": ch,
        }
    return list(seen.values())

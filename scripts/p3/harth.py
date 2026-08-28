"""HARTH loader. Selection was frozen in config before PRIMARY evaluation."""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np

from .config import EXTERNAL_CHANNELS, EXTERNAL_FS, EXTERNAL_SUBJECT_GROUPS, ROOT
from .guard import refuse_path

DATA = ROOT / "data" / "harth_inner"
BASE = "https://raw.githubusercontent.com/ntnu-ai-lab/harth-ml-experiments/main/harth"


def download_subjects(sids: tuple[str, ...]) -> list[str]:
    DATA.mkdir(parents=True, exist_ok=True)
    refuse_path(DATA)
    got = []
    for sid in sids:
        dest = DATA / f"{sid}.csv"
        if dest.exists() and dest.stat().st_size > 10000:
            got.append(sid)
            continue
        url = f"{BASE}/{sid}.csv"
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 10000:
                got.append(sid)
        except Exception:
            if dest.exists():
                dest.unlink()
    return got


def windows_from_csv(sid: str, n_windows: int = 6, start: int = 5000) -> list[dict]:
    path = DATA / f"{sid}.csv"
    refuse_path(path)
    back, thigh = [], []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i < start:
                continue
            try:
                bx, by, bz = float(row["back_x"]), float(row["back_y"]), float(row["back_z"])
                tx, ty, tz = float(row["thigh_x"]), float(row["thigh_y"]), float(row["thigh_z"])
            except Exception:
                continue
            back.append(np.sqrt(bx * bx + by * by + bz * bz))
            thigh.append(np.sqrt(tx * tx + ty * ty + tz * tz))
            if len(back) >= n_windows * 256 + 10:
                break
    back = np.asarray(back, dtype=np.float64)
    thigh = np.asarray(thigh, dtype=np.float64)
    n = min(len(back), len(thigh))
    out = []
    for i in range(n_windows):
        sl = slice(i * 256, (i + 1) * 256)
        if sl.stop > n:
            break
        out.append({
            "window_id": f"HARTH:{sid}:{i}",
            "dataset": "HARTH",
            "subject": sid,
            "fs": EXTERNAL_FS,
            "available_channels": list(EXTERNAL_CHANNELS),
            "channels": {"back_accel": back[sl], "thigh_accel": thigh[sl]},
        })
    return out


def load_eval_windows():
    sids = download_subjects(EXTERNAL_SUBJECT_GROUPS["p3_external_eval"])
    wins = []
    for sid in sids:
        wins.extend(windows_from_csv(sid))
    return wins, sids

"""Window loading with stable identities. Subject-grouped; no claim-level split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATA_MHEALTH,
    DATA_PAMAP2,
    DATA_WISDM,
    FS,
    MAX_WINDOWS_PER_SUBJECT,
    MHEALTH_SPLIT,
    PAMAP2_SPLIT,
    WINDOW,
)


def _hash_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def window_content_hash(channels: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(channels):
        h.update(k.encode("utf-8"))
        h.update(np.asarray(channels[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def window_id(dataset: str, subject: str, session: str, index: int, channels: dict) -> str:
    return _hash_bytes(
        dataset.encode(),
        b"|",
        subject.encode(),
        b"|",
        session.encode(),
        b"|",
        str(index).encode(),
        b"|",
        window_content_hash(channels).encode(),
    )


def wisdm_split(subject_stem: str) -> str:
    # data_1600_accel_watch -> 1600
    parts = subject_stem.split("_")
    sid = None
    for p in parts:
        if p.isdigit():
            sid = int(p)
            break
    if sid is None:
        raise ValueError(f"cannot parse WISDM subject from {subject_stem}")
    return ("development", "challenge", "final_sealed_holdout")[sid % 3]


def assigned_split(dataset: str, subject: str) -> str:
    if dataset == "PAMAP2":
        for split, names in PAMAP2_SPLIT.items():
            if subject in names:
                return split
        raise KeyError(subject)
    if dataset == "MHEALTH":
        for split, names in MHEALTH_SPLIT.items():
            if subject in names:
                return split
        raise KeyError(subject)
    if dataset == "WISDM":
        return wisdm_split(subject)
    raise KeyError(dataset)


def _finite_std_ok(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)) and np.std(arr) > 0.05)


def load_pamap2(max_per_subject: int = MAX_WINDOWS_PER_SUBJECT) -> list[dict]:
    out = []
    for f in sorted(DATA_PAMAP2.glob("subject10*.dat")):
        df = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN")
        hand = df.iloc[:, 4].to_numpy(dtype=float)
        chest = df.iloc[:, 21].to_numpy(dtype=float)
        activity = df.iloc[:, 1].to_numpy(dtype=float) if df.shape[1] > 1 else None
        n_windows = len(hand) // WINDOW
        cnt = 0
        for w in range(n_windows):
            sl = slice(w * WINDOW, (w + 1) * WINDOW)
            h, c = hand[sl], chest[sl]
            if not (_finite_std_ok(h) and _finite_std_ok(c)):
                continue
            act = None
            if activity is not None:
                block = activity[sl]
                # majority activity id, ignore 0 (transient) if possible
                vals, counts = np.unique(block[block != 0], return_counts=True) if np.any(block != 0) else (np.array([]), np.array([]))
                act = int(vals[int(np.argmax(counts))]) if len(vals) else 0
            ch = {"hand_accel": h.astype(float).tolist(), "chest_accel": c.astype(float).tolist()}
            rec = _record("PAMAP2", f.stem, f.name, w, FS["PAMAP2"], ch, act)
            out.append(rec)
            cnt += 1
            if cnt >= max_per_subject:
                break
    return out


def load_wisdm(max_per_subject: int = MAX_WINDOWS_PER_SUBJECT) -> list[dict]:
    out = []
    for f in sorted(DATA_WISDM.glob("data_*_accel_watch.txt")):
        xs, ys, acts = [], [], []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().rstrip(";").split(",")
                if len(parts) >= 6:
                    try:
                        acts.append(parts[1])
                        xs.append(float(parts[3]))
                        ys.append(float(parts[4]))
                    except ValueError:
                        continue
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        n_windows = len(xs) // WINDOW
        cnt = 0
        for w in range(n_windows):
            sl = slice(w * WINDOW, (w + 1) * WINDOW)
            x, y = xs[sl], ys[sl]
            if not (_finite_std_ok(x) and _finite_std_ok(y)):
                continue
            act = None
            if acts:
                block = acts[sl]
                # majority activity letter
                vals, counts = np.unique(block, return_counts=True)
                act = str(vals[int(np.argmax(counts))])
            ch = {"x_accel": x.astype(float).tolist(), "y_accel": y.astype(float).tolist()}
            rec = _record("WISDM", f.stem, f.name, w, FS["WISDM"], ch, act)
            out.append(rec)
            cnt += 1
            if cnt >= max_per_subject:
                break
    return out


def load_mhealth(max_per_subject: int = MAX_WINDOWS_PER_SUBJECT) -> list[dict]:
    out = []
    for f in sorted(DATA_MHEALTH.glob("mHealth_subject*.log")):
        df = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN")
        chest = df.iloc[:, 0].to_numpy(dtype=float)
        ankle = df.iloc[:, 5].to_numpy(dtype=float)
        label = df.iloc[:, 23].to_numpy(dtype=float)
        n_windows = len(chest) // WINDOW
        cnt = 0
        for w in range(n_windows):
            sl = slice(w * WINDOW, (w + 1) * WINDOW)
            c, a, lab = chest[sl], ankle[sl], label[sl]
            if np.any(~np.isfinite(c)) or np.any(~np.isfinite(a)):
                continue
            if np.any(lab == 0) or np.std(c) < 0.05:
                continue
            vals, counts = np.unique(lab, return_counts=True)
            act = int(vals[int(np.argmax(counts))])
            ch = {"chest_accel": c.astype(float).tolist(), "ankle_accel": a.astype(float).tolist()}
            rec = _record("MHEALTH", f.stem, f.name, w, FS["MHEALTH"], ch, act)
            out.append(rec)
            cnt += 1
            if cnt >= max_per_subject:
                break
    return out


def _record(dataset, subject, session, index, fs, channels, activity) -> dict[str, Any]:
    wid = window_id(dataset, subject, session, index, channels)
    split = assigned_split(dataset, subject)
    return {
        "window_id": wid,
        "content_hash": window_content_hash(channels),
        "dataset": dataset,
        "subject": subject,
        "session": session,
        "window_index": int(index),
        "activity": activity,
        "fs": float(fs),
        "channels": channels,
        "split": split,
        "temporal_start_sample": int(index) * WINDOW,
        "temporal_end_sample": (int(index) + 1) * WINDOW,
    }


def load_all_windows() -> list[dict]:
    return load_pamap2() + load_wisdm() + load_mhealth()


def audit_leakage(windows: list[dict]) -> dict:
    by_split: dict[str, list] = {}
    for w in windows:
        by_split.setdefault(w["split"], []).append(w)
    hashes = {s: {w["content_hash"] for w in rows} for s, rows in by_split.items()}
    subjects = {s: {(w["dataset"], w["subject"]) for w in rows} for s, rows in by_split.items()}
    pairs = {}
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"window_{a}_x_{b}"] = len(hashes[a] & hashes[b])
            pairs[f"subject_{a}_x_{b}"] = len(subjects[a] & subjects[b])
    # adjacent window pairs within subject should stay in one split (true by construction)
    return {
        "n_windows": len(windows),
        "n_unique_content": len({w["content_hash"] for w in windows}),
        "per_split": {s: len(v) for s, v in by_split.items()},
        "overlap": pairs,
        "subjects_per_split": {s: sorted(list(map(lambda t: f"{t[0]}:{t[1]}", subjects[s]))) for s in subjects},
    }


def dump_window_manifest(windows: list[dict], path: Path) -> None:
    slim = [{k: w[k] for k in w if k != "channels"} for w in windows]
    path.write_text(json.dumps(slim, indent=1), encoding="utf-8")

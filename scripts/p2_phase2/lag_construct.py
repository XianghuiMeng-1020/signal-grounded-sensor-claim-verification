"""Build equivalent physical delays at several sampling rates.

Construction only. Does not score. Resampling is an evaluation stimulus;
it is not a production preprocessor and is not imported by the kernels.
"""
from __future__ import annotations

import numpy as np

from p2r.schema import ClaimProgram, Predicate
from p35.windows_ir import load_unused_windows

from .lag_config import (
    DELAYS_MS,
    FS_GRID,
    MAX_PER_DATASET,
    MAX_WINDOWS,
    MIN_N_LAG,
    PHYSICAL_BOUND_MS,
    SAMPLE_BOUND,
    delay_samples,
    in_sample_box,
    sample_bound_ms,
)


def resample_to_fs(x, fs_src: float, fs_tgt: float) -> tuple[np.ndarray, int]:
    """Linear interp onto a same-duration grid. Evaluation construction only."""
    src = np.asarray(x, dtype=np.float64).reshape(-1)
    n_src = int(src.size)
    n_tgt = max(1, int(round(n_src / float(fs_src) * float(fs_tgt))))
    t_src = np.linspace(0.0, 1.0, n_src, endpoint=False)
    t_tgt = np.linspace(0.0, 1.0, n_tgt, endpoint=False)
    return np.interp(t_tgt, t_src, src), n_tgt


def inject_physical_delay(x, delay_ms: float, fs: float) -> np.ndarray:
    """Circular shift by +k samples. Production correlate reports lag −k."""
    k = delay_samples(delay_ms, fs)
    return np.roll(np.asarray(x, dtype=np.float64).reshape(-1), k)


def bound_program(channel_a: str, channel_b: str, bound: float) -> ClaimProgram:
    """AND of (lag > -bound) and (lag < +bound). Bound units depend on eval_mode."""
    lo = Predicate(
        measurement="cross_channel_lag_ms",
        channel_a=channel_a,
        comparator="gt",
        channel_b=channel_b,
        reference_value=-float(bound),
        reference_channel=None,
        unit="ms",
    )
    hi = Predicate(
        measurement="cross_channel_lag_ms",
        channel_a=channel_a,
        comparator="lt",
        channel_b=channel_b,
        reference_value=float(bound),
        reference_channel=None,
        unit="ms",
    )
    return ClaimProgram("AND", [lo, hi], parse_status="OK")


def _select_windows() -> list[dict]:
    wins = load_unused_windows()
    by_ds: dict[str, list] = {}
    for w in wins:
        by_ds.setdefault(w["dataset"], []).append(w)
    selected = []
    for ds in sorted(by_ds):
        selected.extend(by_ds[ds][:MAX_PER_DATASET])
    selected = selected[:MAX_WINDOWS]
    selected.sort(key=lambda w: w["window_id"])
    return selected


def build_items() -> list[dict]:
    items = []
    skipped = 0
    for w in _select_windows():
        avail = list(w["available_channels"])
        if not avail:
            skipped += 1
            continue
        src_name = avail[0]
        dst_name = avail[1] if len(avail) > 1 else f"{src_name}__delayed"
        src = w["channels"][src_name]
        fs_src = float(w["fs"])
        if src is None or np.asarray(src).size < 4:
            skipped += 1
            continue
        for fs_tgt in FS_GRID:
            x, n_tgt = resample_to_fs(src, fs_src, fs_tgt)
            for d_ms in DELAYS_MS:
                y = inject_physical_delay(x, d_ms, fs_tgt)
                k = delay_samples(d_ms, fs_tgt)
                named = [src_name, dst_name]
                items.append(
                    {
                        "item_id": f"{w['window_id']}:fs{int(fs_tgt)}:d{int(d_ms)}",
                        "window_id": w["window_id"],
                        "dataset": w["dataset"],
                        "fs_native": fs_src,
                        "fs": float(fs_tgt),
                        "n": int(n_tgt),
                        "available_channels": named,
                        "named_channels": named,
                        "channels": {src_name: x, dst_name: y},
                        "delay_ms": float(d_ms),
                        "delay_samples": int(k),
                        "true_lag_ms": float(k) / float(fs_tgt) * 1000.0,
                        "in_sample_box": in_sample_box(d_ms, fs_tgt),
                        "length_ok": n_tgt >= MIN_N_LAG,
                        "sample_bound_ms": sample_bound_ms(fs_tgt),
                        "physical_bound_ms": PHYSICAL_BOUND_MS,
                        "program_sample": bound_program(src_name, dst_name, SAMPLE_BOUND),
                        "program_physical": bound_program(src_name, dst_name, PHYSICAL_BOUND_MS),
                    }
                )
    if skipped:
        pass
    return items

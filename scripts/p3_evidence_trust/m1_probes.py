"""Constructed invalid probes. Evaluation stimuli only."""
from __future__ import annotations

import numpy as np

from p2r.contracts import contract_spec
from p2_phase2.degrade import dropout

from .config import ABLATION_APPLIES


def _copy_ch(channels: dict) -> dict:
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in channels.items()}


def apply_probe(item: dict, ablation: str) -> dict:
    """Return scoring context: channels, fs, invalid_by_construction."""
    ch = _copy_ch(item["channels"])
    named = list(item["named_channels"])
    fs = item["fs"]
    op = item["op"]
    invalid = True

    if ablation == "drop_nonfinite":
        for n in named:
            ch[n] = dropout(ch[n], 0.10, f"p3m1:{item['item_id']}:{n}")
    elif ablation == "drop_min_n":
        min_n = int(contract_spec(op)["min_finite_n"])
        cut = max(1, min_n - 1)
        for n in named:
            ch[n] = np.asarray(ch[n], dtype=np.float64).reshape(-1)[:cut]
    elif ablation == "drop_fs":
        fs = None
    elif ablation == "drop_second_channel":
        if len(named) >= 2:
            ch[named[1]] = None
    elif ablation == "drop_equal_length":
        if len(named) >= 2:
            arr = np.asarray(ch[named[1]], dtype=np.float64).reshape(-1)
            ch[named[1]] = arr[:-8] if arr.size > 8 else arr[: max(1, arr.size // 2)]
    elif ablation == "drop_variance":
        n0 = named[0]
        ch[n0] = np.zeros_like(np.asarray(ch[n0], dtype=np.float64))
    elif ablation == "drop_output_domain":
        invalid = False
    else:
        raise KeyError(ablation)

    return {"channels": ch, "fs": fs, "invalid_by_construction": invalid}


def applicable(op: str, ablation: str) -> bool:
    return op in ABLATION_APPLIES[ablation]

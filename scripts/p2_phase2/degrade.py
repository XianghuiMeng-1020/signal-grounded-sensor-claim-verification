"""Waveform-only operators. Do not touch programs, contracts, kernels, or thresholds."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import CLIP_FRAC, DROPOUT_P, SEED, SNR_DB


def _rng(key: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h % (2**32))


def awgn(x: np.ndarray, snr_db: float, key: str) -> np.ndarray:
    rng = _rng(key)
    x = np.asarray(x, dtype=np.float64)
    power = float(np.mean(x * x) + 1e-12)
    sigma = float(np.sqrt(power / (10.0 ** (snr_db / 10.0))))
    return x + rng.normal(0.0, sigma, size=x.shape)


def dropout(x: np.ndarray, p: float, key: str) -> np.ndarray:
    rng = _rng(key)
    y = np.asarray(x, dtype=np.float64).copy()
    mask = rng.random(y.size) < p
    y[mask] = np.nan
    return y


def clip_amp(x: np.ndarray, frac: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    tau = frac * peak if peak > 0.0 else 1.0
    return np.clip(x, -tau, tau)


def apply(name: str, channels: dict[str, np.ndarray], named: list[str], key: str) -> dict[str, np.ndarray]:
    """Perturb only named series. Other channels are copied unchanged."""
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in channels.items()}
    for ch in named:
        if ch not in out:
            continue
        x = out[ch]
        if name.startswith("awgn_snr"):
            snr = {"awgn_snr20": SNR_DB[0], "awgn_snr10": SNR_DB[1], "awgn_snr0": SNR_DB[2]}[name]
            out[ch] = awgn(x, snr, f"{key}:{name}:{ch}")
        elif name == "dropout_10pct":
            out[ch] = dropout(x, DROPOUT_P, f"{key}:{name}:{ch}")
        elif name == "clip_0p60":
            out[ch] = clip_amp(x, CLIP_FRAC)
        else:
            raise KeyError(name)
    return out

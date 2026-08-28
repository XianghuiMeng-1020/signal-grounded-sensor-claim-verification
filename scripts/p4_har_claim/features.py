"""Standard statistical HAR features. Not the verification path."""
from __future__ import annotations

import numpy as np

STAT_NAMES = ("mean", "std", "min", "max", "rms", "dom_freq")


def _dom_freq(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return 0.0
    y = x - float(np.mean(x))
    spec = np.abs(np.fft.rfft(y))
    if spec.size == 0:
        return 0.0
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return float(k * fs / x.size)


def _stats(x: np.ndarray, fs: float) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return [0.0] * len(STAT_NAMES)
    rms = float(np.sqrt(np.mean(x * x)))
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        rms,
        _dom_freq(x, fs),
    ]


def feature_names(channel_names: list[str]) -> list[str]:
    names = []
    for ch in channel_names:
        names.extend(f"{ch}:{s}" for s in STAT_NAMES)
    return names


def extract_features(channels: dict, fs: float, channel_order: list[str] | None = None) -> np.ndarray:
    order = list(channel_order or sorted(channels))
    feats = []
    for name in order:
        x = channels.get(name)
        if x is None:
            feats.extend([0.0] * len(STAT_NAMES))
        else:
            feats.extend(_stats(np.asarray(x, dtype=np.float64), fs))
    return np.asarray(feats, dtype=np.float64)

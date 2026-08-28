"""Independent DSP reference layer.

This module is intentionally not a copy of scripts/f_round6_operators.py.
It implements the *resolved* mathematical definitions in config.py using
SciPy code paths (scipy.fft, scipy.signal, scipy.stats, np.linalg.norm).

The production verifier must never import this module.
Gold labels for independent_gold_v2 are produced here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
from scipy.stats import zscore

from .config import (
    LAG_T_MAX_S,
    PERIODICITY_MIN_LAG,
    RELATIVE_FLOOR,
    RELATIVE_FRAC,
    RELATIVE_OPS,
    SPECTRAL_CUTOFF_HZ,
    TOLERANCE_ABS,
    lag_max_samples,
)


class MeasurementError(ValueError):
    """Raised when a primitive cannot be computed from the provided evidence."""


def _as1d(x) -> np.ndarray:
    if x is None:
        raise MeasurementError("missing_channel")
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise MeasurementError("empty_channel")
    if np.any(~np.isfinite(arr)):
        raise MeasurementError("nonfinite_samples")
    return arr


def _require_fs(fs: Optional[float]) -> float:
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise MeasurementError("invalid_sampling_rate")
    return float(fs)


def rms_amplitude(x, fs=None) -> float:
    """Raw RMS including DC: ||x||_2 / sqrt(N)."""
    arr = _as1d(x)
    return float(np.linalg.norm(arr, ord=2) / np.sqrt(arr.size))


def peak_amplitude(x, fs=None) -> float:
    """Max absolute deviation from the sample mean."""
    arr = _as1d(x)
    return float(np.max(np.abs(arr - arr.mean())))


def signal_range(x, fs=None) -> float:
    """Peak-to-peak range via numpy.ptp."""
    arr = _as1d(x)
    return float(np.ptp(arr))


def trend_ratio(x, fs=None) -> float:
    """AC-RMS(second half) / AC-RMS(first half), split at n//2."""
    arr = _as1d(x)
    n = arr.size
    if n < 4:
        raise MeasurementError("insufficient_length")
    first, second = arr[: n // 2], arr[n // 2 :]
    if first.size == 0 or second.size == 0:
        raise MeasurementError("insufficient_length")

    def ac_rms(seg):
        d = seg - seg.mean()
        return float(np.linalg.norm(d, ord=2) / np.sqrt(d.size))

    e1 = ac_rms(first)
    e2 = ac_rms(second)
    return float(e2 / (e1 + 1e-9))


def dominant_frequency(x, fs) -> float:
    """Physical Hz: f_dom = k̂ fs / N, k̂ = argmax_{k>0} |X[k]|, X=rFFT(x-mean)."""
    arr = _as1d(x)
    fs = _require_fs(fs)
    if arr.size < 4:
        raise MeasurementError("insufficient_length")
    centered = arr - arr.mean()
    spec = rfft(centered)
    freqs = rfftfreq(arr.size, d=1.0 / fs)
    mag = np.abs(spec)
    mag[0] = 0.0
    if not np.any(mag > 0):
        return 0.0
    k_hat = int(np.argmax(mag))
    return float(k_hat * fs / arr.size)


def spectral_energy_ratio_low(x, fs, cutoff_hz: float = SPECTRAL_CUTOFF_HZ) -> float:
    """Fraction of rFFT power at frequencies strictly below cutoff_hz."""
    arr = _as1d(x)
    fs = _require_fs(fs)
    if arr.size < 4:
        raise MeasurementError("insufficient_length")
    centered = arr - arr.mean()
    spec = rfft(centered)
    freqs = rfftfreq(arr.size, d=1.0 / fs)
    power = np.square(np.abs(spec))
    total = float(power.sum()) + 1e-9
    low = float(power[freqs < cutoff_hz].sum())
    return float(low / total)


def periodicity_strength(x, fs=None) -> float:
    """Dimensionless P = max_{ℓ=3..N/2-1} |Rxx[ℓ]| / Rxx[0] after mean removal."""
    arr = _as1d(x)
    if arr.size < (PERIODICITY_MIN_LAG + 4):
        raise MeasurementError("insufficient_length")
    centered = arr - arr.mean()
    ac = correlate(centered, centered, mode="full", method="fft")
    mid = arr.size - 1
    ac_pos = ac[mid:]
    denom = float(ac_pos[0]) + 1e-9
    ac_pos = ac_pos / denom
    search = ac_pos[PERIODICITY_MIN_LAG : arr.size // 2]
    if search.size == 0:
        return 0.0
    return float(np.clip(np.max(np.abs(search)), 0.0, 1.0))


def cross_channel_lag_ms(x, y, fs, max_lag: int | None = None) -> float:
    """Lag (ms) of max |z-scored FFT correlation| inside ±L(fs) samples.

    L(fs)=floor(T_max fs), T_max=0.300 s. τ = 1000 ℓ̂ / fs.
    """
    a = _as1d(x)
    b = _as1d(y)
    fs = _require_fs(fs)
    if max_lag is None:
        max_lag = lag_max_samples(fs, LAG_T_MAX_S)
    if a.size != b.size:
        raise MeasurementError("channel_length_mismatch")
    if a.size < 4:
        raise MeasurementError("insufficient_length")
    if a.size <= 2 * max_lag:
        raise MeasurementError("insufficient_length")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        raise MeasurementError("degenerate_channel")
    az = zscore(a, ddof=0)
    bz = zscore(b, ddof=0)
    xc = correlate(az, bz, mode="full", method="fft")
    center = a.size - 1
    window = xc[center - max_lag : center + max_lag + 1]
    peak = int(np.argmax(np.abs(window)))
    lag_samples = peak - max_lag
    return float(lag_samples / fs * 1000.0)


# Diagnostic alternate estimator — NOT used for gold or Gate C pass/fail.
def dominant_frequency_periodogram(x, fs) -> float:
    from scipy.signal import periodogram

    arr = _as1d(x)
    fs = _require_fs(fs)
    freqs, pxx = periodogram(arr - arr.mean(), fs=fs, window="boxcar", scaling="spectrum")
    pxx = np.array(pxx, dtype=np.float64, copy=True)
    if pxx.size:
        pxx[0] = 0.0
    if not np.any(pxx > 0):
        return 0.0
    return float(freqs[int(np.argmax(pxx))])


FN_SINGLE = {
    "dominant_frequency": dominant_frequency,
    "rms_amplitude": rms_amplitude,
    "peak_amplitude": peak_amplitude,
    "signal_range": signal_range,
    "trend_ratio": trend_ratio,
    "periodicity_strength": periodicity_strength,
    "spectral_energy_ratio_low": spectral_energy_ratio_low,
}


def measure(op: str, channels: dict, fs) -> float:
    """channels: ordered name -> array. Lag uses the first two names."""
    if op == "cross_channel_lag_ms":
        names = list(channels.keys())
        if len(names) < 2:
            raise MeasurementError("missing_channels")
        return cross_channel_lag_ms(channels[names[0]], channels[names[1]], fs)
    if op not in FN_SINGLE:
        raise MeasurementError("unknown_operator")
    if not channels:
        raise MeasurementError("missing_channels")
    seg = next(iter(channels.values()))
    fn = FN_SINGLE[op]
    if op in ("dominant_frequency", "spectral_energy_ratio_low"):
        return fn(seg, fs)
    return fn(seg)


def tolerance_for(op: str, actual_value: float) -> float:
    if op in RELATIVE_OPS:
        return max(RELATIVE_FRAC * abs(float(actual_value)), RELATIVE_FLOOR)
    if op not in TOLERANCE_ABS:
        raise MeasurementError("unknown_operator")
    return float(TOLERANCE_ABS[op])

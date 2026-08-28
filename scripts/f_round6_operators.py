"""
Round 6, Part F -- generic signal-operator library.

This module is the SAME code used to (a) generate ground truth for the
benchmark and (b) serve as the "structured signal-grounded verifier"'s
executable evidence engine at pilot-time. Its docstrings are also shown
VERBATIM to the LLM claim-extractor so the LLM must pick the right operator
by reading generic descriptions -- no per-claim/per-family rule is hand
authored after the test set is generated (Round-6 Part 2F requirement).
"""
import numpy as np

from p2.config import LAG_T_MAX_S, lag_max_samples

FS_PAMAP2 = 100.0
FS_WISDM = 20.0

OPERATORS = {
    "dominant_frequency": "Dominant (largest-magnitude, non-DC) frequency component of a channel, in Hz. "
                            "Args: channel name.",
    "rms_amplitude": "Root-mean-square amplitude of a channel over the whole window (raw sensor units). "
                       "Args: channel name.",
    "peak_amplitude": "Maximum absolute, mean-removed amplitude of a channel over the window (raw sensor units). "
                        "Args: channel name.",
    "signal_range": "Max-minus-min (peak-to-peak) raw value range of a channel over the window (raw sensor units). "
                      "Args: channel name.",
    "trend_ratio": "Ratio of RMS energy in the second half of the window to RMS energy in the first half of the "
                    "window, for a channel (>1 means energy increased across the window, <1 means it decreased). "
                    "Args: channel name.",
    "cross_channel_lag_ms": "Cross-correlation-estimated timing lag (milliseconds) between two DIFFERENT channels "
                              "recorded over the same window. Args: two channel names.",
    "periodicity_strength": "Normalized autocorrelation peak strength (0-1) away from lag zero for a channel; "
                              "higher = more periodic/rhythmic, lower = more irregular/noise-like. Args: channel name.",
    "spectral_energy_ratio_low": "Fraction (0-1) of a channel's total spectral energy located below 3 Hz "
                                   "(low-frequency band). Args: channel name.",
}


def _prep(seg):
    return np.asarray(seg, dtype=float)


def dominant_frequency(seg, fs):
    """f_dom = k̂ fs / N, k̂ = argmax_{k>0} |X[k]|."""
    s = _prep(seg) - np.mean(seg)
    n = len(s)
    fft = np.fft.rfft(s)
    mag = np.abs(fft)
    mag[0] = 0
    k_hat = int(np.argmax(mag))
    return float(k_hat * fs / n)


def rms_amplitude(seg, fs=None):
    s = _prep(seg)
    return float(np.sqrt(np.mean(s ** 2)))


def peak_amplitude(seg, fs=None):
    s = _prep(seg)
    return float(np.max(np.abs(s - np.mean(s))))


def signal_range(seg, fs=None):
    s = _prep(seg)
    return float(np.max(s) - np.min(s))


def trend_ratio(seg, fs=None):
    s = _prep(seg)
    n = len(s)
    first, second = s[:n // 2], s[n // 2:]
    e1 = np.sqrt(np.mean((first - np.mean(first)) ** 2))
    e2 = np.sqrt(np.mean((second - np.mean(second)) ** 2))
    return float(e2 / (e1 + 1e-9))


def cross_channel_lag_ms(seg_a, seg_b, fs, max_lag=None):
    """τ = 1000 ℓ̂ / fs with |ℓ̂| ≤ L(fs)=floor(T_max fs), T_max=0.300 s."""
    if max_lag is None:
        max_lag = lag_max_samples(fs, LAG_T_MAX_S)
    a = (_prep(seg_a) - np.mean(seg_a)) / (np.std(seg_a) + 1e-9)
    b = (_prep(seg_b) - np.mean(seg_b)) / (np.std(seg_b) + 1e-9)
    n = len(a)
    xcorr = np.correlate(a, b, mode="full")
    center = n - 1
    window = xcorr[center - max_lag:center + max_lag + 1]
    peak_idx = np.argmax(np.abs(window))
    lag_samples = peak_idx - max_lag
    return float(lag_samples / fs * 1000)


def periodicity_strength(seg, fs=None):
    """P = max |Rxx[ℓ]| / Rxx[0] on ℓ ∈ [3, N/2)."""
    s = _prep(seg) - np.mean(seg)
    n = len(s)
    ac = np.correlate(s, s, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-9)
    search = ac[3:n // 2]
    if len(search) == 0:
        return 0.0
    return float(np.clip(np.max(np.abs(search)), 0, 1))


def spectral_energy_ratio_low(seg, fs, cutoff_hz=3.0):
    s = _prep(seg) - np.mean(seg)
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), d=1 / fs)
    psd = np.abs(fft) ** 2
    total = psd.sum() + 1e-9
    low = psd[freqs < cutoff_hz].sum()
    return float(low / total)


FN = {
    "dominant_frequency": dominant_frequency,
    "rms_amplitude": rms_amplitude,
    "peak_amplitude": peak_amplitude,
    "signal_range": signal_range,
    "trend_ratio": trend_ratio,
    "periodicity_strength": periodicity_strength,
    "spectral_energy_ratio_low": spectral_energy_ratio_low,
}

TOLERANCE = {
    "dominant_frequency": 0.4,
    "rms_amplitude": None,   # relative, see below
    "peak_amplitude": None,  # relative
    "signal_range": None,    # relative
    "trend_ratio": 0.25,
    "cross_channel_lag_ms": 15.0,
    "periodicity_strength": 0.12,
    "spectral_energy_ratio_low": 0.12,
}

UNIT = {
    "dominant_frequency": "Hz", "rms_amplitude": "raw units", "peak_amplitude": "raw units",
    "signal_range": "raw units", "trend_ratio": "ratio", "cross_channel_lag_ms": "ms",
    "periodicity_strength": "0-1 score", "spectral_energy_ratio_low": "0-1 fraction",
}


def compute(op, channels, fs):
    """channels: dict name->array. Returns actual value for the given operator."""
    if op == "cross_channel_lag_ms":
        names = list(channels.keys())
        return cross_channel_lag_ms(channels[names[0]], channels[names[1]], fs)
    fn = FN[op]
    seg = next(iter(channels.values()))
    if op == "dominant_frequency" or op == "spectral_energy_ratio_low":
        return fn(seg, fs)
    return fn(seg)


def tolerance_for(op, actual_value):
    t = TOLERANCE[op]
    if t is not None:
        return t
    return max(0.15 * abs(actual_value), 0.3)

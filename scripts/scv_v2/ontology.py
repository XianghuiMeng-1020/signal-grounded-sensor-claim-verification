"""Registered measurement/channel ontology for SCV V2.

This is the pre-H2 development vocabulary (schema + ordinary sensor names).
It is not derived from original H2 sentences.
"""
from __future__ import annotations

from p2r.schema import COMPARATORS, CONNECTIVES, MEASUREMENTS, UNITS

REGISTERED_CHANNELS = (
    "hand_accel",
    "chest_accel",
    "ankle_accel",
    "x_accel",
    "y_accel",
    "z_accel",
    "back_accel",
    "thigh_accel",
)

# Longest-first aliases. Short tokens such as "x" / "y" are whole-word only.
CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "hand_accel": ("hand_accel",),
    "hand-accel": ("hand_accel",),
    "hand accelerometer": ("hand_accel",),
    "hand": ("hand_accel",),
    "chest_accel": ("chest_accel",),
    "chest-accel": ("chest_accel",),
    "chest accelerometer": ("chest_accel",),
    "chest": ("chest_accel",),
    "ankle_accel": ("ankle_accel",),
    "ankle-accel": ("ankle_accel",),
    "ankle accelerometer": ("ankle_accel",),
    "ankle": ("ankle_accel",),
    "x_accel": ("x_accel",),
    "x-accel": ("x_accel",),
    "x-axis": ("x_accel",),
    "x axis": ("x_accel",),
    "y_accel": ("y_accel",),
    "y-accel": ("y_accel",),
    "y-axis": ("y_accel",),
    "y axis": ("y_accel",),
    "z_accel": ("z_accel",),
    "z-accel": ("z_accel",),
    "z-axis": ("z_accel",),
    "z axis": ("z_accel",),
    "back_accel": ("back_accel",),
    "back-accel": ("back_accel",),
    "lower back": ("back_accel",),
    "back": ("back_accel",),
    "thigh_accel": ("thigh_accel",),
    "thigh-accel": ("thigh_accel",),
    "thigh": ("thigh_accel",),
    "accel": (
        "hand_accel",
        "chest_accel",
        "ankle_accel",
        "x_accel",
        "y_accel",
        "z_accel",
        "back_accel",
        "thigh_accel",
    ),
    "accelerometer": (
        "hand_accel",
        "chest_accel",
        "ankle_accel",
        "x_accel",
        "y_accel",
        "z_accel",
        "back_accel",
        "thigh_accel",
    ),
}

UNSUPPORTED_CHANNEL_NAMES = (
    "wrist_gyro",
    "scalp_eeg",
    "phone_mic",
    "nose_temp",
    "ir_camera",
)

UNSUPPORTED_PRIMITIVES = (
    "jerk",
    "entropy",
    "step_count",
    "activity_label",
    "kalman_gain",
)

MEASUREMENT_UNITS = {
    "dominant_frequency": frozenset({"Hz"}),
    "rms_amplitude": frozenset({"raw"}),
    "peak_amplitude": frozenset({"raw"}),
    "signal_range": frozenset({"raw"}),
    "trend_ratio": frozenset({"ratio"}),
    "cross_channel_lag_ms": frozenset({"ms"}),
    "periodicity_strength": frozenset({"score_0_1"}),
    "spectral_energy_ratio_low": frozenset({"fraction", "percent"}),
}

FS_REQUIRED = frozenset({
    "dominant_frequency",
    "spectral_energy_ratio_low",
    "cross_channel_lag_ms",
})

TWO_CHANNEL = frozenset({"cross_channel_lag_ms"})

READABLE_MEASUREMENTS = {
    "dominant_frequency": ("dominant frequency", "dominant-frequency", "f_dom"),
    "rms_amplitude": ("rms amplitude", "rms-amplitude", "rms"),
    "peak_amplitude": ("peak amplitude", "peak-amplitude"),
    "signal_range": ("signal range", "signal-range", "peak-to-peak"),
    "trend_ratio": ("trend ratio", "trend-ratio"),
    "cross_channel_lag_ms": ("cross-channel lag", "cross channel lag", "lag"),
    "periodicity_strength": ("periodicity strength", "periodicity"),
    "spectral_energy_ratio_low": ("low-band energy", "low band energy", "spectral energy ratio low"),
}

__all__ = [
    "CHANNEL_ALIASES",
    "COMPARATORS",
    "CONNECTIVES",
    "FS_REQUIRED",
    "MEASUREMENTS",
    "MEASUREMENT_UNITS",
    "READABLE_MEASUREMENTS",
    "REGISTERED_CHANNELS",
    "TWO_CHANNEL",
    "UNITS",
    "UNSUPPORTED_CHANNEL_NAMES",
    "UNSUPPORTED_PRIMITIVES",
]

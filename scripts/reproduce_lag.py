"""Lag time-base evaluation-mode contracts. Synthetic windows only.

Does not rerun the 2592-item unused-window cell or write new numbers.
"""
from __future__ import annotations

from _repro import run_pytest


def main() -> int:
    return run_pytest(["tests/p2_phase2/test_lag_timebase.py"])


if __name__ == "__main__":
    raise SystemExit(main())

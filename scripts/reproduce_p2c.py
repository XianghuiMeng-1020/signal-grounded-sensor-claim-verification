"""Phase-2 contract properties: dropout refuses; AWGN/clip stay finite."""
from __future__ import annotations

from _repro import run_pytest


def main() -> int:
    return run_pytest(["tests/p2_phase2/test_degrade_contracts.py"])


if __name__ == "__main__":
    raise SystemExit(main())

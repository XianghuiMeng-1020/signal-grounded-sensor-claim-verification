"""Analytic lag oracles and independent-DSP identity checks. No table rescore."""
from __future__ import annotations

from _repro import run_pytest


def main() -> int:
    return run_pytest(
        [
            "tests/p3cr/test_lag_oracle.py",
            "tests/p4/test_lag_representation_audit.py",
            "tests/p3/test_numeric_domain.py",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Kernel, contract, Kleene, and schema unit checks. No table rescore."""
from __future__ import annotations

from _repro import run_pytest


def main() -> int:
    return run_pytest(
        [
            "tests/p2r",
            "tests/p3/test_semantic_canon.py",
            "tests/p3r_ec/test_property_contracts.py",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

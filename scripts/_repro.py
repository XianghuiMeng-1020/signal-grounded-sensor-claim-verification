"""Shared pytest runner for advertised reproduction commands.

Does not rescore frozen experiment tables. Does not change thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_pytest(paths: list[str]) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import os

    os.chdir(ROOT)
    args = ["-q", "--tb=short", *paths]
    return int(pytest.main(args))

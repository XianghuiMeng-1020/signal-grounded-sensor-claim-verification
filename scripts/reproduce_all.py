"""Run every advertised reproduction command. No experiment rescore."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reproduce_first_principles import main as first_principles  # noqa: E402
from reproduce_oracles import main as oracles  # noqa: E402
from reproduce_p2c import main as p2c  # noqa: E402
from reproduce_lag import main as lag  # noqa: E402


def main() -> int:
    for name, fn in (
        ("first_principles", first_principles),
        ("oracles", oracles),
        ("p2c", p2c),
        ("lag", lag),
    ):
        code = fn()
        if code != 0:
            print(f"FAIL: {name}", file=sys.stderr)
            return code
        print(f"PASS: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

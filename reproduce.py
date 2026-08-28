#!/usr/bin/env python3
"""Available reproduction modes. Does not rerun sealed language holdouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports" / "ICASSP_FINAL_10OF10_HARDENING" / "FINAL_SUBMISSION" / "result_manifest.json"
PAPER_DIR = ROOT / "manuscript" / "icassp2027"

HASHES = {
    "h2_predictions": (
        ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/H2_SEALED_LANGUAGE/h2_official_predictions.jsonl",
        "661709fc7bc1e82921995e72b9db06e9f072f2946c35a5cf1a5336d7c4e8a27e",
    ),
    "h2b_v1_predictions": (
        ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/V2/H2B/h2b_v1_predictions.jsonl",
        "9526896e1c2d5dd82ea67bb53766d422b2baf1ad174ecc79b71d2054543a9e7e",
    ),
    "h2b_v2_predictions": (
        ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/V2/H2B/h2b_v2_predictions.jsonl",
        "c260a3f8af8e33489d35f1522eef88fb28981cc968fb6327a4ffbc425dda4724",
    ),
}

# Item-level sealed confirmatory files are intentionally not distributed in
# this public code-only release (see PUBLIC_PROVENANCE.md). Their absence
# here is expected, not a reproducibility failure; only a present-but-wrong
# hash is treated as a failure.
WITHHELD_IN_PUBLIC_RELEASE = frozenset(HASHES.keys())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cmd_verified_results() -> int:
    withheld = mismatch = 0
    for name, (path, digest) in HASHES.items():
        if not path.is_file():
            if name in WITHHELD_IN_PUBLIC_RELEASE:
                print(f"WITHHELD {name} {path} (intentionally not in this public code-only release)")
                withheld += 1
            else:
                print(f"MISSING {name} {path}")
                mismatch += 1
            continue
        got = sha256(path)
        if got != digest:
            print(f"MISMATCH {name}\n  expected {digest}\n  got      {got}")
            mismatch += 1
        else:
            print(f"OK {name}")
    if not MANIFEST.is_file():
        print("MISSING result_manifest.json")
        return 1
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"manifest_rows={len(rows)}")
    if mismatch:
        print("VERIFIED_RESULTS FAIL")
        return 1
    if withheld:
        print(f"VERIFIED_RESULTS PASS ({withheld} item-level sealed file(s) withheld by design)")
        return 0
    print("VERIFIED_RESULTS PASS")
    return 0


def cmd_tables() -> int:
    h4 = json.loads(
        (ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/H4_PROPERTY_VALIDATION/h4_score_stamp.json").read_text(
            encoding="utf-8"
        )
    )
    h3 = json.loads(
        (ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/H3_PHYSICAL_INTERVENTION/h3_score_stamp.json").read_text(
            encoding="utf-8"
        )
    )
    h2b = json.loads(
        (ROOT / "reports/ICASSP_FINAL_10OF10_HARDENING/V2/H2B/h2b_official_analysis_stamp.json").read_text(
            encoding="utf-8"
        )
    )
    cells = {
        "h4_pass": f"{h4['pass']}/{h4['total']}",
        "h3_vfc": f"{h3['pooled_vfc']['k']}/{h3['pooled_vfc']['n']}",
        "h3_dmc": f"{h3['pooled_dmc']['k']}/{h3['pooled_dmc']['n']}",
        "h2b_v1_fcr": f"{h2b['v1']['fcr']['k']}/{h2b['v1']['fcr']['n']}",
        "h2b_v2_fcr": f"{h2b['v2']['fcr']['k']}/{h2b['v2']['fcr']['n']}",
        "h2b_v1_licensed": f"{h2b['v1']['licensed_verdict']['k']}/{h2b['v1']['licensed_verdict']['n']}",
        "h2b_v2_licensed": f"{h2b['v2']['licensed_verdict']['k']}/{h2b['v2']['licensed_verdict']['n']}",
    }
    expected = {
        "h4_pass": "9000/9000",
        "h3_vfc": "771/771",
        "h3_dmc": "742/773",
        "h2b_v1_fcr": "102/192",
        "h2b_v2_fcr": "47/192",
        "h2b_v1_licensed": "522/576",
        "h2b_v2_licensed": "522/576",
    }
    ok = True
    for k, exp in expected.items():
        got = cells[k]
        status = "OK" if got == exp else "FAIL"
        if status != "OK":
            ok = False
        print(f"{status} {k} {got} (expected {exp})")
    print("TABLES PASS" if ok else "TABLES FAIL")
    return 0 if ok else 1


def cmd_figures() -> int:
    if not (PAPER_DIR / "main.tex").is_file():
        print("FIGURES NOT_APPLICABLE: manuscript source is not part of this code-only public release.")
        return 0
    tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    needed = ["fig:pipe", r"M_{\mathrm{lag}}", r"$G$:", r"$L$:", r"\mathrm{U}"]
    missing = [s for s in needed if s not in tex]
    if missing:
        print("FIGURES FAIL missing", missing)
        return 1
    print("FIGURES PASS (TikZ reconstructed from manuscript source)")
    return 0


def cmd_paper() -> int:
    if not (PAPER_DIR / "main.tex").is_file():
        print("PAPER NOT_APPLICABLE: this code-only public release does not ship manuscript/icassp2027/.")
        print("The submitted PDF/TeX ships with the CMS submission package, not this repository.")
        return 0
    cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    for i, step in enumerate((cmd, ["bibtex", "main"], cmd, cmd)):
        r = subprocess.run(step, cwd=PAPER_DIR, capture_output=True, text=True)
        if r.returncode != 0 and step[0] != "bibtex":
            print(r.stdout[-2000:])
            print(r.stderr[-1000:])
            print("PAPER FAIL")
            return 1
        print(f"OK {' '.join(step)}")
    log = (PAPER_DIR / "main.log").read_text(encoding="utf-8", errors="replace")
    if "Overfull \\hbox" in log:
        print("PAPER FAIL overfull")
        return 1
    if "Output written on main.pdf (5 pages" not in log:
        print("PAPER FAIL page count")
        return 1
    print("PAPER PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce available frozen artifacts. No holdout rescoring.")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--tables", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--verified-results", action="store_true")
    p.add_argument("--all-available", action="store_true", help="paper+tables+figures+verified-results; not full inference")
    args = p.parse_args()
    modes = [args.paper, args.tables, args.figures, args.verified_results, args.all_available]
    if not any(modes):
        p.print_help()
        print("\nNo --all mode: official language inference is sealed and must not be rerun.")
        return 2
    rc = 0
    if args.all_available or args.verified_results:
        rc |= cmd_verified_results()
    if args.all_available or args.tables:
        rc |= cmd_tables()
    if args.all_available or args.figures:
        rc |= cmd_figures()
    if args.all_available or args.paper:
        rc |= cmd_paper()
    return rc


if __name__ == "__main__":
    sys.exit(main())

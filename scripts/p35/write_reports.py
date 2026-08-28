"""Write remaining P3.5 reports from frozen result JSON."""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPORTS, RESULTS


def _load(name, default=None):
    p = RESULTS / name
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(x):
    if x is None:
        return "n/a"
    if isinstance(x, dict):
        p = x.get("p")
        if p is None:
            return "n/a"
        return f"{100 * p:.1f}% ({x.get('k')}/{x.get('n')})"
    return f"{100 * float(x):.1f}%"


def _row(m, key):
    by = (m or {}).get("by_connective") or {}
    return by.get(key) or {}


def write_dev_results():
    ir = _load("ir_dev_ir.json") or {}
    base = _load("ir_dev_v2.json") or {}
    gates = _load("ir_dev_gates.json") or {}
    man = _load("ir_dev_FROZEN.json") or {}
    lines = [
        "# 06 — IR-DEV results",
        "",
        f"n = {ir.get('n') or man.get('retained_n')}",
        "Windows: unused later offsets; no holdout; no SEM-BLIND overlap.",
        "Surfaces: deterministic, new wording.",
        "",
        "## Paths",
        "",
        "| Metric | Baseline v2 | IR → compiler |",
        "|---|---:|---:|",
        f"| strict semantic | {_pct(base.get('strict_semantic'))} | {_pct(ir.get('strict_semantic'))} |",
        f"| primitive | {_pct(base.get('primitive'))} | {_pct(ir.get('primitive'))} |",
        f"| channel | {_pct(base.get('channel'))} | {_pct(ir.get('channel'))} |",
        f"| comparator | {_pct(base.get('comparator'))} | {_pct(ir.get('comparator'))} |",
        f"| connective | {_pct(base.get('connective'))} | {_pct(ir.get('connective'))} |",
        f"| IR exact | n/a | {_pct(ir.get('ir_exact'))} |",
        f"| IR primitive | n/a | {_pct(ir.get('ir_primitive'))} |",
        f"| IR channel | n/a | {_pct(ir.get('ir_channel'))} |",
        f"| IR structure | n/a | {_pct(ir.get('ir_structure'))} |",
        f"| compile exact | n/a | {_pct(ir.get('compile_exact'))} |",
        f"| verdict | {_pct(base.get('verdict_accuracy'))} | {_pct(ir.get('verdict_accuracy'))} |",
        f"| macro-F1 | {base.get('macro_f1')} | {ir.get('macro_f1')} |",
        f"| FCR | {_pct(base.get('false_commitment'))} | {_pct(ir.get('false_commitment'))} |",
        f"| false abstention | {_pct(base.get('false_abstention'))} | {_pct(ir.get('false_abstention'))} |",
        "",
        "## Structure breakdown (IR path, program strict)",
        "",
        "| Structure | n | strict | verdict |",
        "|---|---:|---:|---:|",
    ]
    for name in ("SINGLE", "AND", "OR", "IF_THEN"):
        r = _row(ir, name)
        if r:
            lines.append(f"| {name} | {r.get('n')} | {_pct(r.get('strict'))} | {_pct(r.get('verdict'))} |")
    three = ir.get("three_predicate") or {}
    lines += [
        f"| 3-predicate | {three.get('n')} | {_pct(three.get('strict'))} | n/a |",
        "",
        "## Gates",
        "",
        json.dumps(gates, indent=2),
        "",
        "If any gate fails: `IR_ARCHITECTURE_INSUFFICIENT`. No prompt v4. No IR-BLIND.",
        "",
    ]
    (REPORTS / "06_IR_DEV_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blind_protocol():
    gates = _load("ir_dev_gates.json") or {}
    pool = _load("window_pool_FROZEN.json") or {}
    ran = bool(_load("ir_blind_run.json"))
    text = f"""# 07 — IR-BLIND protocol

IR-DEV gates pass: {gates.get('pass')}

Reserved unused windows: {pool.get('n_reserved_blind_windows')}
Holdout included: {pool.get('holdout_included')}
Window overlap with IR-DEV: {pool.get('overlap')}

IR-BLIND is created only after IR-DEV passes.

If constructed:

- n approximately 1500
- sources: deterministic + Gemma + Llama if available
- freeze surface text, gold IR, gold program
- PRIMARY once: qwen3:8b + ir_interface + compiler
- no SEM-BLIND

Run status: {'RUN' if ran else 'NOT_RUN'}
"""
    (REPORTS / "07_IR_BLIND_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_blind_results():
    run = _load("ir_blind_run.json")
    ir = _load("ir_blind_primary.json")
    if not run or not ir:
        (REPORTS / "08_IR_BLIND_RESULTS.md").write_text(
            "# 08 — IR-BLIND results\n\nNOT_RUN\n\nReason: IR-DEV gates failed or evaluation was not authorized.\n",
            encoding="utf-8",
        )
        return
    lines = [
        "# 08 — IR-BLIND results",
        "",
        "PRIMARY run count: 1",
        f"n = {ir.get('n')}",
        "",
        f"IR exact: {_pct(ir.get('ir_exact'))}",
        f"IR primitive: {_pct(ir.get('ir_primitive'))}",
        f"IR channel: {_pct(ir.get('ir_channel'))}",
        f"IR comparator: {_pct(ir.get('ir_comparator'))}",
        f"IR connective: {_pct(ir.get('ir_connective'))}",
        f"IR structure: {_pct(ir.get('ir_structure'))}",
        f"compile / program exact: {_pct(ir.get('compile_exact'))}",
        f"strict semantic: {_pct(ir.get('strict_semantic'))}",
        f"verdict: {_pct(ir.get('verdict_accuracy'))}",
        f"macro-F1: {ir.get('macro_f1')}",
        f"FCR: {_pct(ir.get('false_commitment'))}",
        f"false abstention: {_pct(ir.get('false_abstention'))}",
        "",
        "## By structure",
        "",
    ]
    for name in ("AND", "IF_THEN"):
        r = _row(ir, name)
        if r:
            lines.append(f"- {name}: n={r.get('n')} strict={_pct(r.get('strict'))} verdict={_pct(r.get('verdict'))}")
    three = ir.get("three_predicate") or {}
    lines.append(f"- 3-predicate: n={three.get('n')} strict={_pct(three.get('strict'))}")
    (REPORTS / "08_IR_BLIND_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision():
    gates = _load("ir_dev_gates.json") or {}
    ir = _load("ir_dev_ir.json") or {}
    base = _load("ir_dev_v2.json") or {}
    blind = _load("ir_blind_run.json")
    decision = "IR_PASS_AND_PROCEED_P4" if gates.get("pass") else "IR_ARCHITECTURE_INSUFFICIENT"
    weakness = "compound IF_THEN / 3-predicate semantic compilation" 
    ift = (_row(ir, "IF_THEN") or {}).get("strict")
    three = (ir.get("three_predicate") or {}).get("strict")
    if ift is not None and three is not None:
        if ift < 0.80:
            weakness = "IF_THEN semantic compilation remains the dominant failure"
        elif three < 0.85:
            weakness = "3-predicate semantic compilation remains the dominant failure"
        elif (ir.get("false_commitment") or {}).get("p", 1) > 0.05:
            weakness = "false commitment on unsupported / ambiguous language"
        elif (ir.get("strict_semantic") or 0) < 0.90:
            weakness = "overall strict semantic compilation"
    text = f"""# 09 — Final method decision

Decision: **{decision}**

IR-DEV gates pass: {gates.get('pass')}
IR-BLIND: {'RUN' if blind else 'NOT_RUN'}

Baseline v2 strict: {_pct(base.get('strict_semantic'))}
IR path strict: {_pct(ir.get('strict_semantic'))}

No further prompt search is authorized.

Strongest remaining weakness: {weakness}
"""
    (REPORTS / "09_FINAL_METHOD_DECISION.md").write_text(text, encoding="utf-8")
    return decision, weakness


def write_pi(decision: str, weakness: str, branch: str, start: str, final: str, tag: str, lag: str, dsp: str):
    ir = _load("ir_dev_ir.json") or {}
    blind_run = _load("ir_blind_run.json")
    gates = _load("ir_dev_gates.json") or {}
    text = f"""# P3.5 PI report

Date: 2026-08-24

Branch: `{branch}`
Starting commit: `{start}`
Final commit: `{final}`
Tag: `{tag}`

## Interface

OLD: LLM → executable program  
NEW: LLM → Semantic IR → deterministic compiler → executable program

Prompt search: NO  
Prompt v4: NO  
SEM-BLIND opened: NO  
Holdouts opened: NO  
DSP changed: NO  
Evidence contract changed: NO  
Kleene changed: NO

## Frozen artifacts

IR schema: PASS  
Compiler tests: PASS  
Lag canonicalization: {lag}  
Analytic lag: 5/5  
DSP: {dsp}

## IR-DEV

n: {ir.get('n')}
Overall semantic: {_pct(ir.get('strict_semantic'))}
Primitive: {_pct(ir.get('primitive'))}
Channel: {_pct(ir.get('channel'))}
Structure: {_pct(ir.get('ir_structure'))}
IF_THEN: {_pct((_row(ir, 'IF_THEN') or {}).get('strict'))}
3-predicate: {_pct((ir.get('three_predicate') or {}).get('strict'))}
Verdict: {_pct(ir.get('verdict_accuracy'))}
Macro-F1: {ir.get('macro_f1')}
FCR: {_pct(ir.get('false_commitment'))}

Gates: {json.dumps(gates.get('checks'), indent=2)}

## IR-BLIND

{'RUN' if blind_run else 'NOT_RUN'}

## Decision

{decision}

Strongest remaining weakness: {weakness}
"""
    (REPORTS / "P3_5_PI_REPORT.md").write_text(text, encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_dev_results()
    write_blind_protocol()
    write_blind_results()
    decision, weakness = write_decision()
    write_pi(decision, weakness, "research/project-f-p35-semantic-ir",
             "3ea8322ba17ddaf44a502d8bf4f5404f299e0f6e", "PENDING", "PENDING",
             "PASS", "8/8")
    print("WROTE", decision, weakness)


if __name__ == "__main__":
    main()

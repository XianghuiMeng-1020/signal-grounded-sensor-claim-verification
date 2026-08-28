import inspect

import numpy as np
import pytest

from scripts.p2r import executor, kleene, pipeline
from scripts.p2r.extractor import extract_b6_baseline, extract_llm, llm_status
from scripts.p2r.schema import ClaimProgram, FORBIDDEN_INFERENCE_KEYS, assert_no_leakage
from scripts.p2r.validator import inference_view


def test_extractor_returns_program_not_verdict():
    prog = extract_b6_baseline(
        "The RMS amplitude of the hand channel is approximately 1.85 raw units.",
        ["hand_accel", "chest_accel"],
        100.0,
    )
    assert isinstance(prog, ClaimProgram)
    assert not hasattr(prog, "verdict") or not callable(getattr(prog, "verdict", None))
    d = prog.to_dict()
    assert "SUPPORTED" not in str(d.get("parse_status"))
    assert "verdict" not in d


def test_llm_interface_does_not_fabricate_without_model():
    st = llm_status()
    if not st["available"]:
        prog = extract_llm("anything", ["hand_accel"], 100.0)
        assert prog.parse_status == "UNAVAILABLE"
        assert prog.predicates == []


def test_inference_view_rejects_gold_keys():
    with pytest.raises(ValueError):
        assert_no_leakage({"surface_text": "x", "split": "challenge"})
    payload = inference_view("hello", ["hand_accel"], 100.0)
    assert not (set(payload) & set(FORBIDDEN_INFERENCE_KEYS))


def test_dsp_executor_rejects_raw_language():
    from scripts.p2r.schema import Predicate

    pred = Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0)
    with pytest.raises(TypeError):
        executor.execute_predicate_measurement(
            pred, {"hand_accel": np.ones(32), "surface_text": "the rms is 1"}, 100.0
        )


def test_adjudicator_source_has_no_llm():
    src = inspect.getsource(kleene)
    for token in ("openai", "OpenAI", "chat.completions", "extract_llm"):
        assert token not in src


def test_pipeline_extractor_cannot_set_verdict():
    def bad_extractor(text, chs, fs):
        return {"verdict": "SUPPORTED"}

    with pytest.raises(TypeError):
        pipeline.run_pipeline(
            "x",
            ["hand_accel"],
            100.0,
            {"hand_accel": np.ones(32)},
            bad_extractor,
        )


def test_nan_does_not_become_numeric_via_executor():
    from scripts.p2r.contracts import INSUFFICIENT_EVIDENCE
    from scripts.p2r.schema import Predicate

    x = np.ones(64)
    x[2] = np.nan
    pred = Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.0)
    res = executor.execute_predicate_measurement(pred, {"hand_accel": x}, 100.0)
    assert res.status == INSUFFICIENT_EVIDENCE
    assert res.value is None

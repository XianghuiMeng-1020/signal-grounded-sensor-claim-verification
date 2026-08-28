from scripts.p2r.ollama_adapter import parse_model_output
from scripts.p2r.schema import ClaimProgram


def test_lag_canonicalizes_missing_channel_b_when_unique():
    from scripts.p2r.validator import validate_program
    from scripts.p2r.schema import ClaimProgram, Predicate

    prog = ClaimProgram(
        "SINGLE",
        [Predicate("cross_channel_lag_ms", "x_accel", "eq", reference_value=50.0, unit="ms")],
    )
    out = validate_program(prog, ["x_accel", "y_accel"])
    assert out.parse_status == "OK"
    assert out.predicates[0].channel_b == "y_accel"


def test_parse_plain_json_program():
    raw = """{
      "connective": "SINGLE",
      "predicates": [{
        "measurement": "rms_amplitude",
        "channel_a": "hand_accel",
        "channel_b": null,
        "comparator": "eq",
        "reference_value": 1.85,
        "reference_channel": null,
        "unit": "raw"
      }],
      "parse_status": "OK",
      "parse_reason": null
    }"""
    rec = parse_model_output(raw, ["hand_accel", "chest_accel"])
    prog = rec["program"]
    assert isinstance(prog, ClaimProgram)
    assert prog.parse_status == "OK"
    assert prog.predicates[0].measurement == "rms_amplitude"
    assert rec["malformed"] is False


def test_parse_strips_qwen_think_block():
    raw = """<think>I should not judge truth.</think>
    {"connective":"SINGLE","predicates":[],"parse_status":"UNPARSEABLE","parse_reason":"unsupported_measurement"}
    """
    rec = parse_model_output(raw, ["hand_accel"])
    assert rec["program"].parse_status == "UNPARSEABLE"
    assert rec["malformed"] is False
    assert rec["program"].predicates == []


def test_non_json_is_malformed_unparseable():
    rec = parse_model_output("sorry I cannot", ["hand_accel"])
    assert rec["malformed"] is True
    assert rec["program"].parse_status == "UNPARSEABLE"
    assert rec["program"].predicates == []


def test_subset_pass_treats_zero_malformed_as_pass():
    from scripts.run_p2r_lm_local import subset_pass

    m = {
        "fields": {"exact": 0.96, "measurement": 0.96, "channel": 0.96, "connective": 0.96},
        "malformed_output_rate": 0.0,
    }
    assert subset_pass(m) is True


def test_unparseable_status_drops_predicates():
    raw = """{"connective":"SINGLE","predicates":[{"measurement":"rms_amplitude","channel_a":"hand_accel","comparator":"eq","reference_value":1}],"parse_status":"UNPARSEABLE","parse_reason":"corrupt"}"""
    rec = parse_model_output(raw, ["hand_accel"])
    assert rec["program"].parse_status == "UNPARSEABLE"
    assert rec["program"].predicates == []

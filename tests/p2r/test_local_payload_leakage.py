"""R7-style payload leakage test for the local Ollama compiler."""
from scripts.p2r.ollama_adapter import (
    FORBIDDEN_PAYLOAD_KEYS,
    build_chat_messages,
    build_inference_payload,
    payload_leakage_audit,
)
from scripts.p2r.schema import FORBIDDEN_INFERENCE_KEYS


def test_inference_payload_keys_are_deployment_legal():
    poison = {
        "surface_text": "The RMS of the hand channel is 1.2 raw units.",
        "available_channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "gold_program": {"CANARY": "GOLD_PROGRAM"},
        "gold_composed_verdict": "SUPPORTED",
        "semantic_program": {"op": "CANARY_PRIMITIVE"},
        "gt_verdict": "CONTRADICTED",
        "template_id": "CANARY_TEMPLATE",
        "paraphrase_family_id": "CANARY_PARA",
        "generation_family": "CANARY_GEN",
        "split": "challenge",
        "difficulty": "hard",
        "primitive": "CANARY_PRIM",
        "source_dataset": "PAMAP2",
        "claim_id": "canary-id",
    }
    payload = build_inference_payload(poison)
    assert set(payload) <= {"surface_text", "available_channels", "fs"}
    assert "CANARY" not in str(payload)


def test_chat_messages_do_not_carry_gold_or_split_canaries():
    poison = {
        "surface_text": "The RMS of the hand channel is 1.2 raw units.",
        "available_channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "gold_program": {"connective": "CANARY_GOLD"},
        "gold_composed_verdict": "CANARY_VERDICT_SUPPORTED",
        "semantic_program": {"predicates": [{"op": "CANARY_OP"}]},
        "template_id": "CANARY_TEMPLATE_ID",
        "paraphrase_family_id": "CANARY_PARA_FAMILY",
        "generation_family": "CANARY_GEN_FAMILY",
        "split": "final_sealed_holdout",
        "difficulty": "CANARY_DIFF",
        "primitive": "CANARY_PRIM_LABEL",
        "filename": "CANARY_FILENAME_gold.jsonl",
    }
    messages = build_chat_messages(poison, prompt_version="v1")
    blob = " ".join(m["content"] for m in messages)
    for token in (
        "CANARY_GOLD",
        "CANARY_VERDICT_SUPPORTED",
        "CANARY_OP",
        "CANARY_TEMPLATE_ID",
        "CANARY_PARA_FAMILY",
        "CANARY_GEN_FAMILY",
        "final_sealed_holdout",
        "CANARY_DIFF",
        "CANARY_PRIM_LABEL",
        "CANARY_FILENAME_gold.jsonl",
    ):
        assert token not in blob


def test_payload_audit_helper_passes_on_clean_row():
    row = {
        "surface_text": "RMS of hand is 2.0 raw units.",
        "available_channels": ["hand_accel"],
        "fs": 100.0,
    }
    result = payload_leakage_audit(row, prompt_version="v1")
    assert result["pass"] is True
    assert result["forbidden_keys_present"] == []


def test_forbidden_key_sets_cover_schema_list():
    assert set(FORBIDDEN_INFERENCE_KEYS) <= set(FORBIDDEN_PAYLOAD_KEYS)

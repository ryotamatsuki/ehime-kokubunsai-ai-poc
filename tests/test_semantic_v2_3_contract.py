from __future__ import annotations

from pathlib import Path

from semantic_atomic_v2_3 import ATOMIC_FRAME_JSON_SCHEMA_V23
from semantic_model_registry import DEFAULT_MODEL_KEY, MODEL_SPECS
from semantic_prompt_v2_3 import ATOMIC_FEW_SHOT_EXAMPLES_V23, build_atomic_frame_messages_v23


def test_v23_prompt_is_bounded_spec_authored_and_does_not_expose_state_ids():
    assert 4 <= len(ATOMIC_FEW_SHOT_EXAMPLES_V23) <= 8
    messages = build_atomic_frame_messages_v23(
        "南予で座って楽しめる催し",
        {"last_result_ids": ["001"], "selected_event_id": "001", "last_command": {"flow": "find_events", "slots": {}}},
        {"regions": ["南予"], "experience_required": ["seated"]},
    )
    rendered = "\n".join(item["content"] for item in messages)
    assert "last_result_ids" not in rendered
    assert "selected_event_id" not in rendered
    assert '"001"' not in rendered
    assert "sealed" not in rendered.lower()
    assert "holdout" not in rendered.lower()


def test_v23_schema_is_identical_for_all_model_backends():
    assert ATOMIC_FRAME_JSON_SCHEMA_V23["additionalProperties"] is False
    source = Path("semantic_v2_3_multimodel_backend.py").read_text(encoding="utf-8")
    assert '"sbintuitions/sarashina2.2-3b-instruct-v0.1"' not in source
    assert '"llm-jp/llm-jp-4-8b-instruct"' not in source
    assert "MODEL_BY_KEY" in source
    assert "build_atomic_frame_messages_v23" in source
    assert "ATOMIC_FRAME_JSON_SCHEMA_V23" in source
    assert 'format_enforcer != "lmfe"' in source
    assert 'do_sample=False' in source
    assert 'repetition_penalty=1.02' in source
    assert 'max_new_tokens=FRAME_MAX_NEW_TOKENS' in source
    assert source.count('gpu="T4"') == 2
    assert "class SarashinaSemanticV23" in source
    assert "class LlmJpSemanticV23" in source


def test_existing_model_selector_contract_is_preserved():
    assert DEFAULT_MODEL_KEY == "sarashina-2.2-3b"
    assert [spec.key for spec in MODEL_SPECS] == ["sarashina-2.2-3b", "llm-jp-4-8b"]
    streamlit_source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "DEFAULT_MODEL_KEY" in streamlit_source
    assert "semantic_model_key" in streamlit_source


def test_v23_has_no_model_specific_post_processing():
    source = Path("semantic_orchestrator_v2_3.py").read_text(encoding="utf-8")
    assert "sarashina" not in source.lower()
    assert "llm-jp" not in source.lower()
    verifier = Path("semantic_verifier_v2_3.py").read_text(encoding="utf-8")
    assert "sarashina" not in verifier.lower()
    assert "llm-jp" not in verifier.lower()


def test_v23_backend_does_not_reference_sealed_holdout_or_live_eval():
    source = Path("semantic_v2_3_multimodel_backend.py").read_text(encoding="utf-8")
    assert "unexpected_utterances_holdout" not in source
    assert "holdout.json.gz" not in source
    assert "live_eval" not in source

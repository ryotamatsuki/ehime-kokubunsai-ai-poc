from __future__ import annotations

from semantic_model_registry import (
    DEFAULT_MODEL_KEY,
    MODEL_BY_KEY,
    MODEL_SPECS,
    configured_model_specs,
    get_model_spec,
)


def test_registry_contains_only_requested_models():
    assert [spec.key for spec in MODEL_SPECS] == ["sarashina-2.2-3b", "llm-jp-4-8b"]
    assert MODEL_BY_KEY["sarashina-2.2-3b"].model_id == "sbintuitions/sarashina2.2-3b-instruct-v0.1"
    assert MODEL_BY_KEY["llm-jp-4-8b"].model_id == "llm-jp/llm-jp-4-8b-instruct"


def test_registry_has_stable_sarashina_default():
    assert DEFAULT_MODEL_KEY == "sarashina-2.2-3b"
    assert get_model_spec(None).key == DEFAULT_MODEL_KEY
    assert get_model_spec("unknown").key == DEFAULT_MODEL_KEY


def test_configured_models_depend_only_on_endpoint_secrets():
    configured = configured_model_specs(
        {
            "MODAL_V22_SARASHINA_URL": "https://example.invalid/sarashina",
            "MODAL_V22_LLMJP_URL": "",
        }
    )
    assert [spec.key for spec in configured] == ["sarashina-2.2-3b"]

    configured = configured_model_specs(
        {
            "MODAL_V22_SARASHINA_URL": "https://example.invalid/sarashina",
            "MODAL_V22_LLMJP_URL": "https://example.invalid/llmjp",
        }
    )
    assert [spec.key for spec in configured] == ["sarashina-2.2-3b", "llm-jp-4-8b"]

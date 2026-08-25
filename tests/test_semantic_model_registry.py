from __future__ import annotations

from semantic_model_registry import (
    DEFAULT_MODEL_KEY,
    MODEL_BY_KEY,
    MODEL_SPECS,
    configured_model_specs,
    get_model_spec,
    resolve_model_url,
)


def test_registry_contains_only_requested_models():
    assert [spec.key for spec in MODEL_SPECS] == ["sarashina-2.2-3b", "llm-jp-4-8b"]
    assert MODEL_BY_KEY["sarashina-2.2-3b"].model_id == "sbintuitions/sarashina2.2-3b-instruct-v0.1"
    assert MODEL_BY_KEY["llm-jp-4-8b"].model_id == "llm-jp/llm-jp-4-8b-instruct"
    assert MODEL_BY_KEY["sarashina-2.2-3b"].endpoint_url.endswith(".modal.run")
    assert MODEL_BY_KEY["llm-jp-4-8b"].endpoint_url.endswith(".modal.run")


def test_registry_has_stable_sarashina_default():
    assert DEFAULT_MODEL_KEY == "sarashina-2.2-3b"
    assert get_model_spec(None).key == DEFAULT_MODEL_KEY
    assert get_model_spec("unknown").key == DEFAULT_MODEL_KEY


def test_deployed_endpoints_make_both_models_available_without_new_streamlit_secrets():
    configured = configured_model_specs({})
    assert [spec.key for spec in configured] == ["sarashina-2.2-3b", "llm-jp-4-8b"]


def test_optional_streamlit_secret_overrides_deployed_url():
    spec = MODEL_BY_KEY["llm-jp-4-8b"]
    assert resolve_model_url(spec, {}) == spec.endpoint_url
    override = "https://override.invalid/llmjp"
    assert resolve_model_url(spec, {spec.modal_url_secret: override}) == override

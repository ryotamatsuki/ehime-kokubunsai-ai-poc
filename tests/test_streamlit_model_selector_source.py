from __future__ import annotations

from pathlib import Path


def test_streamlit_exposes_semantic_model_selector_and_v22_path():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "import semantic_model_registry" in source
    assert "from semantic_v2_2_ui import SemanticEndpointConfig, run_semantic_v22" in source
    assert "st.selectbox(" in source
    assert '"AIモデル"' in source
    assert 'key="semantic_model_key"' in source
    assert "Semantic Operations v2.2" in source
    assert "run_semantic_v22(" in source


def test_model_switch_resets_conversation_instead_of_reusing_cross_model_state():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'active_semantic_model_key' in source
    assert 'model_changed_notice' in source
    assert '_reset()' in source


def test_unconfigured_v22_endpoints_keep_legacy_poc_available():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "configured_model_specs" in source
    assert "Semantic v2.2 endpoint未設定" in source

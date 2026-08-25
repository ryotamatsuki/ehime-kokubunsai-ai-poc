"""Finalize the already-applied model selector in streamlit_app.py.

This temporary branch helper is intentionally idempotent. It does not rebuild
or reformat the Streamlit entrypoint; it only verifies the selector wiring and
ensures the selected model resolves to an optional secret override or the
stable deployed endpoint registered in semantic_model_registry.
"""

from __future__ import annotations

from pathlib import Path


PATH = Path("streamlit_app.py")
OLD_ENDPOINT = "endpoint_url = _optional_secret(selected_model.modal_url_secret)"
NEW_ENDPOINT = "endpoint_url = semantic_model_registry.resolve_model_url(selected_model, st.secrets)"


def main() -> None:
    source = PATH.read_text(encoding="utf-8")
    required_markers = (
        "import semantic_model_registry",
        "from semantic_v2_2_ui import SemanticEndpointConfig, run_semantic_v22",
        "configured_models = _configured_semantic_models()",
        'st.session_state.get("semantic_model_key")',
        'st.selectbox(\n                "AIモデル"',
        "run_semantic_v22(",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(f"selector wiring incomplete: {missing}")

    if NEW_ENDPOINT not in source:
        count = source.count(OLD_ENDPOINT)
        if count != 1:
            raise RuntimeError(f"endpoint resolver: expected exactly one old anchor, got {count}")
        source = source.replace(OLD_ENDPOINT, NEW_ENDPOINT, 1)
        PATH.write_text(source, encoding="utf-8")
        print("streamlit_app.py endpoint resolution: PATCHED")
    else:
        print("streamlit_app.py endpoint resolution: already PASS")


if __name__ == "__main__":
    main()

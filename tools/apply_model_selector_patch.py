"""Apply the model-selector wiring to the large Streamlit entrypoint.

This script is intentionally narrow and idempotent.  It exists because the
GitHub contents API replaces whole files; running the patch in Actions lets us
edit the 120kB Streamlit module without copying or reformatting unrelated UI.
"""

from __future__ import annotations

from pathlib import Path


PATH = Path("streamlit_app.py")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, got {count}")
    return source.replace(old, new, 1)


def main() -> None:
    source = PATH.read_text(encoding="utf-8")

    source = replace_once(
        source,
        "import recommendation_pending\nimport conversation_recovery\n",
        "import recommendation_pending\nimport conversation_recovery\nimport semantic_model_registry\nfrom semantic_v2_2_ui import SemanticEndpointConfig, run_semantic_v22\n",
        "imports",
    )

    source = replace_once(
        source,
        "\n\ndef _validate_modal_url(url: str) -> str:\n",
        '''\n\ndef _optional_secret(name: str) -> str | None:\n    """Read an optional project secret without making rollout all-or-nothing."""\n\n    try:\n        value = st.secrets.get(name)\n    except (KeyError, FileNotFoundError):\n        return None\n    if not isinstance(value, str) or not value.strip():\n        return None\n    return value.strip()\n\n\ndef _configured_semantic_models():\n    try:\n        return semantic_model_registry.configured_model_specs(st.secrets)\n    except (KeyError, FileNotFoundError):\n        return ()\n\n\ndef _validate_modal_url(url: str) -> str:\n''',
        "optional secrets",
    )

    command_anchor = '''    command_payload = _command_plan_payload(command) if command is not None else None\n    if command is not None and command_payload is None:\n        return None\n\n    # Current Agent D public API.  Passing a JSON mapping is intentional: the\n'''
    command_replacement = '''    command_payload = _command_plan_payload(command) if command is not None else None\n    if command is not None and command_payload is None:\n        return None\n\n    # Semantic Operations v2.2 is the common residual-language architecture.\n    # The selector changes only the remote Atomic classifier; deterministic\n    # quick actions and pending explicit CommandPlans remain local/trusted.\n    configured_models = _configured_semantic_models()\n    if command_payload is None and not skip_generation and configured_models:\n        configured_by_key = {spec.key: spec for spec in configured_models}\n        selected_key = str(st.session_state.get("semantic_model_key") or "")\n        selected_model = configured_by_key.get(selected_key) or configured_models[0]\n        endpoint_url = _optional_secret(selected_model.modal_url_secret)\n        if endpoint_url is None:\n            return _CommandOutcome(\n                flow="unsupported",\n                slots={},\n                command={"flow": "unsupported", "slots": {}, "confidence": "low"},\n                events=[],\n                near_events=[],\n                all_event_ids=[],\n                all_near_event_ids=[],\n                pairs=[],\n                message=BACKEND_FAILURE_MESSAGE,\n                handled=True,\n            )\n        try:\n            semantic_result = run_semantic_v22(\n                query,\n                state,\n                SemanticEndpointConfig(\n                    model=selected_model,\n                    url=endpoint_url,\n                    key=modal_config.key,\n                    secret=modal_config.secret,\n                ),\n                events=event_search.load_events(),\n            )\n            normalized = _normalize_command_outcome(semantic_result, None)\n            if normalized is not None:\n                return normalized\n        except Exception:\n            observation = TurnObservation(\n                query,\n                has_search_context=bool(state.get("has_last_search_context")),\n                last_result_count=int(state.get("last_result_count") or 0),\n            )\n            observation.semantic_command_called = True\n            observation.mark_fallback(\n                "semantic_v22_backend_failure",\n                error_type="semantic_v22_backend_failure",\n            )\n            observation.emit()\n        # Never switch models silently when the user explicitly selected a\n        # v2.2 backend.  A backend failure is surfaced as a handled PoC error.\n        return _CommandOutcome(\n            flow="unsupported",\n            slots={},\n            command={"flow": "unsupported", "slots": {}, "confidence": "low"},\n            events=[],\n            near_events=[],\n            all_event_ids=[],\n            all_near_event_ids=[],\n            pairs=[],\n            message=BACKEND_FAILURE_MESSAGE,\n            handled=True,\n        )\n\n    # Current Agent D public API.  Passing a JSON mapping is intentional: the\n'''
    source = replace_once(source, command_anchor, command_replacement, "v2.2 command routing")

    sidebar_anchor = '''with st.sidebar:\n    with st.container(key="iyoshirube-sidebar"):\n        st.markdown("### :material/list: 質問の例")\n'''
    sidebar_replacement = '''with st.sidebar:\n    with st.container(key="iyoshirube-sidebar"):\n        configured_semantic_models = _configured_semantic_models()\n        if configured_semantic_models:\n            configured_keys = [spec.key for spec in configured_semantic_models]\n            if st.session_state.get("semantic_model_key") not in configured_keys:\n                preferred = semantic_model_registry.DEFAULT_MODEL_KEY\n                st.session_state.semantic_model_key = (\n                    preferred if preferred in configured_keys else configured_keys[0]\n                )\n            selected_model_key = st.selectbox(\n                "AIモデル",\n                options=configured_keys,\n                format_func=lambda key: semantic_model_registry.get_model_spec(key).label,\n                key="semantic_model_key",\n            )\n            active_model_key = st.session_state.get("active_semantic_model_key")\n            if active_model_key is None:\n                st.session_state.active_semantic_model_key = selected_model_key\n            elif active_model_key != selected_model_key:\n                st.session_state.active_semantic_model_key = selected_model_key\n                selected_label = semantic_model_registry.get_model_spec(selected_model_key).label\n                st.session_state.model_changed_notice = (\n                    f"AIモデルを{selected_label}に切り替えました。会話をリセットしました。"\n                )\n                _reset()\n            notice = st.session_state.pop("model_changed_notice", None)\n            if notice:\n                st.info(str(notice))\n            selected_spec = semantic_model_registry.get_model_spec(selected_model_key)\n            st.caption(f"{selected_spec.description}")\n            st.caption("Semantic Operations v2.2 / LMFE。意味分類モデルだけを切り替えます。")\n        else:\n            st.caption("Semantic v2.2 endpoint未設定。現在は既存PoCバックエンドを使用します。")\n        st.divider()\n        st.markdown("### :material/list: 質問の例")\n'''
    source = replace_once(source, sidebar_anchor, sidebar_replacement, "sidebar selector")

    PATH.write_text(source, encoding="utf-8")
    print("streamlit_app.py model selector patch: PASS")


if __name__ == "__main__":
    main()

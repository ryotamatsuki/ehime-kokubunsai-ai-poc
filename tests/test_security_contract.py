import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_is_explicitly_fictional_and_complete():
    events = json.loads((ROOT / "data" / "events.json").read_text(encoding="utf-8"))
    assert len(events) == 30
    assert all("example.invalid" in event["公式URL"] for event in events)


def test_new_modal_identifiers_are_isolated():
    source = (ROOT / "modal_backend.py").read_text(encoding="utf-8")
    assert 'modal.App("ehime-kokubunsai-ai-poc-api")' in source
    assert '"ehime-kokubunsai-model-cache"' in source
    assert "sarashina-chat-api" not in source
    assert "sarashina-model-cache" not in source
    assert "requires_proxy_auth=True" in source


def test_streamlit_auth_and_proxy_contract():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "APP_PASSWORD" in source
    assert "hmac.compare_digest" in source
    assert "st.stop()" in source
    assert "allow_redirects=False" in source
    assert '"Modal-Key"' in source
    assert '"Modal-Secret"' in source
    assert '"Authorization"' not in source
    assert "sarashina-chat" not in source


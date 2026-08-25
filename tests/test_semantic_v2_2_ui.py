from __future__ import annotations

import pytest

from command_observability import ModalCallError
from semantic_model_registry import MODEL_SPECS
from semantic_v2_2_ui import (
    MAX_MODAL_RESULT_REDIRECTS,
    SERVICE_ID,
    SemanticEndpointConfig,
    _validate_modal_result_url,
    post_atomic_frame,
    validate_semantic_modal_url,
)


def test_semantic_endpoint_accepts_only_expected_modal_service_host():
    value = "https://owner--ehime-kokubunsai-semantic-v2-2-api-llmjpsemanticv22-semantic.modal.run"
    assert validate_semantic_modal_url(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "http://owner--ehime-kokubunsai-semantic-v2-2-api-x.modal.run",
        "https://example.com",
        "https://owner--ehime-kokubunsai-ai-poc-api-chat.modal.run",
        "https://owner--ehime-kokubunsai-semantic-v2-2-api-x.modal.run/path",
    ],
)
def test_semantic_endpoint_rejects_unexpected_urls(value):
    with pytest.raises(ValueError):
        validate_semantic_modal_url(value)


def test_modal_result_redirect_allows_query_only_on_original_host():
    base = "https://owner--ehime-kokubunsai-semantic-v2-2-api-x.modal.run"
    _validate_modal_result_url(base, f"{base}?modal-result-id=abc")
    with pytest.raises(ModalCallError):
        _validate_modal_result_url(
            base,
            "https://other--ehime-kokubunsai-semantic-v2-2-api-x.modal.run?modal-result-id=abc",
        )


def test_post_atomic_frame_follows_bounded_modal_result_redirects(monkeypatch):
    spec = MODEL_SPECS[0]
    config = SemanticEndpointConfig(
        model=spec,
        url=spec.endpoint_url,
        key="wk-test",
        secret="ws-test",
    )

    class FakeResponse:
        status_code = 200
        history = ()
        url = f"{spec.endpoint_url}?modal-result-id=abc"

        def json(self):
            return {
                "service_id": SERVICE_ID,
                "model_key": spec.key,
                "model_id": spec.model_id,
                "answer": "{}",
            }

    class FakeSession:
        def __init__(self):
            self.max_redirects = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, **kwargs):
            assert url == spec.endpoint_url
            assert self.max_redirects == MAX_MODAL_RESULT_REDIRECTS
            assert kwargs["allow_redirects"] is True
            return FakeResponse()

    monkeypatch.setattr("semantic_v2_2_ui.requests.Session", FakeSession)
    body = post_atomic_frame(config, {"query": "松山市で無料"})
    assert body["model_key"] == spec.key

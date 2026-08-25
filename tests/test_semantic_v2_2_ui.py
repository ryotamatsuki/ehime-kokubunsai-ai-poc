from __future__ import annotations

import pytest

from semantic_v2_2_ui import validate_semantic_modal_url


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

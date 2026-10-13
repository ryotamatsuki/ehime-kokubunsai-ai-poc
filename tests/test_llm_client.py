"""Stage-1 transport tests: fully offline; no provider credentials needed."""
import pytest
import requests

from llm_client import (
    GenerationRequest,
    LLMCallError,
    LLMConfigurationError,
    ModalCredentials,
    ModalTransport,
    SarashinaChatCompletionsTransport,
    SarashinaCredentials,
)


MODAL = ModalCredentials(
    "https://owner--ehime-kokubunsai-ai-poc-api.modal.run", "modal-key", "modal-secret"
)


def sara(**updates):
    fields = {
        "endpoint": "https://api.example.jp/v1/chat/completions",
        "model_id": "official-model-id-from-provider",
        "api_key": "secret-token",
        "allowed_host": "api.example.jp",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    }
    fields.update(updates)
    return SarashinaCredentials(**fields)


class Response:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def recorder(response):
    seen = []

    def post(*args, **kwargs):
        seen.append((args, kwargs))
        return response

    return post, seen


def req():
    return GenerationRequest(
        legacy_payload={"mode": "command", "query": "今治で無料は？"},
        messages=(
            {"role": "system", "content": "JSONのみを返す"},
            {"role": "user", "content": "今治で無料は？"},
        ),
        max_tokens=256,
        temperature=0,
    )


def test_modal_wire_contract_and_service_boundary():
    post, seen = recorder(Response({"service_id": "ehime-kokubunsai-ai-poc", "answer": '{"flow":"search_events"}'}))
    result = ModalTransport(MODAL, post=post).generate(req())
    assert result == '{"flow":"search_events"}'
    args, kwargs = seen[0]
    assert args == (MODAL.url,)
    assert kwargs["json"]["mode"] == "command"
    assert kwargs["headers"]["Modal-Key"] == MODAL.key
    assert kwargs["headers"]["Modal-Secret"] == MODAL.secret
    assert kwargs["allow_redirects"] is False


def test_modal_rejects_wrong_service_and_missing_payload():
    post, _ = recorder(Response({"service_id": "other-app", "answer": "ok"}))
    with pytest.raises(LLMCallError, match="provider_protocol_error"):
        ModalTransport(MODAL, post=post).generate(req())
    with pytest.raises(LLMConfigurationError, match="missing_legacy_payload"):
        ModalTransport(MODAL, post=post).generate(GenerationRequest())


@pytest.mark.parametrize("bad", [
    "http://owner--ehime-kokubunsai-ai-poc-api.modal.run",
    "https://owner--not-the-poc.modal.run",
    "https://owner--ehime-kokubunsai-ai-poc-api.modal.run?token=leak",
    "https://owner--ehime-kokubunsai-ai-poc-api.modal.run@evil.example",
])
def test_modal_invalid_host_fails_closed(bad):
    with pytest.raises(LLMConfigurationError):
        ModalTransport(ModalCredentials(bad, "k", "s"))


def test_sarashina_openai_compatible_request_and_response():
    post, seen = recorder(Response({
        "choices": [{"message": {"role": "assistant", "content": '{"flow":"search_events"}'}}]
    }))
    result = SarashinaChatCompletionsTransport(sara(), post=post).generate(req())
    assert result == '{"flow":"search_events"}'
    args, kwargs = seen[0]
    assert args == ("https://api.example.jp/v1/chat/completions",)
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert kwargs["json"]["model"] == "official-model-id-from-provider"
    assert kwargs["json"]["messages"][1]["content"] == "今治で無料は？"
    assert kwargs["json"]["stream"] is False
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (10, 90)


def test_sarashina_raw_api_key_header_is_explicit_not_guessed():
    post, seen = recorder(Response({"choices": [{"message": {"content": "ok"}}]}))
    client = SarashinaChatCompletionsTransport(
        sara(auth_header="x-api-key", auth_scheme="Raw"), post=post,
    )
    assert client.generate(req()) == "ok"
    assert seen[0][1]["headers"]["x-api-key"] == "secret-token"


@pytest.mark.parametrize("changes", [
    {"endpoint": "http://api.example.jp/v1/chat/completions"},
    {"endpoint": "https://evil.example/v1/chat/completions"},
    {"endpoint": "https://api.example.jp/v1/chat/completions?key=secret"},
    {"endpoint": "https://api.example.jp@evil.example/v1/chat/completions"},
    {"endpoint": "https://api.example.jp:8080/v1/chat/completions"},
    {"model_id": ""},
    {"api_key": ""},
    {"auth_header": "X-Unverified"},
    {"auth_scheme": "Basic"},
    {"auth_header": "Authorization", "auth_scheme": "Raw"},
])
def test_sarashina_bad_configuration_rejected(changes):
    with pytest.raises(LLMConfigurationError):
        SarashinaChatCompletionsTransport(sara(**changes))


@pytest.mark.parametrize("reply,kind", [
    ({"choices": []}, "provider_protocol_error"),
    ({"choices": [{"message": {}}]}, "empty_model_response"),
    ({"choices": [{"message": {"content": None}}]}, "empty_model_response"),
    ({"answer": "not OpenAI style"}, "provider_protocol_error"),
    (["unexpected"], "provider_protocol_error"),
])
def test_provider_response_validation(reply, kind):
    post, _ = recorder(Response(reply))
    with pytest.raises(LLMCallError) as err:
        SarashinaChatCompletionsTransport(sara(), post=post).generate(req())
    assert err.value.kind == kind


@pytest.mark.parametrize("code,expected", [(401, "4xx"), (429, "429"), (503, "5xx")])
def test_http_error_has_only_safe_metadata(code, expected):
    post, _ = recorder(Response({"error": "secret-token should never escape"}, status=code))
    with pytest.raises(LLMCallError) as err:
        SarashinaChatCompletionsTransport(sara(), post=post).generate(req())
    assert err.value.kind == "provider_http_error"
    assert err.value.status_class == expected
    assert "secret-token" not in str(err.value)


def test_timeout_sanitized():
    def post(*args, **kwargs):
        raise requests.Timeout("https://api.example.jp?api_key=secret-token")
    with pytest.raises(LLMCallError, match="provider_timeout") as err:
        SarashinaChatCompletionsTransport(sara(), post=post).generate(req())
    assert "secret-token" not in str(err.value)


def test_json_error_is_not_returned_to_caller():
    post, _ = recorder(Response(ValueError("secret-token")))
    with pytest.raises(LLMCallError, match="provider_protocol_error"):
        SarashinaChatCompletionsTransport(sara(), post=post).generate(req())


@pytest.mark.parametrize("messages", [(), ({"role": "tool", "content": "bad"},), ({"role": "user", "content": ""},)])
def test_message_contract_rejects_bad_inputs_without_http(messages):
    def forbidden(*args, **kwargs):
        raise AssertionError("network should not be touched")
    with pytest.raises(LLMConfigurationError):
        SarashinaChatCompletionsTransport(sara(), post=forbidden).generate(
            GenerationRequest(messages=messages)
        )

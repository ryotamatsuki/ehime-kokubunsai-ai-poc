"""Opt-in, fail-closed LLM transports for the cultural-event PoC.

Stage 1 only: the Streamlit application still uses Modal directly. The
Sarashina adapter assumes an OpenAI-compatible *chat-completions response*;
the free beta's exact contract has not yet been independently verified.
Configure only values confirmed in the provider's own API console/docs.
No requests are made during module import or transport construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import requests

SERVICE_ID = "ehime-kokubunsai-ai-poc"
_MODAL_HOST = re.compile(
    r"^[a-z0-9-]+--ehime-kokubunsai-ai-poc-api(?:-[a-z0-9-]+)*\\.modal\\.run$"
)
_HOST = re.compile(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")
_ALLOWED_AUTH_HEADERS = frozenset({"Authorization", "x-api-key", "api-key"})


class LLMConfigurationError(ValueError):
    """Invalid or missing configuration; never include credential values."""


class LLMCallError(RuntimeError):
    """Provider failure classified without retaining response bodies or keys."""

    def __init__(self, kind: str, *, status_class: str | None = None) -> None:
        self.kind = kind
        self.status_class = status_class
        super().__init__(kind)


@dataclass(frozen=True)
class GenerationRequest:
    """Both representations are prepared by the caller during migration.

    Modal consumes legacy_payload; a chat-completions provider consumes
    messages. No provider may silently synthesize one from the other.
    """

    legacy_payload: Mapping[str, Any] | None = None
    messages: Sequence[Mapping[str, str]] = ()
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class ModalCredentials:
    url: str
    key: str
    secret: str


@dataclass(frozen=True)
class SarashinaCredentials:
    endpoint: str
    model_id: str
    api_key: str
    allowed_host: str
    auth_header: str
    auth_scheme: str  # "Bearer" or "Raw"; select only from official docs


class LLMTransport(Protocol):
    def generate(self, request: GenerationRequest) -> str | Mapping[str, Any]: ...


def _https_url(endpoint: str, host: str) -> str:
    if not isinstance(endpoint, str) or not isinstance(host, str):
        raise LLMConfigurationError("invalid_endpoint")
    endpoint, host = endpoint.strip(), host.strip().lower()
    try:
        parsed = urlsplit(endpoint)
        parsed_host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise LLMConfigurationError("invalid_endpoint") from exc
    if (
        not _HOST.fullmatch(host)
        or parsed.scheme != "https"
        or parsed_host != host
        or port not in (None, 443)
        or not parsed.path.startswith("/")
        or parsed.path in ("", "/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise LLMConfigurationError("invalid_endpoint")
    return endpoint


def _status_class(code: int) -> str:
    if code == 429:
        return "429"
    return f"{code // 100}xx"


def _send(
    post: Callable[..., Any],
    *,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        response = post(
            url, headers=dict(headers), json=dict(body),
            timeout=(10, 90), allow_redirects=False,
        )
    except requests.Timeout as exc:
        raise LLMCallError("provider_timeout") from exc
    except requests.RequestException as exc:
        raise LLMCallError("provider_transport_error") from exc
    code = getattr(response, "status_code", None)
    if not isinstance(code, int):
        raise LLMCallError("provider_protocol_error")
    if not 200 <= code < 300:
        raise LLMCallError("provider_http_error", status_class=_status_class(code))
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise LLMCallError("provider_protocol_error") from exc
    if not isinstance(payload, dict):
        raise LLMCallError("provider_protocol_error")
    return payload


class ModalTransport:
    """Keeps the *existing* authenticated Modal proxy wire contract."""

    def __init__(
        self, credentials: ModalCredentials, *,
        post: Callable[..., Any] | None = None,
    ) -> None:
        parsed = urlsplit(credentials.url)
        host = (parsed.hostname or "").lower()
        if (
            not _MODAL_HOST.fullmatch(host)
            or parsed.scheme != "https"
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")
            or not credentials.key or not credentials.secret
        ):
            raise LLMConfigurationError("invalid_modal_configuration")
        self._creds = credentials
        self._post = post if post is not None else requests.post

    def generate(self, request: GenerationRequest) -> str | Mapping[str, Any]:
        if request.legacy_payload is None:
            raise LLMConfigurationError("missing_legacy_payload")
        payload = _send(
            self._post, url=self._creds.url,
            headers={
                "Modal-Key": self._creds.key,
                "Modal-Secret": self._creds.secret,
                "Content-Type": "application/json",
            },
            body=request.legacy_payload,
        )
        if payload.get("service_id") != SERVICE_ID:
            raise LLMCallError("provider_protocol_error")
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
        if isinstance(answer, dict):
            return answer
        raise LLMCallError("empty_model_response")


class SarashinaChatCompletionsTransport:
    """Conditional OpenAI-style adapter; not an assertion about beta support.

    Do NOT activate for the live PoC without verifying endpoint, auth and
    exact model ID from the Sarashina beta's official technical documentation.
    """

    def __init__(
        self, credentials: SarashinaCredentials, *,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self._endpoint = _https_url(credentials.endpoint, credentials.allowed_host)
        if not credentials.model_id or not credentials.model_id.strip():
            raise LLMConfigurationError("missing_model_id")
        if not credentials.api_key or not credentials.api_key.strip():
            raise LLMConfigurationError("missing_api_key")
        if credentials.auth_header not in _ALLOWED_AUTH_HEADERS:
            raise LLMConfigurationError("unsupported_auth_header")
        if credentials.auth_scheme not in {"Bearer", "Raw"}:
            raise LLMConfigurationError("unsupported_auth_scheme")
        if credentials.auth_header == "Authorization" and credentials.auth_scheme != "Bearer":
            raise LLMConfigurationError("unsupported_auth_scheme")
        self._creds = credentials
        self._post = post if post is not None else requests.post

    def generate(self, request: GenerationRequest) -> str:
        if not isinstance(request.max_tokens, int) or not 1 <= request.max_tokens <= 4096:
            raise LLMConfigurationError("invalid_max_tokens")
        if not isinstance(request.temperature, (int, float)) or not 0 <= request.temperature <= 2:
            raise LLMConfigurationError("invalid_temperature")
        if not request.messages or len(request.messages) > 32:
            raise LLMConfigurationError("invalid_messages")
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if not isinstance(message, Mapping):
                raise LLMConfigurationError("invalid_messages")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise LLMConfigurationError("invalid_messages")
            if not content.strip() or len(content) > 20000:
                raise LLMConfigurationError("invalid_messages")
            messages.append({"role": role, "content": content})
        auth = (
            "Bearer " + self._creds.api_key
            if self._creds.auth_scheme == "Bearer" else self._creds.api_key
        )
        payload = _send(
            self._post,
            url=self._endpoint,
            headers={self._creds.auth_header: auth, "Content-Type": "application/json"},
            body={
                "model": self._creds.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False,
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMCallError("provider_protocol_error")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LLMCallError("provider_protocol_error")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMCallError("provider_protocol_error")
        output = message.get("content")
        if not isinstance(output, str) or not output.strip():
            raise LLMCallError("empty_model_response")
        return output

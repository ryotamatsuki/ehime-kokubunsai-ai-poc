"""UI-side adapter for selectable Semantic Operations v2.2 backends."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import requests

import event_search
from app_config import POC_REFERENCE_DATE
from command_observability import ModalCallError, modal_status_class
from semantic_model_registry import SemanticModelSpec
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22, SemanticV22Result


SERVICE_ID = "ehime-kokubunsai-semantic-v2-2-api"
MAX_MODAL_RESULT_REDIRECTS = 4
_EXPECTED_HOST_RE = re.compile(
    r"^[a-z0-9-]+--ehime-kokubunsai-semantic-v2-2-api(?:-[a-z0-9-]+)*\.modal\.run$"
)


@dataclass(frozen=True)
class SemanticEndpointConfig:
    model: SemanticModelSpec
    url: str
    key: str
    secret: str


def validate_semantic_modal_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not _EXPECTED_HOST_RE.fullmatch(host)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid Semantic v2.2 Modal endpoint")
    return str(url).strip()


def _validate_modal_result_url(initial_url: str, result_url: str) -> None:
    """Allow Modal's long-running result URL only on the original endpoint host.

    Modal Web Functions return HTTP 303 after their synchronous HTTP window and
    point back to the original URL with a result query parameter.  We follow
    that mechanism, but never permit the proxy credentials to cross hosts.
    """

    initial = urlparse(validate_semantic_modal_url(initial_url))
    current = urlparse(str(result_url).strip())
    if (
        current.scheme != "https"
        or (current.hostname or "").lower() != (initial.hostname or "").lower()
        or current.username
        or current.password
        or current.path not in {"", "/"}
        or current.fragment
    ):
        raise ModalCallError("semantic_v22_redirect_error")


def post_atomic_frame(config: SemanticEndpointConfig, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        endpoint_url = validate_semantic_modal_url(config.url)
        with requests.Session() as session:
            # Bound the total number of Modal 303 result hops while still
            # supporting cold model starts that exceed the 150-second HTTP
            # response window of a Web Function.
            session.max_redirects = MAX_MODAL_RESULT_REDIRECTS
            response = session.post(
                endpoint_url,
                headers={
                    "Modal-Key": config.key,
                    "Modal-Secret": config.secret,
                    "Content-Type": "application/json",
                },
                json={**dict(payload), "format_enforcer": "lmfe"},
                timeout=300,
                allow_redirects=True,
            )
        for hop in (*response.history, response):
            _validate_modal_result_url(endpoint_url, hop.url)
        if response.status_code < 200 or response.status_code >= 300:
            raise ModalCallError(
                "semantic_v22_http_error",
                status_class=modal_status_class(response.status_code),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ModalCallError(
                "semantic_v22_protocol_error",
                status_class=modal_status_class(response.status_code),
            ) from exc
        if not isinstance(body, Mapping) or body.get("service_id") != SERVICE_ID:
            raise ModalCallError(
                "semantic_v22_protocol_error",
                status_class=modal_status_class(response.status_code),
            )
        if body.get("model_key") != config.model.key or body.get("model_id") != config.model.model_id:
            raise ModalCallError(
                "semantic_v22_model_mismatch",
                status_class=modal_status_class(response.status_code),
            )
        if not isinstance(body.get("answer"), str):
            raise ModalCallError(
                "semantic_v22_empty_model_response",
                status_class=modal_status_class(response.status_code),
            )
        return body
    except requests.Timeout as exc:
        raise ModalCallError("semantic_v22_timeout") from exc
    except requests.TooManyRedirects as exc:
        raise ModalCallError("semantic_v22_timeout") from exc
    except requests.RequestException as exc:
        raise ModalCallError("semantic_v22_http_error") from exc


def run_semantic_v22(
    query: str,
    state: Mapping[str, Any] | None,
    config: SemanticEndpointConfig,
    *,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> SemanticV22Result:
    orchestrator = SemanticOperationsOrchestratorV22(
        frame_call=lambda payload: post_atomic_frame(config, payload),
        reference_date=POC_REFERENCE_DATE,
        events=events or event_search.load_events(),
    )
    return orchestrator.handle_query(query, state)


__all__ = [
    "MAX_MODAL_RESULT_REDIRECTS",
    "SERVICE_ID",
    "SemanticEndpointConfig",
    "post_atomic_frame",
    "run_semantic_v22",
    "validate_semantic_modal_url",
]

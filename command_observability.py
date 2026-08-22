"""Bounded observability for the semantic-command boundary.

The command generator receives user language but this module deliberately does
not retain the raw utterance.  Production events contain only a coarse query
category and a short hash so failures can be correlated without turning logs
into a transcript store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from typing import Any, Mapping


LOGGER = logging.getLogger("ehime.semantic_command")
UNKNOWN_BUILD = "unknown"
ERROR_TYPES = frozenset(
    {
        "command_import_error",
        "modal_timeout",
        "modal_http_error",
        "modal_protocol_error",
        "empty_model_response",
        "invalid_json",
        "schema_validation_error",
        "repair_failed",
        "orchestrator_exception",
        "missing_search_context",
        "unsupported_semantic_flow",
        "execution_error",
        "invalid_command",
    }
)


class ModalCallError(RuntimeError):
    """Safe, classified failure from the authenticated Modal proxy call."""

    def __init__(self, error_type: str, *, status_class: str | None = None) -> None:
        self.error_type = error_type if error_type in ERROR_TYPES else "modal_http_error"
        self.status_class = status_class
        super().__init__(self.error_type)


def _normalized_query(value: Any) -> str:
    return " ".join(str(value or "").strip().split())[:1200]


def query_hash(value: Any) -> str:
    """Return a correlation hash without retaining the query text."""

    digest = hashlib.sha256(_normalized_query(value).encode("utf-8")).hexdigest()
    return digest[:16]


def query_category(value: Any) -> str:
    """Coarsely classify a turn for metrics; never return the raw utterance."""

    text = _normalized_query(value).replace(" ", "")
    if not text:
        return "empty"
    if any(marker in text.lower() for marker in ("systemprompt", "prompt", "指示を無視", "内部ロジック")):
        return "security_or_injection"
    if any(marker in text for marker in ("株価", "天気", "python", "飲食店", "宿泊")):
        return "out_of_domain"
    if any(marker in text for marker in ("材料", "基準", "観点", "ロジック", "根拠", "理由", "顔ぶれ")):
        return "explanation_or_reference"
    if any(marker in text for marker in ("無料", "屋内", "松山", "今治", "新居浜", "予約", "座って")):
        return "search_or_refinement"
    return "other"


def build_sha() -> str:
    """Resolve a deployment fingerprint without hard-coding a revision."""

    for name in ("STREAMLIT_BUILD_SHA", "BUILD_SHA", "GIT_SHA", "GITHUB_SHA"):
        value = os.getenv(name, "").strip()
        if value:
            return value[:12]
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=0.25,
        ).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return UNKNOWN_BUILD
    return value[:12] if value else UNKNOWN_BUILD


@dataclass
class TurnObservation:
    """Mutable in-process observation for one command turn."""

    query: str
    has_search_context: bool = False
    last_result_count: int = 0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    build_sha: str = field(default_factory=build_sha)
    started: float = field(default_factory=time.perf_counter)
    deterministic_route: str | None = None
    deterministic_confidence: str | None = None
    semantic_command_called: bool = False
    semantic_command_attempts: int = 0
    semantic_command_repaired: bool = False
    semantic_command_error_type: str | None = None
    generated_flow: str | None = None
    validated_flow: str | None = None
    final_flow: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    modal_status_class: str | None = None
    generator_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    _logged: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.query_hash = query_hash(self.query)
        self.query_category = query_category(self.query)

    def mark_generation(self, generated: Any, latency_ms: float) -> None:
        self.semantic_command_called = True
        self.semantic_command_attempts = int(getattr(generated, "attempts", 0) or 0)
        self.semantic_command_repaired = bool(getattr(generated, "repaired", False))
        self.semantic_command_error_type = (
            getattr(generated, "error_type", None)
            or getattr(generated, "first_error_type", None)
        )
        self.modal_status_class = getattr(generated, "modal_status_class", None)
        generated_plan = getattr(generated, "plan", None)
        generated_flow = getattr(generated_plan, "flow", None)
        if not self.semantic_command_error_type and isinstance(generated_flow, str):
            self.generated_flow = generated_flow
        self.generator_latency_ms = round(float(latency_ms), 3)

    def mark_fallback(self, reason: str, *, error_type: str | None = None) -> None:
        self.fallback_used = True
        self.fallback_reason = str(reason)[:80]
        if error_type in ERROR_TYPES:
            self.semantic_command_error_type = error_type

    def finish(self, *, flow: str, status: str) -> None:
        self.validated_flow = flow
        self.final_flow = flow
        if status in {"unavailable", "execution_error", "invalid_command"}:
            self.mark_fallback(status, error_type=status if status in ERROR_TYPES else "orchestrator_exception")
        self.total_latency_ms = round((time.perf_counter() - self.started) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "build_sha": self.build_sha,
            "query_category": self.query_category,
            "query_hash": self.query_hash,
            "deterministic_route": self.deterministic_route,
            "deterministic_confidence": self.deterministic_confidence,
            "semantic_command_called": self.semantic_command_called,
            "semantic_command_attempts": self.semantic_command_attempts,
            "semantic_command_repaired": self.semantic_command_repaired,
            "semantic_command_error_type": self.semantic_command_error_type,
            "generated_flow": self.generated_flow,
            "validated_flow": self.validated_flow,
            "final_flow": self.final_flow,
            "has_search_context": self.has_search_context,
            "last_result_count": self.last_result_count,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "modal_status_class": self.modal_status_class,
            "generator_latency_ms": self.generator_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }

    def emit(self) -> dict[str, Any]:
        """Emit at most one structured event and return its safe payload."""

        payload = self.to_dict()
        if not self._logged:
            LOGGER.info("semantic_command_turn %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
            self._logged = True
        return payload


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, ModalCallError):
        return exc.error_type
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text:
        return "modal_timeout"
    if "json" in name or "json" in text:
        return "invalid_json"
    if "validation" in name or "schema" in text or "contract" in text:
        return "schema_validation_error"
    if "import" in name or "module" in text:
        return "command_import_error"
    return "orchestrator_exception"


def classify_validation_error(exc: BaseException) -> str:
    path = str(getattr(exc, "path", "") or "").lower()
    text = str(exc).lower()
    if path == "json" or "json" in text or "decode" in text:
        return "invalid_json"
    return "schema_validation_error"


def modal_status_class(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    if 500 <= status_code < 600:
        return "5xx"
    return "other"


__all__ = [
    "ERROR_TYPES",
    "ModalCallError",
    "TurnObservation",
    "build_sha",
    "classify_exception",
    "classify_validation_error",
    "modal_status_class",
    "query_category",
    "query_hash",
]

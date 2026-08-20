"""Pure presentation helpers for the いよしるべ character layer.

This module intentionally has no Streamlit, network, or model dependency.  It
keeps avatar selection deterministic and makes the UI metadata safe to test
without importing the application entrypoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


IYOSHIRUBE_NAME = "いよしるべ"
IYOSHIRUBE_TAGLINE = "愛媛の文化、いっしょに探してみん？"
IYOSHIRUBE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "iyoshirube"
IYOSHIRUBE_UI_ASSET_DIR = IYOSHIRUBE_ASSET_DIR.parent / "ui"
IYOSHIRUBE_WAVE_ASSET = IYOSHIRUBE_UI_ASSET_DIR / "seigaiha_wave.png"

EMOTION_NORMAL = "normal"
EMOTION_HAPPY = "happy"
EMOTION_THINKING = "thinking"
EMOTION_TROUBLED = "troubled"
IYOSHIRUBE_AVATAR_STATES = (
    EMOTION_NORMAL,
    EMOTION_HAPPY,
    EMOTION_THINKING,
    EMOTION_TROUBLED,
)
IYOSHIRUBE_EMOTIONS = frozenset(IYOSHIRUBE_AVATAR_STATES)

IYOSHIRUBE_AVATARS: dict[str, Path] = {
    emotion: IYOSHIRUBE_ASSET_DIR / f"{emotion}.png"
    for emotion in IYOSHIRUBE_AVATAR_STATES
}
IYOSHIRUBE_FALLBACK_AVATAR = "🧭"

_NORMAL_FLOWS = frozenset(
    {
        "general_faq",
        "faq",
        "event_detail",
        "detail_followup",
        "reference_followup",
        "count_events",
        "count",
        "attribute",
        "generic_scope",
        "scope_search",
        "nearby",
        "unsupported",
    }
)
_SUCCESS_FLOWS = frozenset(
    {
        "find_events",
        "search",
        "legacy_search",
        "agentic_search",
        "agentic_exact",
        "recommend_next",
        "recommend_similar",
        "plan_event_pair",
        "event_pair",
        "pair_recommendation",
    }
)
_PENDING_FLOWS = frozenset(
    {
        "pending",
        "clarification",
        "date_pending",
        "time_pending",
        "event_selection_pending",
        "recommendation_pending",
        "recommend_next_pending",
        "recommend_similar_pending",
        "recommend_next_without_selection",
        "recommend_similar_without_selection",
        "ask_date",
        "ask_time",
        "ask_event",
    }
)
_TROUBLED_FLOWS = frozenset(
    {
        "backend_failure",
        "modal_failure",
        "command_preparation_failed",
        "invalid_date",
        "invalid_time",
        "event_selection_failed",
        "event_not_found",
        "pair_failure",
        "error",
        "failure",
    }
)
_FIXED_FAILURE_PHRASES = (
    "案内の準備に失敗したけん",
    "日付を解釈できませんでした",
    "時刻を解釈できませんでした",
    "イベントの開催時間内で",
    "条件に合うイベントは見つかりませんでした",
    "候補は見つかりませんでした",
)
_FIXED_PENDING_PHRASES = (
    "何日に行く予定",
    "何時ごろ見終わる予定",
    "必要な条件をもう少し教えて",
    "番号かイベント名を教えて",
    "どのイベントを確認したいか",
)


def normalize_emotion(value: object) -> str:
    """Return one of the four supported states, defaulting to ``normal``."""

    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in IYOSHIRUBE_EMOTIONS:
            return candidate
    return EMOTION_NORMAL


def avatar_path(emotion: object = EMOTION_NORMAL) -> Path | None:
    """Resolve an existing local avatar asset without raising on omissions."""

    path = IYOSHIRUBE_AVATARS.get(normalize_emotion(emotion))
    if path is None or not path.is_file():
        return None
    return path


def avatar_for_streamlit(emotion: object = EMOTION_NORMAL) -> str:
    """Return a local path for Streamlit, or a safe emoji fallback."""

    path = avatar_path(emotion)
    return str(path) if path is not None else IYOSHIRUBE_FALLBACK_AVATAR


def emotion_from_message(message: Mapping[str, Any] | object) -> str:
    """Read persisted UI metadata while supporting the old message shape."""

    if isinstance(message, Mapping):
        return normalize_emotion(message.get("emotion"))
    return EMOTION_NORMAL


def model_history(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Strip presentation metadata before a history is sent to the model."""

    history: list[dict[str, str]] = []
    for message in messages[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _state_value(
    state: Mapping[str, Any] | None,
    *names: str,
    default: Any = None,
) -> Any:
    if state is None:
        return default
    for name in names:
        if name in state:
            return state[name]
    return default


def _flow_key(flow: object, route_action: object) -> str:
    candidate = flow if isinstance(flow, str) and flow.strip() else route_action
    return str(candidate or "").strip().lower()


def _has_flow_token(flow: str, *tokens: str) -> bool:
    return any(token in flow for token in tokens)


def select_assistant_emotion(
    answer: str | None = None,
    result_count: int | None = None,
    flow: str | None = None,
    route_action: str | None = None,
    pending: bool = False,
    backend_failure: bool = False,
    pair_result_count: int | None = None,
    *,
    exact_result_count: int | None = None,
    recommendation_result_count: int | None = None,
    pending_state: Mapping[str, Any] | None = None,
    clarification_required: bool = False,
    modal_failure: bool = False,
    command_preparation_failed: bool = False,
    invalid_input: bool = False,
    event_selection_failed: bool = False,
    explicit_error: bool = False,
    success: bool | None = None,
    state: Mapping[str, Any] | None = None,
) -> str:
    """Select an avatar from structured turn state.

    The order is intentionally fixed: troubled, explicit pending/thinking,
    happy, then normal.  Answer text is used only for a small set of known
    legacy fixed messages; it is never used as a general sentiment classifier.
    """

    state_flow = _state_value(state, "flow", "active_flow")
    state_route = _state_value(state, "route_action", "action_type")
    flow_key = _flow_key(flow or state_flow, route_action or state_route)

    if result_count is None:
        result_count = _state_value(state, "result_count", "results_count")
    if exact_result_count is None:
        exact_result_count = _state_value(state, "exact_result_count", "exact_count")
    if recommendation_result_count is None:
        recommendation_result_count = _state_value(
            state,
            "recommendation_result_count",
            "recommendation_count",
        )
    if pair_result_count is None:
        pair_result_count = _state_value(state, "pair_result_count", "pair_count")
    if not pending:
        pending = bool(_state_value(state, "pending", "is_pending", default=False))
    if pending_state is None:
        candidate_pending = _state_value(state, "pending_state", "pending_command")
        if isinstance(candidate_pending, Mapping):
            pending_state = candidate_pending
    clarification_required = clarification_required or bool(
        _state_value(state, "clarification_required", "needs_clarification", default=False)
    )
    backend_failure = backend_failure or bool(
        _state_value(state, "backend_failure", "failure", default=False)
    )
    invalid_input = invalid_input or bool(
        _state_value(state, "invalid_input", "invalid", default=False)
    )
    event_selection_failed = event_selection_failed or bool(
        _state_value(state, "event_selection_failed", default=False)
    )
    explicit_error = explicit_error or bool(
        _state_value(state, "explicit_error", "error", default=False)
    )
    if success is None:
        state_success = _state_value(state, "success")
        success = state_success if isinstance(state_success, bool) else None

    result_count = _nonnegative_int(result_count)
    exact_result_count = _nonnegative_int(exact_result_count)
    recommendation_result_count = _nonnegative_int(recommendation_result_count)
    pair_result_count = _nonnegative_int(pair_result_count)

    fixed_failure = (
        flow_key not in _NORMAL_FLOWS
        and isinstance(answer, str)
        and any(phrase in answer for phrase in _FIXED_FAILURE_PHRASES)
    )
    fixed_pending = isinstance(answer, str) and any(
        phrase in answer for phrase in _FIXED_PENDING_PHRASES
    )
    flow_failure = flow_key in _TROUBLED_FLOWS or _has_flow_token(
        flow_key,
        "failure",
        "invalid",
        "not_found",
        "selection_failed",
    )
    flow_pending = flow_key in _PENDING_FLOWS or _has_flow_token(
        flow_key,
        "pending",
        "clarif",
        "ask_date",
        "ask_time",
        "ask_event",
    )
    # Explicit pending state outranks a successful candidate set.  A generic
    # clarification flag is intentionally handled after the happy branch so
    # a successful search followed by a refinement invitation stays happy.
    pending_active = bool(pending or pending_state is not None or flow_pending)
    is_pair_flow = flow_key in {"plan_event_pair", "event_pair", "pair_recommendation"}
    is_recommendation_flow = flow_key in {
        "recommend_next",
        "recommend_similar",
        "recommend_next_pending",
        "recommend_similar_pending",
    }
    is_search_flow = flow_key in _SUCCESS_FLOWS or not flow_key

    # Troubled has the highest priority, including failure + clarification.
    # Recommendation and pair branches may retain the previous search cards
    # in ``result_count``; their dedicated counts below are authoritative.
    authoritative_search_count = (
        exact_result_count if exact_result_count is not None else result_count
    )
    zero_search = (
        is_search_flow
        and not is_recommendation_flow
        and not is_pair_flow
        and authoritative_search_count == 0
        and not pending_active
    )
    zero_exact = (
        is_search_flow
        and not is_recommendation_flow
        and not is_pair_flow
        and exact_result_count == 0
        and not pending_active
    )
    zero_recommendation = (
        is_recommendation_flow
        and recommendation_result_count == 0
        and not pending_active
    )
    zero_pair = is_pair_flow and pair_result_count == 0 and not pending_active
    if (
        backend_failure
        or modal_failure
        or command_preparation_failed
        or invalid_input
        or event_selection_failed
        or explicit_error
        or flow_failure
        or fixed_failure
        or zero_search
        or zero_exact
        or zero_recommendation
        or zero_pair
    ):
        return EMOTION_TROUBLED

    # A normal FAQ/detail/count response must not become happy merely because
    # its source event set happens to contain one record.
    if pending_active:
        return EMOTION_THINKING

    positive_results = any(
        count is not None and count > 0
        for count in (
            result_count,
            exact_result_count,
            recommendation_result_count,
            pair_result_count,
        )
    )
    successful_search = (
        (success is True and (is_search_flow or is_recommendation_flow or is_pair_flow))
        or positive_results
        and (is_search_flow or is_recommendation_flow or is_pair_flow)
    )
    if successful_search and flow_key not in _NORMAL_FLOWS:
        return EMOTION_HAPPY

    if clarification_required or fixed_pending:
        return EMOTION_THINKING

    return EMOTION_NORMAL

"""Streamlit UI for the separate Ehime cultural-event PoC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from collections.abc import Mapping
from html import escape
import importlib
import inspect
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st

import agent_orchestrator
import event_search
import event_details
import event_recommendation
import experience_preferences
import recommendation_pending
import conversation_recovery
from command_observability import ModalCallError, TurnObservation, build_sha, modal_status_class
from agent_planner import ModalConfig
from conversation_router import route_conversation
from app_config import (
    REGION_CITIES,
    MAX_RESULT_SET_SIZE,
    MAX_WRITER_CANDIDATES,
    POC_REFERENCE_DATE,
    POC_REFERENCE_DATE_TEXT,
    RESULT_PAGE_SIZE,
)
from command_models import (
    ALLOWED_AGE_GROUPS,
    ALLOWED_AUDIENCES,
    ALLOWED_DETAIL_FIELDS,
    ALLOWED_EXPERIENCE_CONCEPTS,
    ALLOWED_REFERENCE_KINDS,
    ALLOWED_TIME_SLOTS,
    ALLOWED_VENUES,
    CANONICAL_MUNICIPALITIES,
    COMMAND_SLOT_FIELDS,
    GENRE_VALUES,
    MAX_REFERENCE_INDEX,
    MAX_VISIT_COUNT,
)
from flow_registry import FLOW_REGISTRY
from event_image_assets import event_image_path
from result_pagination import next_visible_count, normalize_visible_count, visible_items
from result_context import classify_result_context_source, transition_result_context
from iyoshirube_ui import (
    EMOTION_NORMAL,
    EMOTION_THINKING,
    IYOSHIRUBE_NAME,
    IYOSHIRUBE_TAGLINE,
    IYOSHIRUBE_WAVE_ASSET,
    avatar_path,
    avatar_for_streamlit,
    emotion_from_message,
    model_history,
    select_assistant_emotion,
)


PAGE_TITLE = "🎭 伊予の文化案内人"
SERVICE_ID = "ehime-kokubunsai-ai-poc"
BUILD_SHA = build_sha()


@dataclass(frozen=True)
class _UICommandPlan:
    """Small UI-side CommandPlan adapter used when the command package is absent.

    The eventual command package may provide its own CommandPlan dataclass.
    Keeping this payload JSON-shaped lets the UI remain importable before that
    package is merged and gives the optional bridge a stable input contract.
    """

    flow: str
    slots: dict[str, Any]
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow,
            "slots": dict(self.slots),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class _QuickAction:
    label: str
    command: _UICommandPlan
    fallback_query: str


@dataclass
class _CommandOutcome:
    """Normalized result of the optional Command/Flow executor."""

    flow: str
    slots: dict[str, Any]
    command: dict[str, Any]
    events: list[dict[str, Any]]
    near_events: list[dict[str, Any]]
    all_event_ids: list[str]
    all_near_event_ids: list[str]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    total_matches: int | None = None
    filters: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    question: str | None = None
    writer: Any = None
    message: str | None = None
    relaxed_condition: str | None = None
    handled: bool = True


@dataclass(frozen=True)
class _CommandRender:
    answer: str
    results: list[dict[str, Any]]
    near_results: list[dict[str, Any]]
    relaxed_condition: str | None
    filters: dict[str, Any] | None
    selected_event: dict[str, Any] | None
    pair_results: list[tuple[dict[str, Any], dict[str, Any]]]
    pending_state: dict[str, Any] | None


@dataclass(frozen=True)
class _PendingCommandDate:
    applicable: bool
    invalid: bool = False
    command: dict[str, Any] | None = None
    value: date | None = None


@dataclass(frozen=True)
class _PendingCommandTime:
    applicable: bool
    invalid: bool = False
    command: dict[str, Any] | None = None
    value: int | None = None


_ALLOWED_COMMAND_FLOWS = frozenset(FLOW_REGISTRY)
_COMMAND_SLOT_KEYS = frozenset(COMMAND_SLOT_FIELDS)
_COMMAND_LIST_SLOTS = frozenset(
    {
        "dates",
        "municipalities",
        "regions",
        "genres",
        "topics",
        "experience_required",
        "experience_preferred",
        "experience_excluded",
        "time_slots",
        "detail_fields",
    }
)
_COMMAND_VENUES = frozenset(ALLOWED_VENUES)
_COMMAND_AGE_GROUPS = frozenset(ALLOWED_AGE_GROUPS)
_COMMAND_AUDIENCES = frozenset(ALLOWED_AUDIENCES)
_COMMAND_DETAIL_FIELDS = frozenset(ALLOWED_DETAIL_FIELDS)
_COMMAND_REFERENCE_KINDS = frozenset(ALLOWED_REFERENCE_KINDS)


QUICK_ACTIONS = (
    _QuickAction(
        label="今日のイベント",
        command=_UICommandPlan(
            flow="find_events",
            slots={"dates": [POC_REFERENCE_DATE.isoformat()]},
        ),
        fallback_query="今日やっているイベント",
    ),
    _QuickAction(
        label="子どもと楽しむ",
        command=_UICommandPlan(
            flow="find_events",
            slots={"audience": "family"},
        ),
        fallback_query="子どもと楽しめるイベント",
    ),
    _QuickAction(
        label="雨でもOK",
        command=_UICommandPlan(
            flow="find_events",
            slots={"venue": "indoor", "rain_preferred": True},
        ),
        fallback_query="雨でも楽しめる屋内イベント",
    ),
    _QuickAction(
        label="無料イベント",
        command=_UICommandPlan(
            flow="find_events",
            slots={"entry_free": True},
        ),
        fallback_query="無料のイベント",
    ),
    _QuickAction(
        label="伝統芸能",
        command=_UICommandPlan(
            flow="find_events",
            slots={"genres": ["伝統芸能"]},
        ),
        fallback_query="伝統芸能のイベント",
    ),
    # The old label said "地域から探す" but silently sent 南予.  Keep the
    # quick action useful while making the selected region explicit.
    _QuickAction(
        label="南予のイベント",
        command=_UICommandPlan(
            flow="find_events",
            slots={"regions": ["南予"]},
        ),
        fallback_query="南予でイベント",
    ),
)
# Preserve the old constant for lightweight callers and regression fixtures.
QUICK_QUESTIONS = tuple(
    (action.label, action.fallback_query) for action in QUICK_ACTIONS
)
GENERIC_SCOPE_MESSAGE = "このPoCは文化祭イベントを探す機能の検証が中心なんよ。関連するイベントなら探せるよ。"
NEARBY_MESSAGE = "「近く」の範囲はまだ自動判定していません。探したい市町を教えてみん？"
NO_RESULT_MESSAGE = "条件に合うイベントは見つかりませんでした。日付・地域・料金のどれかを少し変えて探してみん？"
BACKEND_FAILURE_MESSAGE = "案内の準備に失敗したけん、条件を短くしてもう一度試してみて。"
EVENT_FACT_FIELDS = ("イベント名", "日時", "場所", "料金", "公式URL")
_EXPECTED_MODAL_HOST_RE = re.compile(
    r"^[a-z0-9-]+--ehime-kokubunsai-ai-poc-api(?:-[a-z0-9-]+)*\.modal\.run$"
)


st.set_page_config(page_title=PAGE_TITLE, page_icon="🎭", layout="wide")


def _inject_ui_css() -> None:
    """Make the standard Streamlit layout follow the supplied UI reference."""

    st.markdown(
        """
        <style>
        :root {
            --iyoshirube-navy: #1f3c72;
            --iyoshirube-blue: #315a97;
            --iyoshirube-border: #d7dce5;
            --iyoshirube-soft: #f6f8fc;
        }
        [data-testid="stAppViewContainer"] { background: #ffffff; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 5.5rem;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--iyoshirube-soft);
            border-right: 1px solid #e2e6ee;
        }
        .st-key-iyoshirube-sidebar {
            padding: 0.25rem 0.15rem 1rem;
        }
        .st-key-iyoshirube-hero {
            overflow: hidden;
        }
        .st-key-iyoshirube-wave img {
            max-width: 100%;
            height: auto;
            opacity: 0.72;
        }
        .st-key-iyoshirube-sidebar button {
            min-height: 2.75rem;
            margin: 0.18rem 0;
            border: 1px solid var(--iyoshirube-border);
            border-radius: 0.75rem;
            background: #ffffff;
            color: #263650;
            text-align: left;
            box-shadow: 0 1px 2px rgba(31, 60, 114, 0.04);
        }
        .st-key-iyoshirube-sidebar button:hover {
            border-color: #9bb1d4;
            color: var(--iyoshirube-navy);
        }
        [data-testid="stChatMessage"] { padding: 0.28rem 0.45rem; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.55;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--iyoshirube-border);
            border-radius: 0.8rem;
        }
        .iyoshirube-event-fact {
            display: flex;
            gap: 0.38rem;
            align-items: flex-start;
            margin: 0.13rem 0;
            color: #334155;
            font-size: 0.86rem;
            line-height: 1.42;
        }
        .iyoshirube-event-fact-icon {
            min-width: 1.1rem;
            color: var(--iyoshirube-blue);
            font-weight: 700;
            text-align: center;
        }
        .iyoshirube-event-overview {
            margin-top: 0.32rem;
            padding-top: 0.32rem;
            border-top: 1px solid #edf0f5;
            color: #526174;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        [class*="st-key-iyoshirube-event-image-"] img {
            aspect-ratio: 4 / 5;
            object-fit: cover;
            border-radius: 0.55rem;
        }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 0.8rem;
                padding-bottom: 4.5rem;
            }
            .st-key-iyoshirube-sidebar button { min-height: 2.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_ui_css()


def _required_secret(name: str) -> str:
    """Read a required project secret without exposing its value."""

    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        st.error("このPoCは現在利用できません。管理者が設定を確認してください。")
        st.stop()
    if not isinstance(value, str) or not value.strip():
        st.error("このPoCは現在利用できません。管理者が設定を確認してください。")
        st.stop()
    return value.strip()


def _validate_modal_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not _EXPECTED_MODAL_HOST_RE.fullmatch(host)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        st.error("新PoC用のModal接続先が正しく設定されていません。")
        st.stop()
    return url


def _value(value: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either a mapping or a small result dataclass."""

    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            converted = asdict(value)
        except (TypeError, ValueError):
            return None
        return converted if isinstance(converted, dict) else None
    return None


@lru_cache(maxsize=8)
def _optional_entrypoint(
    module_name: str,
    function_names: tuple[str, ...],
) -> Any:
    """Load a new Command API only when it is available.

    The semantic-command work is being landed by several agents.  The UI must
    therefore tolerate a checkout in which the command package is not yet
    present, or in which an intermediate module is not importable.  Import
    failures are deliberately converted into ``None`` so the existing
    Agentic/legacy path remains the safe fallback.
    """

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    for function_name in function_names:
        candidate = getattr(module, function_name, None)
        if callable(candidate):
            return candidate
    return None


def _command_entrypoint() -> Any:
    return _optional_entrypoint(
        "command_orchestrator",
        (
            "handle_command",
            "handle_command_query",
            "handle_turn",
            "run_command",
            "execute_command",
            "orchestrate_command",
        ),
    )


@lru_cache(maxsize=1)
def _command_orchestrator_class() -> Any:
    try:
        module = importlib.import_module("command_orchestrator")
    except Exception:
        return None
    candidate = getattr(module, "CommandOrchestrator", None)
    return candidate if callable(candidate) else None


def _post_command_modal(
    modal_config: ModalConfig,
    payload: Mapping[str, Any],
) -> Any:
    """Call the existing authenticated Modal proxy for Command generation."""

    try:
        response = requests.post(
            modal_config.url,
            headers={
                "Modal-Key": modal_config.key,
                "Modal-Secret": modal_config.secret,
                "Content-Type": "application/json",
            },
            json=dict(payload),
            timeout=300,
            allow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise ModalCallError("modal_http_error", status_class=modal_status_class(response.status_code))
        try:
            body = response.json()
        except ValueError as exc:
            raise ModalCallError("modal_protocol_error", status_class=modal_status_class(response.status_code)) from exc
        if not isinstance(body, Mapping) or body.get("service_id") != SERVICE_ID:
            raise ModalCallError("modal_protocol_error", status_class=modal_status_class(response.status_code))
        answer = body.get("answer")
        if not isinstance(answer, (str, Mapping)):
            raise ModalCallError("empty_model_response", status_class=modal_status_class(response.status_code))
        return answer
    except requests.Timeout as exc:
        raise ModalCallError("modal_timeout") from exc
    except requests.RequestException as exc:
        raise ModalCallError("modal_http_error") from exc


def _pair_entrypoint() -> Any:
    """Resolve the optional deterministic pair executor from Agent E."""

    for module_name in ("event_pair_recommendation", "event_recommendation"):
        candidate = _optional_entrypoint(
            module_name,
            ("recommend_event_pairs", "plan_event_pair"),
        )
        if candidate is not None:
            return candidate
    return None


def _normalize_command_slots(value: Any) -> dict[str, Any] | None:
    """Validate the untrusted CommandSlots boundary without executing tools."""

    raw = _mapping(value)
    if raw is None:
        return {} if value in (None, {}) else None
    if set(raw) - _COMMAND_SLOT_KEYS:
        return None

    slots: dict[str, Any] = {}
    for key, item in raw.items():
        # The native CommandSlots dataclass serializes omitted optional slots
        # as None and empty tuples.  Treat those canonical defaults as absent
        # while keeping explicit False values (notably refine_previous).
        if item is None or (
            key in _COMMAND_LIST_SLOTS
            and isinstance(item, (list, tuple))
            and not item
        ):
            continue
        if key in _COMMAND_LIST_SLOTS:
            if not isinstance(item, (list, tuple)) or len(item) > 20:
                return None
            if not all(isinstance(entry, str) and entry.strip() for entry in item):
                return None
            values = [str(entry).strip() for entry in item]
            if key == "dates":
                try:
                    values = [date.fromisoformat(entry).isoformat() for entry in values]
                except ValueError:
                    return None
            elif key == "municipalities":
                canonical: list[str] = []
                for entry in values:
                    municipality = entry
                    if municipality not in CANONICAL_MUNICIPALITIES:
                        return None
                    canonical.append(municipality)
                values = list(dict.fromkeys(canonical))
            elif key == "regions":
                if not all(entry in REGION_CITIES for entry in values):
                    return None
                values = list(dict.fromkeys(values))
            elif key == "genres":
                if not all(entry in GENRE_VALUES for entry in values):
                    return None
                values = list(dict.fromkeys(values))
            elif key == "time_slots":
                if not all(entry in ALLOWED_TIME_SLOTS for entry in values):
                    return None
            elif key == "detail_fields":
                if not all(entry in _COMMAND_DETAIL_FIELDS for entry in values):
                    return None
            elif key in {
                "experience_required",
                "experience_preferred",
                "experience_excluded",
            }:
                if not all(entry in ALLOWED_EXPERIENCE_CONCEPTS for entry in values):
                    return None
                if len(set(values)) != len(values):
                    return None
            else:
                values = list(dict.fromkeys(values))
            slots[key] = values
            continue

        if key == "audience":
            if not isinstance(item, str) or item not in _COMMAND_AUDIENCES:
                return None
            slots[key] = item
        elif key == "age_group":
            if not isinstance(item, str) or item not in _COMMAND_AGE_GROUPS:
                return None
            slots[key] = item
        elif key == "age_intent":
            if item not in {"eligible", "recommended"}:
                return None
            slots[key] = item
        elif key == "venue":
            if not isinstance(item, str) or item not in _COMMAND_VENUES:
                return None
            slots[key] = item
        elif key in {"entry_free", "paid_only", "reservation_required", "rain_preferred", "refine_previous"}:
            if not isinstance(item, bool):
                return None
            slots[key] = item
        elif key in {"age", "max_entry_fee", "time_after", "visit_count", "reference_index"}:
            if isinstance(item, bool) or not isinstance(item, int):
                return None
            upper = {
                "age": 120,
                "max_entry_fee": 1_000_000,
                "time_after": 24 * 60,
                "visit_count": MAX_VISIT_COUNT,
                "reference_index": MAX_REFERENCE_INDEX,
            }[key]
            lower = 1 if key == "visit_count" else 0
            if not lower <= item <= upper:
                return None
            slots[key] = item
        elif key in {"topics", "event_name"}:
            # Topics are semantic labels, never direct tool names or event
            # facts.  Keep them bounded for the optional adapter.
            if key == "topics":
                if not isinstance(item, (list, tuple)) or len(item) > 8:
                    return None
                if not all(isinstance(entry, str) and 0 < len(entry.strip()) <= 80 for entry in item):
                    return None
                slots[key] = list(dict.fromkeys(entry.strip() for entry in item))
            elif isinstance(item, str) and 0 < len(item.strip()) <= 240:
                slots[key] = item.strip()
            else:
                return None
        elif key == "reference_kind":
            if not isinstance(item, str) or item not in _COMMAND_REFERENCE_KINDS:
                return None
            slots[key] = item

    return slots


def _command_plan_payload(value: Any) -> dict[str, Any] | None:
    raw = _mapping(value)
    if raw is None:
        return None
    if set(raw) - {"flow", "slots", "confidence"}:
        return None
    flow = raw.get("flow")
    slots = _normalize_command_slots(raw.get("slots", {}))
    confidence = raw.get("confidence", "medium")
    if flow not in _ALLOWED_COMMAND_FLOWS or slots is None:
        return None
    if flow == "plan_event_pair":
        if len(slots.get("dates", [])) > 1:
            return None
        if slots.get("visit_count") not in (None, 2):
            return None
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {"flow": flow, "slots": slots, "confidence": confidence}


def _external_command_plan(payload: dict[str, Any]) -> Any:
    """Adapt the JSON payload to Agent A's dataclass when it exists."""

    try:
        module = importlib.import_module("command_models")
    except Exception:
        return payload
    plan_class = getattr(module, "CommandPlan", None)
    slots_class = getattr(module, "CommandSlots", None)
    if not callable(plan_class):
        return payload
    slots_value: Any = payload["slots"]
    if callable(slots_class):
        try:
            slots_value = slots_class(**payload["slots"])
        except (TypeError, ValueError):
            slots_value = payload["slots"]
    try:
        return plan_class(
            flow=payload["flow"],
            slots=slots_value,
            confidence=payload.get("confidence", "medium"),
        )
    except (TypeError, ValueError):
        return payload


def _invoke_with_signature(function: Any, values: dict[str, Any]) -> Any:
    """Call a future-agent API without guessing unsupported keyword names."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        try:
            return function(
                values.get("query", ""),
                values.get("state", {}),
                values.get("reference_date"),
                values.get("modal_config"),
            )
        except Exception:
            return None

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        kwargs = dict(values)
    else:
        kwargs = {
            name: values[name]
            for name in signature.parameters
            if name in values
        }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and name not in kwargs
    ]
    if missing:
        return None
    try:
        return function(**kwargs)
    except Exception:
        # Command output is untrusted and optional.  A failed new path must
        # never prevent the legacy path from serving the PoC.
        return None


def _command_state(
    previous_results: list[dict[str, Any]],
    pending_command: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_event = st.session_state.get("selected_event") or {}
    selected_event_id = st.session_state.get("selected_event_id") or selected_event.get("id")
    pending_plan = (
        pending_command.get("command")
        if pending_command and isinstance(pending_command.get("command"), Mapping)
        else None
    )
    search_context = conversation_recovery.SearchContext.from_value(
        st.session_state.get("last_search_context")
    )
    result_ids = list(
        st.session_state.get("last_result_ids") or _event_ids(previous_results)
    )
    state: dict[str, Any] = {
        "reference_date": POC_REFERENCE_DATE.isoformat(),
        "selected_event_id": str(selected_event_id or "") or None,
        "last_result_ids": result_ids,
        "last_command": pending_plan or st.session_state.get("last_command"),
        "active_flow": pending_command.get("flow") if pending_command else None,
        "pending_slots": dict(pending_plan.get("slots", {})) if pending_plan else {},
        "pending_required_slots": list(pending_command.get("missing_slots", []))
        if pending_command
        else [],
        # The raw context is passed only to the trusted Python orchestrator so
        # its fixed recovery executor can render grounded evidence. The
        # orchestrator's generation projection deliberately excludes it.
        "last_search_context": search_context.to_dict() if search_context else None,
        "last_action": st.session_state.get("last_action"),
        "has_last_search_context": search_context is not None,
        "last_result_count": (
            search_context.total_matches
            if search_context is not None
            else len(result_ids)
        ),
    }
    return state


def _call_new_command(
    query: str,
    *,
    state: dict[str, Any],
    modal_config: ModalConfig,
    command: Mapping[str, Any] | None = None,
    skip_generation: bool = False,
) -> _CommandOutcome | None:
    """Invoke Agent D's Command API if present, then normalize its result."""

    command_payload = _command_plan_payload(command) if command is not None else None
    if command is not None and command_payload is None:
        return None

    # Current Agent D public API.  Passing a JSON mapping is intentional: the
    # repository's command_orchestrator owns its own validated CommandPlan
    # class, while command_models.py is an optional contract mirror.
    orchestrator_class = _command_orchestrator_class()
    if orchestrator_class is not None:
        try:
            orchestrator = orchestrator_class(
                modal_call=lambda payload: _post_command_modal(modal_config, payload),
                reference_date=POC_REFERENCE_DATE,
                events=event_search.load_events(),
            )
            pending_state = bool(
                state.get("active_flow")
                and state.get("pending_required_slots")
                and state.get("last_command")
            )
            native_plan = None if skip_generation and pending_state else command_payload
            raw = orchestrator.handle_query(
                query,
                state,
                command_plan=native_plan,
            )
            outcome = _normalize_command_outcome(raw, command_payload)
            # Some intermediate CommandOrchestrator revisions still include
            # the registry's required ``dates`` slot after the pending date
            # fast path has filled it.  Complete that narrow pair flow with
            # Agent E's deterministic executor instead of asking the user for
            # the same date again.
            if (
                skip_generation
                and outcome is not None
                and outcome.flow == "plan_event_pair"
                and outcome.pending is not None
                and outcome.slots.get("dates")
            ):
                pair_outcome = _run_optional_pair_executor(outcome.command)
                if pair_outcome is not None:
                    return pair_outcome
            return outcome
        except Exception:
            # A broken or partially deployed Command path falls back below to
            # the existing Agentic/legacy route.
            observation = TurnObservation(
                query,
                has_search_context=bool(state.get("has_last_search_context")),
                last_result_count=int(state.get("last_result_count") or 0),
            )
            observation.semantic_command_called = True
            observation.mark_fallback("orchestrator_exception", error_type="orchestrator_exception")
            observation.emit()
            return None

    # Compatibility adapter for an intermediate checkout that exposes a
    # function rather than CommandOrchestrator.
    function = _command_entrypoint()
    if function is None:
        return None
    external_plan = (
        _external_command_plan(command_payload) if command_payload is not None else None
    )
    values = {
        "query": query,
        "user_query": query,
        "prompt": query,
        "command": command_payload,
        "command_payload": command_payload,
        "command_plan": external_plan,
        "plan": external_plan or command_payload,
        "state": state,
        "conversation_state": state,
        "reference_date": POC_REFERENCE_DATE,
        "modal_config": modal_config,
        "config": modal_config,
        "skip_generation": skip_generation,
        "generate": not skip_generation,
    }
    raw = _invoke_with_signature(function, values)
    return _normalize_command_outcome(raw, command_payload)


def _event_key(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        value = _mapping(value) or {}
    if isinstance(value, Mapping):
        for key in ("id", "event_id", "eventId"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and str(candidate).strip():
                return str(candidate).strip()
        url = value.get("公式URL")
        if isinstance(url, str) and url.rstrip("/"):
            return url.rstrip("/").rsplit("/", 1)[-1]
        name = value.get("イベント名")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _event_catalog() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    events = [dict(event) for event in event_search.load_events()]
    by_key: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = _event_key(event)
        if event_id:
            by_key[event_id] = event
        name = event.get("イベント名")
        if isinstance(name, str):
            by_key[name] = event
    return events, by_key


def _ground_event(value: Any, by_key: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = _event_key(value)
    if key is None:
        return None
    event = by_key.get(key)
    return dict(event) if event is not None else None


def _ground_event_list(value: Any, by_key: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping) and not any(
        key in value for key in ("id", "event_id", "eventId", "イベント名", "公式URL")
    ):
        value = value.get("events", value.get("event_ids", []))
    if isinstance(value, (str, Mapping)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    grounded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        event = _ground_event(item, by_key)
        if event is None:
            continue
        event_id = _event_key(event) or ""
        if event_id not in seen:
            grounded.append(event)
            seen.add(event_id)
    return grounded


def _event_ids(events: Any) -> list[str]:
    """Return stable, ordered, duplicate-free event IDs from grounded values."""

    if not isinstance(events, (list, tuple)):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        event_id = _event_key(event)
        if event_id and event_id not in seen:
            ids.append(event_id)
            seen.add(event_id)
    return ids


def _ground_ordered_ids(
    value: Any,
    by_key: Mapping[str, dict[str, Any]],
) -> list[str]:
    return _event_ids(_ground_event_list(value, by_key))


def _events_for_ids(ids: list[str]) -> list[dict[str, Any]]:
    """Rehydrate cards from the local catalog in the supplied order."""

    if not ids:
        return []
    try:
        _, by_key = _event_catalog()
    except (OSError, TypeError, ValueError):
        return []
    return _ground_event_list(ids, by_key)


def _ground_pairs(value: Any, by_key: Mapping[str, dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("pairs", value.get("event_pairs", []))
    if not isinstance(value, (list, tuple)):
        return []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_pair in value:
        if is_dataclass(raw_pair) and not isinstance(raw_pair, type):
            raw_pair = _mapping(raw_pair) or {}
        first: Any = None
        second: Any = None
        if isinstance(raw_pair, Mapping):
            first = raw_pair.get("first_event_id", raw_pair.get("first", raw_pair.get("event_a")))
            second = raw_pair.get("second_event_id", raw_pair.get("second", raw_pair.get("event_b")))
            if first is None or second is None:
                ids = raw_pair.get("event_ids", raw_pair.get("events"))
                if isinstance(ids, (list, tuple)) and len(ids) >= 2:
                    first, second = ids[0], ids[1]
        elif isinstance(raw_pair, (list, tuple)) and len(raw_pair) >= 2:
            first, second = raw_pair[0], raw_pair[1]
        first_event = _ground_event(first, by_key)
        second_event = _ground_event(second, by_key)
        if first_event is not None and second_event is not None:
            pairs.append((first_event, second_event))
    return pairs


def _normalize_command_outcome(
    raw: Any,
    command_payload: dict[str, Any] | None,
) -> _CommandOutcome | None:
    if raw is None or raw is False:
        return None
    raw_map = _mapping(raw)
    if raw_map is not None and raw_map.get("handled") is False:
        return None
    status = _value(raw, "status", default=None)
    if status == "clarification" and _value(raw, "error", default=None):
        # Modal/generator failure is allowed to fall back to the existing
        # Agentic/legacy route.  A clean unsupported/injection clarification
        # is handled below without executing a search.
        return None
    plan = _value(raw, "command", "command_plan", "plan", default=None)
    plan_map = _mapping(plan)
    flow = _value(raw, "flow", default=None) or _value(plan_map, "flow", default=None)
    slots_value = _value(raw, "slots", default=None)
    if slots_value is None:
        slots_value = _value(plan_map, "slots", default={})
    slots = _normalize_command_slots(slots_value)
    if flow is None and status in {"clarification", "unsupported"}:
        flow = "unsupported"
        slots = {}
    if flow not in _ALLOWED_COMMAND_FLOWS or slots is None:
        return None
    normalized_command = command_payload or _command_plan_payload(
        {
            "flow": flow,
            "slots": slots,
            "confidence": _value(plan_map, "confidence", default="medium"),
        }
    )
    if normalized_command is None:
        return None
    result_value = _value(raw, "result", default=None)
    events_raw = _value(raw, "events", "exact_events", "results", "candidates", default=None)
    if events_raw in (None, []):
        events_raw = _value(result_value, "events", "exact_events", "results", default=None)
        if events_raw is None and isinstance(result_value, (list, tuple)):
            events_raw = result_value
    near_raw = _value(raw, "near_events", "relaxed_events", "near_results", default=None)
    if near_raw in (None, []):
        near_raw = _value(result_value, "near_events", "relaxed_events", "near_results", default=[])
    pairs_raw = _value(raw, "pairs", "pair_results", "event_pairs", default=None)
    if pairs_raw in (None, []):
        pairs_raw = _value(result_value, "pairs", "pair_results", "event_pairs", default=None)
        if pairs_raw is None and isinstance(result_value, (list, tuple)):
            pairs_raw = result_value
    try:
        _, by_key = _event_catalog()
    except (OSError, ValueError, TypeError):
        return None
    all_ids_raw = _value(
        raw,
        "all_event_ids",
        "ordered_event_ids",
        "result_ids",
        default=None,
    )
    if all_ids_raw is None:
        all_ids_raw = _value(
            result_value,
            "all_event_ids",
            "ordered_event_ids",
            "result_ids",
            default=None,
        )
    all_near_ids_raw = _value(
        raw,
        "all_near_event_ids",
        "ordered_near_event_ids",
        "near_result_ids",
        default=None,
    )
    if all_near_ids_raw is None:
        all_near_ids_raw = _value(
            result_value,
            "all_near_event_ids",
            "ordered_near_event_ids",
            "near_result_ids",
            default=None,
        )
    all_event_ids = _ground_ordered_ids(
        all_ids_raw if all_ids_raw is not None else events_raw,
        by_key,
    )
    all_near_event_ids = _ground_ordered_ids(
        all_near_ids_raw if all_near_ids_raw is not None else near_raw,
        by_key,
    )
    events = _ground_event_list(
        all_ids_raw if all_ids_raw is not None else events_raw,
        by_key,
    )
    near_events = _ground_event_list(
        all_near_ids_raw if all_near_ids_raw is not None else near_raw,
        by_key,
    )
    pairs = _ground_pairs(pairs_raw, by_key)
    filters = _value(raw, "filters", "search_filters", default=None)
    filters = dict(filters) if isinstance(filters, Mapping) else None
    try:
        known_filter_keys = set(event_search.SearchFilters().to_dict())
        if filters is None or set(filters) - known_filter_keys:
            filters = event_search.command_slots_to_search_filters(
                slots,
                flow=str(flow),
                reference_date=POC_REFERENCE_DATE,
            ).to_dict()
    except (AttributeError, TypeError, ValueError):
        # Keep the result usable when an older event_search checkout lacks
        # the semantic adapter; the main renderer will simply omit unsafe
        # filter state from the next turn.
        pass
    total_matches = _value(raw, "total_matches", "count", default=None)
    if total_matches is None:
        total_matches = _value(result_value, "total_matches", "count", default=None)
    if (
        isinstance(total_matches, bool)
        or not isinstance(total_matches, int)
        or not 0 <= total_matches <= MAX_RESULT_SET_SIZE
    ):
        total_matches = None
    pending_raw = _value(raw, "pending", "pending_state", default=None)
    missing_slots = _value(raw, "missing_slots", "required_slots", default=[])
    if not isinstance(missing_slots, (list, tuple)):
        missing_slots = []
    missing_slots = [
        str(slot) for slot in missing_slots if str(slot) in _COMMAND_SLOT_KEYS
    ][:8]
    pending: dict[str, Any] | None = None
    if isinstance(pending_raw, Mapping) or pending_raw is True or missing_slots:
        pending = dict(pending_raw) if isinstance(pending_raw, Mapping) else {}
        pending["missing_slots"] = missing_slots or list(pending.get("missing_slots", []))
        pending["awaiting"] = pending.get("awaiting") or (
            "dates" if "dates" in pending["missing_slots"] else ""
        )
    writer = _value(raw, "writer", "writer_output", default=None)
    question = _value(raw, "question", "clarification", "pending_question", default=None)
    if not isinstance(question, str) or len(question) > 500:
        question = None
    message = _value(raw, "message", "answer", "text", default=None)
    if message is None:
        message = _value(result_value, "message", "answer", "text", default=None)
    if not isinstance(message, str) or len(message) > 500:
        message = None
    relaxed_condition = _value(raw, "relaxed_condition", default=None)
    if not isinstance(relaxed_condition, str) or len(relaxed_condition) > 120:
        relaxed_condition = None
    has_result = bool(
        events
        or near_events
        or pairs
        or pending
        or question
        or message
        or flow in {"explain_search", "explain_result"}
        or flow == "unsupported"
    )
    if not has_result:
        return None
    return _CommandOutcome(
        flow=str(flow),
        slots=slots,
        command=normalized_command,
        events=events,
        near_events=near_events,
        all_event_ids=all_event_ids,
        all_near_event_ids=all_near_event_ids,
        pairs=pairs,
        total_matches=total_matches,
        filters=filters,
        pending=pending,
        question=question,
        writer=writer,
        message=message,
        relaxed_condition=relaxed_condition,
    )


def _search_result(
    query: str,
    *,
    previous_filters: dict[str, object] | None = None,
    inherit_previous: bool = False,
) -> event_search.SearchResult:
    """Run the local structured search and retain its explanation metadata."""

    return event_search.search_events(
        query,
        reference_date=POC_REFERENCE_DATE,
        previous_filters=previous_filters,
        inherit_previous=inherit_previous,
        limit=MAX_RESULT_SET_SIZE,
    )


def _search_candidates(query: str) -> list[dict[str, object]]:
    """Backward-compatible helper for callers that only need exact cards."""

    return list(_search_result(query).events[:MAX_WRITER_CANDIDATES])


def _quick_action_fallback(action: _QuickAction) -> _CommandOutcome:
    """Execute a structured quick action locally when the new API is absent."""

    search_result = _search_result(action.fallback_query)
    return _CommandOutcome(
        flow=action.command.flow,
        slots=dict(action.command.slots),
        command=action.command.to_dict(),
        events=list(search_result.events),
        near_events=list(search_result.near_matches),
        all_event_ids=list(search_result.all_event_ids),
        all_near_event_ids=list(search_result.all_near_event_ids),
        pairs=[],
        total_matches=search_result.total_matches,
        filters=search_result.filters.to_dict(),
        relaxed_condition=search_result.relaxed_condition,
        message=search_result.message,
    )


def _pending_command_date(
    pending_command: Mapping[str, Any] | None,
    query: str,
) -> _PendingCommandDate:
    """Parse a date only while a Command flow explicitly asks for one."""

    if not isinstance(pending_command, Mapping) or pending_command.get("kind") != "flow":
        return _PendingCommandDate(False)
    command = _command_plan_payload(pending_command.get("command"))
    if command is None:
        return _PendingCommandDate(False)
    missing_slots = pending_command.get("missing_slots", [])
    awaiting = str(pending_command.get("awaiting") or "")
    if not (
        "dates" in missing_slots
        or awaiting in {"date", "dates"}
        or (command["flow"] == "plan_event_pair" and not command["slots"].get("dates"))
    ):
        return _PendingCommandDate(False)
    parsed = event_recommendation.parse_recommendation_date_answer(
        query,
        POC_REFERENCE_DATE,
    )
    if not parsed.is_date_like:
        return _PendingCommandDate(False)
    if parsed.invalid or parsed.value is None:
        return _PendingCommandDate(True, invalid=True, command=command)
    slots = dict(command["slots"])
    slots["dates"] = [parsed.value.isoformat()]
    return _PendingCommandDate(
        True,
        command={
            "flow": command["flow"],
            "slots": slots,
            "confidence": command.get("confidence", "medium"),
        },
        value=parsed.value,
    )


def _pending_command_time(
    pending_command: Mapping[str, Any] | None,
    query: str,
) -> _PendingCommandTime:
    """Parse a time only while a recommend-next flow asks for one."""

    if not isinstance(pending_command, Mapping) or pending_command.get("kind") != "flow":
        return _PendingCommandTime(False)
    command = _command_plan_payload(pending_command.get("command"))
    if command is None or command["flow"] != "recommend_next":
        return _PendingCommandTime(False)
    missing_slots = pending_command.get("missing_slots", [])
    awaiting = str(pending_command.get("awaiting") or "")
    if "time_after" not in missing_slots and awaiting != "time":
        return _PendingCommandTime(False)
    parsed = event_recommendation.parse_recommendation_time_answer(query)
    if not parsed.is_time_like:
        return _PendingCommandTime(False)
    if parsed.invalid or parsed.value is None:
        return _PendingCommandTime(True, invalid=True, command=command)
    slots = dict(command["slots"])
    slots["time_after"] = parsed.value.hour * 60 + parsed.value.minute
    return _PendingCommandTime(
        True,
        command={
            "flow": command["flow"],
            "slots": slots,
            "confidence": command.get("confidence", "medium"),
        },
        value=slots["time_after"],
    )


def _run_optional_pair_executor(
    command: Mapping[str, Any],
) -> _CommandOutcome | None:
    """Bridge Agent E's deterministic pair API when it has landed."""

    function = _pair_entrypoint()
    payload = _command_plan_payload(command)
    if function is None or payload is None:
        return None
    dates = payload["slots"].get("dates", [])
    if not dates:
        return None
    try:
        recommendation_date = date.fromisoformat(str(dates[0]))
        events = event_search.load_events()
    except (ValueError, OSError, TypeError):
        return None
    slots = payload["slots"]
    try:
        pair_filters = event_search.command_slots_to_search_filters(
            slots,
            flow="plan_event_pair",
            reference_date=POC_REFERENCE_DATE,
        ).to_dict()
    except (TypeError, ValueError):
        pair_filters = dict(slots)
    raw = _invoke_with_signature(
        function,
        {
            "events": events,
            "day": recommendation_date,
            "date": recommendation_date,
            "recommendation_date": recommendation_date,
            "municipalities": slots.get("municipalities", []),
            "regions": slots.get("regions", []),
            "filters": pair_filters,
            "limit": 3,
            "command": payload,
            "slots": slots,
        },
    )
    if raw is None:
        return None
    raw_map = _mapping(raw)
    if raw_map is None:
        raw_map = {"pairs": raw}
    else:
        raw_map = dict(raw_map)
    raw_map.setdefault("pairs", raw_map.get("event_pairs", []))
    raw_map.setdefault("flow", "plan_event_pair")
    raw_map.setdefault("slots", slots)
    normalized = _normalize_command_outcome(raw_map, payload)
    if normalized is not None:
        return normalized
    # An empty deterministic pair list is still a completed flow.  Preserve
    # that distinction from an unavailable optional executor so a date answer
    # is not asked for again when there simply is no compatible pair.
    if isinstance(raw, (list, tuple)) and not raw:
        return _CommandOutcome(
            flow="plan_event_pair",
            slots=dict(slots),
            command=dict(payload),
            events=[],
            near_events=[],
            all_event_ids=[],
            all_near_event_ids=[],
            pairs=[],
            total_matches=0,
            message="同日・時間順で組み合わせられる候補は見つかりませんでした。",
        )
    return None


def _pending_state_for_outcome(outcome: _CommandOutcome) -> dict[str, Any] | None:
    if outcome.pending is None:
        return None
    pending = {
        "kind": "flow",
        "flow": outcome.flow,
        "command": dict(outcome.command),
        "missing_slots": list(outcome.pending.get("missing_slots", [])),
        "awaiting": str(outcome.pending.get("awaiting") or ""),
    }
    return pending


def _command_pending_question(outcome: _CommandOutcome) -> str:
    """Render a safe slot question without exposing model-generated facts."""

    if outcome.flow == "explain_result":
        return "どのイベントの根拠か、番号かイベント名を教えてみて。"
    missing_slots = (outcome.pending or {}).get("missing_slots", [])
    if "time_after" in missing_slots or (outcome.pending or {}).get("awaiting") == "time":
        return "何時ごろ見終わる予定？"
    if outcome.flow == "plan_event_pair":
        municipalities = outcome.slots.get("municipalities", [])
        city = municipalities[0] if municipalities else "指定した地域"
        if city not in {city_name for cities in REGION_CITIES.values() for city_name in cities}:
            city = "指定した地域"
        count = outcome.slots.get("visit_count", 2)
        if not isinstance(count, int) or not 2 <= count <= MAX_VISIT_COUNT:
            count = 2
        return f"{city}で{count}つ回れる組み合わせを探せるよ。何日に行く予定？"
    if "dates" in missing_slots:
        return "何日に行く予定か教えてみて。"
    return "必要な条件をもう少し教えてみて。"


def _facts_answer(
    events: list[dict[str, object]],
    requested_field: str,
) -> str:
    """Answer factual fields only from the JSON-backed event records."""

    if len(events) == 1:
        return event_search.attribute_answer(events[0], requested_field)
    lines = [f"条件に合うイベントは{len(events)}件あります。"]
    lines.extend(
        f"- {event_search.attribute_answer(event, requested_field)}"
        for event in events
    )
    return "\n".join(lines)


def _command_detail_answer(
    outcome: _CommandOutcome,
    *,
    prompt: str,
    previous_results: list[dict[str, Any]],
    route: Any,
    detail_field: str | None,
) -> tuple[str, dict[str, Any] | None]:
    selected = outcome.events[0] if outcome.events else None
    if selected is None:
        reference_index = outcome.slots.get("reference_index")
        if isinstance(reference_index, int) and 1 <= reference_index <= len(previous_results):
            selected = dict(previous_results[reference_index - 1])
    if selected is None:
        selected_route_event = _value(route, "selected_event", default=None)
        if isinstance(selected_route_event, Mapping):
            selected = dict(selected_route_event)
    if selected is None:
        return "どのイベントを確認したいか、番号かイベント名を教えてみて。", None

    requested_detail = detail_field
    if not requested_detail:
        detail_fields = outcome.slots.get("detail_fields", [])
        if detail_fields:
            requested_detail = str(detail_fields[0])
    requested_detail = {
        "日時": "datetime",
        "場所": "place",
        "料金": "fee",
        "ジャンル": "genre",
        "概要": "overview",
        "対象": "target",
        "申込": "application_required",
        "アクセス": "public_transport",
        "雨天": "rain_policy",
    }.get(requested_detail, requested_detail)
    if requested_detail in {
        "fee",
        "place",
        "datetime",
        "genre",
        "overview",
        "child_friendly",
        "venue",
    }:
        return event_search.attribute_answer(selected, requested_detail), selected
    if requested_detail:
        return event_details.answer_event_detail(selected, requested_detail, prompt), selected
    return f"選択中のイベントは「{selected['イベント名']}」です。カードを確認してみて。", selected


def _safe_command_writer_lead(
    writer: Any,
    candidates: list[dict[str, Any]],
) -> str | None:
    """Allow Writer language only after deterministic candidate validation."""

    lead = _value(writer, "lead", "text", "summary", default=None)
    if not isinstance(lead, str) or not lead.strip() or len(lead) > 800:
        return None
    if _has_unapproved_event_claims(lead, candidates):
        return None
    safe = _redact_event_facts(lead, candidates)
    return safe if safe and safe != "候補カードを確認してみてください。こんなんもあるよ。" else None


def _render_command_outcome(
    outcome: _CommandOutcome,
    *,
    prompt: str,
    previous_results: list[dict[str, Any]],
    previous_search_context: conversation_recovery.SearchContext | None = None,
    route: Any,
    detail_field: str | None,
) -> _CommandRender:
    """Render Command output from deterministic event records only."""

    pending_state = _pending_state_for_outcome(outcome)
    if pending_state is not None:
        return _CommandRender(
            answer=_command_pending_question(outcome),
            results=previous_results,
            near_results=[],
            relaxed_condition=None,
            filters=outcome.filters,
            selected_event=None,
            pair_results=[],
            pending_state=pending_state,
        )

    if outcome.flow == "explain_search":
        answer = (
            outcome.message
            or conversation_recovery.render_search_explanation(previous_search_context)
            if previous_search_context is not None
            else outcome.message
            or "直前の検索結果がないけん、まずイベントを探してみて。"
        )
        return _CommandRender(
            answer=answer,
            results=previous_results,
            near_results=[],
            relaxed_condition=None,
            filters=None,
            selected_event=None,
            pair_results=[],
            pending_state=None,
        )

    if outcome.flow == "explain_result":
        selected = outcome.events[0] if outcome.events else None
        answer = outcome.message or conversation_recovery.render_result_explanation(
            previous_search_context,
            selected,
            query=prompt,
        )
        return _CommandRender(
            answer=answer,
            results=previous_results,
            near_results=[],
            relaxed_condition=None,
            filters=None,
            selected_event=selected,
            pair_results=[],
            pending_state=None,
        )

    if outcome.pairs:
        flattened: list[dict[str, Any]] = []
        seen: set[str] = set()
        for first, second in outcome.pairs:
            for event in (first, second):
                event_id = _event_key(event) or ""
                if event_id not in seen:
                    flattened.append(event)
                    seen.add(event_id)
        answer = (
            "同日・時間順で組み合わせられる候補です。"
            "実際の移動時間ではなく、PoC上の簡易判定です。"
        )
        return _CommandRender(
            answer=answer,
            results=flattened,
            near_results=[],
            relaxed_condition=None,
            filters=outcome.filters,
            selected_event=None,
            pair_results=outcome.pairs,
            pending_state=None,
        )

    events = list(outcome.events)
    near_events = list(outcome.near_events)
    selected_event: dict[str, Any] | None = events[0] if len(events) == 1 else None
    # Explain an executed Command from its trusted filters, not by parsing the
    # natural-language prompt a second time.  This keeps the response tied to
    # the actual deterministic search contract.
    outcome_filters = outcome.filters if isinstance(outcome.filters, Mapping) else {}
    experience_required = list(outcome_filters.get("experience_required", []) or [])
    experience_preferred = list(outcome_filters.get("experience_preferred", []) or [])
    experience_excluded = list(outcome_filters.get("experience_excluded", []) or [])

    if outcome.flow == "event_detail":
        answer, selected_event = _command_detail_answer(
            outcome,
            prompt=prompt,
            previous_results=previous_results,
            route=route,
            detail_field=detail_field,
        )
        return _CommandRender(
            answer=answer,
            results=events or previous_results,
            near_results=[],
            relaxed_condition=None,
            filters=outcome.filters,
            selected_event=selected_event,
            pair_results=[],
            pending_state=None,
        )

    if outcome.flow == "general_faq":
        faq_match = _value(route, "faq_match", default=None)
        answer = _value(faq_match, "answer", default=None) or outcome.message
        if not isinstance(answer, str) or not answer.strip():
            answer = GENERIC_SCOPE_MESSAGE
        return _CommandRender(
            answer=answer,
            results=previous_results,
            near_results=[],
            relaxed_condition=None,
            filters=outcome.filters,
            selected_event=None,
            pair_results=[],
            pending_state=None,
        )

    if experience_required or experience_preferred or experience_excluded:
        total = outcome.total_matches if outcome.total_matches is not None else len(events)
        if events:
            answer = experience_preferences.render_result_message(
                total,
                required=experience_required,
                preferred=experience_preferred,
                excluded=experience_excluded,
            )
        else:
            answer = experience_preferences.render_result_message(
                0,
                required=experience_required,
                preferred=experience_preferred,
                excluded=experience_excluded,
            )
            if experience_required or experience_excluded:
                answer += " 条件を広げる場合は、立ったり歩いたりするイベントを含めてよいか教えてね。"
    elif outcome.flow == "unsupported":
        # Security/out-of-scope guards are deterministic messages from the
        # command executor.  Keep them when present; never ask the Writer to
        # invent a rationale for an unsupported request.
        answer = outcome.message or GENERIC_SCOPE_MESSAGE
    elif outcome.flow == "count_events":
        total = outcome.total_matches if outcome.total_matches is not None else len(events)
        answer = f"条件に合うイベントは{total}件あります。"
    elif outcome.flow == "plan_event_pair":
        answer = "同日・時間順で組み合わせられる候補は見つかりませんでした。"
    elif not events:
        answer = outcome.message or NO_RESULT_MESSAGE
    elif outcome.flow == "recommend_next":
        answer = (
            "同日・終了後・簡易移動バッファで次に行けそうな候補です。"
            "実際の移動時間は計算していません。"
        )
    elif outcome.flow == "recommend_similar":
        answer = "ジャンル・検索タグ・地域などが近い候補です。"
    else:
        total = outcome.total_matches if outcome.total_matches is not None else len(events)
        answer = (
            f"条件に合うイベントが{total}件見つかりました。"
            "下のカードを確認してみて。"
        )

    # Writer is an optional subjective layer only.  It is never used for
    # dates, fees, places, counts, URLs, or participation facts.
    if outcome.writer is not None and outcome.flow in {
        "find_events",
        "recommend_next",
        "recommend_similar",
    }:
        writer_lead = _safe_command_writer_lead(outcome.writer, events)
        if writer_lead:
            answer = f"{writer_lead}\n\n{answer}"

    return _CommandRender(
        answer=answer,
        results=events,
        near_results=near_events,
        relaxed_condition=outcome.relaxed_condition,
        filters=outcome.filters,
        selected_event=selected_event,
        pair_results=[],
        pending_state=None,
    )


def _recommendation_preferences(query: str) -> dict[str, object]:
    filters = event_search.parse_query(query, POC_REFERENCE_DATE)
    preferences: dict[str, object] = {
        "child_friendly": filters.child_friendly,
        "entry_free": filters.entry_free,
    }
    if "同じ市町" in query or "同じ市" in query:
        preferences["scope"] = "city"
    elif "同じ地域" in query:
        preferences["scope"] = "region"
    return preferences


def _llm_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Pass only current candidates; URLs remain card-only facts."""

    safe_fields = (
        "イベント名",
        "日時",
        "場所",
        "ジャンル",
        "子ども向け",
        "屋内/屋外",
        "料金",
        "概要",
    )
    return [
        {key: candidate[key] for key in safe_fields if key in candidate}
        for candidate in candidates[:MAX_WRITER_CANDIDATES]
    ]


def _call_modal(
    *,
    modal_url: str,
    modal_key: str,
    modal_secret: str,
    user_query: str,
    candidates: list[dict[str, object]],
    history: list[dict[str, str]],
) -> str:
    payload = {
        "user_query": user_query,
        "candidates": _llm_candidates(candidates),
        # ``emotion`` is presentation metadata and must not enter the
        # existing LLM history contract.
        "history": model_history(history),
    }
    try:
        response = requests.post(
            modal_url,
            headers={
                "Modal-Key": modal_key,
                "Modal-Secret": modal_secret,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
            allow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            return BACKEND_FAILURE_MESSAGE
        result = response.json()
        if not isinstance(result, dict):
            return BACKEND_FAILURE_MESSAGE
        if result.get("service_id") != SERVICE_ID:
            return BACKEND_FAILURE_MESSAGE
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return BACKEND_FAILURE_MESSAGE
        if _has_unapproved_event_claims(answer, candidates):
            return _fallback_guidance(candidates)
        return _redact_event_facts(answer, candidates)
    except (requests.RequestException, ValueError, TypeError):
        # Do not display exception text: it can contain URLs, headers, or
        # provider-specific details that are not useful to PoC users.
        return BACKEND_FAILURE_MESSAGE


def _event_fact_redactions(
    candidates: list[dict[str, object]],
) -> set[str]:
    """Build display-only redactions from the JSON-backed candidate cards."""

    values: set[str] = set()
    for event in candidates:
        for field in EVENT_FACT_FIELDS:
            raw_value = event.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue

            value = raw_value.strip()
            values.add(value)

            if field == "日時":
                for date_text in re.findall(r"\d{4}-\d{2}-\d{2}", value):
                    values.add(date_text)
                    try:
                        parsed = date.fromisoformat(date_text)
                    except ValueError:
                        continue
                    values.update(
                        {
                            f"{parsed.year}年{parsed.month}月{parsed.day}日",
                            f"{parsed.month}月{parsed.day}日",
                            f"{parsed.month}/{parsed.day}",
                        }
                    )

            if field == "場所":
                for match in re.finditer(
                    r"([一-龯々ぁ-んァ-ヶー]{1,10})(市|町|村)",
                    value,
                ):
                    values.add(match.group(0))
                    values.add(match.group(1))

            if field == "料金":
                values.update(re.findall(r"無料", value))
                values.update(
                    re.findall(r"[0-9０-９][0-9０-９,，]*\s*円", value)
                )

    return {value for value in values if len(value) >= 2}


def _fallback_guidance(candidates: list[dict[str, object]]) -> str:
    """Give deterministic guidance when the model leaves the candidate set."""

    count = len(candidates)
    if count == 1:
        return "条件に合うイベントが1件見つかりました。日時・場所・料金は、下のイベントカードを確認してみて。"
    return f"条件に合うイベントが{count}件見つかりました。気になる番号や、さらに絞りたい条件を教えてみて。"


def _has_unapproved_event_claims(
    answer: str,
    candidates: list[dict[str, object]],
) -> bool:
    """Reject model text that introduces an event outside the JSON candidates."""

    def normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).strip("「」『』")

    approved_names = {
        normalize(str(event.get("イベント名", "")))
        for event in candidates
        if str(event.get("イベント名", "")).strip()
    }
    approved_titles = {
        name.removeprefix("【PoC架空】").strip()
        for name in approved_names
    }

    def is_approved(fragment: str) -> bool:
        normalized = normalize(fragment)
        return any(
            normalized == name
            or normalized in name
            or name in normalized
            for name in approved_names
        ) or any(
            normalized == title
            or normalized in title
            or title in normalized
            for title in approved_titles
        )

    # The model must never invent another fictional event label.
    for fragment in re.findall(r"【PoC架空】[^\n。！？]{2,160}", answer):
        if not is_approved(fragment):
            return True

    # Also reject quoted or marker-bearing event titles that are not cards.
    event_markers = (
        "フェス",
        "フェスタ",
        "常設展",
        "企画展",
        "展示会",
        "ウォーク",
        "サロン",
        "シアター",
        "ラボ",
        "コンサート",
        "講座",
        "ワークショップ",
        "公演",
        "演奏会",
        "祭",
        "まつり",
        "工芸",
    )
    for fragment in re.findall(r"[「『]([^」』\n]{3,160})[」』]", answer):
        if any(marker in fragment for marker in event_markers) and not is_approved(
            fragment
        ):
            return True
    for marker in event_markers:
        for match in re.finditer(
            rf"[^\n。！？]{{0,50}}{re.escape(marker)}[^\n。！？]{{0,50}}",
            answer,
        ):
            if not is_approved(match.group(0)):
                return True

    # The app supplies a fixed PoC date, so stock uncertainty/refusal text is
    # unsafe even when the candidate cards themselves are correct.
    unsafe_phrases = (
        "現在日時が不明",
        "現在日時では",
        "具体的にお答えすることができません",
        "具体的にお答えできません",
        "具体的なイベント情報がないため",
        "確定的な回答はできません",
        "特定の日やイベントに関するご質問であれば",
        "情報は提供できません",
        "お答えすることができません",
        "お答えできません",
        "公式サイト",
        "観光情報サイト",
        "最新情報を確認",
    )
    return any(phrase in answer for phrase in unsafe_phrases) or bool(
        re.search(r"(?:https?://|www\.)", answer)
    )


def _redact_event_facts(
    answer: str,
    candidates: list[dict[str, object]],
) -> str:
    """Keep LLM output as guidance text, never as the event fact channel."""

    redacted = re.sub(
        r"\[[^\]]*\]\(https?://[^)]+\)",
        "",
        answer,
    )
    redacted = re.sub(r"<https?://[^>]+>", "", redacted)
    redacted = re.sub(r"https?://\S+", "", redacted)
    redacted = re.sub(r"\s*[（(]\s*<\s*$", "", redacted)
    for value in sorted(_event_fact_redactions(candidates), key=len, reverse=True):
        redacted = redacted.replace(value, "この候補")

    redacted = re.sub(r"\n{3,}", "\n\n", redacted).strip()
    return redacted or "候補カードを確認してみてください。こんなんもあるよ。"


def _render_event_card(
    event: dict[str, object],
    index: int,
    *,
    scope: str = "exact",
) -> None:
    def fact(icon: str, label: str, value: object) -> None:
        st.markdown(
            "<div class=\"iyoshirube-event-fact\">"
            f"<span class=\"iyoshirube-event-fact-icon\">{escape(icon)}</span>"
            f"<span><strong>{escape(label)}</strong> {escape(str(value))}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    def render_facts() -> None:
        fact("◷", "日時", event["日時"])
        fact("⌖", "場所", event["場所"])
        fact("✦", "ジャンル", event["ジャンル"])
        # ``False`` means only that the source record is not marked
        # child-friendly; it does not establish an adult/general audience.
        audience = "子ども向け" if event.get("子ども向け") is True else "子ども向けの明示なし"
        fact("♧", "対象", audience)
        fact("⌂", "会場", event["屋内/屋外"])
        fact("¥", "料金", event["料金"])

    event_key = str(event.get("id", index))
    card_key = f"iyoshirube-event-card-{scope}-{event_key}-{index}"
    image_key = f"iyoshirube-event-image-{scope}-{event_key}-{index}"

    with st.container(border=True, key=card_key):
        st.markdown(f"**{index}. {event['イベント名']}**")
        image_path = event_image_path(event.get("id"))
        if image_path is None:
            render_facts()
        else:
            facts_col, image_col = st.columns([1.65, 1])
            with facts_col:
                render_facts()
            with image_col:
                with st.container(key=image_key):
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )
        st.markdown(
            "<div class=\"iyoshirube-event-overview\">"
            f"<strong>▣ 概要</strong> {escape(str(event['概要']))}"
            "</div>",
            unsafe_allow_html=True,
        )
        if event_details.V2_FIELDS.issubset(event):
            with st.expander("♧ 参加案内・アクセスを見る"):
                for line in event_details.compact_participation_lines(event):
                    st.write(line)
        st.link_button(
            "公式URL（PoC架空）",
            str(event["公式URL"]),
            icon=":material/open_in_new:",
            use_container_width=True,
        )


def _render_event_grid(
    events: list[dict[str, object]],
    *,
    start_index: int = 1,
    scope: str = "exact",
) -> None:
    """Render ordered cards in pages without re-running the search pipeline."""

    ordered_events = list(events)
    state_key = f"{scope}_visible_count"
    visible_count = normalize_visible_count(
        len(ordered_events),
        st.session_state.get(state_key),
        page_size=RESULT_PAGE_SIZE,
    )
    st.session_state[state_key] = visible_count
    visible_events = visible_items(
        ordered_events,
        visible_count,
        page_size=RESULT_PAGE_SIZE,
    )

    for row_start in range(0, len(visible_events), 3):
        row_events = visible_events[row_start : row_start + 3]
        columns = st.columns(3 if len(row_events) == 3 else len(row_events))
        for column, event_offset in zip(columns, range(len(row_events))):
            with column:
                _render_event_card(
                    row_events[event_offset],
                    start_index + row_start + event_offset,
                    scope=scope,
                )

    shown = len(visible_events)
    total = len(ordered_events)
    if total > 0:
        st.caption(f"{total}件中{shown}件を表示")
    if shown < total:
        remaining = total - shown
        next_page = min(RESULT_PAGE_SIZE, remaining)
        if st.button(
            f"さらに{next_page}件表示（残り{remaining}件）",
            key=f"load_more_{scope}",
            use_container_width=True,
        ):
            st.session_state[state_key] = next_visible_count(
                total,
                visible_count,
                page_size=RESULT_PAGE_SIZE,
            )
            st.rerun()


def _render_avatar_image(emotion: str, *, width: int) -> None:
    """Render a local avatar asset while keeping missing-asset fallback safe."""

    path = avatar_path(emotion)
    if path is not None:
        st.image(str(path), width=width)
    else:
        st.markdown("🧭")


def _render_assistant_message(content: str, emotion: object) -> None:
    """Render one persisted or temporary assistant turn with its avatar."""

    with st.chat_message(
        "assistant",
        avatar=avatar_for_streamlit(emotion),
    ):
        st.caption(IYOSHIRUBE_NAME)
        st.markdown(content)


def _render_user_message(content: str) -> None:
    with st.chat_message("user"):
        st.markdown(content)


def _render_pair_results(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    if not pairs:
        return
    st.subheader("一緒に回れるイベントの組み合わせ")
    st.caption(
        "同日開催・時間順の候補です。実際の道路や交通を使った移動時間ではなく、"
        "PoC上の簡易判定です。"
    )
    for index, (first, second) in enumerate(pairs, start=1):
        with st.container(border=True):
            st.markdown(f"**組み合わせ {index}**")
            first_col, second_col = st.columns(2)
            with first_col:
                st.caption("1件目")
                st.markdown(f"**{first['イベント名']}**")
                st.write(f"日時：{first['日時']}")
                st.write(f"場所：{first['場所']}")
                st.write(f"料金：{first['料金']}")
                st.link_button("公式URL", str(first["公式URL"]), key=f"pair_{index}_first")
            with second_col:
                st.caption("2件目")
                st.markdown(f"**{second['イベント名']}**")
                st.write(f"日時：{second['日時']}")
                st.write(f"場所：{second['場所']}")
                st.write(f"料金：{second['料金']}")
                st.link_button("公式URL", str(second["公式URL"]), key=f"pair_{index}_second")


def _run_next_recommendation(
    selected_event: dict[str, object],
    recommendation_date: date,
    *,
    selected_end_override=None,
) -> tuple[event_recommendation.RecommendationResult | None, str | None]:
    source_events = event_search.load_events()
    if not event_details.V2_FIELDS.issubset(selected_event) or any(
        not event_details.V2_FIELDS.issubset(event) for event in source_events
    ):
        return None, "次のイベント推薦に必要な構造化データが、まだ読み込み中です。少し待ってからもう一度試してみて。"
    return (
        event_recommendation.recommend_next_events(
            selected_event,
            source_events,
            recommendation_date,
            selected_end_override=selected_end_override,
        ),
        None,
    )


def _build_recommendation_context(
    query: str,
    events: list[dict[str, Any]],
    *,
    policy: str,
) -> conversation_recovery.SearchContext:
    """Persist a fresh context for legacy recommendation result sets.

    Recommendations are deterministic Python results too. Keeping their
    ordered IDs in a new SearchContext prevents a later explanation/refinement
    turn from silently reusing the search that produced the seed event.
    """

    parsed = event_search.parse_query(query, POC_REFERENCE_DATE)
    return conversation_recovery.build_search_context(
        query,
        parsed,
        events,
        result_ids=_event_ids(events),
        total_matches=len(events),
        selection_policy=policy,
    )


def _is_reset_query(query: str) -> bool:
    return recommendation_pending.is_reset_query(query)


def _reset() -> None:
    for key in (
        "messages",
        "last_results",
        "last_near_results",
        "last_result_ids",
        "last_near_result_ids",
        "exact_visible_count",
        "near_visible_count",
        "last_relaxed_condition",
        "last_filters",
        "selected_event",
        "selected_event_id",
        "last_plan",
        "last_search_context",
        "suppress_result_cards",
        "recovery_display_results",
        "last_query",
        "feedback",
        "pending_prompt",
        "pending_recommendation",
        "pending_command",
        "last_command",
        "last_action",
        "last_pair_results",
    ):
        st.session_state.pop(key, None)
    st.rerun()


modal_url = _validate_modal_url(_required_secret("MODAL_URL"))
modal_key = _required_secret("MODAL_KEY")
modal_secret = _required_secret("MODAL_SECRET")

title_col, mascot_col = st.columns([6, 1])
with title_col:
    st.title("🎭  伊予の文化案内人")
    st.caption("愛顔えひめの文化祭2028を想定したイベント案内PoC")
with mascot_col:
    _render_avatar_image(EMOTION_NORMAL, width=88)

warning_col, date_col = st.columns([1.1, 1])
with warning_col:
    st.warning(
        "- 生成AIを利用した技術検証PoC\n"
        "- 掲載イベントはすべて架空\n"
        "- 愛媛県・愛顔えひめの文化祭2028の公式サービスではありません\n"
        "- AI回答に誤りが含まれる場合があります"
    )
with date_col:
    st.info(f"**PoC現在日**\n\n### {POC_REFERENCE_DATE_TEXT}")

with st.container(border=True, key="iyoshirube-hero"):
    hero_avatar_col, hero_text_col, hero_wave_col = st.columns([1.15, 5.25, 2.6])
    with hero_avatar_col:
        _render_avatar_image(EMOTION_NORMAL, width=128)
    with hero_text_col:
        st.subheader(IYOSHIRUBE_NAME)
        st.write(IYOSHIRUBE_TAGLINE)
        st.caption(
            "愛顔えひめの文化祭2028を想定したPoC用の架空キャラクターです。"
        )
    with hero_wave_col:
        with st.container(key="iyoshirube-wave"):
            if IYOSHIRUBE_WAVE_ASSET.is_file():
                st.image(str(IYOSHIRUBE_WAVE_ASSET), use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_near_results" not in st.session_state:
    st.session_state.last_near_results = []
if "last_result_ids" not in st.session_state:
    st.session_state.last_result_ids = []
if "last_near_result_ids" not in st.session_state:
    st.session_state.last_near_result_ids = []
if "exact_visible_count" not in st.session_state:
    st.session_state.exact_visible_count = 0
if "near_visible_count" not in st.session_state:
    st.session_state.near_visible_count = 0
if "last_relaxed_condition" not in st.session_state:
    st.session_state.last_relaxed_condition = None
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None
if "last_search_context" not in st.session_state:
    st.session_state.last_search_context = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "suppress_result_cards" not in st.session_state:
    st.session_state.suppress_result_cards = False
if "recovery_display_results" not in st.session_state:
    st.session_state.recovery_display_results = None
if "last_pair_results" not in st.session_state:
    st.session_state.last_pair_results = []

with st.sidebar:
    with st.container(key="iyoshirube-sidebar"):
        st.markdown("### :material/list: 質問の例")
        quick_action_icons = {
            "今日のイベント": ":material/calendar_month:",
            "子どもと楽しむ": ":material/groups:",
            "雨でもOK": ":material/umbrella:",
            "無料イベント": ":material/local_activity:",
            "伝統芸能": ":material/celebration:",
            "南予のイベント": ":material/location_on:",
        }
        for action in QUICK_ACTIONS:
            if st.button(
                action.label,
                icon=quick_action_icons[action.label],
                use_container_width=True,
                key=f"quick_{action.label}",
            ):
                st.session_state.pending_prompt = action.fallback_query
                st.session_state.pending_command = {
                    "kind": "quick_action",
                    "command": action.command.to_dict(),
                }
                st.rerun()
        st.divider()
        if st.button(
            "会話をリセット",
            icon=":material/refresh:",
            use_container_width=True,
            key="reset_conversation",
        ):
            _reset()
        st.caption("⚠ 個人情報・機密情報・未公開情報は入力しないでください。")
        st.caption(f"Build: {BUILD_SHA}")

_render_assistant_message(
    "こんにちは、いよしるべです。\n\n"
    "愛媛の文化イベントをいっしょに探してみん？",
    EMOTION_NORMAL,
)

for message in st.session_state.messages:
    if message.get("role") == "assistant":
        _render_assistant_message(
            str(message.get("content", "")),
            emotion_from_message(message),
        )
    else:
        _render_user_message(str(message.get("content", "")))

recovery_display_results = st.session_state.get("recovery_display_results")
if recovery_display_results is not None:
    st.subheader("対象のイベント")
    _render_event_grid(list(recovery_display_results), scope="exact")
elif st.session_state.last_pair_results and not st.session_state.get("suppress_result_cards"):
    _render_pair_results(st.session_state.last_pair_results)
elif (
    st.session_state.last_results
    and not st.session_state.last_pair_results
    and not st.session_state.get("suppress_result_cards")
):
    st.subheader("条件に合うイベント")
    _render_event_grid(st.session_state.last_results, scope="exact")

if st.session_state.last_near_results and not st.session_state.get("suppress_result_cards"):
    relaxed = st.session_state.last_relaxed_condition or "一部の条件"
    st.subheader(f"参考候補（「{relaxed}」を外した場合）")
    st.caption("上の検索結果には含めていません。条件を緩めた候補として表示しています。")
    _render_event_grid(st.session_state.last_near_results, scope="near")

if st.session_state.get("last_query"):
    st.divider()
    feedback_label_col, feedback_col1, feedback_col2 = st.columns([2.8, 1, 1])
    with feedback_label_col:
        st.caption("この案内は役に立ちましたか？")
    with feedback_col1:
        if st.button(
            "役に立った",
            icon=":material/thumb_up:",
            key="feedback_yes",
        ):
            st.session_state.feedback = "yes"
    with feedback_col2:
        if st.button(
            "改善が必要",
            icon=":material/thumb_down:",
            key="feedback_no",
        ):
            st.session_state.feedback = "no"
    if st.session_state.get("feedback") == "yes":
        st.success("フィードバックを受け取りました。")
    elif st.session_state.get("feedback") == "no":
        st.info("ありがとうございます。PoCの改善に使います。")

prompt = st.session_state.pop("pending_prompt", None)
if prompt is None:
    prompt = st.chat_input("例：11月3日に子どもと行けるイベント")

pending_command_request = st.session_state.get("pending_command")
quick_action_command: dict[str, Any] | None = None
active_command_pending: dict[str, Any] | None = None
if isinstance(pending_command_request, Mapping):
    if pending_command_request.get("kind") == "quick_action":
        quick_action_command = _command_plan_payload(
            pending_command_request.get("command")
        )
    elif pending_command_request.get("kind") == "flow":
        active_command_pending = dict(pending_command_request)

if prompt:
    prompt = prompt.strip()
    if len(prompt) > 500:
        st.error("質問は500文字以内にしてください。")
        st.stop()
    if _is_reset_query(prompt):
        _reset()

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Render only this new turn while the synchronous backend work is running.
    # The temporary nodes are not added to session_state and disappear on the
    # final rerun, so the persisted history cannot be duplicated.
    _render_user_message(prompt)
    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        _render_assistant_message("いよしるべが考え中…", EMOTION_THINKING)

    results: list[dict[str, object]] = []
    near_results: list[dict[str, object]] = []
    relaxed_condition: str | None = None
    selected_event: dict[str, object] | None = None
    filters: event_search.SearchFilters | None = None
    search_result: event_search.SearchResult | None = None
    search_context_for_turn: conversation_recovery.SearchContext | None = None
    exact_result_count: int | None = None
    recommendation_result_count: int | None = None
    pair_result_count: int | None = None
    invalid_input = False
    event_selection_failed = False
    backend_failure = False
    command_preparation_failed = False
    previous_results = list(st.session_state.get("last_results", []))
    route = route_conversation(
        prompt,
        previous_results,
        st.session_state.get("selected_event"),
        st.session_state.get("last_filters"),
        POC_REFERENCE_DATE,
    )
    detail_field = route.detail_field
    # Explicit high-confidence recovery/detail branches stay deterministic.
    # A bare reference_followup without a factual detail field is allowed to
    # reach the existing Semantic Command Generator: phrases such as
    # "2番目が条件に合う理由" are not a card lookup and must be classified
    # semantically as explain_result.
    prefer_router_reference = route.action_type in {
        "detail_followup",
        "explain_search",
        "explain_result",
        "clarify_reference",
    } or (
        route.action_type == "reference_followup"
        and detail_field is not None
    )
    pending_decision = recommendation_pending.PendingDecision(False)
    pending_handled = False
    pending_result_set_replaced = False
    pending_state_to_store: dict[str, str] | None = None
    pending_state = st.session_state.get("pending_recommendation")
    if pending_state:
        pending_decision = recommendation_pending.resolve_pending_input(
            prompt,
            pending_state,
            event_search.load_events(),
            POC_REFERENCE_DATE,
        )
        if pending_decision.handled:
            pending_handled = True
            pending_state_to_store = pending_decision.next_state
            if pending_decision.event is None and pending_decision.answer and any(
                marker in pending_decision.answer
                for marker in (
                    "推薦対象のイベント情報を再取得できませんでした",
                    "推薦条件を確認できませんでした",
                )
            ):
                event_selection_failed = True
            if pending_decision.answer and any(
                marker in pending_decision.answer
                for marker in ("解釈できませんでした", "開催時間内で", "開催期間外")
            ):
                invalid_input = True
        else:
            # A non-date/time turn is a new question.  Do not force it through
            # the pending flow; the normal router handles named events, FAQs,
            # and ordinary search instead.
            st.session_state.pop("pending_recommendation", None)

    command_outcome: _CommandOutcome | None = None
    command_render: _CommandRender | None = None
    command_handled = False
    command_pending_to_store: dict[str, Any] | None = active_command_pending
    command_state = _command_state(previous_results, active_command_pending)
    previous_search_context = conversation_recovery.SearchContext.from_value(
        st.session_state.get("last_search_context")
    )
    suppress_cards_for_turn = False
    recovery_display_results_for_turn: list[dict[str, Any]] | None = None

    if not pending_handled:
        if quick_action_command is not None:
            command_outcome = _call_new_command(
                prompt,
                state=command_state,
                modal_config=ModalConfig(modal_url, modal_key, modal_secret),
                command=quick_action_command,
            )
            if command_outcome is None:
                quick_action = next(
                    (
                        action
                        for action in QUICK_ACTIONS
                        if action.command.to_dict() == quick_action_command
                    ),
                    None,
                )
                if quick_action is not None:
                    command_outcome = _quick_action_fallback(quick_action)
        elif active_command_pending is not None:
            pending_date = _pending_command_date(active_command_pending, prompt)
            pending_time = _pending_command_time(active_command_pending, prompt)
            if pending_date.applicable or pending_time.applicable:
                command_pending_to_store = None
                if pending_date.applicable:
                    invalid_pending = pending_date.invalid
                    pending_command = pending_date.command
                    invalid_answer = "日付を解釈できませんでした。例：11月3日、11/3、3日"
                else:
                    invalid_pending = pending_time.invalid
                    pending_command = pending_time.command
                    invalid_answer = "時刻を解釈できませんでした。例：13時、13:30"
                if invalid_pending or pending_command is None:
                    invalid_input = invalid_pending
                    command_pending_to_store = active_command_pending
                    command_handled = True
                    command_render = _CommandRender(
                        answer=invalid_answer,
                        results=previous_results,
                        near_results=[],
                        relaxed_condition=None,
                        filters=None,
                        selected_event=None,
                        pair_results=[],
                        pending_state=active_command_pending,
                    )
                else:
                    command_outcome = _call_new_command(
                        prompt,
                        state=_command_state(previous_results, active_command_pending),
                        modal_config=ModalConfig(modal_url, modal_key, modal_secret),
                        command=pending_command,
                        skip_generation=True,
                    )
                    if command_outcome is None and pending_date.applicable:
                        command_outcome = _run_optional_pair_executor(pending_command)
                    if command_outcome is not None:
                        command_handled = True
                    else:
                        # The pending state was created by the new path, but
                        # its executor may be in a partially deployed
                        # checkout.  Do not reinterpret the date as a search.
                        command_handled = True
                        command_render = _CommandRender(
                            answer=(
                                "時刻は確認したけど、次の候補計算がまだ準備中です。"
                                "少し待ってからもう一度試してみて。"
                                if pending_time.applicable
                                else "日付は確認したけど、イベントの組み合わせ計算が"
                                "まだ準備中です。少し待ってからもう一度試してみて。"
                            ),
                            results=previous_results,
                            near_results=[],
                            relaxed_condition=None,
                            filters=None,
                            selected_event=None,
                            pair_results=[],
                            pending_state=None,
                        )
            else:
                awaiting = str(active_command_pending.get("awaiting") or "")
                pending_kind = event_recommendation.classify_pending_answer(
                    prompt,
                    awaiting=(
                        "date"
                        if awaiting in {"date", "dates"}
                        or active_command_pending.get("flow") == "plan_event_pair"
                        else "time"
                    ),
                    reference_date=POC_REFERENCE_DATE,
                )
                if pending_kind == event_recommendation.PENDING_AMBIGUOUS:
                    # Preserve the authoritative flow until the user gives a
                    # recognizable slot answer or an explicit new request.
                    command_handled = True
                    command_render = _CommandRender(
                        answer=(
                            "何時ごろ見終わる予定か、13時のように教えてみて。"
                            if awaiting in {"time", "time_after", "end_time"}
                            else "何日に行く予定か、11/4のように教えてみて。"
                        ),
                        results=previous_results,
                        near_results=[],
                        relaxed_condition=None,
                        filters=None,
                        selected_event=None,
                        pair_results=[],
                        pending_state=active_command_pending,
                    )
                else:
                    # An explicit new request interrupts the pending Command
                    # flow and is routed from a clean conversation state.
                    st.session_state.pop("pending_command", None)
                    active_command_pending = None
                    command_pending_to_store = None
                    if not prefer_router_reference:
                        command_outcome = _call_new_command(
                            prompt,
                            state=_command_state(previous_results),
                            modal_config=ModalConfig(modal_url, modal_key, modal_secret),
                        )
        else:
            if not prefer_router_reference:
                command_outcome = _call_new_command(
                    prompt,
                    state=command_state,
                    modal_config=ModalConfig(modal_url, modal_key, modal_secret),
                )

        if command_outcome is not None:
            command_handled = True
            command_render = _render_command_outcome(
                command_outcome,
                prompt=prompt,
                previous_results=previous_results,
                previous_search_context=previous_search_context,
                route=route,
                detail_field=detail_field,
            )
            command_pending_to_store = command_render.pending_state
            if command_outcome.flow == "find_events":
                exact_result_count = (
                    command_outcome.total_matches
                    if command_outcome.total_matches is not None
                    else len(command_outcome.events)
                )
            if command_outcome.flow in {"recommend_next", "recommend_similar"}:
                recommendation_result_count = len(command_outcome.events)
            if command_outcome.flow in {
                "plan_event_pair",
                "event_pair",
                "pair_recommendation",
            }:
                pair_result_count = len(command_outcome.pairs)

    agentic_response = None
    parsed_for_agentic = event_search.parse_query(prompt, POC_REFERENCE_DATE)
    if (
        not pending_handled
        and not command_handled
        and agent_orchestrator.should_use_agentic_search(
        prompt,
        route,
        parsed_for_agentic,
        )
    ):
        conversation_state = {
            "selected_event_id": (
                st.session_state.get("selected_event_id")
                or (st.session_state.get("selected_event", {}) or {}).get("id")
            ),
            "last_result_ids": list(
                st.session_state.get("last_result_ids") or _event_ids(previous_results)
            ),
            "last_filters": st.session_state.get("last_filters") or {},
            "last_results": previous_results,
            "selected_event": st.session_state.get("selected_event"),
        }
        agentic_response = agent_orchestrator.handle_agentic_query(
            prompt,
            conversation_state,
            reference_date=POC_REFERENCE_DATE,
            modal_config=ModalConfig(modal_url, modal_key, modal_secret),
        )
        answer = agent_orchestrator.render_agentic_response(agentic_response)
        results = list(agentic_response.exact_events)
        exact_result_count = agentic_response.total_matches
        near_results = list(agentic_response.relaxed_events)
        relaxed_condition = "・".join(
            agent_orchestrator.humanize_relaxed_fields(agentic_response.relaxed_fields)
        ) or None
        filters = parsed_for_agentic
        search_context_for_turn = conversation_recovery.build_search_context(
            prompt,
            filters,
            results,
            result_ids=agentic_response.exact_event_ids or _event_ids(results),
            total_matches=agentic_response.total_matches,
            search_specs=agentic_response.search_specs,
        )

    if command_handled:
        # The Command path has already executed its deterministic Flow and
        # produced a safe render.  Do not pass this turn through Agentic or
        # the legacy parser a second time.
        if command_render is None:
            command_render = _CommandRender(
                answer=BACKEND_FAILURE_MESSAGE,
                results=previous_results,
                near_results=[],
                relaxed_condition=None,
                filters=None,
                selected_event=None,
                pair_results=[],
                pending_state=None,
            )
        answer = command_render.answer
        results = list(command_render.results)
        near_results = list(command_render.near_results)
        relaxed_condition = command_render.relaxed_condition
        selected_event = command_render.selected_event
        if command_outcome is not None and command_outcome.flow in {
            "explain_search",
            "explain_result",
            "general_faq",
            "unsupported",
        }:
            # Recovery/FAQ/boundary turns must not re-render stale result cards.
            suppress_cards_for_turn = True
        if (
            command_outcome is not None
            and command_outcome.flow == "explain_result"
            and selected_event is not None
        ):
            recovery_display_results_for_turn = [selected_event]
        if command_outcome is not None and command_outcome.flow in {
            "find_events",
            "count_events",
            "recommend_next",
            "recommend_similar",
            "plan_event_pair",
        }:
            command_ids = list(command_outcome.all_event_ids) or _event_ids(results)
            trace_events = _events_for_ids(command_ids) if command_ids else list(results)
            search_context_for_turn = conversation_recovery.build_search_context(
                prompt,
                command_outcome.filters,
                trace_events,
                result_ids=command_ids,
                total_matches=command_outcome.total_matches,
                selection_policy=(
                    f"semantic_command_{command_outcome.flow}"
                    if command_outcome.flow in {"recommend_next", "recommend_similar"}
                    else "deterministic_hard_filters_then_existing_ranker"
                ),
            )
    elif pending_handled:
        if pending_decision.event is not None:
            selected_event = dict(pending_decision.event)
        results = previous_results
        if pending_decision.answer is not None:
            answer = pending_decision.answer
        elif pending_decision.event is not None and pending_decision.recommendation_date is not None:
            recommendation, recommendation_error = _run_next_recommendation(
                dict(pending_decision.event),
                pending_decision.recommendation_date,
                selected_end_override=pending_decision.selected_end_override,
            )
            if recommendation_error is not None or recommendation is None:
                answer = recommendation_error or BACKEND_FAILURE_MESSAGE
                recommendation_result_count = 0
            else:
                answer = recommendation.message
                recommendation_result_count = len(recommendation.events)
                results = list(recommendation.events)
                pending_result_set_replaced = True
                search_context_for_turn = _build_recommendation_context(
                    prompt,
                    results,
                    policy="legacy_recommend_next_pending",
                )
        else:
            answer = "推薦条件を確認できませんでした。イベント名からもう一度探してみて。"
    elif agentic_response is not None:
        # Agentic Search already produced the deterministic result set and the
        # Writer language above.  Event cards remain rendered from events.json.
        selected_event = results[0] if len(results) == 1 else None
    elif route.action_type == "explain_search":
        answer = conversation_recovery.render_search_explanation(previous_search_context)
        results = previous_results
        suppress_cards_for_turn = True
    elif route.action_type == "explain_result":
        selected_event = None
        if previous_search_context is not None and route.reference_index is not None:
            selected_event = dict(
                conversation_recovery.reference_event(
                    previous_search_context,
                    previous_results,
                    route.reference_index,
                )
                or {}
            ) or None
        elif (
            previous_search_context is not None
            and isinstance(route.selected_event, Mapping)
            and _event_key(route.selected_event) in set(previous_search_context.result_ids)
        ):
            selected_event = dict(route.selected_event)
        answer = conversation_recovery.render_result_explanation(
            previous_search_context,
            selected_event,
            query=prompt,
        )
        results = previous_results
        suppress_cards_for_turn = True
        if selected_event is not None:
            recovery_display_results_for_turn = [selected_event]
    elif route.action_type == "clarify_reference":
        answer = (
            "まだ参照できるイベント一覧がないけん、どのイベントのことか分からないよ。"
            "条件かイベント名を教えてみて。"
        )
        results = previous_results
        suppress_cards_for_turn = True
    elif route.action_type == "reference_followup":
        # Ordinals and pronouns resolve against the last exact result set.
        # Participation facts are answered locally and never sent to Modal.
        filters = event_search.parse_query(prompt, POC_REFERENCE_DATE)
        if route.selected_event is not None:
            selected_event = dict(route.selected_event)
        if selected_event and detail_field:
            answer = event_details.answer_event_detail(selected_event, detail_field, prompt)
            results = previous_results
        elif selected_event and filters.requested_field:
            answer = event_search.attribute_answer(selected_event, filters.requested_field)
            results = previous_results
        elif selected_event:
            answer = f"選択中のイベントは「{selected_event['イベント名']}」です。カードを確認してみて。"
            results = previous_results
        else:
            answer = "どのイベントを指しているか、番号かイベント名を教えてみて。"
            results = previous_results
    elif route.action_type == "detail_followup":
        selected_event = dict(route.selected_event) if route.selected_event is not None else None
        answer = event_details.answer_event_detail(selected_event, detail_field, prompt)
        results = previous_results
    elif route.action_type == "recommend_next_without_selection":
        answer = "まずイベントを1件選んでから、「このあと何か行ける？」と聞いてみて。"
        results = previous_results
        suppress_cards_for_turn = True
    elif route.action_type == "recommend_next":
        selected_event = dict(route.selected_event) if route.selected_event is not None else None
        if selected_event is None:
            answer = "次のイベント推薦に必要なイベントを特定できませんでした。イベント名を教えてみて。"
            event_selection_failed = True
            results = previous_results
        else:
            date_resolution = event_recommendation.resolve_recommendation_date(
                selected_event,
                prompt,
                POC_REFERENCE_DATE,
                previous_filters=st.session_state.get("last_filters"),
            )
            if date_resolution.recommendation_date is None:
                answer = date_resolution.message
                results = previous_results
                invalid_input = invalid_input or (
                    "開催期間外" in date_resolution.message
                )
                if date_resolution.message.startswith("期間開催のイベントなので"):
                    pending_state_to_store = recommendation_pending.make_pending_state(
                        selected_event,
                        awaiting="date",
                    )
            elif str(selected_event.get("参加形式")) == event_recommendation.DROP_IN_ENTRY:
                answer = "何時ごろ見終わる予定？"
                results = previous_results
                pending_state_to_store = recommendation_pending.make_pending_state(
                    selected_event,
                    awaiting="end_time",
                    recommendation_date=date_resolution.recommendation_date,
                )
            else:
                recommendation, recommendation_error = _run_next_recommendation(
                    selected_event,
                    date_resolution.recommendation_date,
                )
                if recommendation_error is not None or recommendation is None:
                    answer = recommendation_error or BACKEND_FAILURE_MESSAGE
                    recommendation_result_count = 0
                    results = previous_results
                else:
                    answer = recommendation.message
                    recommendation_result_count = len(recommendation.events)
                    results = list(recommendation.events)
                    search_context_for_turn = _build_recommendation_context(
                        prompt,
                        results,
                        policy="legacy_recommend_next",
                    )
    elif route.action_type == "recommend_similar_without_selection":
        answer = "まず基準にするイベントを1件選んでみて。"
        results = previous_results
        suppress_cards_for_turn = True
    elif route.action_type == "recommend_similar":
        selected_event = dict(route.selected_event) if route.selected_event is not None else None
        source_events = event_search.load_events()
        if selected_event is None or not event_details.V2_FIELDS.issubset(selected_event) or any(
            not event_details.V2_FIELDS.issubset(event) for event in source_events
        ):
            answer = "類似イベントの推薦に必要な構造化データが、まだ読み込み中です。少し待ってからもう一度試してみて。"
            recommendation_result_count = 0
            if selected_event is None:
                event_selection_failed = True
            results = previous_results
        else:
            recommendation = event_recommendation.recommend_similar_events(
                selected_event,
                source_events,
                POC_REFERENCE_DATE,
                preferences=_recommendation_preferences(prompt),
            )
            answer = recommendation.message
            recommendation_result_count = len(recommendation.events)
            results = list(recommendation.events)
            search_context_for_turn = _build_recommendation_context(
                prompt,
                results,
                policy="legacy_recommend_similar",
            )
    elif route.action_type == "nearby":
        answer = NEARBY_MESSAGE
        suppress_cards_for_turn = True
    elif route.action_type == "scope_search":
        search_result = _search_result(prompt)
        exact_result_count = search_result.total_matches
        answer = search_result.message or GENERIC_SCOPE_MESSAGE
        suppress_cards_for_turn = True
    elif route.action_type == "general_faq":
        answer = route.faq_match.answer if route.faq_match is not None else GENERIC_SCOPE_MESSAGE
        results = previous_results
        suppress_cards_for_turn = True
    elif route.action_type == "generic_scope":
        answer = GENERIC_SCOPE_MESSAGE
        suppress_cards_for_turn = True
    else:
        inherit_previous = event_search.is_refinement_query(prompt) and bool(
            st.session_state.get("last_filters")
        )
        search_result = _search_result(
            prompt,
            previous_filters=st.session_state.get("last_filters"),
            inherit_previous=inherit_previous,
        )
        filters = search_result.filters
        invalid_input = invalid_input or bool(filters.invalid_date)
        exact_result_count = search_result.total_matches
        results = list(search_result.events)
        near_results = list(search_result.near_matches)
        relaxed_condition = search_result.relaxed_condition
        search_context_for_turn = conversation_recovery.build_search_context(
            prompt,
            filters,
            _events_for_ids(search_result.all_event_ids)
            if search_result.all_event_ids
            else results,
            result_ids=search_result.all_event_ids or _event_ids(results),
            total_matches=search_result.total_matches,
        )
        if not results:
            answer = search_result.message or NO_RESULT_MESSAGE
        elif filters.intent == "count":
            answer = f"条件に合うイベントは{search_result.total_matches}件あります。"
        elif detail_field and len(results) == 1:
            answer = event_details.answer_event_detail(results[0], detail_field, prompt)
        elif filters.requested_field:
            # Event facts remain grounded in events.json.  Modal is reserved
            # for discovery guidance over already-filtered candidates.
            answer = _facts_answer(results, filters.requested_field)
        elif (
            filters.experience_required
            or filters.experience_preferred
            or filters.experience_excluded
        ):
            # Keep the user-facing explanation deterministic when the legacy
            # UI path is used.  Experience facts are already resolved and
            # matched by Python; the optional Writer must not replace that
            # grounded explanation with a generic or speculative lead.
            answer = experience_preferences.render_result_message(
                search_result.total_matches,
                required=filters.experience_required,
                preferred=filters.experience_preferred,
                excluded=filters.experience_excluded,
            )
        else:
            answer = _call_modal(
                modal_url=modal_url,
                modal_key=modal_key,
                modal_secret=modal_secret,
                user_query=prompt,
                candidates=results,
                history=history,
            )
    if command_handled:
        turn_flow = (
            command_outcome.flow
            if command_outcome is not None
            else str((active_command_pending or {}).get("flow") or "command")
        )
        command_preparation_failed = command_preparation_failed or (
            command_outcome is None
            and command_render is not None
            and command_render.pending_state is None
        )
        if command_render is not None and command_render.pair_results:
            pair_result_count = len(command_render.pair_results)
        if (
            command_outcome is not None
            and command_outcome.flow == "event_detail"
            and command_render is not None
            and command_render.selected_event is None
        ):
            event_selection_failed = True
    elif pending_handled:
        # The pending recommendation is still a recommend_next flow even
        # when the current query is only a date/time reply.
        turn_flow = "recommend_next"
    elif agentic_response is not None:
        turn_flow = "agentic_search"
    else:
        turn_flow = route.action_type
        if route.action_type == "search" and filters is not None:
            if filters.intent == "count":
                turn_flow = "count_events"
            elif detail_field or filters.requested_field:
                # The legacy search branch can answer a factual field without
                # taking the router's dedicated detail-followup path.
                turn_flow = "event_detail"

    if turn_flow in {"reference_followup", "event_detail", "detail_followup"}:
        event_selection_failed = event_selection_failed or (
            selected_event is None
            and any(
                marker in answer
                for marker in ("どのイベント", "番号かイベント名")
            )
        )
    backend_failure = backend_failure or answer == BACKEND_FAILURE_MESSAGE
    pending_for_emotion = bool(
        (command_render is not None and command_render.pending_state is not None)
        or pending_state_to_store is not None
        or turn_flow
        in {
            "recommend_next_without_selection",
            "recommend_similar_without_selection",
        }
    )
    assistant_emotion = select_assistant_emotion(
        answer=answer,
        result_count=len(results),
        exact_result_count=exact_result_count,
        flow=turn_flow,
        route_action=route.action_type,
        pair_result_count=pair_result_count,
        recommendation_result_count=recommendation_result_count,
        pending=pending_for_emotion,
        backend_failure=backend_failure,
        command_preparation_failed=command_preparation_failed,
        invalid_input=invalid_input,
        event_selection_failed=event_selection_failed,
    )
    thinking_placeholder.empty()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "emotion": assistant_emotion,
        }
    )
    previous_result_ids = list(st.session_state.get("last_result_ids", []))
    previous_near_results = list(st.session_state.get("last_near_results", []))
    previous_near_result_ids = list(st.session_state.get("last_near_result_ids", []))
    result_ids = _event_ids(results)
    near_result_ids = _event_ids(near_results)
    if command_outcome is not None and command_outcome.all_event_ids:
        result_ids = list(command_outcome.all_event_ids)
    elif agentic_response is not None and agentic_response.exact_event_ids:
        result_ids = list(agentic_response.exact_event_ids)
    elif search_result is not None and search_result.all_event_ids:
        result_ids = list(search_result.all_event_ids)
    if command_outcome is not None and command_outcome.all_near_event_ids:
        near_result_ids = list(command_outcome.all_near_event_ids)
    elif agentic_response is not None and agentic_response.relaxed_event_ids:
        near_result_ids = list(agentic_response.relaxed_event_ids)
    elif search_result is not None and search_result.all_near_event_ids:
        near_result_ids = list(search_result.all_near_event_ids)

    # If an adapter returned only IDs, rehydrate the complete ordered set from
    # the local catalog before persisting it.  This keeps card facts local and
    # makes the session's result IDs the single source for pagination and
    # conversation references.
    if len(result_ids) > len(_event_ids(results)):
        results = _events_for_ids(result_ids)
    if len(near_result_ids) > len(_event_ids(near_results)):
        near_results = _events_for_ids(near_result_ids)
    if len(previous_result_ids) > len(_event_ids(previous_results)):
        previous_results = _events_for_ids(previous_result_ids)
    if len(previous_near_result_ids) > len(_event_ids(previous_near_results)):
        previous_near_results = _events_for_ids(previous_near_result_ids)

    context_source = classify_result_context_source(
        command_flow=command_outcome.flow if command_outcome is not None else None,
        search_result_present=search_result is not None,
        agentic_response_present=agentic_response is not None,
        command_pending_response=bool(
            command_render is not None and command_render.pending_state is not None
        ),
        pending_response_preserving=(
            pending_handled and not pending_result_set_replaced
        ),
    )
    context_flow = turn_flow if command_outcome is None else command_outcome.flow

    result_context = transition_result_context(
        previous_results=previous_results,
        previous_result_ids=previous_result_ids,
        previous_near_results=previous_near_results,
        previous_near_result_ids=previous_near_result_ids,
        previous_visible_count=st.session_state.get("exact_visible_count"),
        previous_near_visible_count=st.session_state.get("near_visible_count"),
        flow=context_flow,
        source=context_source,
        new_results=results,
        new_result_ids=result_ids,
        new_near_results=near_results,
        new_near_result_ids=near_result_ids,
        page_size=RESULT_PAGE_SIZE,
    )
    results = result_context.results
    near_results = result_context.near_results
    result_ids = result_context.result_ids
    near_result_ids = result_context.near_result_ids
    st.session_state.exact_visible_count = result_context.visible_count
    st.session_state.near_visible_count = result_context.near_visible_count
    st.session_state.last_results = results
    st.session_state.last_near_results = near_results
    st.session_state.last_result_ids = result_ids
    st.session_state.last_near_result_ids = near_result_ids
    if result_context.replace_result_set:
        st.session_state.last_relaxed_condition = relaxed_condition
    else:
        relaxed_condition = st.session_state.get("last_relaxed_condition") or relaxed_condition
    if command_handled and command_render is not None:
        command_filters = command_render.filters
        known_filter_keys = set(event_search.SearchFilters().to_dict())
        if not isinstance(command_filters, Mapping) or set(command_filters) - known_filter_keys:
            command_filters = None
        if result_context.replace_result_set:
            st.session_state.last_filters = (
                dict(command_filters) if command_filters is not None else None
            )
        if result_context.replace_result_set:
            st.session_state.last_plan = {
                "mode": "command",
                "flow": command_outcome.flow if command_outcome is not None else (
                    quick_action_command or (active_command_pending or {}).get("command")
                ),
                "slots": command_outcome.slots if command_outcome is not None else {},
                "total_matches": (
                    command_outcome.total_matches
                    if command_outcome is not None and command_outcome.total_matches is not None
                    else len(results)
                ),
                "pending": command_render.pending_state is not None,
            }
        if result_context.replace_result_set and command_outcome is not None:
            st.session_state.last_command = dict(command_outcome.command)
        elif result_context.replace_result_set and quick_action_command is not None:
            st.session_state.last_command = dict(quick_action_command)
        elif result_context.replace_result_set and active_command_pending is not None:
            command_value = active_command_pending.get("command")
            if isinstance(command_value, Mapping):
                st.session_state.last_command = dict(command_value)
    elif filters is not None and search_result is not None:
        st.session_state.last_filters = filters.to_dict()
        st.session_state.last_plan = {
            "intent": filters.intent,
            "entity": filters.entity,
            "requested_field": filters.requested_field,
            "confidence": search_result.confidence,
            "total_matches": search_result.total_matches,
            "relaxed_condition": search_result.relaxed_condition,
        }
    elif agentic_response is not None:
        st.session_state.last_filters = filters.to_dict() if filters is not None else None
        st.session_state.last_plan = {
            "mode": "agentic",
            "answer_type": agentic_response.answer_type,
            "planner_rounds": agentic_response.planner_rounds,
            "search_count": agentic_response.search_count,
            "total_matches": agentic_response.total_matches,
            "relaxed_fields": list(agentic_response.relaxed_fields),
            "strong_event_ids": list(agentic_response.strong_event_ids),
            "reference_event_ids": list(agentic_response.reference_event_ids),
            "latency_ms": {
                "planner": agentic_response.latency.planner_ms,
                "replan": agentic_response.latency.replan_ms,
                "writer": agentic_response.latency.writer_ms,
                "total": agentic_response.latency.total_ms,
            },
            "writer_skipped": agentic_response.writer_skipped,
        }
    elif search_context_for_turn is not None and (
        route.action_type in {"recommend_next", "recommend_similar"}
        or pending_handled
    ):
        # A legacy recommendation is a fresh deterministic result set, not a
        # continuation of the seed search's filter object.
        st.session_state.last_filters = None
        st.session_state.last_plan = {
            "mode": "recommendation",
            "policy": search_context_for_turn.selection_policy,
            "total_matches": search_context_for_turn.total_matches,
        }
    if route.action_type == "scope_search":
        # A domain/security fallback must not leave its old search contract
        # available to a later refinement turn.
        st.session_state.last_filters = None
        st.session_state.last_plan = None
        st.session_state.last_search_context = None
    if search_context_for_turn is not None and result_context.replace_result_set:
        st.session_state.last_search_context = search_context_for_turn.to_dict()
    st.session_state.last_pair_results = (
        list(command_render.pair_results)
        if command_handled and command_render is not None
        else []
    )
    if selected_event is not None:
        st.session_state.selected_event = selected_event
        st.session_state.selected_event_id = str(selected_event.get("id") or "") or None
    elif command_handled and command_render is not None and command_render.pair_results:
        st.session_state.selected_event = None
        st.session_state.selected_event_id = None
    elif search_result is not None:
        st.session_state.selected_event = results[0] if len(results) == 1 else None
        st.session_state.selected_event_id = (
            str(results[0].get("id") or "") if len(results) == 1 else None
        )
    elif result_context.replace_result_set:
        # A new multi-event search/recommendation has no single selected
        # event.  Do not let a stale event answer a later pronoun question.
        st.session_state.selected_event = None
        st.session_state.selected_event_id = None
    if pending_state_to_store is None:
        st.session_state.pop("pending_recommendation", None)
    else:
        st.session_state.pending_recommendation = pending_state_to_store
    if command_handled:
        if command_pending_to_store is None:
            st.session_state.pop("pending_command", None)
        else:
            st.session_state.pending_command = command_pending_to_store
    st.session_state.suppress_result_cards = suppress_cards_for_turn
    st.session_state.recovery_display_results = recovery_display_results_for_turn
    st.session_state.last_query = prompt
    st.session_state.last_action = turn_flow
    st.session_state.feedback = None
    st.rerun()

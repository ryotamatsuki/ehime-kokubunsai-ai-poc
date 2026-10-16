"""Trusted Semantic Operations v2 state reducer.

This module is the key anti-rule-explosion boundary:
- deterministic parsers extract only explicit, already-supported event filters,
- Sarashina contributes a *small* semantic frame for intent/refinement/release/
  references/free-language experience concepts,
- this reducer applies those meanings to bounded conversation state,
- the existing CommandPlan/CommandSlots contract remains the only executable
  interface to search/recommendation tools.

No phrase-specific rules belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import event_search
import experience_preferences
from app_config import POC_REFERENCE_DATE
from command_models import CommandPlan, CommandSlots, CommandValidationError, parse_command_plan
from semantic_frame_v2 import SemanticFrame


INTENT_TO_FLOW = {
    "search": "find_events",
    "count": "count_events",
    "detail": "event_detail",
    "next": "recommend_next",
    "similar": "recommend_similar",
    "pair": "plan_event_pair",
    "explain_search": "explain_search",
    "explain_result": "explain_result",
    "faq": "general_faq",
    "unsupported": "unsupported",
}

RELEASE_GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "fee": ("entry_free", "paid_only", "max_entry_fee"),
    "venue": ("venue",),
    "rain": ("rain_preferred",),
    "reservation": ("reservation_required",),
    "experience": (
        "experience_required",
        "experience_preferred",
        "experience_excluded",
    ),
    "age": ("audience", "age", "age_group", "age_intent"),
    "location": ("municipalities", "regions"),
    "date": ("dates",),
    "time": ("time_slots", "time_after"),
    "topic": ("genres", "topics"),
}

_COLLECTION_FIELDS = {
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


@dataclass(frozen=True)
class SemanticReduction:
    status: str
    frame: SemanticFrame
    plan: CommandPlan | None = None
    message: str = ""
    grounded_slots: Mapping[str, Any] | None = None
    applied_release_groups: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.plan is not None


def _flatten(groups: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(groups, (list, tuple)):
        return result
    for group in groups:
        values = group if isinstance(group, (list, tuple)) else (group,)
        for value in values:
            normalized = str(value).strip()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def _sparse_slots(slots: CommandSlots) -> dict[str, Any]:
    raw = slots.to_dict()
    sparse: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "refine_previous":
            if value:
                sparse[key] = True
            continue
        if value in (None, "", [], (), {}):
            continue
        if isinstance(value, bool) and value is False and key != "reservation_required":
            continue
        sparse[key] = value
    return sparse


def _grounded_slots_from_query(
    query: str,
    reference_date: date = POC_REFERENCE_DATE,
) -> dict[str, Any]:
    """Extract only parser-supported explicit constraints.

    ``parse_query_strict`` deliberately excludes unknown residual text. This is
    the deterministic half of the hybrid architecture: it does not attempt to
    infer subjective suitability or unsupported event facts.
    """

    parsed = event_search.parse_query_strict(query, reference_date)
    topics = [
        term
        for term in event_search.topics_to_soft_terms(parsed.soft_terms)
        if term not in parsed.genres
    ]
    values: dict[str, Any] = {
        "dates": list(parsed.dates),
        "municipalities": _flatten(parsed.city_groups),
        "regions": _flatten(parsed.region_groups),
        "genres": list(parsed.genres),
        "topics": topics,
        "experience_required": list(parsed.experience_required),
        "experience_preferred": list(parsed.experience_preferred),
        "experience_excluded": list(parsed.experience_excluded),
        "audience": (
            "family"
            if parsed.child_friendly is True
            and parsed.age is None
            and parsed.age_group is None
            else None
        ),
        "age": parsed.age,
        "age_group": parsed.age_group,
        "age_intent": parsed.age_intent,
        "venue": {"屋内": "indoor", "屋外": "outdoor"}.get(parsed.venue),
        "entry_free": parsed.entry_free,
        "paid_only": True if parsed.paid_only else None,
        "max_entry_fee": parsed.max_entry_fee,
        "reservation_required": parsed.reservation_required,
        "rain_preferred": True if parsed.rain_preferred else None,
        "time_slots": list(parsed.time_slots),
        "time_after": parsed.time_after,
    }
    return _sparse_slots(CommandSlots.from_dict(values))


def _previous_slots(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    raw = state.get("last_command")
    if not isinstance(raw, Mapping):
        return {}
    try:
        plan = parse_command_plan(raw)
    except (CommandValidationError, TypeError, ValueError):
        return {}
    return _sparse_slots(plan.slots)


def _clear_field(values: dict[str, Any], field_name: str) -> None:
    if field_name in _COLLECTION_FIELDS:
        values[field_name] = []
    elif field_name == "refine_previous":
        values[field_name] = False
    else:
        values[field_name] = None


def _apply_release_groups(values: dict[str, Any], groups: tuple[str, ...]) -> None:
    for group in groups:
        for field_name in RELEASE_GROUP_FIELDS[group]:
            _clear_field(values, field_name)


_REPLACE_ON_EXPLICIT_GROUPS = (
    "fee",
    "venue",
    "rain",
    "reservation",
    "age",
    "location",
    "date",
    "topic",
)


def _has_grounded_member(grounded: Mapping[str, Any], group: str) -> bool:
    return any(
        field_name in grounded and grounded[field_name] not in (None, "", [], (), {})
        for field_name in RELEASE_GROUP_FIELDS[group]
    )


def _merge_experience(values: dict[str, Any], source: Mapping[str, Any]) -> None:
    fields = (
        "experience_required",
        "experience_preferred",
        "experience_excluded",
    )
    if not any(source.get(field_name) for field_name in fields):
        return
    required = list(values.get("experience_required") or [])
    preferred = list(values.get("experience_preferred") or [])
    excluded = list(values.get("experience_excluded") or [])

    for concept in source.get("experience_required") or []:
        preferred = [value for value in preferred if value != concept]
        excluded = [value for value in excluded if value != concept]
        if concept not in required:
            required.append(concept)
    for concept in source.get("experience_preferred") or []:
        required = [value for value in required if value != concept]
        excluded = [value for value in excluded if value != concept]
        if concept not in preferred:
            preferred.append(concept)
    for concept in source.get("experience_excluded") or []:
        required = [value for value in required if value != concept]
        preferred = [value for value in preferred if value != concept]
        if concept not in excluded:
            excluded.append(concept)

    values["experience_required"] = required
    values["experience_preferred"] = preferred
    values["experience_excluded"] = excluded


def _overlay_grounded(values: dict[str, Any], grounded: Mapping[str, Any]) -> None:
    # An explicit new value replaces its prior constraint family. This avoids
    # impossible combinations such as previous paid_only + current free.
    for group in _REPLACE_ON_EXPLICIT_GROUPS:
        if _has_grounded_member(grounded, group):
            for field_name in RELEASE_GROUP_FIELDS[group]:
                _clear_field(values, field_name)

    ordinary = {
        key: value
        for key, value in grounded.items()
        if not key.startswith("experience_")
    }
    values.update(ordinary)
    _merge_experience(values, grounded)


def _apply_reference(values: dict[str, Any], frame: SemanticFrame, state: Mapping[str, Any] | None) -> None:
    reference = frame.reference
    if reference is not None:
        values["reference_kind"] = reference.kind
        values["reference_index"] = reference.index
        values["event_name"] = reference.event_name
        return

    selected = None
    if isinstance(state, Mapping):
        raw_selected = state.get("selected_event_id")
        if raw_selected not in (None, ""):
            selected = str(raw_selected).strip()
    if selected and frame.intent in {"detail", "next", "similar", "explain_result"}:
        values["reference_kind"] = "selected"


def _apply_experience_semantics(values: dict[str, Any], query: str, frame: SemanticFrame) -> None:
    deterministic = experience_preferences.resolve_experience_query(query)
    if deterministic.recognized:
        _merge_experience(
            values,
            {
                "experience_required": list(deterministic.required),
                "experience_preferred": list(deterministic.preferred),
                "experience_excluded": list(deterministic.excluded),
            },
        )
        return
    if frame.experience_required or frame.experience_preferred or frame.experience_excluded:
        _merge_experience(
            values,
            {
                "experience_required": list(frame.experience_required),
                "experience_preferred": list(frame.experience_preferred),
                "experience_excluded": list(frame.experience_excluded),
            },
        )


def _clarification_message(reason: str) -> str:
    return {
        "ambiguous_suitability": (
            "どんな点を重視したい？ たとえば、座って楽しめる・あまり歩かない・"
            "見る／聞く中心、のように教えてみて。"
        ),
        "ambiguous_request": "どんな条件を重視したいか、もう一つだけ教えてみて。",
        "missing_reference": "基準にするイベントを番号かイベント名で教えてみて。",
        "missing_date": "何日に行く予定か教えてみて。",
        "other": "もう少しだけ条件を教えてみて。",
    }.get(reason, "もう少しだけ条件を教えてみて。")


def _data_gap_message(gap: str) -> str:
    labels = {
        "crowding": "混雑度",
        "noise": "静かさ・騒音",
        "wheelchair_access": "車椅子で確実に利用できるか",
        "medical_safety": "個別の健康状態に対する安全性",
        "parking_distance": "駐車場から会場までの実測距離",
        "toilet_proximity": "トイレまでの近さ",
        "weather_guarantee": "天候による中止が絶対にないこと",
        "social_fit": "一人参加や人間関係上の気まずさ",
        "other": "その適性条件",
    }
    label = labels.get(gap, "その適性条件")
    return (
        f"{label}は、このPoCの検索条件として確定できる根拠データが足りません。"
        "分かっている日時・地域・料金・参加形式などの条件なら絞り込めます。"
    )


def reduce_semantic_frame(
    frame: SemanticFrame,
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
) -> SemanticReduction:
    """Reduce one semantic frame into a validated executable CommandPlan."""

    if frame.data_gap != "none":
        return SemanticReduction(
            status="data_limit",
            frame=frame,
            message=_data_gap_message(frame.data_gap),
            applied_release_groups=frame.release,
        )

    if frame.intent == "clarify" or frame.clarification_reason != "none":
        reason = (
            frame.clarification_reason
            if frame.clarification_reason != "none"
            else "ambiguous_request"
        )
        return SemanticReduction(
            status="clarification",
            frame=frame,
            message=_clarification_message(reason),
            applied_release_groups=frame.release,
        )

    if frame.intent == "unsupported":
        return SemanticReduction(
            status="unsupported",
            frame=frame,
            message="このPoCは文化祭イベントの検索・参加案内が中心です。",
            applied_release_groups=frame.release,
        )

    if frame.refine_previous:
        has_previous_results = bool(
            isinstance(state, Mapping) and state.get("last_result_ids")
        )
        has_previous_command = bool(_previous_slots(state))
        if not has_previous_results and not has_previous_command:
            return SemanticReduction(
                status="clarification",
                frame=frame,
                message="直前の検索結果がないけん、まずイベントを探してみて。",
                applied_release_groups=frame.release,
            )

    grounded = _grounded_slots_from_query(query, reference_date)
    values = _previous_slots(state) if frame.refine_previous else {}
    _overlay_grounded(values, grounded)
    _apply_experience_semantics(values, query, frame)
    _apply_reference(values, frame, state)

    if frame.intent == "pair":
        values["visit_count"] = 2
    values["refine_previous"] = bool(frame.refine_previous)

    # Release is applied after deterministic extraction. Lexical matches such
    # as "無料" inside "無料じゃなくてもいい" therefore cannot create the
    # opposite hard constraint.
    _apply_release_groups(values, frame.release)

    flow = INTENT_TO_FLOW.get(frame.intent)
    if flow is None:
        return SemanticReduction(
            status="invalid_frame",
            frame=frame,
            message="意味フレームを実行可能な操作へ変換できませんでした。",
            grounded_slots=grounded,
            applied_release_groups=frame.release,
        )

    try:
        slots = CommandSlots.from_dict(values)
        plan = CommandPlan(flow=flow, slots=slots, confidence=frame.confidence)
    except (CommandValidationError, TypeError, ValueError) as exc:
        return SemanticReduction(
            status="invalid_frame",
            frame=frame,
            message=f"意味フレームの状態適用に失敗しました: {type(exc).__name__}",
            grounded_slots=grounded,
            applied_release_groups=frame.release,
        )

    return SemanticReduction(
        status="ready",
        frame=frame,
        plan=plan,
        grounded_slots=grounded,
        applied_release_groups=frame.release,
    )


__all__ = [
    "INTENT_TO_FLOW",
    "RELEASE_GROUP_FIELDS",
    "SemanticReduction",
    "reduce_semantic_frame",
]

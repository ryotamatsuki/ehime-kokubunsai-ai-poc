"""Trusted sparse-operation reducer for Semantic Operations v2.1.

The reducer applies finite state operations; it never contains utterance-level
phrase rules. Explicit supported filters come from deterministic parsing. A
bounded canonical ``set`` patch can supplement explicit filters that parsing
missed; unset and Experience operations remain separate state operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import event_search
from app_config import POC_REFERENCE_DATE
from command_models import CommandPlan, CommandSlots, CommandValidationError, parse_command_plan
from semantic_frame_v2_1 import SparseSemanticFrame


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

GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "fee": ("entry_free", "paid_only", "max_entry_fee"),
    "venue": ("venue",),
    "rain": ("rain_preferred",),
    "reservation": ("reservation_required",),
    "age": ("audience", "age", "age_group", "age_intent"),
    "location": ("municipalities", "regions"),
    "date": ("dates",),
    "time": ("time_slots", "time_after"),
    "topic": ("genres", "topics"),
    "experience_all": ("experience_required", "experience_preferred", "experience_excluded"),
}
COLLECTION_FIELDS = {
    "dates", "municipalities", "regions", "genres", "topics",
    "experience_required", "experience_preferred", "experience_excluded",
    "time_slots", "detail_fields",
}
REPLACE_GROUPS = ("fee", "venue", "rain", "reservation", "age", "location", "date", "time", "topic")


@dataclass(frozen=True)
class SparseReduction:
    status: str
    frame: SparseSemanticFrame
    plan: CommandPlan | None = None
    message: str = ""
    grounded_slots: Mapping[str, Any] | None = None
    applied_unset: tuple[str, ...] = ()

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
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
    return result


def _sparse(slots: CommandSlots) -> dict[str, Any]:
    raw = slots.to_dict()
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "refine_previous":
            if value:
                result[key] = True
            continue
        if value in (None, "", [], (), {}):
            continue
        if isinstance(value, bool) and value is False and key != "reservation_required":
            continue
        result[key] = value
    return result


def grounded_slots_from_query(query: str, reference_date: date = POC_REFERENCE_DATE) -> dict[str, Any]:
    parsed = event_search.parse_query_strict(query, reference_date)
    topics = event_search.topics_to_soft_terms(parsed.soft_terms)
    values: dict[str, Any] = {
        "dates": list(parsed.dates),
        "municipalities": _flatten(parsed.city_groups),
        "regions": _flatten(parsed.region_groups),
        "genres": list(parsed.genres),
        "topics": topics,
        "experience_required": list(parsed.experience_required),
        "experience_preferred": list(parsed.experience_preferred),
        "experience_excluded": list(parsed.experience_excluded),
        "audience": "family" if parsed.child_friendly is True and parsed.age is None and parsed.age_group is None else None,
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
    return _sparse(CommandSlots.from_dict(values))


def _previous_slots(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not isinstance(state.get("last_command"), Mapping):
        return {}
    try:
        return _sparse(parse_command_plan(state["last_command"]).slots)
    except (CommandValidationError, TypeError, ValueError):
        return {}


def _clear(values: dict[str, Any], field_name: str) -> None:
    values[field_name] = [] if field_name in COLLECTION_FIELDS else None


def _group_present(source: Mapping[str, Any], group: str) -> bool:
    return any(source.get(field_name) not in (None, "", [], (), {}) for field_name in GROUP_FIELDS[group])


def _replace_explicit_groups(values: dict[str, Any], source: Mapping[str, Any]) -> None:
    for group in REPLACE_GROUPS:
        if _group_present(source, group):
            for field_name in GROUP_FIELDS[group]:
                _clear(values, field_name)
    for key, value in source.items():
        if not key.startswith("experience_"):
            values[key] = value


def _merge_experience(values: dict[str, Any], required: Any = (), preferred: Any = (), excluded: Any = ()) -> None:
    req = list(values.get("experience_required") or [])
    pref = list(values.get("experience_preferred") or [])
    exc = list(values.get("experience_excluded") or [])
    for concept in required or ():
        pref = [item for item in pref if item != concept]
        exc = [item for item in exc if item != concept]
        if concept not in req:
            req.append(concept)
    for concept in preferred or ():
        req = [item for item in req if item != concept]
        exc = [item for item in exc if item != concept]
        if concept not in pref:
            pref.append(concept)
    for concept in excluded or ():
        req = [item for item in req if item != concept]
        pref = [item for item in pref if item != concept]
        if concept not in exc:
            exc.append(concept)
    values["experience_required"] = req
    values["experience_preferred"] = pref
    values["experience_excluded"] = exc


def _apply_grounded_experience(values: dict[str, Any], grounded: Mapping[str, Any]) -> None:
    _merge_experience(
        values,
        grounded.get("experience_required", ()),
        grounded.get("experience_preferred", ()),
        grounded.get("experience_excluded", ()),
    )


def _apply_sparse_experience(values: dict[str, Any], frame: SparseSemanticFrame) -> None:
    _merge_experience(values, frame.require, frame.prefer, frame.exclude)


def _apply_audience_mode(values: dict[str, Any], frame: SparseSemanticFrame) -> None:
    if frame.audience_mode == "family":
        for field_name in ("age", "age_group", "age_intent"):
            _clear(values, field_name)
        values["audience"] = "family"
    elif frame.audience_mode == "adult":
        for field_name in ("age", "age_group", "age_intent"):
            _clear(values, field_name)
        values["audience"] = "adult"


def _apply_reference(values: dict[str, Any], frame: SparseSemanticFrame, state: Mapping[str, Any] | None) -> None:
    if frame.reference is not None:
        values["reference_kind"] = frame.reference.kind
        values["reference_index"] = frame.reference.index
        values["event_name"] = frame.reference.event_name
        return
    if isinstance(state, Mapping) and state.get("selected_event_id") and frame.intent in {"detail", "next", "similar", "explain_result"}:
        values["reference_kind"] = "selected"


def _unset_experience_concept(values: dict[str, Any], concept: str) -> None:
    for field_name in ("experience_required", "experience_preferred", "experience_excluded"):
        values[field_name] = [item for item in values.get(field_name, []) if item != concept]


def _apply_unset(values: dict[str, Any], tokens: tuple[str, ...]) -> None:
    for token in tokens:
        if token.startswith("experience:"):
            _unset_experience_concept(values, token.split(":", 1)[1])
            continue
        for field_name in GROUP_FIELDS[token]:
            _clear(values, field_name)


def _clarification_message(reason: str | None) -> str:
    return {
        "ambiguous_suitability": "どんな点を重視したい？ 座って楽しめる、あまり歩かない、見る・聞く中心などから教えてみて。",
        "missing_reference": "基準にするイベントを番号かイベント名で教えてみて。",
        "missing_date": "何日に行く予定か教えてみて。",
        "ambiguous_request": "どんな条件を重視したいか、もう一つだけ教えてみて。",
        "other": "もう少しだけ条件を教えてみて。",
    }.get(reason or "other", "もう少しだけ条件を教えてみて。")


def _data_gap_message(gap: str) -> str:
    labels = {
        "crowding": "混雑度", "noise": "静かさ・騒音", "wheelchair_access": "車椅子で確実に利用できるか",
        "medical_safety": "個別の健康状態に対する安全性", "parking_distance": "駐車場から会場までの実測距離",
        "toilet_proximity": "トイレまでの近さ", "weather_guarantee": "天候による中止保証", "social_fit": "社会的な参加しやすさ",
        "popularity": "人気順位", "fame": "知名度", "duration_fit": "所要時間への適合", "localness": "愛媛らしさ",
        "other": "その適性条件",
    }
    return f"{labels.get(gap, 'その適性条件')}は、このPoCの根拠データだけでは確定できません。分かっている日時・地域・料金・参加形式などなら絞り込めます。"


def reduce_sparse_frame(
    frame: SparseSemanticFrame,
    query: str,
    state: Mapping[str, Any] | None = None,
    *,
    reference_date: date = POC_REFERENCE_DATE,
) -> SparseReduction:
    if frame.data_gap is not None:
        return SparseReduction("data_limit", frame, message=_data_gap_message(frame.data_gap), applied_unset=frame.unset)
    if frame.intent == "clarify" or frame.clarification is not None:
        return SparseReduction("clarification", frame, message=_clarification_message(frame.clarification), applied_unset=frame.unset)
    if frame.intent == "unsupported":
        return SparseReduction("unsupported", frame, message="このPoCは文化祭イベントの検索・参加案内が中心です。", applied_unset=frame.unset)

    if frame.scope == "previous":
        has_context = bool(isinstance(state, Mapping) and (state.get("last_result_ids") or state.get("last_command")))
        if not has_context:
            return SparseReduction("clarification", frame, message="直前の検索結果がないけん、まずイベントを探してみて。", applied_unset=frame.unset)

    grounded = grounded_slots_from_query(query, reference_date)
    values = _previous_slots(state) if frame.scope == "previous" else {}
    _replace_explicit_groups(values, grounded)
    _apply_grounded_experience(values, grounded)
    # Canonical set is a supplement to deterministic parsing. It uses the same
    # replacement semantics as any explicit current-turn filter and therefore
    # cannot accidentally combine old and new values from the same family.
    _replace_explicit_groups(values, dict(frame.set_slots or {}))
    _apply_sparse_experience(values, frame)
    _apply_audience_mode(values, frame)
    _apply_reference(values, frame, state)
    if frame.intent == "pair":
        values["visit_count"] = 2
    values["refine_previous"] = frame.scope == "previous"
    # Unset is deliberately last: lexical matches inside a release utterance
    # cannot recreate the just-released constraint.
    _apply_unset(values, frame.unset)

    flow = INTENT_TO_FLOW.get(frame.intent)
    if flow is None:
        return SparseReduction("invalid_frame", frame, message="意味操作を実行可能なflowへ変換できませんでした。", grounded_slots=grounded, applied_unset=frame.unset)
    try:
        slots = CommandSlots.from_dict(values)
        plan = CommandPlan(flow=flow, slots=slots, confidence="high")
    except (CommandValidationError, TypeError, ValueError) as exc:
        return SparseReduction("invalid_frame", frame, message=f"意味操作の状態適用に失敗しました: {type(exc).__name__}", grounded_slots=grounded, applied_unset=frame.unset)
    return SparseReduction("ready", frame, plan=plan, grounded_slots=grounded, applied_unset=frame.unset)


__all__ = ["GROUP_FIELDS", "INTENT_TO_FLOW", "SparseReduction", "grounded_slots_from_query", "reduce_sparse_frame"]

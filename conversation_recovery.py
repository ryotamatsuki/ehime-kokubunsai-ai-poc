"""Grounded recovery helpers for follow-up turns.

This module is deliberately small and deterministic.  It does not attempt to
reproduce model reasoning; it records only the normalized search contract and
public event facts that were used by the existing Python search pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
import unicodedata
from typing import Any, Mapping, Sequence

import event_search
import experience_matcher
import experience_preferences


MAX_TRACE_RESULTS = 100
MAX_TRACE_EVIDENCE_PER_EVENT = 24


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", normalized)


def _has_any(value: str, phrases: Sequence[str]) -> bool:
    return any(phrase in value for phrase in phrases)


# These are semantic phrase families, not a growing one-keyword patch.  They
# are used only for the conversation-meta layer; event facts remain governed
# by event_search and the structured data model.
_SEARCH_EXPLANATION_MARKERS = (
    "どういう基準",
    "どういう条件",
    "何を基準",
    "どんな基準",
    "どんな条件",
    "どうやって選",
    "どう選んだ",
    "どうやって探",
    "何を見て選",
    "何を見て探",
    "何を見てこの候補",
    "なんでこの",
    "なぜこの",
    "どうしてこの",
    "条件は",
    "検索条件",
    "選定基準",
    "選定理由を教えて",
    "この結果の選定理由",
    "検索の基準",
)
_RESULT_EXPLANATION_MARKERS = (
    "なんで入",
    "なぜ入",
    "どうして入",
    "入った理由",
    "選んだ理由",
    "なんで選",
    "なぜ選",
    "どうして選",
    "どうやって選",
    "どうやって入",
    "どういう理由",
    "選ばれた理由",
    "選定理由",
    "根拠",
    "本当に座",
    "本当に歩",
    "本当に雨",
)
_REFERENCE_MARKERS = (
    "これ",
    "それ",
    "このイベント",
    "そのイベント",
    "この中",
    "さっき",
    "今の",
    "結果",
    "候補",
    "番目",
    "最初",
    "最後",
)
_DETAIL_WORDS = (
    "予約",
    "申込",
    "料金",
    "いくら",
    "駐車場",
    "雨",
    "座",
    "歩",
    "バリアフリー",
    "アクセス",
    "公式",
    "本物",
)
_DOMAIN_BOUNDARY_MARKERS = (
    "株価",
    "おすすめの株",
    "量子力学",
    "天気",
    "ニュース",
    "python",
    "システムプロンプト",
    "個人情報",
    "秘密の設定",
)


def is_search_explanation_query(query: str) -> bool:
    """Recognize a question about how the immediately preceding search ran."""

    compact = _compact(query)
    # An explicit event reference makes this an item-level question when the
    # wording asks why that event was selected.  Keep the broader "基準" and
    # "条件" families as search-level explanations.
    if _has_any(compact, ("このイベント", "そのイベント")) and _has_any(
        compact, ("なぜ", "なんで", "どうして", "選ばれ", "入って")
    ) and not _has_any(compact, ("基準", "条件", "検索")):
        return False
    if "選んだ理由" in compact:
        return not _has_any(compact, ("番目", "最初", "最後", "このイベント", "そのイベント"))
    return _has_any(compact, _SEARCH_EXPLANATION_MARKERS)


def is_result_explanation_query(query: str) -> bool:
    """Recognize a question about why a referenced result was included."""

    compact = _compact(query)
    if is_search_explanation_query(compact):
        return False
    has_reason = _has_any(compact, _RESULT_EXPLANATION_MARKERS) or "なぜ" in compact
    return has_reason and _has_any(compact, _REFERENCE_MARKERS)


def is_ambiguous_reference_query(query: str) -> bool:
    """Return true for a context-dependent question whose target is unclear."""

    compact = _compact(query)
    has_reference = _has_any(compact, _REFERENCE_MARKERS)
    has_detail = _has_any(compact, _DETAIL_WORDS)
    return has_reference and not has_detail and not is_search_explanation_query(compact)


def is_domain_out_of_scope(query: str) -> bool:
    """Recognize a small set of unambiguous non-event domain requests."""

    return _has_any(_compact(query).lower(), _DOMAIN_BOUNDARY_MARKERS)


@dataclass(frozen=True)
class SearchContext:
    """Bounded public context used to answer a later follow-up."""

    original_query: str
    normalized_query: str
    normalized_conditions: dict[str, Any]
    result_ids: list[str]
    selection_policy: str
    result_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    total_matches: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "SearchContext | None":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        try:
            result_ids = [str(item) for item in value.get("result_ids", [])][:MAX_TRACE_RESULTS]
            raw_evidence = value.get("result_evidence", {})
            evidence = {
                str(event_id): [dict(item) for item in items[:MAX_TRACE_EVIDENCE_PER_EVENT]]
                for event_id, items in raw_evidence.items()
                if isinstance(items, list)
                and all(isinstance(item, Mapping) for item in items)
            } if isinstance(raw_evidence, Mapping) else {}
            conditions = value.get("normalized_conditions", {})
            if not isinstance(conditions, Mapping):
                conditions = {}
            return cls(
                original_query=str(value.get("original_query", "")),
                normalized_query=str(value.get("normalized_query", "")),
                normalized_conditions=dict(conditions),
                result_ids=result_ids,
                selection_policy=str(value.get("selection_policy", "")),
                result_evidence=evidence,
                total_matches=int(value.get("total_matches", len(result_ids))),
                created_at=str(value.get("created_at", "")),
            )
        except (TypeError, ValueError, AttributeError):
            return None


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id") or str(event.get("公式URL", "")).rstrip("/").rsplit("/", 1)[-1])


def _filters_dict(filters: Any) -> dict[str, Any]:
    if filters is None:
        return {}
    if hasattr(filters, "to_dict"):
        try:
            value = filters.to_dict()
            return dict(value) if isinstance(value, Mapping) else {}
        except (TypeError, ValueError):
            return {}
    if isinstance(filters, Mapping):
        return dict(filters)
    return {}


def _present_conditions(filters: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"intent", "entity", "requested_field", "invalid_date", "keywords"}
    result: dict[str, Any] = {}
    for key, value in filters.items():
        if key in ignored or value in (None, False, "", [], ()):
            continue
        result[str(key)] = value
    return result


def _condition_label(key: str, value: Any) -> str:
    labels = {
        "dates": "日付",
        "city": "市町",
        "city_groups": "市町",
        "region": "地域",
        "region_groups": "地域",
        "genres": "ジャンル",
        "genre_groups": "ジャンル",
        "child_friendly": "子ども向け",
        "age": "年齢",
        "age_group": "対象年齢層",
        "venue": "屋内外",
        "rain_preferred": "雨でも参加しやすい条件",
        "entry_free": "入場無料",
        "paid_only": "有料",
        "reservation_required": "申込要否",
        "max_entry_fee": "料金上限",
        "time_slots": "時間帯",
        "time_after": "終了時刻",
        "soft_terms": "テーマ・検索語",
        "experience_required": "体験条件（必須）",
        "experience_preferred": "体験条件（希望）",
        "experience_excluded": "体験条件（除外）",
    }
    label = labels.get(key, key)
    if key.startswith("experience_") and isinstance(value, (list, tuple)):
        rendered: list[str] = []
        for item in value:
            try:
                rendered.append(experience_preferences.concept(str(item)).label)
            except experience_preferences.ExperienceVocabularyError:
                rendered.append(str(item))
        return f"{label}: {', '.join(rendered)}"
    return f"{label}: {value}"


def _fact_evidence(event: Mapping[str, Any], filters: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for concept_field in ("experience_required", "experience_excluded", "experience_preferred"):
        values = filters.get(concept_field) or []
        for concept_id in values:
            match = experience_matcher.concept_match(event, str(concept_id))
            if match is True:
                profile = experience_matcher.experience_profile(event) or {}
                evidence.append(
                    {
                        "condition": str(concept_id),
                        "evidence_level": "explicit",
                        "evidence_source": "data_model_v3.experience_profile",
                        "evidence_value": {
                            "posture": profile.get("posture"),
                            "seating": profile.get("seating"),
                            "mobility_load": profile.get("mobility_load"),
                            "engagement_modes": profile.get("engagement_modes", []),
                        },
                    }
                )
            elif match is None:
                evidence.append(
                    {
                        "condition": str(concept_id),
                        "evidence_level": "unknown",
                        "evidence_source": "data_model_v3.experience_profile",
                        "evidence_value": None,
                    }
                )

    # Keep general filter evidence factual and compact.  The filter itself is
    # the result of the trusted deterministic search; the value below points
    # back to the public event field rather than to model reasoning.
    field_map = {
        "dates": "日時",
        "city_groups": "市町",
        "region_groups": "地域",
        "genres": "ジャンル",
        "venue": "屋内/屋外",
        "entry_free": "料金",
        "reservation_required": "参加案内.申込要否",
        "child_friendly": "子ども向け",
    }
    for key, field_name in field_map.items():
        value = filters.get(key)
        if value in (None, False, [], ()):
            continue
        if key == "reservation_required":
            current = event.get("参加案内", {}).get("申込要否") if isinstance(event.get("参加案内"), Mapping) else None
        else:
            current = event.get(field_name)
        evidence.append(
            {
                "condition": key,
                "evidence_level": "explicit" if current is not None else "unknown",
                "evidence_source": field_name,
                "evidence_value": current,
            }
        )
    return evidence[:MAX_TRACE_EVIDENCE_PER_EVENT]


def build_search_context(
    query: str,
    filters: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    result_ids: Sequence[Any] | None = None,
    total_matches: int | None = None,
    selection_policy: str = "deterministic_hard_filters_then_existing_ranker",
) -> SearchContext:
    """Create a bounded trace from the same filters and records used to search."""

    filters_dict = _filters_dict(filters)
    ids = [str(value) for value in (result_ids or (_event_id(event) for event in events))]
    ids = list(dict.fromkeys(ids))[:MAX_TRACE_RESULTS]
    evidence = {
        _event_id(event): _fact_evidence(event, filters_dict)
        for event in events
        if _event_id(event) in ids
    }
    return SearchContext(
        original_query=str(query),
        normalized_query=event_search.normalize_query(query),
        normalized_conditions=_present_conditions(filters_dict),
        result_ids=ids,
        selection_policy=selection_policy,
        result_evidence=evidence,
        total_matches=int(total_matches if total_matches is not None else len(ids)),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _experience_labels(conditions: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("experience_required", "experience_preferred", "experience_excluded"):
        for value in conditions.get(key, []) or []:
            try:
                label = experience_preferences.concept(str(value)).label
            except experience_preferences.ExperienceVocabularyError:
                label = str(value)
            if label not in labels:
                labels.append(label)
    return labels


def render_search_explanation(context: SearchContext | None) -> str:
    if context is None or not context.result_ids:
        return "直前の検索条件を確認できませんでした。もう一度、探したい条件を教えてみて。"
    conditions = context.normalized_conditions
    labels = [_condition_label(key, value) for key, value in conditions.items()]
    experience_labels = _experience_labels(conditions)
    if "座って楽しめる" in experience_labels:
        return (
            f"「座って楽しめる」を、イベントの体験特性で主に着席して楽しめることを"
            f"確認できるものとして検索したよ。{context.total_matches}件が条件に合ったけん、"
            "単に屋内というだけでは対象にしていないよ。"
        )
    condition_text = "、".join(labels) if labels else "イベント情報の条件"
    return (
        f"今回は{condition_text}を、イベントDBの構造化された情報に照合して選んだよ。"
        f"条件に合ったのは{context.total_matches}件。"
    )


def _format_evidence(evidence: Mapping[str, Any]) -> str:
    level = str(evidence.get("evidence_level", "unknown"))
    value = evidence.get("evidence_value")
    condition = str(evidence.get("condition", "条件"))
    if level == "unknown":
        return f"{condition}の根拠はデータ上で確認できません"
    if condition == "seated" and isinstance(value, Mapping):
        return (
            "体験特性に「主に着席」と記録され、"
            f"座席情報は「{value.get('seating', '不明')}」"
        )
    return f"{evidence.get('evidence_source', 'イベント情報')}に「{value}」と記録されている"


def render_result_explanation(
    context: SearchContext | None,
    event: Mapping[str, Any] | None,
    *,
    query: str = "",
) -> str:
    if context is None or event is None:
        return "どのイベントの根拠か確認できませんでした。番号かイベント名を教えてみて。"
    event_id = _event_id(event)
    evidence = context.result_evidence.get(event_id, [])
    name = str(event.get("イベント名", "このイベント"))
    if not evidence:
        return f"「{name}」を選んだ根拠は、このPoCデータでは確認できません。"
    explicit = [item for item in evidence if item.get("evidence_level") == "explicit"]
    unknown = [item for item in evidence if item.get("evidence_level") == "unknown"]
    if explicit:
        reasons = "、".join(_format_evidence(item) for item in explicit[:3])
        answer = f"「{name}」は、{reasons}ため、今回の条件に含まれているよ。"
        if unknown:
            answer += "一部の情報は確認できないため、断定していないよ。"
        return answer
    return f"「{name}」は、必要な根拠がデータ上で確認できないため、確実とは言い切れないよ。"


def reference_event(
    context: SearchContext | None,
    events: Sequence[Mapping[str, Any]],
    index: int | None,
) -> Mapping[str, Any] | None:
    if context is None or index is None or index < 0 or index >= len(context.result_ids):
        return None
    target_id = context.result_ids[index]
    return next((event for event in events if _event_id(event) == target_id), None)


__all__ = [
    "SearchContext",
    "build_search_context",
    "is_ambiguous_reference_query",
    "is_domain_out_of_scope",
    "is_result_explanation_query",
    "is_search_explanation_query",
    "reference_event",
    "render_result_explanation",
    "render_search_explanation",
]

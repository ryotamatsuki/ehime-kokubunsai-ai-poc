"""Compact semantic-frame contract for Semantic Operations v2.

The language model is intentionally *not* asked to reproduce the complete
CommandSlots schema. Python remains authoritative for explicit dates, places,
fees, ages, reservation flags, venue, time filters, and event facts.

The model only classifies the small set of meanings that are hard to recover
reliably with deterministic parsing:
- user intent,
- whether the previous result set is being refined,
- explicit constraint-release groups,
- experience concepts expressed in free language,
- references to a prior/selected result,
- clarification / data-capability boundaries.

This keeps the natural-language surface open while the internal state machine
stays finite and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import experience_preferences
from command_models import MAX_REFERENCE_INDEX


INTENT_VALUES = frozenset(
    {
        "search",
        "count",
        "detail",
        "next",
        "similar",
        "pair",
        "explain_search",
        "explain_result",
        "faq",
        "unsupported",
        "clarify",
    }
)
RELEASE_GROUP_VALUES = frozenset(
    {
        "fee",
        "venue",
        "rain",
        "reservation",
        "experience",
        "age",
        "location",
        "date",
        "time",
        "topic",
    }
)
REFERENCE_KIND_VALUES = frozenset({"ordinal", "event_name", "selected", "last_result"})
CLARIFICATION_REASON_VALUES = frozenset(
    {
        "none",
        "ambiguous_suitability",
        "ambiguous_request",
        "missing_reference",
        "missing_date",
        "other",
    }
)
DATA_GAP_VALUES = frozenset(
    {
        "none",
        "crowding",
        "noise",
        "wheelchair_access",
        "medical_safety",
        "parking_distance",
        "toilet_proximity",
        "weather_guarantee",
        "social_fit",
        "other",
    }
)
CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
MAX_RELEASE_GROUPS = 10
MAX_EXPERIENCE_ITEMS = 6
MAX_EVENT_NAME_LENGTH = 240


class SemanticFrameError(ValueError):
    """Raised when untrusted semantic-frame output violates the v2 contract."""


def _fail(message: str) -> None:
    raise SemanticFrameError(message)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate field: {key}")
        result[key] = value
    return result


def _strict_enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field_name} must be boolean")
    return value


def _strict_list(value: Any, allowed: frozenset[str], field_name: str, max_items: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be an array")
    if len(value) > max_items:
        _fail(f"{field_name} has too many items")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            _fail(f"{field_name} contains invalid value: {item!r}")
        if item in result:
            _fail(f"{field_name} contains duplicate value: {item!r}")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class SemanticReference:
    kind: str
    index: int | None = None
    event_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _strict_enum(self.kind, REFERENCE_KIND_VALUES, "reference.kind"))
        if self.index is not None:
            if isinstance(self.index, bool) or not isinstance(self.index, int):
                _fail("reference.index must be an integer")
            if not 1 <= self.index <= MAX_REFERENCE_INDEX:
                _fail(f"reference.index must be between 1 and {MAX_REFERENCE_INDEX}")
        if self.event_name is not None:
            if not isinstance(self.event_name, str):
                _fail("reference.event_name must be a string")
            name = self.event_name.strip()
            if not name or len(name) > MAX_EVENT_NAME_LENGTH:
                _fail("reference.event_name is invalid")
            object.__setattr__(self, "event_name", name)

        if self.kind == "ordinal":
            if self.index is None or self.event_name is not None:
                _fail("ordinal reference requires index only")
        elif self.kind == "event_name":
            if self.event_name is None or self.index is not None:
                _fail("event_name reference requires event_name only")
        elif self.index is not None or self.event_name is not None:
            _fail(f"{self.kind} reference cannot carry index/event_name")

    @classmethod
    def from_value(cls, raw: Any) -> "SemanticReference | None":
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("reference must be an object or null")
        allowed = {"kind", "index", "event_name"}
        unknown = set(raw) - allowed
        if unknown:
            _fail(f"reference has unknown fields: {sorted(unknown)}")
        if "kind" not in raw:
            _fail("reference.kind is required")
        return cls(
            kind=raw["kind"],
            index=raw.get("index"),
            event_name=raw.get("event_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.index is not None:
            result["index"] = self.index
        if self.event_name is not None:
            result["event_name"] = self.event_name
        return result


@dataclass(frozen=True)
class SemanticFrame:
    intent: str
    refine_previous: bool = False
    release: tuple[str, ...] = ()
    experience_required: tuple[str, ...] = ()
    experience_preferred: tuple[str, ...] = ()
    experience_excluded: tuple[str, ...] = ()
    reference: SemanticReference | None = None
    clarification_reason: str = "none"
    data_gap: str = "none"
    confidence: str = "medium"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _strict_enum(self.intent, INTENT_VALUES, "intent"))
        object.__setattr__(
            self,
            "refine_previous",
            _strict_bool(self.refine_previous, "refine_previous"),
        )
        release = _strict_list(
            list(self.release),
            RELEASE_GROUP_VALUES,
            "release",
            MAX_RELEASE_GROUPS,
        )
        object.__setattr__(self, "release", release)

        concepts = experience_preferences.EXPERIENCE_CONCEPT_IDS
        required = _strict_list(
            list(self.experience_required),
            concepts,
            "experience_required",
            MAX_EXPERIENCE_ITEMS,
        )
        preferred = _strict_list(
            list(self.experience_preferred),
            concepts,
            "experience_preferred",
            MAX_EXPERIENCE_ITEMS,
        )
        excluded = _strict_list(
            list(self.experience_excluded),
            concepts,
            "experience_excluded",
            MAX_EXPERIENCE_ITEMS,
        )
        if (set(required) & set(preferred)) or (set(required) & set(excluded)) or (set(preferred) & set(excluded)):
            _fail("experience concepts must not overlap across required/preferred/excluded")
        object.__setattr__(self, "experience_required", required)
        object.__setattr__(self, "experience_preferred", preferred)
        object.__setattr__(self, "experience_excluded", excluded)

        if self.reference is not None and not isinstance(self.reference, SemanticReference):
            _fail("reference must be SemanticReference or null")
        object.__setattr__(
            self,
            "clarification_reason",
            _strict_enum(
                self.clarification_reason,
                CLARIFICATION_REASON_VALUES,
                "clarification_reason",
            ),
        )
        object.__setattr__(self, "data_gap", _strict_enum(self.data_gap, DATA_GAP_VALUES, "data_gap"))
        object.__setattr__(self, "confidence", _strict_enum(self.confidence, CONFIDENCE_VALUES, "confidence"))

        if self.intent == "clarify" and self.clarification_reason == "none":
            _fail("clarify intent requires clarification_reason")
        if self.data_gap != "none" and self.intent not in {"search", "detail", "clarify", "unsupported"}:
            _fail("data_gap is only valid for search/detail/clarify/unsupported intents")

    @classmethod
    def from_dict(cls, raw: Any) -> "SemanticFrame":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("semantic frame must be an object")
        allowed = {
            "intent",
            "refine_previous",
            "release",
            "experience_required",
            "experience_preferred",
            "experience_excluded",
            "reference",
            "clarification_reason",
            "data_gap",
            "confidence",
        }
        unknown = set(raw) - allowed
        if unknown:
            _fail(f"semantic frame has unknown fields: {sorted(unknown)}")
        if "intent" not in raw:
            _fail("intent is required")
        return cls(
            intent=raw["intent"],
            refine_previous=raw.get("refine_previous", False),
            release=tuple(raw.get("release", [])),
            experience_required=tuple(raw.get("experience_required", [])),
            experience_preferred=tuple(raw.get("experience_preferred", [])),
            experience_excluded=tuple(raw.get("experience_excluded", [])),
            reference=SemanticReference.from_value(raw.get("reference")),
            clarification_reason=raw.get("clarification_reason", "none"),
            data_gap=raw.get("data_gap", "none"),
            confidence=raw.get("confidence", "medium"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SemanticFrame":
        if not isinstance(raw, str):
            _fail("semantic frame output must be a string")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        if not text:
            _fail("semantic frame output is empty")
        try:
            parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            _fail(f"invalid semantic-frame JSON: {exc.msg}")
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "refine_previous": self.refine_previous,
            "release": list(self.release),
            "experience_required": list(self.experience_required),
            "experience_preferred": list(self.experience_preferred),
            "experience_excluded": list(self.experience_excluded),
            "reference": self.reference.to_dict() if self.reference is not None else None,
            "clarification_reason": self.clarification_reason,
            "data_gap": self.data_gap,
            "confidence": self.confidence,
        }


SEMANTIC_FRAME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "refine_previous",
        "release",
        "experience_required",
        "experience_preferred",
        "experience_excluded",
        "reference",
        "clarification_reason",
        "data_gap",
        "confidence",
    ],
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENT_VALUES)},
        "refine_previous": {"type": "boolean"},
        "release": {
            "type": "array",
            "maxItems": MAX_RELEASE_GROUPS,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(RELEASE_GROUP_VALUES)},
        },
        "experience_required": {
            "type": "array",
            "maxItems": MAX_EXPERIENCE_ITEMS,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(experience_preferences.EXPERIENCE_CONCEPT_IDS)},
        },
        "experience_preferred": {
            "type": "array",
            "maxItems": MAX_EXPERIENCE_ITEMS,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(experience_preferences.EXPERIENCE_CONCEPT_IDS)},
        },
        "experience_excluded": {
            "type": "array",
            "maxItems": MAX_EXPERIENCE_ITEMS,
            "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(experience_preferences.EXPERIENCE_CONCEPT_IDS)},
        },
        "reference": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind"],
                    "properties": {
                        "kind": {"type": "string", "enum": sorted(REFERENCE_KIND_VALUES)},
                        "index": {"type": "integer", "minimum": 1, "maximum": MAX_REFERENCE_INDEX},
                        "event_name": {"type": "string", "minLength": 1, "maxLength": MAX_EVENT_NAME_LENGTH},
                    },
                },
            ]
        },
        "clarification_reason": {
            "type": "string",
            "enum": sorted(CLARIFICATION_REASON_VALUES),
        },
        "data_gap": {"type": "string", "enum": sorted(DATA_GAP_VALUES)},
        "confidence": {"type": "string", "enum": sorted(CONFIDENCE_VALUES)},
    },
}


def build_semantic_frame_system_prompt() -> str:
    concepts = ", ".join(sorted(experience_preferences.EXPERIENCE_CONCEPT_IDS))
    releases = ", ".join(sorted(RELEASE_GROUP_VALUES))
    gaps = ", ".join(sorted(DATA_GAP_VALUES - {"none"}))
    return f"""あなたは愛媛の文化祭イベント案内の Semantic Frame Normalizer です。
回答文やイベント事実は生成しません。利用者の発話を小さな意味フレームへ変換します。

出力は必ず次の10キーを持つJSONオブジェクト1個だけです。説明文やMarkdownは禁止です。
{{\"intent\":\"search\",\"refine_previous\":false,\"release\":[],\"experience_required\":[],\"experience_preferred\":[],\"experience_excluded\":[],\"reference\":null,\"clarification_reason\":\"none\",\"data_gap\":\"none\",\"confidence\":\"high\"}}

重要:
- 日付、市町、地域、料金、年齢、予約、屋内外、時間帯、ジャンルは抽出しないでください。
  それらは後段Pythonが利用者の明示表現から決定します。
- あなたが担当するのは intent / 前回結果のrefine / 条件解除 / Experience / reference /
  clarification / data-gapだけです。
- 言い回しを列挙して一致判定するのではなく、意味で分類してください。
- 「〜でなくてもよい」「気にしない」「条件から外す」は反対条件のSETではなく release です。
  release は現在の会話stateからその制約群を消す操作です。
- release の意味は次の抽象グループだけです。個別の日本語表現をコピーしないでください。
  fee=無料/有料/料金上限、venue=屋内外、rain=雨天条件、reservation=予約条件、
  experience=体験特性、age=年齢/対象層、location=市町/地域、date=日付、
  time=時間帯/時刻、topic=ジャンル/テーマ。
- release に使える値: {releases}
- Experience IDは次だけです: {concepts}
- Experienceは利用者が体験特性を述べた場合だけ設定します。年齢属性、病名、性格、
  同行者属性からExperienceやadult等を推測してはいけません。
- required=必須、preferred=できれば、excluded=避けたい。解除はExperience配列ではなくreleaseを使います。
- 混雑、静かさ、医学的安全性、駐車場からの距離、トイレ近接、天候中止保証など、
  イベントDBに根拠がない適性要求は data_gap に分類してください: {gaps}
- 「その中から」「さっきの候補から」のように前回結果集合をさらに絞る意味なら refine_previous=true。
- 「2番目」なら reference={{\"kind\":\"ordinal\",\"index\":2}}。
  選択中のものなら kind=selected、前回結果全体への参照なら kind=last_result、
  明示されたイベント名なら kind=event_name と event_name を使います。
- 曖昧な適性要求で具体的な必要条件が分からない場合は intent=clarify と
  clarification_reason を使い、属性から条件を推測しません。
- 文化祭イベント案内の範囲外やイベント捏造要求は intent=unsupported。
- confidenceは意味判定の確信度だけです。イベントが実在するか、条件を満たすかの確信度ではありません。
"""


def build_semantic_frame_payload(query: str, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    safe_state: dict[str, Any] = {}
    if isinstance(state, Mapping):
        if state.get("selected_event_id") is not None:
            safe_state["selected_event_id"] = str(state.get("selected_event_id"))[:64]
        result_ids = state.get("last_result_ids")
        if isinstance(result_ids, (list, tuple)):
            safe_state["last_result_ids"] = [str(value)[:64] for value in list(result_ids)[:32]]
        safe_state["has_previous_results"] = bool(safe_state.get("last_result_ids"))
        safe_state["has_previous_command"] = isinstance(state.get("last_command"), Mapping)
    return {
        "query": str(query).strip()[:1200],
        "state": safe_state,
    }


__all__ = [
    "CLARIFICATION_REASON_VALUES",
    "CONFIDENCE_VALUES",
    "DATA_GAP_VALUES",
    "INTENT_VALUES",
    "REFERENCE_KIND_VALUES",
    "RELEASE_GROUP_VALUES",
    "SEMANTIC_FRAME_JSON_SCHEMA",
    "SemanticFrame",
    "SemanticFrameError",
    "SemanticReference",
    "build_semantic_frame_payload",
    "build_semantic_frame_system_prompt",
]

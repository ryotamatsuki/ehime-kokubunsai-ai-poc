"""Sparse semantic-operation contract for Semantic Operations v2.1.

Only ``intent`` is required.  Every other field is omitted unless the current
utterance actually carries that meaning.  This removes the v2 requirement for
a 3B model to emit many unrelated empty/null fields on every turn.

Natural-language variety remains the model's job; the contract itself stays
small and finite.  Explicit event filters are still grounded by Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import experience_preferences
from command_models import MAX_REFERENCE_INDEX


INTENT_VALUES = frozenset({
    "search", "count", "detail", "next", "similar", "pair",
    "explain_search", "explain_result", "faq", "unsupported", "clarify",
})
SCOPE_VALUES = frozenset({"new", "previous"})
REFERENCE_KIND_VALUES = frozenset({"ordinal", "event_name", "selected", "last_result"})
CLARIFICATION_VALUES = frozenset({
    "ambiguous_suitability", "ambiguous_request", "missing_reference",
    "missing_date", "other",
})
DATA_GAP_VALUES = frozenset({
    "crowding", "noise", "wheelchair_access", "medical_safety",
    "parking_distance", "toilet_proximity", "weather_guarantee",
    "social_fit", "other",
})
AUDIENCE_MODE_VALUES = frozenset({"family", "adult", "target"})
UNSET_GROUP_VALUES = frozenset({
    "fee", "venue", "rain", "reservation", "age", "location", "date",
    "time", "topic", "experience_all",
})
EXPERIENCE_CONCEPTS = frozenset(experience_preferences.EXPERIENCE_CONCEPT_IDS)
EXPERIENCE_UNSET_VALUES = frozenset(f"experience:{value}" for value in EXPERIENCE_CONCEPTS)
UNSET_VALUES = UNSET_GROUP_VALUES | EXPERIENCE_UNSET_VALUES
MAX_UNSET_ITEMS = 12
MAX_EXPERIENCE_ITEMS = 6
MAX_EVENT_NAME_LENGTH = 240


class SparseFrameError(ValueError):
    pass


def _fail(message: str) -> None:
    raise SparseFrameError(message)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate field: {key}")
        result[key] = value
    return result


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _list(value: Any, allowed: frozenset[str], field_name: str, maximum: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > maximum:
        _fail(f"{field_name} must be an array with at most {maximum} items")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            _fail(f"{field_name} contains invalid value: {item!r}")
        if item in result:
            _fail(f"{field_name} contains duplicate value: {item!r}")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class SparseReference:
    kind: str
    index: int | None = None
    event_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, REFERENCE_KIND_VALUES, "reference.kind"))
        if self.index is not None:
            if isinstance(self.index, bool) or not isinstance(self.index, int) or not 1 <= self.index <= MAX_REFERENCE_INDEX:
                _fail("reference.index is out of range")
        if self.event_name is not None:
            if not isinstance(self.event_name, str):
                _fail("reference.event_name must be a string")
            value = self.event_name.strip()
            if not value or len(value) > MAX_EVENT_NAME_LENGTH:
                _fail("reference.event_name is invalid")
            object.__setattr__(self, "event_name", value)
        if self.kind == "ordinal" and self.index is None:
            _fail("ordinal reference requires index")
        if self.kind == "event_name" and self.event_name is None:
            _fail("event_name reference requires event_name")
        if self.kind in {"selected", "last_result"} and (self.index is not None or self.event_name is not None):
            _fail(f"{self.kind} reference cannot carry index/event_name")

    @classmethod
    def from_value(cls, raw: Any) -> "SparseReference | None":
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("reference must be an object")
        allowed = {"kind", "index", "event_name"}
        unknown = set(raw) - allowed
        if unknown or "kind" not in raw:
            _fail("reference has invalid fields")
        return cls(kind=raw["kind"], index=raw.get("index"), event_name=raw.get("event_name"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.index is not None:
            result["index"] = self.index
        if self.event_name is not None:
            result["event_name"] = self.event_name
        return result


@dataclass(frozen=True)
class SparseSemanticFrame:
    intent: str
    scope: str = "new"
    unset: tuple[str, ...] = ()
    require: tuple[str, ...] = ()
    prefer: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    reference: SparseReference | None = None
    clarification: str | None = None
    data_gap: str | None = None
    audience_mode: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _enum(self.intent, INTENT_VALUES, "intent"))
        object.__setattr__(self, "scope", _enum(self.scope, SCOPE_VALUES, "scope"))
        object.__setattr__(self, "unset", _list(list(self.unset), UNSET_VALUES, "unset", MAX_UNSET_ITEMS))
        object.__setattr__(self, "require", _list(list(self.require), EXPERIENCE_CONCEPTS, "require", MAX_EXPERIENCE_ITEMS))
        object.__setattr__(self, "prefer", _list(list(self.prefer), EXPERIENCE_CONCEPTS, "prefer", MAX_EXPERIENCE_ITEMS))
        object.__setattr__(self, "exclude", _list(list(self.exclude), EXPERIENCE_CONCEPTS, "exclude", MAX_EXPERIENCE_ITEMS))
        if (set(self.require) & set(self.prefer)) or (set(self.require) & set(self.exclude)) or (set(self.prefer) & set(self.exclude)):
            _fail("experience concepts overlap across require/prefer/exclude")
        if self.reference is not None and not isinstance(self.reference, SparseReference):
            _fail("reference must be SparseReference")
        if self.clarification is not None:
            object.__setattr__(self, "clarification", _enum(self.clarification, CLARIFICATION_VALUES, "clarification"))
        if self.data_gap is not None:
            object.__setattr__(self, "data_gap", _enum(self.data_gap, DATA_GAP_VALUES, "data_gap"))
        if self.audience_mode is not None:
            object.__setattr__(self, "audience_mode", _enum(self.audience_mode, AUDIENCE_MODE_VALUES, "audience_mode"))
        if self.intent == "clarify" and self.clarification is None:
            _fail("clarify intent requires clarification")

    @classmethod
    def from_dict(cls, raw: Any) -> "SparseSemanticFrame":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("sparse frame must be an object")
        allowed = {"intent", "scope", "unset", "require", "prefer", "exclude", "reference", "clarification", "data_gap", "audience_mode"}
        unknown = set(raw) - allowed
        if unknown:
            _fail(f"unknown sparse frame fields: {sorted(unknown)}")
        if "intent" not in raw:
            _fail("intent is required")
        return cls(
            intent=raw["intent"],
            scope=raw.get("scope", "new"),
            unset=tuple(raw.get("unset", [])),
            require=tuple(raw.get("require", [])),
            prefer=tuple(raw.get("prefer", [])),
            exclude=tuple(raw.get("exclude", [])),
            reference=SparseReference.from_value(raw.get("reference")),
            clarification=raw.get("clarification"),
            data_gap=raw.get("data_gap"),
            audience_mode=raw.get("audience_mode"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SparseSemanticFrame":
        if not isinstance(raw, str) or not raw.strip():
            _fail("sparse frame output is empty")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            _fail(f"invalid sparse frame JSON: {exc.msg}")
        return cls.from_dict(parsed)

    def to_dict(self, *, sparse: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"intent": self.intent}
        optional = {
            "scope": self.scope if self.scope != "new" else None,
            "unset": list(self.unset) or None,
            "require": list(self.require) or None,
            "prefer": list(self.prefer) or None,
            "exclude": list(self.exclude) or None,
            "reference": self.reference.to_dict() if self.reference else None,
            "clarification": self.clarification,
            "data_gap": self.data_gap,
            "audience_mode": self.audience_mode,
        }
        if sparse:
            result.update({key: value for key, value in optional.items() if value is not None})
        else:
            result.update(optional)
        return result


SPARSE_FRAME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent"],
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENT_VALUES)},
        "scope": {"type": "string", "enum": sorted(SCOPE_VALUES)},
        "unset": {"type": "array", "maxItems": MAX_UNSET_ITEMS, "uniqueItems": True, "items": {"type": "string", "enum": sorted(UNSET_VALUES)}},
        "require": {"type": "array", "maxItems": MAX_EXPERIENCE_ITEMS, "uniqueItems": True, "items": {"type": "string", "enum": sorted(EXPERIENCE_CONCEPTS)}},
        "prefer": {"type": "array", "maxItems": MAX_EXPERIENCE_ITEMS, "uniqueItems": True, "items": {"type": "string", "enum": sorted(EXPERIENCE_CONCEPTS)}},
        "exclude": {"type": "array", "maxItems": MAX_EXPERIENCE_ITEMS, "uniqueItems": True, "items": {"type": "string", "enum": sorted(EXPERIENCE_CONCEPTS)}},
        "reference": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind"],
            "properties": {
                "kind": {"type": "string", "enum": sorted(REFERENCE_KIND_VALUES)},
                "index": {"type": "integer", "minimum": 1, "maximum": MAX_REFERENCE_INDEX},
                "event_name": {"type": "string", "minLength": 1, "maxLength": MAX_EVENT_NAME_LENGTH},
            },
        },
        "clarification": {"type": "string", "enum": sorted(CLARIFICATION_VALUES)},
        "data_gap": {"type": "string", "enum": sorted(DATA_GAP_VALUES)},
        "audience_mode": {"type": "string", "enum": sorted(AUDIENCE_MODE_VALUES)},
    },
}


def build_sparse_frame_system_prompt() -> str:
    concepts = ", ".join(sorted(EXPERIENCE_CONCEPTS))
    unset_values = ", ".join(sorted(UNSET_VALUES))
    return f"""あなたは文化祭イベント案内の意味正規化器です。回答文やイベント事実は作りません。
JSONオブジェクト1個だけを返してください。必須キーは intent だけです。意味がない任意キーは出力しません。

intent: search/count/detail/next/similar/pair/explain_search/explain_result/faq/unsupported/clarify
前回結果をさらに絞る意味なら scope=previous。新規探索はscopeを省略します。
条件を外す意味は unset を使います。反対条件を設定してはいけません。
unset values: {unset_values}
Experienceの必須/希望/除外だけ require/prefer/exclude を使います。values: {concepts}
特定のExperienceだけ解除する場合は unset の experience:<concept> を使い、全部解除だけ experience_all を使います。
日付、市町、地域、料金、予約、屋内外、時間、ジャンルは出力しません。後段Pythonが明示表現から確定します。
ordinal/event_name/selected/last_result の参照が必要なときだけ reference を出します。
曖昧で確認が必要なら intent=clarify と clarification を出します。
DBに根拠がない混雑・騒音・医学的安全性・駐車距離・トイレ近接・天候保証・社会的適性は data_gap を出します。
同行者の年齢説明と検索対象年齢を混同しないでください。家族・世代混合の同行者説明は audience_mode=family、明示的な大人向けはadult、明示的な対象年齢はtargetです。
通常の言い換え、口語、方言は語句一致ではなく意味で分類してください。
"""


def build_sparse_frame_payload(query: str, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    if isinstance(state, Mapping):
        result_ids = state.get("last_result_ids")
        if isinstance(result_ids, (list, tuple)):
            safe["last_result_ids"] = [str(item)[:64] for item in list(result_ids)[:32]]
        if state.get("selected_event_id") not in (None, ""):
            safe["selected_event_id"] = str(state.get("selected_event_id"))[:64]
        safe["has_previous_results"] = bool(safe.get("last_result_ids"))
        safe["has_previous_command"] = isinstance(state.get("last_command"), Mapping)
    return {"query": str(query).strip()[:1200], "state": safe}


__all__ = [
    "AUDIENCE_MODE_VALUES", "CLARIFICATION_VALUES", "DATA_GAP_VALUES",
    "EXPERIENCE_CONCEPTS", "INTENT_VALUES", "REFERENCE_KIND_VALUES",
    "SPARSE_FRAME_JSON_SCHEMA", "SCOPE_VALUES", "SparseFrameError",
    "SparseReference", "SparseSemanticFrame", "UNSET_VALUES",
    "build_sparse_frame_payload", "build_sparse_frame_system_prompt",
]

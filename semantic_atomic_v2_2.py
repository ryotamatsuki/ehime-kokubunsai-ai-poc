"""Atomic semantic-classification contract for Semantic Operations v2.2.

v2.2 deliberately removes the open-ended ``set`` patch emitted by the model.
Python owns ordinary explicit filters. The model only emits a fixed vector of
small categorical decisions for residual semantics that deterministic parsing
cannot safely decide. Every experience concept has exactly one action, so
require/prefer/exclude overlap is structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Mapping

import experience_preferences
from command_models import CANONICAL_MUNICIPALITIES, CANONICAL_REGIONS


ATOMIC_INTENT_VALUES = frozenset({"search", "count", "pair", "faq", "unsupported", "clarify"})
SCOPE_VALUES = frozenset({"new", "previous"})
EXPERIENCE_ACTION_VALUES = frozenset({"none", "require", "prefer", "exclude", "unset"})
EXPERIENCE_CONCEPTS = tuple(sorted(experience_preferences.EXPERIENCE_CONCEPT_IDS))
MUNICIPALITY_VALUES = frozenset({"none", "release", *CANONICAL_MUNICIPALITIES})
REGION_VALUES = frozenset({"none", "release", *CANONICAL_REGIONS})
FEE_VALUES = frozenset({"none", "free", "paid", "release"})
RESERVATION_VALUES = frozenset({"none", "required", "not_required", "release"})
VENUE_VALUES = frozenset({"none", "indoor", "outdoor", "release"})
RAIN_VALUES = frozenset({"none", "prefer", "release"})
AUDIENCE_MODE_VALUES = frozenset({"none", "family", "adult", "target", "release"})
CLARIFICATION_VALUES = frozenset({
    "none", "ambiguous_suitability", "ambiguous_request", "missing_reference", "missing_date", "other",
})
DATA_GAP_VALUES = frozenset({
    "none", "crowding", "noise", "wheelchair_access", "medical_safety",
    "parking_distance", "toilet_proximity", "weather_guarantee", "social_fit",
    "popularity", "fame", "duration_fit", "localness", "other",
})


class AtomicFrameError(ValueError):
    pass


def _fail(message: str) -> None:
    raise AtomicFrameError(message)


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate field: {key}")
        result[key] = value
    return result


def neutral_experience() -> dict[str, str]:
    return {concept: "none" for concept in EXPERIENCE_CONCEPTS}


def _experience(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail("experience must be an object")
    if set(value) != set(EXPERIENCE_CONCEPTS):
        _fail("experience must contain exactly one action for every canonical concept")
    return {
        concept: _enum(value.get(concept), EXPERIENCE_ACTION_VALUES, f"experience.{concept}")
        for concept in EXPERIENCE_CONCEPTS
    }


@dataclass(frozen=True)
class AtomicSemanticFrame:
    intent: str
    scope: str
    municipality: str
    region: str
    fee: str
    reservation: str
    venue: str
    rain: str
    audience_mode: str
    clarification: str
    data_gap: str
    experience: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _enum(self.intent, ATOMIC_INTENT_VALUES, "intent"))
        object.__setattr__(self, "scope", _enum(self.scope, SCOPE_VALUES, "scope"))
        object.__setattr__(self, "municipality", _enum(self.municipality, MUNICIPALITY_VALUES, "municipality"))
        object.__setattr__(self, "region", _enum(self.region, REGION_VALUES, "region"))
        object.__setattr__(self, "fee", _enum(self.fee, FEE_VALUES, "fee"))
        object.__setattr__(self, "reservation", _enum(self.reservation, RESERVATION_VALUES, "reservation"))
        object.__setattr__(self, "venue", _enum(self.venue, VENUE_VALUES, "venue"))
        object.__setattr__(self, "rain", _enum(self.rain, RAIN_VALUES, "rain"))
        object.__setattr__(self, "audience_mode", _enum(self.audience_mode, AUDIENCE_MODE_VALUES, "audience_mode"))
        object.__setattr__(self, "clarification", _enum(self.clarification, CLARIFICATION_VALUES, "clarification"))
        object.__setattr__(self, "data_gap", _enum(self.data_gap, DATA_GAP_VALUES, "data_gap"))
        object.__setattr__(self, "experience", _experience(self.experience))
        if self.intent == "clarify" and self.clarification == "none":
            _fail("clarify intent requires clarification")

    @classmethod
    def from_dict(cls, raw: Any) -> "AtomicSemanticFrame":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("atomic frame must be an object")
        required = {
            "intent", "scope", "municipality", "region", "fee", "reservation",
            "venue", "rain", "audience_mode", "clarification", "data_gap", "experience",
        }
        if set(raw) != required:
            _fail(f"atomic frame fields must be exactly {sorted(required)}")
        return cls(
            intent=raw["intent"], scope=raw["scope"], municipality=raw["municipality"],
            region=raw["region"], fee=raw["fee"], reservation=raw["reservation"],
            venue=raw["venue"], rain=raw["rain"], audience_mode=raw["audience_mode"],
            clarification=raw["clarification"], data_gap=raw["data_gap"],
            experience=raw["experience"],
        )

    @classmethod
    def from_json(cls, raw: str) -> "AtomicSemanticFrame":
        if not isinstance(raw, str) or not raw.strip():
            _fail("atomic frame output is empty")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            _fail(f"invalid atomic frame JSON: {exc.msg}")
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "scope": self.scope,
            "municipality": self.municipality,
            "region": self.region,
            "fee": self.fee,
            "reservation": self.reservation,
            "venue": self.venue,
            "rain": self.rain,
            "audience_mode": self.audience_mode,
            "clarification": self.clarification,
            "data_gap": self.data_gap,
            "experience": dict(self.experience),
        }


# Finite orthographic aliases of the canonical location vocabulary, not
# per-utterance rules. They let the deterministic parser handle kana-only
# place names before the model is consulted.
_KANA_MUNICIPALITY_ALIASES: dict[str, str] = {
    "いまばり": "今治", "にいはま": "新居浜", "さいじょう": "西条",
    "しこくちゅうおう": "四国中央", "かみじま": "上島", "まつやま": "松山",
    "いよ": "伊予", "とうおん": "東温", "くまこうげん": "久万高原",
    "まさき": "松前", "とべ": "砥部", "うわじま": "宇和島",
    "やわたはま": "八幡浜", "おおず": "大洲", "せいよ": "西予",
    "うちこ": "内子", "いかた": "伊方", "まつの": "松野",
    "きほく": "鬼北", "あいなん": "愛南",
}
_KANA_REGION_ALIASES: dict[str, str] = {"とうよ": "東予", "ちゅうよ": "中予", "なんよ": "南予"}
_LOCATION_SUFFIX = r"(?=(?:市|町|で|の|に|から|だけ|周辺|近く|、|。|,|\s|$))"


def normalize_query_for_grounding(value: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()
    if not text:
        return text
    for alias, canonical in sorted(_KANA_MUNICIPALITY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(alias) + _LOCATION_SUFFIX, canonical, text)
    for alias, canonical in sorted(_KANA_REGION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(alias) + _LOCATION_SUFFIX, canonical, text)
    # Small compositional orthographic normalizations for stable control
    # vocabulary, rather than complete utterance templates.
    text = text.replace("よやく", "予約").replace("もうしこみ", "申し込み").replace("もうしこみ", "申し込み")
    text = text.replace("いらん", "いらない")
    return text


ATOMIC_FRAME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent", "scope", "municipality", "region", "fee", "reservation",
        "venue", "rain", "audience_mode", "clarification", "data_gap", "experience",
    ],
    "properties": {
        "intent": {"type": "string", "enum": sorted(ATOMIC_INTENT_VALUES)},
        "scope": {"type": "string", "enum": sorted(SCOPE_VALUES)},
        "municipality": {"type": "string", "enum": sorted(MUNICIPALITY_VALUES)},
        "region": {"type": "string", "enum": sorted(REGION_VALUES)},
        "fee": {"type": "string", "enum": sorted(FEE_VALUES)},
        "reservation": {"type": "string", "enum": sorted(RESERVATION_VALUES)},
        "venue": {"type": "string", "enum": sorted(VENUE_VALUES)},
        "rain": {"type": "string", "enum": sorted(RAIN_VALUES)},
        "audience_mode": {"type": "string", "enum": sorted(AUDIENCE_MODE_VALUES)},
        "clarification": {"type": "string", "enum": sorted(CLARIFICATION_VALUES)},
        "data_gap": {"type": "string", "enum": sorted(DATA_GAP_VALUES)},
        "experience": {
            "type": "object",
            "additionalProperties": False,
            "required": list(EXPERIENCE_CONCEPTS),
            "properties": {
                concept: {"type": "string", "enum": sorted(EXPERIENCE_ACTION_VALUES)}
                for concept in EXPERIENCE_CONCEPTS
            },
        },
    },
}


def build_atomic_frame_system_prompt() -> str:
    concepts = ", ".join(EXPERIENCE_CONCEPTS)
    return f"""あなたは文化祭イベント案内の原子的な意味分類器です。回答文やイベント事実は作りません。
固定JSONオブジェクト1個だけを返します。キーを増減しません。

Pythonが日時・通常表記の市町/地域・ジャンル/テーマ・年齢・料金・予約・屋内外・雨・時間・既知Experienceを先に抽出します。
groundedに既にある条件はモデル側で繰り返さず、対応するatomはnoneにしてください。
モデルが補うのは、Pythonが落とした明示的な口語/かな表記の municipality / region / fee / reservation / venue / rain と、残余の意味分類だけです。
利用者が言っていない条件を推測してはいけません。

intentは search/count/pair/faq/unsupported/clarify だけです。イベント詳細・前回結果参照・似た候補・説明はPython側で先に処理され、ここでは分類しません。
scopeは新規ならnew、前回結果をさらに絞る意味だけpreviousです。
municipality/regionは明示された場所をPythonが取りこぼした時だけ正規形、場所条件を外す時だけrelease、それ以外noneです。
feeは none/free/paid/release、reservationは none/required/not_required/release、venueは none/indoor/outdoor/release、rainは none/prefer/release です。
audience_modeは none/family/adult/target/release。同行者の年齢を検索対象年齢へ変換しません。
clarificationは確認が必要な時だけ理由を設定し、それ以外none。data_gapはDBに根拠がない属性の時だけ設定し、それ以外noneです。
Experienceは各conceptにつき exactly one: none/require/prefer/exclude/unset。concepts: {concepts}
同じconceptを複数の意味にすることはできません。
通常の言い換え・口語・方言は語句一致ではなく意味で分類してください。
"""


def build_atomic_frame_payload(
    query: str,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_state: dict[str, Any] = {}
    if isinstance(state, Mapping):
        ids = state.get("last_result_ids")
        if isinstance(ids, (list, tuple)):
            safe_state["last_result_ids"] = [str(item)[:64] for item in list(ids)[:32]]
        if state.get("selected_event_id") not in (None, ""):
            safe_state["selected_event_id"] = str(state.get("selected_event_id"))[:64]
        safe_state["has_previous_results"] = bool(safe_state.get("last_result_ids"))
        safe_state["has_previous_command"] = isinstance(state.get("last_command"), Mapping)
    safe_grounded = {
        str(key): value
        for key, value in dict(grounded or {}).items()
        if value not in (None, "", [], (), {})
    }
    return {"query": str(query).strip()[:1200], "state": safe_state, "grounded": safe_grounded}


__all__ = [
    "ATOMIC_FRAME_JSON_SCHEMA", "ATOMIC_INTENT_VALUES", "AtomicFrameError",
    "AtomicSemanticFrame", "DATA_GAP_VALUES", "EXPERIENCE_ACTION_VALUES",
    "EXPERIENCE_CONCEPTS", "build_atomic_frame_payload", "build_atomic_frame_system_prompt",
    "neutral_experience", "normalize_query_for_grounding",
]

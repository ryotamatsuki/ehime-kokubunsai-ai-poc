"""Closed atomic interpretation schema for Evidence-Bounded Semantics v2.3.

The LLM does not emit final clarification/data-gap control.  It classifies the
requested evidence domain, a bounded semantic-resolution signal, and bounded
supported atoms.  Python derives supportability, clarification/data-gap and
final flow from those signals plus independent grounding and trusted state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from semantic_atomic_v2_2 import (
    AUDIENCE_MODE_VALUES,
    EXPERIENCE_ACTION_VALUES,
    EXPERIENCE_CONCEPTS,
    FEE_VALUES,
    MUNICIPALITY_VALUES,
    RAIN_VALUES,
    REGION_VALUES,
    RESERVATION_VALUES,
    SCOPE_VALUES,
    VENUE_VALUES,
    neutral_experience,
    normalize_query_for_grounding,
)
from semantic_evidence_v2_3 import (
    EVIDENCE_REQUEST_VALUES,
    SEMANTIC_RESOLUTION_VALUES,
    EvidenceRequest,
    SemanticResolution,
    evidence_request,
    semantic_resolution,
)


ATOMIC_INTENT_VALUES_V23 = frozenset({"search", "count", "pair", "faq"})


class AtomicFrameV23Error(ValueError):
    pass


def _fail(message: str) -> None:
    raise AtomicFrameV23Error(message)


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


def _experience(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(EXPERIENCE_CONCEPTS):
        _fail("experience must contain exactly one action for every canonical concept")
    return {
        concept: _enum(value.get(concept), EXPERIENCE_ACTION_VALUES, f"experience.{concept}")
        for concept in EXPERIENCE_CONCEPTS
    }


@dataclass(frozen=True)
class AtomicSemanticFrameV23:
    intent: str
    scope: str
    evidence_request: str
    semantic_resolution: str
    municipality: str
    region: str
    fee: str
    reservation: str
    venue: str
    rain: str
    audience_mode: str
    experience: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _enum(self.intent, ATOMIC_INTENT_VALUES_V23, "intent"))
        object.__setattr__(self, "scope", _enum(self.scope, SCOPE_VALUES, "scope"))
        object.__setattr__(self, "evidence_request", evidence_request(self.evidence_request).value)
        object.__setattr__(self, "semantic_resolution", semantic_resolution(self.semantic_resolution).value)
        object.__setattr__(self, "municipality", _enum(self.municipality, MUNICIPALITY_VALUES, "municipality"))
        object.__setattr__(self, "region", _enum(self.region, REGION_VALUES, "region"))
        object.__setattr__(self, "fee", _enum(self.fee, FEE_VALUES, "fee"))
        object.__setattr__(self, "reservation", _enum(self.reservation, RESERVATION_VALUES, "reservation"))
        object.__setattr__(self, "venue", _enum(self.venue, VENUE_VALUES, "venue"))
        object.__setattr__(self, "rain", _enum(self.rain, RAIN_VALUES, "rain"))
        object.__setattr__(self, "audience_mode", _enum(self.audience_mode, AUDIENCE_MODE_VALUES, "audience_mode"))
        object.__setattr__(self, "experience", _experience(self.experience))

    @classmethod
    def from_dict(cls, raw: Any) -> "AtomicSemanticFrameV23":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            _fail("atomic v2.3 frame must be an object")
        required = {
            "intent", "scope", "evidence_request", "semantic_resolution",
            "municipality", "region", "fee", "reservation", "venue", "rain",
            "audience_mode", "experience",
        }
        if set(raw) != required:
            _fail(f"atomic v2.3 frame fields must be exactly {sorted(required)}")
        return cls(
            intent=raw["intent"], scope=raw["scope"], evidence_request=raw["evidence_request"],
            semantic_resolution=raw["semantic_resolution"], municipality=raw["municipality"],
            region=raw["region"], fee=raw["fee"], reservation=raw["reservation"],
            venue=raw["venue"], rain=raw["rain"], audience_mode=raw["audience_mode"],
            experience=raw["experience"],
        )

    @classmethod
    def from_json(cls, raw: str) -> "AtomicSemanticFrameV23":
        if not isinstance(raw, str) or not raw.strip():
            _fail("atomic v2.3 frame output is empty")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            _fail(f"invalid atomic v2.3 frame JSON: {exc.msg}")
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "scope": self.scope,
            "evidence_request": self.evidence_request,
            "semantic_resolution": self.semantic_resolution,
            "municipality": self.municipality,
            "region": self.region,
            "fee": self.fee,
            "reservation": self.reservation,
            "venue": self.venue,
            "rain": self.rain,
            "audience_mode": self.audience_mode,
            "experience": dict(self.experience),
        }


ATOMIC_FRAME_JSON_SCHEMA_V23: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent", "scope", "evidence_request", "semantic_resolution",
        "municipality", "region", "fee", "reservation", "venue", "rain",
        "audience_mode", "experience",
    ],
    "properties": {
        "intent": {"type": "string", "enum": sorted(ATOMIC_INTENT_VALUES_V23)},
        "scope": {"type": "string", "enum": sorted(SCOPE_VALUES)},
        "evidence_request": {"type": "string", "enum": sorted(EVIDENCE_REQUEST_VALUES)},
        "semantic_resolution": {"type": "string", "enum": sorted(SEMANTIC_RESOLUTION_VALUES)},
        "municipality": {"type": "string", "enum": sorted(MUNICIPALITY_VALUES)},
        "region": {"type": "string", "enum": sorted(REGION_VALUES)},
        "fee": {"type": "string", "enum": sorted(FEE_VALUES)},
        "reservation": {"type": "string", "enum": sorted(RESERVATION_VALUES)},
        "venue": {"type": "string", "enum": sorted(VENUE_VALUES)},
        "rain": {"type": "string", "enum": sorted(RAIN_VALUES)},
        "audience_mode": {"type": "string", "enum": sorted(AUDIENCE_MODE_VALUES)},
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


def neutral_frame_v23(
    *,
    evidence: EvidenceRequest = EvidenceRequest.NONE,
    resolution: SemanticResolution = SemanticResolution.RESOLVED,
    scope: str = "new",
) -> AtomicSemanticFrameV23:
    return AtomicSemanticFrameV23(
        intent="search",
        scope=scope,
        evidence_request=evidence.value,
        semantic_resolution=resolution.value,
        municipality="none",
        region="none",
        fee="none",
        reservation="none",
        venue="none",
        rain="none",
        audience_mode="none",
        experience=neutral_experience(),
    )


def build_atomic_frame_system_prompt_v23() -> str:
    concepts = ", ".join(EXPERIENCE_CONCEPTS)
    evidence = ", ".join(sorted(EVIDENCE_REQUEST_VALUES))
    resolutions = ", ".join(sorted(SEMANTIC_RESOLUTION_VALUES))
    return f"""あなたは文化祭イベント案内のEvidence-Bounded原子的意味分類器です。回答文やイベント事実は作りません。
固定JSONオブジェクト1個だけを返し、キーを増減しません。

Pythonが日時・通常表記の市町/地域・ジャンル/テーマ・年齢・料金・予約・屋内外・雨・時間・既知Experienceを先に抽出します。
groundedにある条件は繰り返さず対応atomをnoneにしてください。trusted groundingが常に優先です。

最重要: ユーザーが求める意味領域と、DBがその要求に答えられるかを分離してください。
evidence_requestは次のclosed vocabularyのみです: {evidence}
relational_suitabilityは person/group と event の適性判断です。人物属性・性格・経験量・知識量などから、seated/low_mobility/watch_listen等へ勝手に変換してはいけません。
subjective_judgment/absolute_guarantee/realtime_state/external_logistics/unsupported_fact/unknown_capabilityも、別のsupported atomへ置き換えてはいけません。
ただし同じ発話に独立して明示されたsupported条件は保持します。grounded済みならモデル側はnoneです。

semantic_resolutionは {resolutions} のみです。resolvedは要求が意味的に解決済み、underspecifiedは必要情報不足、conditionalは現状schemaで表現できない条件分岐、ambiguousは複数解釈が残る場合です。これは最終clarification命令ではありません。PythonがCapability Registry・grounding・stateと組み合わせて最終flow/statusを決めます。

intentは search/count/pair/faq のbounded classificationだけです。clarification/data-gap/最終flow/statusは出力しません。
scopeは新規ならnew、前回の検索条件を継続して変更する意味だけpreviousです。
municipality/region/fee/reservation/venue/rain/audience_modeは明示された意味だけ。人物属性からaudienceやExperienceを推測しません。
releaseは条件解除であり、反対条件のrequireではありません。
Experienceは各concept exactly one: none/require/prefer/exclude/unset。concepts: {concepts}
機能的な要望が直接述べられている場合だけ対応Experienceへ分類し、人口統計・性格・診断名をproxyにしないでください。
"""


__all__ = [
    "ATOMIC_FRAME_JSON_SCHEMA_V23",
    "ATOMIC_INTENT_VALUES_V23",
    "AtomicFrameV23Error",
    "AtomicSemanticFrameV23",
    "build_atomic_frame_system_prompt_v23",
    "neutral_frame_v23",
    "neutral_experience",
    "normalize_query_for_grounding",
]

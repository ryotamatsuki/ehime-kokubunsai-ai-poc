"""Bounded product-capability and security policy for Semantic Operations v2.1.

This is not an utterance dictionary.  The rules encode stable product
boundaries: the PoC can search/describe/recommend its cultural-event catalog,
but it cannot perform travel/dining commerce, fabricate catalog items, or
exfiltrate internal instructions.  Ordinary unknown event themes are not
rejected here and remain available to the semantic normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class CapabilityDecision:
    allowed: bool
    reason: str | None = None
    message: str = ""


# Stable external entity classes, not individual test utterances.  These are
# deliberately narrow so an unknown cultural theme is not mistaken for an
# out-of-scope request merely because the vocabulary is unseen.
_EXTERNAL_SERVICE_ENTITY = re.compile(
    r"(?:ホテル|旅館|宿泊(?:先|施設)?|民宿|ゲストハウス|"
    r"居酒屋|レストラン|飲食店|食事処|カフェ|喫茶店|バー|"
    r"航空券|飛行機(?:の)?チケット|新幹線(?:の)?切符|レンタカー)"
)
_EXTERNAL_SERVICE_ACTION = re.compile(
    r"(?:予約(?:して|する|取って|を取)|手配(?:して|する)|申し込(?:んで|む)|"
    r"おすすめ(?:を)?(?:教えて|出して|探して)|探して|紹介して|教えて)"
)

# Security invariants are phrased around protected resources/behaviour rather
# than exact sentences.  The catalog must remain closed-world and internal
# prompts/configuration are never an answer surface.
_INTERNAL_RESOURCE = re.compile(
    r"(?:system\s*prompt|システム\s*プロンプト|内部(?:の)?(?:指示|命令|設定|ロジック)|"
    r"検索ロジック|開発者(?:の)?指示|隠し(?:た)?指示)",
    re.IGNORECASE,
)
_INTERNAL_EXFIL_ACTION = re.compile(r"(?:見せて|表示して|教えて|全部出して|公開して|開示して)")
_FABRICATION_TARGET = re.compile(
    r"(?:(?:events?\.json|イベントDB|データベース|掲載(?:済み)?|候補|登録)[^。！？]{0,30}"
    r"(?:ない|外|未登録)[^。！？]{0,15}イベント|"
    r"(?:架空|存在しない|でっち上げ|捏造)[^。！？]{0,12}イベント|"
    r"イベント[^。！？]{0,12}(?:作って|作れ|捏造して|でっち上げて))",
    re.IGNORECASE,
)
_RULE_BYPASS = re.compile(
    r"(?:(?:ルール|制約|指示|命令|プロンプト)[^。！？]{0,15}(?:無視|破って|従わず|解除))"
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def evaluate_capability(query: str) -> CapabilityDecision:
    text = _normalize(query)
    if not text:
        return CapabilityDecision(True)

    if _INTERNAL_RESOURCE.search(text) and _INTERNAL_EXFIL_ACTION.search(text):
        return CapabilityDecision(
            False,
            "internal_exfiltration",
            "内部の指示や検索ロジックは開示できません。文化祭イベントの案内なら手伝えます。",
        )
    if _FABRICATION_TARGET.search(text) or _RULE_BYPASS.search(text):
        return CapabilityDecision(
            False,
            "closed_world_security",
            "掲載されていないイベントを作ったり、案内の制約を無効にしたりはできません。掲載済みのイベントから探してみて。",
        )
    if _EXTERNAL_SERVICE_ENTITY.search(text) and _EXTERNAL_SERVICE_ACTION.search(text):
        return CapabilityDecision(
            False,
            "external_service",
            "このPoCは文化祭イベントの検索・参加案内が中心で、宿泊・飲食・交通の予約や店舗推薦は扱いません。",
        )
    return CapabilityDecision(True)


__all__ = ["CapabilityDecision", "evaluate_capability"]

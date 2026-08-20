"""Small local matcher for the eight general PoC FAQs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


FAQ_PATH = Path(__file__).resolve().parent / "data" / "general_faq.json"


@dataclass(frozen=True)
class FAQMatch:
    faq_id: str
    answer: str
    score: float


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


@lru_cache(maxsize=2)
def load_faq(path: str | Path = FAQ_PATH) -> tuple[dict[str, Any], ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 8:
        raise ValueError("general_faq.json は8件の配列である必要があります。")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict) or not {"id", "カテゴリ", "質問例", "回答", "データ区分"}.issubset(item):
            raise ValueError("general_faq.json の項目が不正です。")
        if item["id"] in ids or item["データ区分"] != "PoC説明":
            raise ValueError("general_faq.json のIDまたはデータ区分が不正です。")
        if not isinstance(item["質問例"], list) or not item["質問例"]:
            raise ValueError("general_faq.json の質問例が不正です。")
        ids.add(item["id"])
        result.append(item)
    return tuple(result)


_FAQ_HINTS: dict[str, tuple[str, ...]] = {
    "faq-001": ("公式", "本物", "実際のイベント", "実在"),
    "faq-002": ("今日って", "今日の日付", "今日いつ", "poc上の今日"),
    "faq-003": ("近くで探したい", "近場で何か", "現在地"),
    "faq-004": ("予約って必要", "申込は必要", "申し込みは必要"),
    "faq-005": ("雨でも開催", "雨でもやる", "雨の日はどう", "雨天時対応"),
    "faq-006": ("車いすで", "多目的トイレ", "バリアフリー"),
    "faq-007": ("体験も無料", "無料？", "完全無料"),
    "faq-008": ("このあと何か", "似たイベント", "似たイベントある", "同じ系統", "同じ地域で他"),
}


def find_faq(query: str, threshold: float = 0.68) -> FAQMatch | None:
    normalized = _compact(query)
    if not normalized:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for item in load_faq():
        candidates = [str(item["カテゴリ"]), *map(str, item["質問例"]), *_FAQ_HINTS.get(item["id"], ())]
        score = 0.0
        for candidate in candidates:
            compact = _compact(candidate)
            if compact and (compact in normalized or normalized in compact):
                score = max(score, 1.0 if compact == normalized else 0.90)
            elif len(normalized) >= 4 and len(compact) >= 4:
                score = max(score, SequenceMatcher(None, normalized, compact).ratio())
        if best is None or score > best[0]:
            best = (score, item)
    if best is None or best[0] < threshold:
        return None
    return FAQMatch(str(best[1]["id"]), str(best[1]["回答"]), best[0])

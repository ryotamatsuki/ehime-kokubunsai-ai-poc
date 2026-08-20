"""Small, deterministic natural-language search over the PoC event JSON.

The dataset is intentionally tiny.  This module is deliberately not an RAG
system: every filter is applied in Python before any candidate is sent to the
language model.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Iterable

from app_config import (
    EVENT_DATA_PATH,
    MAX_EVENT_CANDIDATES,
    MUNICIPALITY_ALIASES,
    POC_REFERENCE_DATE,
    REGION_ALIASES,
    REGION_MUNICIPALITIES,
)


_DATE_RANGE_RE = re.compile(
    r"(?P<start>\d{4}-\d{2}-\d{2})(?:〜(?P<end>\d{4}-\d{2}-\d{2}))?"
)
_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})日?")
_SLASH_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[\-/](\d{1,2})(?!\d)")
_PRICE_RE = re.compile(r"(?<!\d)([0-9０-９][0-9０-９,，]*)\s*円")

_CHILD_TERMS = (
    "子ども",
    "子供",
    "こども",
    "小学生",
    "小学",
    "親子",
    "子ども向け",
)
_INDOOR_TERMS = ("屋内", "室内", "雨", "雨でも")
_OUTDOOR_TERMS = ("屋外", "外で")
_FREE_TERMS = ("無料", "タダ", "0円", "0 円")
_PAID_TERMS = ("有料", "料金がかかる", "お金がかかる")

# These are intentionally narrow.  They turn common cultural terms into a
# deterministic filter without attempting open-ended semantic retrieval.
_GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "伝統芸能": ("伝統芸能", "民俗芸能"),
    "伝統文化": ("伝統", "民俗", "祭り"),
    "歴史": ("歴史", "文化財", "産業遺産"),
    "俳句": ("俳句",),
    "陶芸": ("陶芸", "砥部焼"),
    "砥部焼": ("砥部焼",),
    "舞台": ("舞台", "演劇", "芝居"),
    "工芸": ("工芸", "紙", "水引", "陶芸"),
}

_FILLER_RE = re.compile(
    r"(イベント|を探して|探して|教えて|知りたい|おすすめ|行きたい|行ける|"
    r"楽しめる|楽しみたい|興味がある|興味|やっている|やって|ありますか|あります|ください|"
    r"してみん\??|してみたい|もの|こと|なら|で|の|に|を|は|が|と|や|から|まで)"
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def event_date_range(event: dict[str, Any]) -> tuple[date, date]:
    """Extract an inclusive date range from the JSON's Japanese date field."""

    match = _DATE_RANGE_RE.search(_normalize(str(event["日時"])))
    if not match:
        raise ValueError(f"日時を解釈できません: {event.get('日時')}")
    start = _parse_date(match.group("start"))
    end = _parse_date(match.group("end") or match.group("start"))
    return start, end


def event_occurs_on(event: dict[str, Any], target: date) -> bool:
    start, end = event_date_range(event)
    return start <= target <= end


@lru_cache(maxsize=1)
def load_events() -> tuple[dict[str, Any], ...]:
    with EVENT_DATA_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("PoCイベントデータは30件の配列である必要があります")
    return tuple(data)


def _query_dates(query: str, reference_date: date) -> set[date]:
    dates: set[date] = set()
    normalized = _normalize(query)

    if "今日" in normalized:
        dates.add(reference_date)
    if "明日" in normalized:
        dates.add(reference_date + timedelta(days=1))
    if "今週末" in normalized or "週末" in normalized:
        # Saturday/Sunday of the next/current weekend.  The fixed PoC date is
        # a Friday, so this maps to 11/4 and 11/5 as users expect.
        saturday = reference_date + timedelta(days=(5 - reference_date.weekday()) % 7)
        dates.update({saturday, saturday + timedelta(days=1)})

    for month, day in _MONTH_DAY_RE.findall(normalized):
        dates.add(date(reference_date.year, int(month), int(day)))
    for month, day in _SLASH_DATE_RE.findall(normalized):
        dates.add(date(reference_date.year, int(month), int(day)))
    return dates


def _query_municipalities(query: str) -> set[str]:
    normalized = _normalize(query)
    found: set[str] = set()
    for alias, municipality in sorted(MUNICIPALITY_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in normalized:
            found.add(municipality)
    return found


def _query_regions(query: str) -> set[str]:
    normalized = _normalize(query)
    return {canonical for alias, canonical in REGION_ALIASES.items() if alias in normalized}


def _query_genres(query: str) -> set[str]:
    normalized = _normalize(query)
    return {
        alias
        for alias in sorted(_GENRE_ALIASES, key=len, reverse=True)
        if alias in normalized
    }


def _is_fully_free(event: dict[str, Any]) -> bool:
    fee = _normalize(str(event["料金"])).replace(" ", "")
    return fee in {"無料", "無料（事前申込制）", "無料(事前申込制)"}


def _price_numbers(event: dict[str, Any]) -> list[int]:
    fee = _normalize(str(event["料金"]))
    numbers: list[int] = []
    for raw in _PRICE_RE.findall(fee):
        numbers.append(int(raw.replace(",", "").replace("，", "")))
    return numbers


def _has_unknown_paid_component(event: dict[str, Any]) -> bool:
    fee = _normalize(str(event["料金"]))
    if _is_fully_free(event):
        return False
    # A non-free string without a numeric amount cannot be proven to fit a
    # user's price cap (e.g. "食体験は有料").
    return not _price_numbers(event)


def _query_max_price(query: str) -> int | None:
    normalized = _normalize(query)
    patterns = (
        r"([0-9０-９][0-9０-９,，]*)\s*円\s*(?:以内|まで|以下)",
        r"(?:上限|予算)[^0-9０-９]{0,8}([0-9０-９][0-9０-９,，]*)\s*円",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1).replace(",", "").replace("，", ""))
    return None


def _search_terms(
    query: str, events: Iterable[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    normalized = _normalize(query)
    # Remove date literals and recognised filters before extracting residual
    # event-specific terms.
    normalized = _DATE_RANGE_RE.sub(" ", normalized)
    normalized = _MONTH_DAY_RE.sub(" ", normalized)
    normalized = _SLASH_DATE_RE.sub(" ", normalized)
    normalized = re.sub(r"[0-9０-９][0-9０-９,，]*\s*円(?:以内|まで|以下)?", " ", normalized)
    for term in (
        "今日",
        "明日",
        "今週末",
        "週末",
        "子ども向け",
        "子ども",
        "子供",
        "こども",
        "小学生",
        "小学3年生",
        "親子",
        "雨でもOK",
        "雨でも",
        "雨",
        "屋内",
        "室内",
        "屋外",
        "無料",
        "タダ",
        "0円",
        "有料",
        "東予",
        "中予",
        "南予",
    ):
        normalized = normalized.replace(term, " ")
    for alias in sorted(MUNICIPALITY_ALIASES, key=len, reverse=True):
        normalized = normalized.replace(alias, " ")
    for alias in sorted(_GENRE_ALIASES, key=len, reverse=True):
        normalized = normalized.replace(alias, " ")
    normalized = _FILLER_RE.sub(" ", normalized)
    raw_terms = re.findall(r"[A-Za-z0-9一-龯々ぁ-んァ-ヶー]{2,}", normalized)

    searchable_texts = [
        " ".join(str(event.get(key, "")) for key in ("イベント名", "概要", "ジャンル"))
        for event in events
    ]
    known_terms = [
        term
        for term in raw_terms
        if any(term.casefold() in text.casefold() for text in searchable_texts)
    ]
    unknown_terms = [term for term in raw_terms if term not in known_terms]
    return known_terms, unknown_terms


def _event_municipality(event: dict[str, Any]) -> str | None:
    location = str(event.get("場所", ""))
    for municipality in MUNICIPALITY_ALIASES.values():
        if municipality in location:
            return municipality
    return None


def _matches_terms(event: dict[str, Any], terms: list[str]) -> bool:
    haystack = " ".join(
        str(event.get(key, "")) for key in ("イベント名", "概要", "ジャンル", "場所")
    ).casefold()
    return all(term.casefold() in haystack for term in terms)


def _matches_genres(event: dict[str, Any], genres: set[str]) -> bool:
    if not genres:
        return True
    haystack = " ".join(
        str(event.get(key, "")) for key in ("イベント名", "概要", "ジャンル")
    )
    return all(
        any(keyword in haystack for keyword in _GENRE_ALIASES[genre])
        for genre in genres
    )


def _sort_key(event: dict[str, Any]) -> tuple[date, str]:
    return event_date_range(event)[0], str(event.get("イベント名", ""))


def search_events(
    query: str,
    *,
    reference_date: date = POC_REFERENCE_DATE,
    limit: int = MAX_EVENT_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return at most ``limit`` events satisfying every detected condition."""

    normalized = _normalize(query)
    if not normalized:
        return []
    events = load_events()
    dates = _query_dates(normalized, reference_date)
    municipalities = _query_municipalities(normalized)
    regions = _query_regions(normalized)
    genres = _query_genres(normalized)
    max_price = _query_max_price(normalized)
    child_only = any(term in normalized for term in _CHILD_TERMS)
    indoor_only = any(term in normalized for term in _INDOOR_TERMS)
    outdoor_only = any(term in normalized for term in _OUTDOOR_TERMS)
    free_only = any(term in normalized for term in _FREE_TERMS)
    paid_only = any(term in normalized for term in _PAID_TERMS)
    terms, unknown_terms = _search_terms(normalized, events)
    # If nothing remains after known filters and the residual words cannot be
    # found in any event, do not silently turn an unknown request into "show
    # the first eight events".
    if unknown_terms:
        return []

    filtered: list[dict[str, Any]] = []
    for event in events:
        municipality = _event_municipality(event)
        region = next(
            (
                region_name
                for region_name, members in REGION_MUNICIPALITIES.items()
                if municipality in members
            ),
            None,
        )
        if dates and not any(event_occurs_on(event, target) for target in dates):
            continue
        if municipalities and municipality not in municipalities:
            continue
        if regions and region not in regions:
            continue
        if not _matches_genres(event, genres):
            continue
        if child_only and not bool(event.get("子ども向け")):
            continue
        venue_type = str(event.get("屋内/屋外", ""))
        if indoor_only and "屋内" not in venue_type:
            continue
        if outdoor_only and "屋外" not in venue_type:
            continue
        if free_only and not _is_fully_free(event):
            continue
        if paid_only and _is_fully_free(event):
            continue
        if max_price is not None:
            if _has_unknown_paid_component(event):
                continue
            prices = _price_numbers(event)
            if prices and max(prices) > max_price:
                continue
        if terms and not _matches_terms(event, terms):
            continue
        filtered.append(event)

    filtered.sort(key=_sort_key)
    return filtered[: max(0, limit)]


def looks_like_event_query(query: str) -> bool:
    """Return whether the UI should route a question to the event search."""

    normalized = _normalize(query)
    hints = (
        "イベント",
        "文化祭",
        "今日",
        "明日",
        "週末",
        "無料",
        "雨",
        "屋内",
        "屋外",
        "子ども",
        "子供",
        "小学生",
        "親子",
        "行きたい",
        "行ける",
        "おすすめ",
        "探して",
        "伝統芸能",
        "伝統文化",
        "砥部焼",
    )
    return bool(
        any(hint in normalized for hint in hints)
        or _query_municipalities(normalized)
        or _query_regions(normalized)
    )


def asks_for_nearby(query: str) -> bool:
    return "近く" in _normalize(query) or "近場" in _normalize(query)

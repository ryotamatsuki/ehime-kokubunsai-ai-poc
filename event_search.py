"""Deterministic AND search for the 30 fictional cultural events.

The source JSON is the sole source of event facts.  This module performs only
local parsing and filtering; it does not call an LLM and does not use an
embedding, vector database, RAG pipeline, web search, or external API.

The public entry point for the UI is :func:`search_events`::

    result = search_events("11月3日に子どもと行けるイベント", events)
    for event in result.events:
        # Render these fields directly from the JSON in the UI.
        print(event["イベント名"], event["日時"], event["料金"])

Every active condition is an AND condition.  Aliases inside one semantic
condition (for example, ``子ども``/``小学生``) are OR alternatives.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from app_config import (
    CHILD_TERMS,
    CITY_ALIASES,
    EVENTS_PATH,
    FREE_TERMS,
    GENRE_ALIASES,
    INDOOR_TERMS,
    INJECTION_PATTERNS,
    KNOWN_KEYWORDS,
    MAX_SEARCH_RESULTS,
    NEAR_TERMS,
    OUTDOOR_TERMS,
    OUT_OF_SCOPE_PATTERNS,
    POC_REFERENCE_DATE,
    RAIN_TERMS,
    REGION_CITIES,
)


EVENT_FIELDS: tuple[str, ...] = (
    "イベント名",
    "日時",
    "場所",
    "ジャンル",
    "子ども向け",
    "屋内/屋外",
    "料金",
    "概要",
    "公式URL",
)

ALL_CITIES: tuple[str, ...] = tuple(
    city for cities in REGION_CITIES.values() for city in cities
)

_SORTED_CITY_ALIASES: tuple[str, ...] = tuple(
    sorted(CITY_ALIASES, key=len, reverse=True)
)
_SORTED_GENRE_LABELS: tuple[str, ...] = tuple(
    sorted(GENRE_ALIASES, key=len, reverse=True)
)
_SORTED_KNOWN_KEYWORDS: tuple[str, ...] = tuple(
    sorted(KNOWN_KEYWORDS, key=len, reverse=True)
)

_EVENT_DATE_RE = re.compile(
    r"(?P<start>20\d{2}-\d{2}-\d{2})"
    r"(?:\s*[〜~]\s*(?P<end>20\d{2}-\d{2}-\d{2}))?"
)
_EXPLICIT_DATE_RE = re.compile(
    r"(?:(?P<year>20\d{2})年)?"
    r"(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
)
_SLASH_DATE_RE = re.compile(
    r"(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?!\d)"
)
_FEE_LIMIT_RE = re.compile(
    r"(?P<amount>\d[\d,]*)\s*円\s*(?:以内|以下|まで)"
)
_YEN_AMOUNT_RE = re.compile(r"\d[\d,]*\s*円")

_GENERIC_STOPWORDS = frozenset(
    {
        "イベント",
        "文化祭",
        "もの",
        "何か",
        "楽しめる",
        "楽しむ",
        "楽しみたい",
        "行ける",
        "行きたい",
        "やっている",
        "探す",
        "探して",
        "探し",
        "あります",
        "ある",
        "ください",
        "教えて",
        "お願い",
        "興味",
        "参加",
        "できる",
        "できそう",
        "おすすめ",
        "だけ",
        "条件",
        "地域",
        "市町",
        "料金",
        "上限",
        "以内",
        "以下",
        "まで",
        "今日",
        "本日",
        "明日",
        "今週末",
        "週末",
        "雨でも",
        "雨の日",
        "雨天",
        "無料",
        "タダ",
        "ただ",
        "屋内",
        "室内",
        "屋外",
        "子ども",
        "子供",
        "こども",
        "小学生",
        "小学",
        "親子",
        "家族",
        "ファミリー",
        "近く",
        "近い",
        "周辺",
        "伝統文化",
        "伝統芸能",
    }
)


@dataclass
class SearchFilters:
    """Structured interpretation of a natural-language search.

    ``dates`` contains target calendar days in ISO format.  A period event
    matches when at least one target day lies between its start and end,
    inclusive.  ``genres`` contains canonical labels from
    :data:`app_config.GENRE_ALIASES`.
    """

    dates: list[str] = field(default_factory=list)
    city: str | None = None
    region: str | None = None
    genres: list[str] = field(default_factory=list)
    child_friendly: bool | None = None
    venue: str | None = None
    rain_preferred: bool = False
    entry_free: bool | None = None
    max_entry_fee: int | None = None
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/session-state-friendly primitive values."""

        return asdict(self)


@dataclass
class SearchResult:
    """Result returned to the UI without any generated event facts."""

    events: list[dict[str, Any]]
    filters: SearchFilters
    intent: str = "search"
    message: str | None = None
    total_matches: int = 0


def normalize_query(value: str) -> str:
    """Normalize Japanese/full-width spelling without changing meaning."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("～", "〜")
    return re.sub(r"\s+", " ", normalized).strip()


def load_events(path: str | Path = EVENTS_PATH) -> list[dict[str, Any]]:
    """Load and validate the 30-record fictional-event source file.

    Validation is intentionally strict at the data boundary.  Search itself
    never invents a missing field or repairs malformed event facts.
    """

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("events.json は30件の配列である必要があります。")

    expected_fields = set(EVENT_FIELDS)
    names: set[str] = set()
    urls: set[str] = set()
    for index, event in enumerate(data, start=1):
        if not isinstance(event, dict) or set(event) != expected_fields:
            raise ValueError(f"events.json の {index} 件目の項目が不正です。")

        if not isinstance(event["子ども向け"], bool):
            raise ValueError(f"events.json の {index} 件目の子ども向け値が不正です。")
        for field_name in EVENT_FIELDS:
            if field_name == "子ども向け":
                continue
            if not isinstance(event[field_name], str) or not event[field_name].strip():
                raise ValueError(
                    f"events.json の {index} 件目の{field_name}が空または型不正です。"
                )
        if event["屋内/屋外"] not in {"屋内", "屋外", "屋内・屋外"}:
            raise ValueError(f"events.json の {index} 件目の屋内外区分が不正です。")
        if not event["イベント名"].startswith("【PoC架空】"):
            raise ValueError(f"events.json の {index} 件目に架空表示がありません。")
        if not event["公式URL"].startswith("https://example.invalid/"):
            raise ValueError(f"events.json の {index} 件目のURLがPoC用ではありません。")
        if event["イベント名"] in names or event["公式URL"] in urls:
            raise ValueError(f"events.json の {index} 件目に重複があります。")

        start, end = parse_event_dates(event["日時"])
        if end < start:
            raise ValueError(f"events.json の {index} 件目の日付範囲が不正です。")
        names.add(event["イベント名"])
        urls.add(event["公式URL"])

    return data


def parse_event_dates(value: str) -> tuple[date, date]:
    """Parse a single date or an inclusive ``start〜end`` event period."""

    normalized = normalize_query(value)
    match = _EVENT_DATE_RE.search(normalized)
    if not match:
        raise ValueError(f"日時を解釈できません: {value}")
    start = date.fromisoformat(match.group("start"))
    end = date.fromisoformat(match.group("end") or match.group("start"))
    return start, end


def event_city(event: Mapping[str, Any]) -> str | None:
    """Extract the canonical municipality from the place field."""

    place = str(event["場所"])
    return next(
        (city for city in sorted(ALL_CITIES, key=len, reverse=True) if city in place),
        None,
    )


def event_region(event: Mapping[str, Any]) -> str | None:
    """Map an event's municipality to 東予・中予・南予."""

    city = event_city(event)
    if city is None:
        return None
    return next(
        (region for region, cities in REGION_CITIES.items() if city in cities),
        None,
    )


def query_city(query: str) -> str | None:
    """Return the longest matching canonical municipality alias."""

    normalized = normalize_query(query)
    # Avoid treating the municipality prefix in a domain keyword such as
    # ``砥部焼`` as a municipality constraint.  An explicit ``砥部町`` in the
    # same query is still retained because only the exact known keyword is
    # masked here.
    for keyword in _SORTED_KNOWN_KEYWORDS:
        normalized = normalized.replace(keyword, " ")
    for alias in _SORTED_CITY_ALIASES:
        if alias in normalized:
            return CITY_ALIASES[alias]
    return None


def is_entry_free(fee: str) -> bool:
    """Whether the advertised event entry is universally free.

    This is deliberately stricter than ``"無料" in fee``.  In particular,
    these are *not* generally free events:

    - ``無料（一部ワークショップ500円）``
    - ``無料（一部体験300円）``
    - ``入場無料・絵付け体験1,000円``
    - ``一般800円・高校生以下無料``

    ``無料（事前申込制）`` remains free because the parenthetical text adds
    a procedure, not a paid option.
    """

    compact = normalize_query(fee).replace(" ", "")
    if compact in {"無料", "入場無料", "0円"}:
        return True
    if not compact.startswith("無料"):
        return False
    paid_or_limited_markers = (
        "円",
        "有料",
        "一部",
        "一般",
        "高校生",
        "体験",
        "ワークショップ",
        "試食",
        "参加費",
    )
    return not any(marker in compact for marker in paid_or_limited_markers)


def base_entry_fee(fee: str) -> int:
    """Return the lowest advertised *entry* price used by a fee-cap query.

    Optional paid workshops do not change the base admission price.  Thus a
    record advertised as ``無料（一部体験300円）`` has a base entry fee of
    zero, while ``一般800円・高校生以下無料`` has a base fee of 800 yen.
    This function is for ``500円以内``-style filtering and is intentionally
    separate from the stricter all-purpose ``is_entry_free`` predicate.
    """

    compact = normalize_query(fee).replace(" ", "")
    if compact.startswith(("無料", "入場無料", "0円")):
        return 0
    match = _YEN_AMOUNT_RE.search(compact)
    if match:
        return int(match.group(0).replace(",", "").replace("円", ""))
    return 0


def _weekend_dates(reference_date: date) -> tuple[date, date]:
    """Return the Saturday/Sunday belonging to the current simulated week."""

    weekday = reference_date.weekday()  # Monday=0 ... Sunday=6
    if weekday <= 4:
        saturday = reference_date + timedelta(days=5 - weekday)
    else:
        saturday = reference_date - timedelta(days=weekday - 5)
    return saturday, saturday + timedelta(days=1)


def _query_dates(query: str, reference_date: date) -> list[date]:
    """Extract one or more target days from a natural-language query."""

    normalized = normalize_query(query)
    found: set[date] = set()

    if "今週末" in normalized or "週末" in normalized:
        found.update(_weekend_dates(reference_date))
    if "今日" in normalized or "本日" in normalized:
        found.add(reference_date)
    if "明日" in normalized:
        found.add(reference_date + timedelta(days=1))

    for match in _EXPLICIT_DATE_RE.finditer(normalized):
        year = int(match.group("year") or reference_date.year)
        found.add(date(year, int(match.group("month")), int(match.group("day"))))
    for match in _SLASH_DATE_RE.finditer(normalized):
        found.add(date(reference_date.year, int(match.group("month")), int(match.group("day"))))

    return sorted(found)


def _extract_genres(query: str) -> list[str]:
    """Extract canonical genre filters in deterministic longest-first order."""

    return [label for label in _SORTED_GENRE_LABELS if label in query]


def _extract_keywords(query: str) -> list[str]:
    """Extract name/overview terms after removing structured query language."""

    work = query
    keywords: list[str] = []

    # Preserve known domain terms before masking all control vocabulary.
    for term in _SORTED_KNOWN_KEYWORDS:
        if term in work:
            keywords.append(term)
            work = work.replace(term, " ")

    # Dates and fee expressions are not event keywords.
    work = _EXPLICIT_DATE_RE.sub(" ", work)
    work = _SLASH_DATE_RE.sub(" ", work)
    work = re.sub(r"\d[\d,]*\s*円(?:以内|以下|まで)?", " ", work)
    work = re.sub(r"小学\s*\d*\s*年生", " ", work)
    work = re.sub(r"\d+", " ", work)

    control_terms: set[str] = set(_SORTED_GENRE_LABELS)
    control_terms.update(REGION_CITIES)
    control_terms.update(CITY_ALIASES)
    control_terms.update(CHILD_TERMS)
    control_terms.update(INDOOR_TERMS)
    control_terms.update(OUTDOOR_TERMS)
    control_terms.update(RAIN_TERMS)
    control_terms.update(FREE_TERMS)
    control_terms.update(NEAR_TERMS)
    control_terms.update(("今日", "本日", "明日", "今週末", "週末"))
    control_terms.update(_GENERIC_STOPWORDS)

    for term in sorted(control_terms, key=len, reverse=True):
        work = work.replace(term, " ")

    # Japanese does not require spaces between words.  This conservative
    # tokenizer catches residual two-or-more-character terms while avoiding
    # particles and common request boilerplate.
    tokens = re.findall(r"[一-龥々ぁ-んァ-ヶA-Za-z][一-龥々ぁ-んァ-ヶA-Za-z0-9ー・]{1,}", work)
    for token in tokens:
        # A condition often leaves a Japanese particle attached to the next
        # word (for example, ``子どもと楽しむ`` -> ``と楽しむ``).  Strip only
        # grammatical particles at the edge; the substantive word is then
        # checked against the stopword set below.
        token = re.sub(r"^[はがをにでとやへも]+", "", token)
        token = re.sub(r"[はがをにでとやへも]+$", "", token)
        if token in _GENERIC_STOPWORDS or token in keywords:
            continue
        if len(token) >= 2:
            keywords.append(token)

    return list(dict.fromkeys(keywords))


def parse_query(query: str, reference_date: date = POC_REFERENCE_DATE) -> SearchFilters:
    """Convert natural language into explicit deterministic filters."""

    normalized = normalize_query(query)
    free_requested = any(term in normalized for term in FREE_TERMS if term != "0円")
    # Do not let ``500円`` or ``1,000円`` satisfy a literal ``0円`` query.
    zero_yen_requested = re.search(r"(?<![\d,])0\s*円", normalized) is not None
    fee_limit = _FEE_LIMIT_RE.search(normalized)

    venue: str | None = None
    if any(term in normalized for term in INDOOR_TERMS):
        venue = "屋内"
    elif any(term in normalized for term in OUTDOOR_TERMS):
        venue = "屋外"

    return SearchFilters(
        dates=[day.isoformat() for day in _query_dates(normalized, reference_date)],
        city=query_city(normalized),
        region=next(
            (region for region in REGION_CITIES if region in normalized),
            None,
        ),
        genres=_extract_genres(normalized),
        child_friendly=True if any(term in normalized for term in CHILD_TERMS) else None,
        venue=venue,
        rain_preferred=any(term in normalized for term in RAIN_TERMS),
        entry_free=True if free_requested or zero_yen_requested else None,
        max_entry_fee=(
            int(fee_limit.group("amount").replace(",", ""))
            if fee_limit
            else None
        ),
        keywords=_extract_keywords(normalized),
    )


def merge_filters(
    current: SearchFilters,
    previous: Mapping[str, Any] | None,
) -> SearchFilters:
    """Optionally inherit omitted structured filters from a previous turn."""

    if not previous:
        return current
    merged = current.to_dict()
    for key in (
        "dates",
        "city",
        "region",
        "genres",
        "child_friendly",
        "venue",
        "entry_free",
        "max_entry_fee",
        "keywords",
    ):
        if merged[key] in (None, []):
            previous_value = previous.get(key)
            if previous_value not in (None, []):
                merged[key] = previous_value
    if not current.rain_preferred and previous.get("rain_preferred"):
        merged["rain_preferred"] = True
    return SearchFilters(**merged)


def _matches_dates(event: Mapping[str, Any], dates: Iterable[str]) -> bool:
    """Match any requested day against an inclusive event period."""

    start, end = parse_event_dates(str(event["日時"]))
    return any(start <= date.fromisoformat(value) <= end for value in dates)


def _matches_genres(event: Mapping[str, Any], genres: Iterable[str]) -> bool:
    event_genre = str(event["ジャンル"])
    return all(
        any(alias in event_genre for alias in GENRE_ALIASES[genre])
        for genre in genres
    )


def _event_text(event: Mapping[str, Any]) -> str:
    return f"{event['イベント名']} {event['概要']}"


def _matches_keywords(event: Mapping[str, Any], keywords: Iterable[str]) -> bool:
    text = _event_text(event)
    return all(keyword in text for keyword in keywords)


def _text_score(event: Mapping[str, Any], filters: SearchFilters) -> int:
    """Rank already-matching source records without changing membership."""

    name = str(event["イベント名"])
    overview = str(event["概要"])
    genre = str(event["ジャンル"])
    score = 0
    for keyword in filters.keywords:
        if keyword in name:
            score += 12
        elif keyword in overview:
            score += 8
    for label in filters.genres:
        if any(alias in genre for alias in GENRE_ALIASES[label]):
            score += 8
    if filters.dates:
        score += 10
    if filters.city:
        score += 6
    if filters.region:
        score += 5
    if filters.child_friendly:
        score += 3
    if filters.venue:
        score += 3
    if filters.rain_preferred:
        score += 3 if event["屋内/屋外"] == "屋内" else 1
    if filters.entry_free:
        score += 3
    if filters.max_entry_fee is not None:
        score += 2
    return score


def _matches_event(event: Mapping[str, Any], filters: SearchFilters) -> bool:
    """Apply every active condition as an AND filter."""

    if filters.dates and not _matches_dates(event, filters.dates):
        return False
    if filters.city and event_city(event) != filters.city:
        return False
    if filters.region and event_region(event) != filters.region:
        return False
    if filters.genres and not _matches_genres(event, filters.genres):
        return False
    if filters.child_friendly is True and event["子ども向け"] is not True:
        return False
    if filters.venue == "屋内" and event["屋内/屋外"] != "屋内":
        return False
    if filters.venue == "屋外" and "屋外" not in event["屋内/屋外"]:
        return False
    if filters.rain_preferred and "屋内" not in event["屋内/屋外"]:
        return False
    if filters.entry_free is True and not is_entry_free(str(event["料金"])):
        return False
    if (
        filters.max_entry_fee is not None
        and base_entry_fee(str(event["料金"])) > filters.max_entry_fee
    ):
        return False
    if filters.keywords and not _matches_keywords(event, filters.keywords):
        return False
    return True


def _date_distance(event: Mapping[str, Any], reference_date: date) -> int:
    start, end = parse_event_dates(str(event["日時"]))
    if start <= reference_date <= end:
        return 0
    if start > reference_date:
        return (start - reference_date).days
    return 10_000 + (reference_date - end).days


def classify_intent(query: str) -> str:
    """Classify safety/clarification intents before data filtering."""

    normalized = normalize_query(query)
    if any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in INJECTION_PATTERNS
    ):
        return "injection"
    if any(term in normalized for term in NEAR_TERMS) and query_city(normalized) is None:
        return "needs_location"
    if normalized in {"地域から探す", "地域で探す", "地域"}:
        return "needs_region"
    if any(re.search(pattern, normalized) for pattern in OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"
    return "search"


def asks_for_nearby(query: str) -> bool:
    """Whether the user asked for proximity without naming a municipality."""

    return classify_intent(query) == "needs_location"


def looks_like_event_query(query: str) -> bool:
    """Keep unrelated cultural questions out of the event-search path."""

    intent = classify_intent(query)
    if intent != "search":
        return False
    filters = parse_query(query)
    return bool(
        filters.dates
        or filters.city
        or filters.region
        or filters.genres
        or filters.child_friendly
        or filters.venue
        or filters.rain_preferred
        or filters.entry_free
        or filters.max_entry_fee is not None
        or filters.keywords
    )


def search_events(
    query: str,
    events: Iterable[dict[str, Any]] | None = None,
    reference_date: date = POC_REFERENCE_DATE,
    *,
    previous_filters: Mapping[str, Any] | None = None,
    inherit_previous: bool = False,
    limit: int = MAX_SEARCH_RESULTS,
) -> SearchResult:
    """Return at most eight source records matching all parsed conditions."""

    if not isinstance(query, str):
        raise TypeError("検索語は文字列で指定してください。")
    normalized = normalize_query(query)
    intent = classify_intent(normalized)

    if intent == "injection":
        return SearchResult(
            [],
            SearchFilters(),
            intent,
            "イベントを新しく作ることはできんのよ。掲載済みの架空イベントから探してみる？",
        )
    if intent == "needs_location":
        return SearchResult(
            [],
            SearchFilters(),
            intent,
            "今どの市町あたりにおる？ 市町名を教えてくれたら、その地域で探すよ。",
        )
    if intent == "needs_region":
        return SearchResult(
            [],
            SearchFilters(),
            intent,
            "東予・中予・南予のうち、どの地域で探してみる？",
        )
    if intent == "out_of_scope":
        return SearchResult(
            [],
            SearchFilters(),
            intent,
            "このPoCは文化祭イベントを探す機能の検証が中心なんよ。関連するイベントなら探せるよ。",
        )

    filters = parse_query(normalized, reference_date)
    if inherit_previous:
        filters = merge_filters(filters, previous_filters)

    has_condition = any(
        (
            filters.dates,
            filters.city,
            filters.region,
            filters.genres,
            filters.child_friendly,
            filters.venue,
            filters.rain_preferred,
            filters.entry_free,
            filters.max_entry_fee is not None,
            filters.keywords,
        )
    )
    if not has_condition:
        return SearchResult(
            [],
            filters,
            "needs_condition",
            "いつ頃・どの地域のイベントを探しよる？ 「今日」「今週末」のように教えてみて。",
        )

    source_events = load_events() if events is None else list(events)
    matched: list[tuple[int, int, dict[str, Any]]] = []
    for source_index, event in enumerate(source_events):
        if _matches_event(event, filters):
            matched.append(
                (_text_score(event, filters), source_index, event)
            )

    matched.sort(
        key=lambda item: (
            -item[0],
            _date_distance(item[2], reference_date),
            item[1],
        )
    )
    safe_limit = min(max(limit, 0), MAX_SEARCH_RESULTS)
    selected = [event for _, _, event in matched[:safe_limit]]
    if not selected:
        return SearchResult(
            [],
            filters,
            "no_results",
            "その条件にぴったり合うイベントは見つからんかったよ。条件を少し変えて探してみる？",
            0,
        )
    return SearchResult(
        selected,
        filters,
        "search",
        total_matches=len(matched),
    )


def is_refinement_query(query: str) -> bool:
    """Whether the query looks like a follow-up narrowing request."""

    normalized = normalize_query(query)
    return any(term in normalized for term in ("だけ", "その中", "さらに", "もっと", "絞", "なら", "じゃあ"))


def follow_up_suggestion(filters: SearchFilters) -> str:
    """Provide a deterministic UI suggestion after a successful search."""

    if not filters.venue and not filters.rain_preferred:
        return "屋内だけに絞ってみる？"
    if not filters.entry_free and filters.max_entry_fee is None:
        return "無料だけで探してみる？"
    if not filters.child_friendly:
        return "子どもと楽しめるものに絞ってみる？"
    if not filters.region and not filters.city:
        return "東予・中予・南予の地域で絞ってみる？"
    return "別の日でも探してみる？"

"""Coverage assessment for the legacy deterministic query parser.

Fast Path is safe only when the legacy parser has accounted for the meaningful
search conditions in the whole query.  Recognizing one city or one count word
is not sufficient if another semantic constraint would be silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import event_search
from age_semantics import extract_query_age, extract_query_age_group
from app_config import (
    CHILD_TERMS,
    CITY_ALIASES,
    FREE_TERMS,
    GENRE_ALIASES,
    INDOOR_TERMS,
    OUTDOOR_TERMS,
    PAID_TERMS,
    RAIN_TERMS,
    REGION_CITIES,
)


@dataclass(frozen=True)
class ParserCoverage:
    complete: bool
    recognized_constraints: tuple[str, ...]
    unresolved_terms: tuple[str, ...]
    reason: str


_COLLOQUIAL_COUNT = ("どれくらい", "どのくらい", "どの程度")
_AGE_GROUP_TERMS = ("幼稚園児", "未就学児", "保育園児", "幼児", "小さい子", "小学生", "中学生", "高校生", "大人", "成人")
_RESERVATION_RE = re.compile(
    r"(?:予約|事前予約|申込|申し込み).{0,5}(?:不要|なし|なくても|しなくても|せず|必要|必須)"
)
_COLLOQUIAL_FREE_RE = re.compile(
    r"(?:お金|費用|料金)(?:が|の)?(?:全く)?かからない|(?:お金|費用|料金)をかけず"
)

_BOILERPLATE = (
    "イベント", "文化祭", "ありますか", "あるかな", "ある", "探して", "探す", "探し",
    "楽しめる", "楽しみたい", "楽しむ", "行きたい", "行ける", "おすすめ", "教えて",
    "ください", "何か", "どんな", "やっている", "参加", "できる", "できそう", "向け",
    "今日", "本日", "明日", "今週末", "週末", "来週", "土曜日", "日曜日",
    "開催", "開催日", "開催日時", "日程", "いつ", "何時", "何日に", "どこ", "場所",
    "会場", "いくら", "料金", "何円", "費用", "何件", "いくつ", "件数", "何個",
    "どれくらい", "どのくらい", "どの程度", "が好き", "好き", "興味", "関連",
    "大丈夫", "ですか", "でしょうか", "かな", "くらい", "ほど", "やる", "やって",
)

_QUERY_GENRE_WORDS = (
    "伝統文化", "伝統芸能", "民俗芸能", "語り芸", "祭り", "まつり", "祭礼", "民俗",
    "歴史", "文化財", "建築", "城下町", "工芸", "伝統工芸", "陶芸", "水引", "紙",
    "アート", "美術", "芸術", "デザイン", "文学", "俳句", "ことば", "食文化", "農文化",
    "演劇", "芝居", "舞台", "自然", "海洋文化", "学習",
)


def _recognized(parsed: event_search.SearchFilters) -> list[str]:
    result: list[str] = []
    if parsed.dates:
        result.append("date")
    if parsed.city_groups:
        result.append("municipality")
    if parsed.region_groups:
        result.append("region")
    if parsed.genre_groups or parsed.genres:
        result.append("genre")
    if parsed.child_friendly is True:
        result.append("child_friendly")
    if parsed.venue:
        result.append("venue")
    if parsed.rain_preferred:
        result.append("rain")
    if parsed.entry_free is True:
        result.append("entry_free")
    if parsed.paid_only:
        result.append("paid_only")
    if parsed.max_entry_fee is not None:
        result.append("max_entry_fee")
    if parsed.time_slots or parsed.time_after is not None:
        result.append("time")
    if parsed.soft_terms:
        result.append("soft_terms")
    if parsed.requested_field:
        result.append("requested_field")
    if event_search.classify_intent(parsed.entity or "") == "count":
        result.append("count")
    return result


def _semantic_gaps(query: str, parsed: event_search.SearchFilters) -> list[str]:
    gaps: list[str] = []
    age = extract_query_age(query)
    age_group = extract_query_age_group(query)
    if age is not None:
        gaps.append(f"age={age}")
    elif age_group is not None or any(term in query for term in _AGE_GROUP_TERMS):
        gaps.append(f"age_group={age_group or 'unresolved'}")

    reservation = _RESERVATION_RE.search(query)
    if reservation:
        gaps.append(reservation.group(0))

    if any(term in query for term in ("建物の中", "建物内")) and not parsed.venue:
        gaps.append("venue=indoor")

    if _COLLOQUIAL_FREE_RE.search(query) and parsed.entry_free is not True:
        gaps.append("entry_free")

    if any(term in query for term in _COLLOQUIAL_COUNT) and event_search.classify_intent(query) != "count":
        gaps.append("answer_type=count")
    return gaps


def _remove_many(value: str, terms: Iterable[str]) -> str:
    work = value
    for term in sorted({term for term in terms if term}, key=len, reverse=True):
        work = work.replace(term, " ")
    return work


def _meaningful_residual(query: str, parsed: event_search.SearchFilters) -> tuple[str, ...]:
    work = event_search.normalize_query(query)

    # Search-content soft terms are valid parser output, not unknown syntax.
    work = _remove_many(work, parsed.soft_terms)

    city_terms: list[str] = []
    if parsed.city_groups:
        for group in parsed.city_groups:
            for city in group:
                city_terms.extend((city, city.removesuffix("市").removesuffix("町")))
        city_terms.extend(CITY_ALIASES)
    work = _remove_many(work, city_terms)

    if parsed.region_groups:
        work = _remove_many(work, REGION_CITIES)
    if parsed.genres or parsed.genre_groups:
        work = _remove_many(work, _QUERY_GENRE_WORDS)
        for label in parsed.genres:
            work = _remove_many(work, (label, *GENRE_ALIASES.get(label, ())))
    if parsed.child_friendly is True:
        work = _remove_many(work, CHILD_TERMS)
    if parsed.venue:
        work = _remove_many(work, (*INDOOR_TERMS, *OUTDOOR_TERMS))
    if parsed.rain_preferred:
        work = _remove_many(work, RAIN_TERMS)
    if parsed.entry_free is True:
        work = _remove_many(work, FREE_TERMS)
    if parsed.paid_only:
        work = _remove_many(work, PAID_TERMS)

    # Remove deterministic date/time/fee syntax already represented by fields.
    work = re.sub(r"(?:(?:20\d{2})年)?\d{1,2}月\d{1,2}日", " ", work)
    work = re.sub(r"\d{1,2}[/-]\d{1,2}", " ", work)
    work = re.sub(r"\d{1,2}月(?:前半|後半)?", " ", work)
    work = re.sub(r"\d{1,2}時(?:以降|から)?", " ", work)
    work = re.sub(r"\d[\d,]*\s*円(?:以内|以下|まで)?", " ", work)

    if parsed.requested_field:
        work = _remove_many(
            work,
            ("開催日", "開催日時", "日程", "いつ", "何時", "何日に", "どこ", "場所", "会場", "いくら", "料金", "何円", "費用", "ジャンル", "分野", "種類", "子ども向け", "対象", "屋内外", "概要", "内容"),
        )

    work = _remove_many(work, _BOILERPLATE)
    work = re.sub(r"[、。,.!?！？「」『』（）()・:/〜~]", " ", work)
    work = re.sub(r"(?:^|\s)[のはがをにでとやへもか]+(?:\s|$)", " ", work)
    # Japanese particles can remain attached after phrase removal.
    work = re.sub(r"^[のはがをにでとやへもか]+", "", work.strip())
    work = re.sub(r"[のはがをにでとやへもか]+$", "", work.strip())
    tokens = re.findall(r"[一-龥々ぁ-んァ-ヶA-Za-z][一-龥々ぁ-んァ-ヶA-Za-z0-9ー]{1,}", work)
    return tuple(dict.fromkeys(token for token in tokens if len(token) >= 2))


def evaluate_parser_coverage(
    query: str,
    parsed: event_search.SearchFilters,
) -> ParserCoverage:
    """Assess whether the legacy parser covered the whole search meaning."""

    normalized = event_search.normalize_query(query)
    recognized = _recognized(parsed)
    if event_search.classify_intent(normalized) == "count":
        recognized.append("count")

    gaps = _semantic_gaps(normalized, parsed)
    if gaps:
        return ParserCoverage(
            False,
            tuple(dict.fromkeys(recognized)),
            tuple(dict.fromkeys(gaps)),
            "legacy parser would drop a semantic constraint",
        )

    residual = _meaningful_residual(normalized, parsed)
    if residual:
        return ParserCoverage(
            False,
            tuple(dict.fromkeys(recognized)),
            residual,
            "meaningful residual text is not covered by parser fields",
        )

    # A generic discovery question with no actual parsed condition is not high
    # confidence merely because all boilerplate disappeared.
    if not recognized:
        return ParserCoverage(False, (), (), "no meaningful search constraint was parsed")

    return ParserCoverage(
        True,
        tuple(dict.fromkeys(recognized)),
        (),
        "all meaningful search constraints are represented by parser fields",
    )

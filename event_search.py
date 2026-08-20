"""Deterministic Search v2 for the 30 fictional cultural events.

The event JSON remains the sole source of truth for facts. Search v2 keeps
hard constraints strict, but separates natural-language question boilerplate
from soft discovery terms and ranks approximate matches instead of requiring
every residual token to occur verbatim in an event record.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import age_semantics
from app_config import (
    CHILD_TERMS,
    CITY_ALIASES,
    EVENTS_PATH,
    FREE_TERMS,
    GENRE_ALIASES,
    INDOOR_TERMS,
    INJECTION_PATTERNS,
    MAX_SEARCH_RESULTS,
    NEAR_TERMS,
    OUTDOOR_TERMS,
    OUT_OF_SCOPE_PATTERNS,
    PAID_TERMS,
    POC_REFERENCE_DATE,
    RAIN_TERMS,
    REGION_CITIES,
    SEARCH_METADATA_PATH,
)
from event_details import V2_FIELDS, validate_events_v2


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
# ``参加形式`` lives in events.json as recommendation-only metadata.  Keep it
# in the legacy loader's tolerated fields as well, so a Streamlit checkout
# that briefly mixes the v2 event data with the legacy validation path can
# still start and serve the app.
METADATA_FIELDS = ("id", "aliases", "search_tags", "参加形式")
OPTIONAL_EVENT_FIELDS = frozenset(METADATA_FIELDS) | V2_FIELDS
ALL_CITIES: tuple[str, ...] = tuple(
    city for cities in REGION_CITIES.values() for city in cities
)
_SORTED_CITY_ALIASES: tuple[str, ...] = tuple(sorted(CITY_ALIASES, key=len, reverse=True))
_SORTED_GENRE_LABELS: tuple[str, ...] = tuple(sorted(GENRE_ALIASES, key=len, reverse=True))

_EVENT_DATE_RE = re.compile(
    r"(?P<start>20\d{2}-\d{2}-\d{2})"
    r"(?:\s*[〜~]\s*(?P<end>20\d{2}-\d{2}-\d{2}))?"
)
_EXPLICIT_DATE_RE = re.compile(
    r"(?:(?P<year>20\d{2})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
)
_SLASH_DATE_RE = re.compile(r"(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?!\d)")
_MONTH_RE = re.compile(r"(?P<month>\d{1,2})月(?P<half>前半|後半)?(?!\d{1,2}日)")
_FEE_LIMIT_RE = re.compile(r"(?P<amount>\d[\d,]*)\s*円\s*(?:以内|以下|まで)")
_YEN_AMOUNT_RE = re.compile(r"\d[\d,]*\s*円")
_TIME_AFTER_RE = re.compile(r"(?P<hour>\d{1,2})時(?:以降|から)")

_GENERIC_STOPWORDS = frozenset(
    {
        "イベント", "文化祭", "もの", "何か", "楽しめる", "楽しむ", "楽しみたい",
        "行ける", "行きたい", "やっている", "探す", "探して", "探し", "ありますか",
        "ある", "あるかな", "ください", "教えて", "知りたい", "お願い", "興味",
        "参加", "できる", "できそう", "おすすめ", "だけ", "条件", "地域", "市町",
        "料金", "上限", "以内", "以下", "まで", "今日", "本日", "明日", "今週末",
        "週末", "来週", "土曜日", "日曜日", "午前", "午後", "夕方", "雨でも", "雨の日",
        "雨天", "無料", "タダ", "ただ", "屋内", "室内", "屋外", "外で", "子ども",
        "子供", "こども", "向け", "小学生", "小学", "親子", "家族", "ファミリー",
        "近く", "近い", "近場", "周辺", "伝統文化", "伝統芸能", "いつ", "いつですか",
        "行われますか", "開催されますか", "開催日", "開催日時", "開催", "何日に", "何時", "何時から",
        "どこ", "どこですか", "どこで", "場所は", "会場は", "いくら", "料金は", "何円",
        "いくつくらい", "何件くらい", "何個くらい", "件くらい",
        "費用", "予約必要", "予約は必要", "ですか", "かな", "でしょうか", "系", "っぽい",
        "が好き", "がいい", "に興味", "に関係する", "関連する", "イベントある", "文化イベント",
        "体験", "したい", "見たい", "入れる", "有料", "やつ", "時以降", "時から",
    }
)

_QUESTION_PHRASES = (
    "子ども向けですか", "開催されますか", "行われますか", "いつ行われますか",
    "いつ開催されますか", "いつですか", "何時から", "何日に", "開催日時", "開催日は",
    "予約は必要ですか", "予約必要", "どこですか", "どこで", "場所は", "会場は",
    "いくらですか", "料金は", "何円ですか", "教えて", "知りたい", "ありますか",
)

_QUERY_SYNONYMS: dict[str, str] = {
    "開幕": "オープニング", "開会": "オープニング", "開幕イベント": "オープニング",
    "オープニングフェス": "オープニング", "閉幕": "クロージング", "フィナーレ": "クロージング",
    "閉幕イベント": "クロージング", "お祭り": "祭り", "まつり": "祭り", "焼き物": "砥部焼",
    "やきもの": "砥部焼", "陶芸": "陶芸", "美術": "アート", "芸術": "アート",
    "舞台": "演劇", "芝居": "演劇", "海": "海", "海洋": "海洋", "子供": "子ども",
    "こども": "子ども",
}
_COUNT_PATTERNS = (r"何件", r"いくつ", r"件数", r"何個")
_REFINE_TERMS = ("その中", "だけ", "じゃあ", "さらに", "もっと", "絞って")
_REFERENCE_RE = re.compile(r"(?:それ|そのイベント|第?(\d+|一|二|三|最後)番目|最初|第一|最后)")


@dataclass
class SearchFilters:
    dates: list[str] = field(default_factory=list)
    city: str | None = None
    region: str | None = None
    city_groups: list[list[str]] = field(default_factory=list)
    region_groups: list[list[str]] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    genre_groups: list[list[str]] = field(default_factory=list)
    child_friendly: bool | None = None
    age: int | None = None
    age_group: str | None = None
    age_intent: str | None = None
    venue: str | None = None
    rain_preferred: bool = False
    entry_free: bool | None = None
    paid_only: bool = False
    reservation_required: bool | None = None
    max_entry_fee: int | None = None
    time_slots: list[str] = field(default_factory=list)
    time_after: int | None = None
    keywords: list[str] = field(default_factory=list)
    soft_terms: list[str] = field(default_factory=list)
    intent: str = "discover"
    entity: str | None = None
    requested_field: str | None = None
    invalid_date: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    events: list[dict[str, Any]]
    filters: SearchFilters
    intent: str = "search"
    message: str | None = None
    total_matches: int = 0
    confidence: str = "none"
    near_matches: list[dict[str, Any]] = field(default_factory=list)
    relaxed_condition: str | None = None


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("～", "〜")
    return re.sub(r"\s+", " ", normalized).strip()


def _compact_intent_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_query(value), flags=re.UNICODE)


def _event_id(event: Mapping[str, Any]) -> str:
    if isinstance(event.get("id"), str) and event["id"]:
        return str(event["id"])
    return str(event["公式URL"]).rstrip("/").rsplit("/", 1)[-1]


def load_events(path: str | Path = EVENTS_PATH) -> list[dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("events.json は30件の配列である必要があります。")
    if data and isinstance(data[0], dict) and V2_FIELDS.issubset(data[0]):
        return validate_events_v2(data)
    expected_fields = set(EVENT_FIELDS)
    names: set[str] = set()
    urls: set[str] = set()
    ids: set[str] = set()
    for index, event in enumerate(data, start=1):
        if not isinstance(event, dict) or not expected_fields.issubset(event):
            raise ValueError(f"events.json の {index} 件目の項目が不正です。")
        extras = set(event) - expected_fields
        # A Streamlit restart can briefly observe a mixed checkout while the
        # repository update is propagating.  Keep the legacy search path
        # tolerant of v2 fields, but never weaken the strict v2 validator when
        # all v2 fields are present.
        if extras - OPTIONAL_EVENT_FIELDS:
            raise ValueError(f"events.json の {index} 件目に未知の項目があります。")
        if not isinstance(event["子ども向け"], bool):
            raise ValueError(f"events.json の {index} 件目の子ども向け値が不正です。")
        for field_name in EVENT_FIELDS:
            if field_name == "子ども向け":
                continue
            if not isinstance(event[field_name], str) or not event[field_name].strip():
                raise ValueError(f"events.json の {index} 件目の{field_name}が空です。")
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
        event_id = _event_id(event)
        if event_id in ids:
            raise ValueError(f"events.json の {index} 件目のIDが重複しています。")
        ids.add(event_id)
        names.add(event["イベント名"])
        urls.add(event["公式URL"])
    return data


@lru_cache(maxsize=4)
def load_search_metadata(path: str | Path = SEARCH_METADATA_PATH) -> dict[str, dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("search_metadata.json は30件の配列である必要があります。")
    result: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict) or set(item) != {"id", "aliases", "search_tags"}:
            raise ValueError("検索メタデータの項目が不正です。")
        if not isinstance(item["id"], str) or not isinstance(item["aliases"], list):
            raise ValueError("検索メタデータの型が不正です。")
        if not all(isinstance(value, str) and value.strip() for value in item["aliases"]):
            raise ValueError("検索aliasが不正です。")
        if not isinstance(item["search_tags"], list) or not all(
            isinstance(value, str) and value.strip() for value in item["search_tags"]
        ):
            raise ValueError("検索タグが不正です。")
        if item["id"] in result:
            raise ValueError("検索メタデータIDが重複しています。")
        result[item["id"]] = item
    return result


def _metadata_for(event: Mapping[str, Any]) -> dict[str, Any]:
    embedded = {
        "id": _event_id(event),
        "aliases": list(event.get("aliases", [])),
        "search_tags": list(event.get("search_tags", [])),
    }
    external = load_search_metadata().get(_event_id(event))
    if external:
        embedded["aliases"] = list(dict.fromkeys(external["aliases"] + embedded["aliases"]))
        embedded["search_tags"] = list(dict.fromkeys(external["search_tags"] + embedded["search_tags"]))
    return embedded


def parse_event_dates(value: str) -> tuple[date, date]:
    match = _EVENT_DATE_RE.search(normalize_query(value))
    if not match:
        raise ValueError(f"日時を解釈できません: {value}")
    start = date.fromisoformat(match.group("start"))
    end = date.fromisoformat(match.group("end") or match.group("start"))
    return start, end


def event_city(event: Mapping[str, Any]) -> str | None:
    if isinstance(event.get("市町"), str) and event["市町"]:
        return str(event["市町"])
    place = str(event["場所"])
    return next((city for city in sorted(ALL_CITIES, key=len, reverse=True) if city in place), None)


def event_region(event: Mapping[str, Any]) -> str | None:
    if isinstance(event.get("地域"), str) and event["地域"]:
        return str(event["地域"])
    city = event_city(event)
    return next((region for region, cities in REGION_CITIES.items() if city in cities), None) if city else None


def _city_in_location_context(query: str, alias: str) -> bool:
    return bool(re.search(
        rf"{re.escape(alias)}(?:市|町)?\s*(?:で|のイベント|の会場|周辺|近く|あたり|なら|では|か|または)",
        query,
    ))


def _query_city_groups(query: str) -> list[list[str]]:
    normalized = normalize_query(query)
    found: list[str] = []
    for alias in _SORTED_CITY_ALIASES:
        canonical = CITY_ALIASES[alias]
        if alias.endswith(("市", "町")):
            if alias in normalized and canonical not in found:
                found.append(canonical)
        elif _city_in_location_context(normalized, alias) and canonical not in found:
            found.append(canonical)
    if not found:
        return []
    if len(found) == 1:
        return [[found[0]]]
    return [found] if re.search(r"(?:か|または|、|/|・)", normalized) else [[city] for city in found]


def query_city(query: str) -> str | None:
    groups = _query_city_groups(query)
    return groups[0][0] if len(groups) == 1 and len(groups[0]) == 1 else None


def is_entry_free(fee: str) -> bool:
    compact = normalize_query(fee).replace(" ", "")
    if compact in {"無料", "入場無料", "0円"}:
        return True
    if not compact.startswith("無料"):
        return False
    return not any(marker in compact for marker in ("円", "有料", "一部", "一般", "高校生", "体験", "ワークショップ", "試食", "参加費"))


def base_entry_fee(fee: str) -> int:
    compact = normalize_query(fee).replace(" ", "")
    if compact.startswith(("無料", "入場無料", "0円")):
        return 0
    match = _YEN_AMOUNT_RE.search(compact)
    return int(match.group(0).replace(",", "").replace("円", "")) if match else 0


def _weekend_dates(reference_date: date) -> tuple[date, date]:
    weekday = reference_date.weekday()
    saturday = reference_date + timedelta(days=5 - weekday) if weekday <= 4 else reference_date - timedelta(days=weekday - 5)
    return saturday, saturday + timedelta(days=1)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _query_dates(query: str, reference_date: date) -> tuple[list[date], bool]:
    normalized = normalize_query(query)
    found: set[date] = set()
    invalid = False
    if "今週末" in normalized or "週末" in normalized:
        found.update(_weekend_dates(reference_date))
    if "今日" in normalized or "本日" in normalized:
        found.add(reference_date)
    if "明日" in normalized:
        found.add(reference_date + timedelta(days=1))
    if "来週" in normalized:
        start = reference_date + timedelta(days=7)
        found.update(start + timedelta(days=offset) for offset in range(7))
    if "土曜日" in normalized:
        found.add(reference_date + timedelta(days=(5 - reference_date.weekday()) % 7))
    if "日曜日" in normalized:
        found.add(reference_date + timedelta(days=(6 - reference_date.weekday()) % 7))
    for match in _EXPLICIT_DATE_RE.finditer(normalized):
        parsed = _safe_date(int(match.group("year") or reference_date.year), int(match.group("month")), int(match.group("day")))
        if parsed is None:
            invalid = True
        else:
            found.add(parsed)
    for match in _SLASH_DATE_RE.finditer(normalized):
        parsed = _safe_date(reference_date.year, int(match.group("month")), int(match.group("day")))
        if parsed is None:
            invalid = True
        else:
            found.add(parsed)
    for match in _MONTH_RE.finditer(normalized):
        month = int(match.group("month"))
        first = _safe_date(reference_date.year, month, 1)
        if first is None:
            invalid = True
            continue
        next_month = date(reference_date.year + (month == 12), (month % 12) + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        half = match.group("half")
        start_day = 1 if half != "後半" else 16
        end_day = 15 if half == "前半" else last_day
        found.update(first + timedelta(days=offset) for offset in range(start_day - 1, end_day))
    return sorted(found), invalid


def _requested_field(query: str) -> str | None:
    normalized = normalize_query(query)
    if re.search(r"いつ|開催日|開催日時|何日に|何時|時間|日程", normalized):
        return "datetime"
    if re.search(r"どこ|場所|会場", normalized):
        return "place"
    if re.search(r"いくら|料金|何円|費用", normalized):
        return "fee"
    if re.search(r"ジャンル|分野|種類", normalized):
        return "genre"
    if re.search(r"子ども向け|子供向け|対象", normalized):
        return "child_friendly"
    if re.search(r"屋内外|屋内ですか|屋外ですか|室内ですか|会場", normalized):
        return "venue"
    if re.search(r"概要|どんな|内容", normalized):
        return "overview"
    return None


def _phrase_pattern(phrase: str) -> str:
    return r"\s*".join(re.escape(part) for part in phrase.split())


def _remove_question_language(value: str) -> str:
    work = value
    for phrase in sorted(_QUESTION_PHRASES, key=len, reverse=True):
        work = re.sub(_phrase_pattern(phrase), " ", work, flags=re.IGNORECASE)
    return work


@lru_cache(maxsize=2)
def _metadata_aliases() -> tuple[str, ...]:
    aliases: set[str] = set()
    for item in load_search_metadata().values():
        aliases.update(item["aliases"])
    return tuple(sorted(aliases, key=len, reverse=True))


def _genre_candidates(query: str) -> list[str]:
    normalized = normalize_query(query)
    # These are query-side vocabulary.  Do not reuse every event-side genre
    # alias here: doing so turns a soft phrase such as "歴史かアート" into
    # the broader hard filter 伝統文化, and turns "体験したい" into 学習.
    query_aliases: dict[str, tuple[str, ...]] = {
        "伝統文化": ("伝統文化",),
        "伝統芸能": ("伝統芸能", "民俗芸能", "語り芸"),
        "祭り": ("祭り", "まつり", "祭礼"),
        "民俗": ("民俗",),
        "歴史": ("歴史", "文化財", "建築", "城下町"),
        "工芸": ("工芸", "伝統工芸", "陶芸", "水引", "紙"),
        "アート": ("アート", "美術", "芸術", "デザイン"),
        "文学": ("文学", "俳句", "ことば"),
        "食文化": ("食文化", "農文化"),
        "演劇": ("演劇", "芝居", "舞台", "語り芸"),
        "自然": ("自然",),
        "海洋文化": ("海洋文化",),
        "学習": ("学習",),
    }
    found: list[str] = []
    for label, aliases in query_aliases.items():
        for alias in sorted({label, *aliases}, key=len, reverse=True):
            if alias in normalized and not re.search(
                rf"{re.escape(alias)}(?:系|っぽい|が好き|がいい|に興味|に関係|関連|したい)",
                normalized,
            ):
                if label not in found:
                    found.append(label)
                break
    return found


def _genre_soft_terms(query: str) -> list[str]:
    normalized = normalize_query(query)
    terms: list[str] = []
    for source, target in (("美術", "美術"), ("芸術", "芸術"), ("お祭り", "祭り"), ("まつり", "祭り"), ("歴史", "歴史"), ("海", "海"), ("焼き物", "焼き物"), ("陶芸", "陶芸")):
        if source in normalized and re.search(rf"{re.escape(source)}(?:系|っぽい|が好き|がいい|に興味|に関係|関連)", normalized):
            terms.append(target)
    return terms


def _extract_soft_terms(query: str) -> list[str]:
    normalized = normalize_query(query)
    work = _remove_question_language(normalized)

    # Remove municipality vocabulary before matching metadata aliases.  This
    # prevents a one-character alias such as 山 from being extracted out of
    # 松山, and keeps city OR filters hard rather than turning city names into
    # soft relevance hints.
    for region, cities in REGION_CITIES.items():
        work = work.replace(region, " ")
        for city in cities:
            work = work.replace(city, " ")
            short = city.removesuffix("市").removesuffix("町")
            if _city_in_location_context(normalized, short):
                work = work.replace(short, " ")
    terms: list[str] = []
    for alias in _metadata_aliases():
        if alias in work:
            terms.append(alias)
            work = work.replace(alias, " ")
    for source, target in sorted(_QUERY_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if source in work:
            terms.append(target)
            work = work.replace(source, " ")
    terms.extend(_genre_soft_terms(normalized))
    work = _EXPLICIT_DATE_RE.sub(" ", work)
    work = _SLASH_DATE_RE.sub(" ", work)
    work = _MONTH_RE.sub(" ", work)
    work = re.sub(r"\d{1,2}時(?:以降|から)", " ", work)
    work = re.sub(r"\d[\d,]*\s*円(?:以内|以下|まで)?", " ", work)
    work = re.sub(r"小学\s*\d*\s*年生", " ", work)
    work = re.sub(r"\d+", " ", work)
    for region, cities in REGION_CITIES.items():
        work = work.replace(region, " ")
        for city in cities:
            work = work.replace(city, " ")
            short = city.removesuffix("市").removesuffix("町")
            if _city_in_location_context(normalized, short):
                work = work.replace(short, " ")
    control_terms = set(_GENERIC_STOPWORDS) | set(CHILD_TERMS) | set(INDOOR_TERMS) | set(OUTDOOR_TERMS) | set(RAIN_TERMS) | set(FREE_TERMS) | set(PAID_TERMS) | set(NEAR_TERMS) | {
        "予約なし", "予約不要", "予約はいらない", "予約いらない", "申込不要", "申し込み不要",
        "申込なし", "申し込みなし", "建物の中", "建物内", "建物", "中でやる",
        "か", "または", "と", "で", "の", "は", "が", "を", "に", "や", "へ", "も",
    }
    for term in sorted(control_terms, key=len, reverse=True):
        work = work.replace(term, " ")
    tokens = re.findall(r"[一-龥々ぁ-んァ-ヶA-Za-z][一-龥々ぁ-んァ-ヶA-Za-z0-9ー・]{1,}", work)
    for token in tokens:
        token = re.sub(r"^[はがをにでとやへも]+", "", token)
        token = re.sub(r"[はがをにでとやへも]+$", "", token)
        if len(token) >= 2 and token not in _GENERIC_STOPWORDS:
            terms.append(token)
    result: list[str] = []
    hard_control_terms = set(CHILD_TERMS) | set(INDOOR_TERMS) | set(OUTDOOR_TERMS) | set(RAIN_TERMS) | set(FREE_TERMS) | set(PAID_TERMS) | set(NEAR_TERMS)
    for term in terms:
        term = _QUERY_SYNONYMS.get(term, term)
        if term in hard_control_terms:
            continue
        if term not in result:
            result.append(term)
    return result


def _extract_genre_groups(query: str) -> tuple[list[str], list[list[str]]]:
    genres = _genre_candidates(query)
    if not genres:
        return [], []
    normalized = normalize_query(query)
    return genres, [genres] if len(genres) > 1 and re.search(r"か|または|/|・", normalized) else [[genre] for genre in genres]


def _extract_time_filters(query: str) -> tuple[list[str], int | None]:
    normalized = normalize_query(query)
    slots = [slot for slot in ("午前", "午後", "夕方") if slot in normalized]
    match = _TIME_AFTER_RE.search(normalized)
    return slots, int(match.group("hour")) * 60 if match else None


def _extract_entity(soft_terms: list[str], query: str) -> str | None:
    return soft_terms[0] if soft_terms and _requested_field(query) else None


def parse_query(query: str, reference_date: date = POC_REFERENCE_DATE) -> SearchFilters:
    normalized = normalize_query(query)
    age_query = age_semantics.query_age_semantics(normalized)
    dates, invalid_date = _query_dates(normalized, reference_date)
    city_groups = _query_city_groups(normalized)
    region_found = [region for region in REGION_CITIES if region in normalized]
    region_groups = [region_found] if len(region_found) > 1 and re.search(r"か|または", normalized) else [[region] for region in region_found]
    genres, genre_groups = _extract_genre_groups(normalized)
    soft_terms = [term for term in _extract_soft_terms(normalized) if not age_semantics.is_age_query_term(term)]
    requested_field = _requested_field(normalized)
    child_requested = any(term in normalized for term in CHILD_TERMS) or bool(
        re.search(r"(?:小|小学)\s*[1-6](?:年生)?", normalized)
    )
    venue = (
        "屋内"
        if any(term in normalized for term in INDOOR_TERMS)
        or any(term in normalized for term in ("建物の中", "建物内", "建物", "中でやる"))
        else "屋外"
        if any(term in normalized for term in OUTDOOR_TERMS)
        else None
    )
    free_requested = any(term in normalized for term in ("無料", "タダ")) or bool(
        re.search(r"ただ(?:で|だけ|入場)", normalized)
    )
    paid_requested = any(term in normalized for term in PAID_TERMS)
    zero_yen_requested = re.search(r"(?<![\d,])0\s*円", normalized) is not None
    fee_limit = _FEE_LIMIT_RE.search(normalized)
    time_slots, time_after = _extract_time_filters(normalized)
    reservation_required = None
    if any(term in normalized for term in ("予約なし", "予約不要", "予約はいらない", "予約いらない", "申込不要", "申し込み不要", "申込なし", "申し込みなし")):
        reservation_required = False
    elif any(term in normalized for term in ("予約必要", "予約が必要", "申込必要", "申し込み必要")):
        reservation_required = True
    city = city_groups[0][0] if len(city_groups) == 1 and len(city_groups[0]) == 1 else None
    region = region_groups[0][0] if len(region_groups) == 1 and len(region_groups[0]) == 1 else None
    intent = classify_intent(normalized, _parsed_hint=True, parsed_values=(soft_terms, requested_field))
    return SearchFilters(
        dates=[day.isoformat() for day in dates], city=city, region=region,
        city_groups=city_groups, region_groups=region_groups, genres=genres, genre_groups=genre_groups,
        child_friendly=True if child_requested or age_query.recognized else None,
        age=age_query.age, age_group=age_query.age_group, age_intent=age_query.age_intent,
        venue=venue,
        rain_preferred=any(term in normalized for term in RAIN_TERMS),
        entry_free=True if free_requested or zero_yen_requested else None,
        paid_only=paid_requested,
        max_entry_fee=int(fee_limit.group("amount").replace(",", "")) if fee_limit else None,
        time_slots=time_slots, time_after=time_after, keywords=soft_terms, soft_terms=soft_terms,
        reservation_required=reservation_required,
        intent=intent, entity=_extract_entity(soft_terms, normalized), requested_field=requested_field,
        invalid_date=invalid_date,
    )


def merge_filters(current: SearchFilters, previous: Mapping[str, Any] | None) -> SearchFilters:
    if not previous:
        return current
    merged = current.to_dict()
    for key in ("dates", "city", "region", "city_groups", "region_groups", "genres", "genre_groups", "child_friendly", "age", "age_group", "age_intent", "venue", "rain_preferred", "entry_free", "paid_only", "max_entry_fee", "reservation_required", "time_slots", "time_after", "soft_terms", "keywords"):
        if key in {"age", "age_group", "age_intent", "reservation_required"}:
            current_missing = merged[key] is None
            previous_present = previous.get(key) is not None
        else:
            current_missing = merged[key] in (None, [], False)
            previous_present = previous.get(key) not in (None, [], False)
        if current_missing and previous_present:
            merged[key] = previous[key]
    if current.entity is None and previous.get("entity"):
        merged["entity"] = previous["entity"]
    return SearchFilters(**merged)


def _compact_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", value).lower())


def _fuzzy_contains(term: str, value: str) -> float:
    compact_term = _compact_text(term)
    compact_value = _compact_text(value)
    if not compact_term or not compact_value:
        return 0.0
    if compact_term in compact_value:
        return 1.0
    best = 0.0
    for length in {len(compact_term), max(1, len(compact_term) - 1), len(compact_term) + 1}:
        if length > len(compact_value):
            continue
        for start in range(len(compact_value) - length + 1):
            best = max(best, SequenceMatcher(None, compact_term, compact_value[start : start + length]).ratio())
    return best


def _soft_term_score(event: Mapping[str, Any], term: str) -> int:
    name = str(event["イベント名"])
    genre = str(event["ジャンル"])
    overview = str(event["概要"])
    metadata = _metadata_for(event)
    aliases = " ".join(str(value) for value in metadata["aliases"])
    tags = " ".join(str(value) for value in metadata["search_tags"])
    if term in name:
        return 140
    if term in aliases:
        return 125
    if term in tags:
        return 95
    if term in genre:
        return 85
    if term in overview:
        return 55
    compact_term = _compact_text(term)
    # Short tokens create dangerous collisions (e.g. 伊予弁 -> 伊予市 or
    # オープニング -> オープン工房).  Fuzzy matching is a last resort and
    # only applies to longer search vocabulary in metadata/name fields.
    if len(compact_term) < 4:
        return 0
    fuzzy = max((_fuzzy_contains(term, value) for value in (name, aliases, tags)), default=0.0)
    return int(25 + fuzzy * 25) if fuzzy >= 0.80 else 0


def _soft_score(event: Mapping[str, Any], filters: SearchFilters) -> int:
    return sum(_soft_term_score(event, term) for term in filters.soft_terms)


def _matches_soft_terms(event: Mapping[str, Any], filters: SearchFilters) -> bool:
    if not filters.soft_terms:
        return True
    scores = [_soft_term_score(event, term) for term in filters.soft_terms]
    # A factual lookup containing several entity terms is an AND query at the
    # entity level (牛鬼 + 宇和島, 提灯 + 展示).  Broad discovery remains
    # ranked OR retrieval so a theme can return related events.
    if filters.intent == "lookup" and len(scores) > 1:
        return all(score > 0 for score in scores)
    return any(score > 0 for score in scores)


def _matches_dates(event: Mapping[str, Any], dates: Iterable[str]) -> bool:
    start, end = parse_event_dates(str(event["日時"]))
    return any(start <= date.fromisoformat(value) <= end for value in dates)


def _matches_genres(event: Mapping[str, Any], filters: SearchFilters) -> bool:
    event_genre = str(event["ジャンル"])
    groups = filters.genre_groups or [[genre] for genre in filters.genres]
    return all(any(any(alias in event_genre for alias in GENRE_ALIASES[genre]) for genre in group) for group in groups)


def _matches_time(event: Mapping[str, Any], filters: SearchFilters) -> bool:
    times = re.findall(r"(\d{1,2}):(\d{2})", str(event["日時"]))
    if not times:
        return True
    start_minutes = int(times[0][0]) * 60 + int(times[0][1])
    end_minutes = int(times[-1][0]) * 60 + int(times[-1][1])
    for slot in filters.time_slots:
        if slot == "午前" and start_minutes >= 12 * 60:
            return False
        if slot == "午後" and end_minutes <= 12 * 60:
            return False
        if slot == "夕方" and end_minutes < 17 * 60:
            return False
    return filters.time_after is None or end_minutes >= filters.time_after


def _matches_hard(event: Mapping[str, Any], filters: SearchFilters) -> bool:
    if filters.dates and not _matches_dates(event, filters.dates):
        return False
    event_city_name = event_city(event)
    if filters.city_groups and not all(event_city_name in group for group in filters.city_groups):
        return False
    if filters.region_groups and not all(event_region(event) in group for group in filters.region_groups):
        return False
    if filters.genre_groups or filters.genres:
        if not _matches_genres(event, filters):
            return False
    if filters.child_friendly is True and event["子ども向け"] is not True:
        return False
    if filters.venue == "屋内" and event["屋内/屋外"] != "屋内":
        return False
    if filters.venue == "屋外" and "屋外" not in str(event["屋内/屋外"]):
        return False
    if filters.rain_preferred and "屋内" not in str(event["屋内/屋外"]):
        return False
    if filters.entry_free is True and not is_entry_free(str(event["料金"])):
        return False
    if filters.paid_only and is_entry_free(str(event["料金"])):
        return False
    if filters.max_entry_fee is not None and base_entry_fee(str(event["料金"])) > filters.max_entry_fee:
        return False
    if filters.reservation_required is not None:
        required = str(event.get("参加案内", {}).get("申込要否", ""))
        if filters.reservation_required is False and required != "不要":
            return False
        if filters.reservation_required is True and required != "必要":
            return False
    return _matches_time(event, filters)


def _date_distance(event: Mapping[str, Any], reference_date: date) -> int:
    start, end = parse_event_dates(str(event["日時"]))
    if start <= reference_date <= end:
        return 0
    if start > reference_date:
        return (start - reference_date).days
    return 10_000 + (reference_date - end).days


def _ranking_score(event: Mapping[str, Any], filters: SearchFilters, reference_date: date) -> int:
    score = _soft_score(event, filters)
    if filters.dates:
        score += 20
    if filters.city_groups:
        score += 12
    if filters.region_groups:
        score += 10
    if filters.child_friendly:
        score += 5
    if filters.venue:
        score += 5
    if filters.rain_preferred:
        score += 8 if str(event["屋内/屋外"]) == "屋内" else 2
    if filters.entry_free:
        score += 5
    if filters.paid_only:
        score += 5
    if filters.max_entry_fee is not None:
        score += 3
    return score + max(0, 12 - min(_date_distance(event, reference_date), 12))


def _search_with_filters(source_events: list[dict[str, Any]], filters: SearchFilters, reference_date: date, limit: int) -> tuple[list[dict[str, Any]], int]:
    matched: list[tuple[int, int, dict[str, Any]]] = []
    for source_index, event in enumerate(source_events):
        if not _matches_hard(event, filters):
            continue
        soft_score = _soft_score(event, filters)
        if filters.soft_terms and (soft_score <= 0 or not _matches_soft_terms(event, filters)):
            continue
        matched.append((_ranking_score(event, filters, reference_date), source_index, event))
    matched.sort(key=lambda item: (-item[0], _date_distance(item[2], reference_date), item[1]))
    safe_limit = min(max(limit, 0), MAX_SEARCH_RESULTS)
    return [event for _, _, event in matched[:safe_limit]], len(matched)


def _copy_filters(filters: SearchFilters) -> SearchFilters:
    return SearchFilters(**filters.to_dict())


def _relaxation_candidates(source_events: list[dict[str, Any]], filters: SearchFilters, reference_date: date, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    relaxable: list[tuple[str, str]] = []
    if filters.genre_groups or filters.genres:
        relaxable.append(("genre_groups", "ジャンル"))
    if filters.child_friendly:
        relaxable.append(("child_friendly", "子ども向け"))
    if filters.venue:
        relaxable.append(("venue", "屋内外"))
    if filters.rain_preferred:
        relaxable.append(("rain_preferred", "雨対応"))
    if filters.entry_free:
        relaxable.append(("entry_free", "無料条件"))
    if filters.paid_only:
        relaxable.append(("paid_only", "有料条件"))
    if filters.max_entry_fee is not None:
        relaxable.append(("max_entry_fee", "料金上限"))
    if filters.time_slots or filters.time_after is not None:
        relaxable.append(("time_slots", "時間帯"))
    for field_name, label in relaxable:
        relaxed = _copy_filters(filters)
        if field_name == "genre_groups":
            relaxed.genres = []
            relaxed.genre_groups = []
        elif field_name == "time_slots":
            relaxed.time_slots = []
            relaxed.time_after = None
        elif field_name == "rain_preferred":
            relaxed.rain_preferred = False
        elif field_name in {"child_friendly", "entry_free"}:
            setattr(relaxed, field_name, None)
        elif field_name == "paid_only":
            relaxed.paid_only = False
        else:
            setattr(relaxed, field_name, None)
        candidates, _ = _search_with_filters(source_events, relaxed, reference_date, limit)
        if candidates:
            return candidates, label
    return [], None


def classify_intent(query: str, *, _parsed_hint: bool = False, parsed_values: tuple[list[str], str | None] | None = None) -> str:
    normalized = normalize_query(query)
    compact = _compact_intent_text(normalized)
    intent_texts = (normalized, compact)
    if any(re.search(pattern, text, flags=re.IGNORECASE) for text in intent_texts for pattern in INJECTION_PATTERNS):
        return "injection"
    if any(term in text for text in intent_texts for term in NEAR_TERMS) and query_city(normalized) is None:
        return "needs_location"
    if normalized in {"地域から探す", "地域で探す", "地域"} or compact in {"地域から探す", "地域で探す", "地域"}:
        return "needs_region"
    if any(re.search(pattern, text) for text in intent_texts for pattern in OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"
    if any(re.search(pattern, normalized) for pattern in _COUNT_PATTERNS):
        return "count"
    requested = parsed_values[1] if _parsed_hint and parsed_values else _requested_field(normalized)
    soft = parsed_values[0] if _parsed_hint and parsed_values else _extract_soft_terms(normalized)
    if requested:
        return "attribute" if _REFERENCE_RE.search(normalized) and not soft else "lookup" if soft else "attribute"
    return "refine" if any(term in normalized for term in _REFINE_TERMS) else "discover"


def looks_like_event_query(query: str) -> bool:
    intent = classify_intent(query)
    if intent in {"injection", "out_of_scope", "needs_location", "needs_region"}:
        return False
    # Age-oriented discovery is intentionally left for the bounded Agentic
    # Search planner when the legacy parser has no dedicated age field yet.
    normalized = normalize_query(query)
    if age_semantics.query_age_semantics(normalized).recognized:
        return True
    filters = parse_query(query)
    return bool(filters.dates or filters.city_groups or filters.region_groups or filters.genre_groups or filters.child_friendly or filters.age is not None or filters.age_group or filters.venue or filters.reservation_required is not None or filters.rain_preferred or filters.entry_free or filters.paid_only or filters.max_entry_fee is not None or filters.time_slots or filters.time_after is not None or filters.soft_terms or filters.requested_field or intent in {"count", "refine", "attribute"})


def asks_for_nearby(query: str) -> bool:
    return classify_intent(query) == "needs_location"


def is_refinement_query(query: str) -> bool:
    normalized = normalize_query(query)
    return any(term in normalized for term in _REFINE_TERMS)


def resolve_reference_index(query: str, result_count: int) -> int | None:
    normalized = normalize_query(query)
    if "最初" in normalized or "第一" in normalized:
        return 0
    if "最後" in normalized or "最后" in normalized:
        return result_count - 1 if result_count else None
    match = re.search(r"第?(\d+|一|二|三)番目", normalized)
    if match:
        value = {"一": 1, "二": 2, "三": 3}.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 0)
        return value - 1 if 1 <= value <= result_count else None
    return None


def attribute_answer(event: Mapping[str, Any], requested_field: str) -> str:
    name = str(event["イベント名"])
    answers = {
        "datetime": f"「{name}」は、{event['日時']}に開催予定です。",
        "place": f"「{name}」の場所は、{event['場所']}です。",
        "fee": f"「{name}」の料金は、{event['料金']}です。",
        "genre": f"「{name}」のジャンルは、{event['ジャンル']}です。",
        "child_friendly": f"「{name}」は、{'子ども向け' if event['子ども向け'] else '一般向け'}です。",
        "venue": f"「{name}」の会場は、{event['屋内/屋外']}です。",
        "overview": f"「{name}」は、{event['概要']}",
    }
    return answers.get(requested_field, "詳しい内容は、表示されたイベントカードを確認してみてください。")


def _confidence(events: list[dict[str, Any]], filters: SearchFilters) -> str:
    if not events:
        return "none"
    if filters.requested_field and len(events) == 1 and filters.soft_terms:
        return "high"
    return "high" if len(events) == 1 else "medium"


def search_events(query: str, events: Iterable[dict[str, Any]] | None = None, reference_date: date = POC_REFERENCE_DATE, *, previous_filters: Mapping[str, Any] | None = None, inherit_previous: bool = False, limit: int = MAX_SEARCH_RESULTS) -> SearchResult:
    if not isinstance(query, str):
        raise TypeError("検索語は文字列で指定してください。")
    normalized = normalize_query(query)
    intent = classify_intent(normalized)
    if intent == "injection":
        return SearchResult([], SearchFilters(intent=intent), intent, "イベントを新しく作ることはできんのよ。掲載済みの架空イベントから探してみる？")
    if intent == "needs_location":
        return SearchResult([], SearchFilters(intent=intent), intent, "今どの市町あたりにおる？ 市町名を教えてくれたら、その地域で探すよ。")
    if intent == "needs_region":
        return SearchResult([], SearchFilters(intent=intent), intent, "東予・中予・南予のうち、どの地域で探してみる？")
    if intent == "out_of_scope":
        return SearchResult([], SearchFilters(intent=intent), intent, "このPoCは文化祭イベントを探す機能の検証が中心なんよ。関連するイベントなら探せるよ。")
    filters = parse_query(normalized, reference_date)
    if inherit_previous or is_refinement_query(normalized):
        filters = merge_filters(filters, previous_filters)
    filters.intent = intent if intent not in {"discover", "refine"} else filters.intent
    source_events = load_events() if events is None else list(events)
    if filters.invalid_date and not filters.dates:
        return SearchResult([], filters, "no_results", "その日付は確認できんかったよ。別の日付で探してみて。")
    has_condition = bool(filters.dates or filters.city_groups or filters.region_groups or filters.genre_groups or filters.child_friendly or filters.venue or filters.rain_preferred or filters.entry_free or filters.paid_only or filters.max_entry_fee is not None or filters.time_slots or filters.time_after is not None or filters.soft_terms)
    if not has_condition:
        return SearchResult([], filters, "needs_condition", "いつ頃・どの地域のイベントを探しよる？ 「今日」「今週末」のように教えてみて。")
    selected, total = _search_with_filters(source_events, filters, reference_date, limit)
    if selected:
        return SearchResult(selected, filters, "search", total_matches=total, confidence=_confidence(selected, filters))
    near_matches, relaxed_condition = _relaxation_candidates(source_events, filters, reference_date, limit)
    message = f"ぴったりの条件では見つからんかったよ。「{relaxed_condition}」を外すと候補があるけん、見てみる？" if near_matches and relaxed_condition else "その条件にぴったり合うイベントは見つからんかったよ。条件を少し変えて探してみる？"
    return SearchResult([], filters, "no_results", message, near_matches=near_matches, relaxed_condition=relaxed_condition)

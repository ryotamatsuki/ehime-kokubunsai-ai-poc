"""Configuration for the independent cultural-event PoC.

Only deterministic local configuration lives here.  The search layer does
not depend on an LLM, an embedding model, a vector database, RAG, or a
network service.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


POC_REFERENCE_DATE = date(2028, 11, 3)
POC_REFERENCE_DATE_TEXT = "2028年11月3日"
MAX_EVENT_CANDIDATES = 8
MAX_SEARCH_RESULTS = MAX_EVENT_CANDIDATES
EVENT_DATA_PATH = Path(__file__).resolve().parent / "data" / "events.json"
SEARCH_METADATA_PATH = Path(__file__).resolve().parent / "data" / "search_metadata.json"


# The 20 municipalities represented by the attached 30-event dataset.
REGION_CITIES: dict[str, tuple[str, ...]] = {
    "東予": (
        "今治市",
        "新居浜市",
        "西条市",
        "四国中央市",
        "上島町",
    ),
    "中予": (
        "松山市",
        "伊予市",
        "東温市",
        "久万高原町",
        "松前町",
        "砥部町",
    ),
    "南予": (
        "宇和島市",
        "八幡浜市",
        "大洲市",
        "西予市",
        "内子町",
        "伊方町",
        "松野町",
        "鬼北町",
        "愛南町",
    ),
}

# Backward-compatible names used by the existing PoC UI and its isolation
# checks.  Frozensets prevent accidental mutation of the region table.
REGION_MUNICIPALITIES: dict[str, frozenset[str]] = {
    region: frozenset(cities) for region, cities in REGION_CITIES.items()
}


MUNICIPALITY_ALIASES: dict[str, str] = {
    alias: municipality
    for municipality in (city for cities in REGION_CITIES.values() for city in cities)
    for alias in (municipality, municipality.removesuffix("市").removesuffix("町"))
}

# Explicit aliases make the accepted query vocabulary visible and stable.
MUNICIPALITY_ALIASES.update(
    {
        "久万高原": "久万高原町",
        "四国中央": "四国中央市",
        "上島": "上島町",
    }
)

REGION_ALIASES = {region: region for region in REGION_CITIES}

# Compatibility names used by the deterministic search module.
EVENTS_PATH = EVENT_DATA_PATH
CITY_ALIASES = MUNICIPALITY_ALIASES
MAX_SEARCH_RESULTS = MAX_EVENT_CANDIDATES


# A genre label is an OR over its aliases.  Two different labels in a query
# are ANDed by event_search.py.  The broad 伝統文化 label is intentionally
# explicit and does not match every occurrence of the word 文化.
GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "伝統文化": (
        "祭り",
        "民俗",
        "伝統芸能",
        "歴史",
        "生活文化",
        "伝統工芸",
        "工芸",
        "信仰",
        "遍路",
    ),
    "伝統芸能": (
        "伝統芸能",
        "民俗芸能",
        "語り芸",
        "芝居",
    ),
    "祭り": ("祭り", "祭礼"),
    "民俗": ("民俗", "祭り"),
    "歴史": ("歴史", "産業遺産", "文化財", "建築", "城下町"),
    "工芸": ("工芸", "陶芸", "水引", "紙", "タオル"),
    "アート": ("アート", "美術", "芸術", "デザイン"),
    "文学": ("文学", "俳句", "ことば"),
    "食文化": ("食文化", "農文化"),
    "演劇": ("演劇", "芝居", "舞台芸術", "語り芸"),
    "自然": ("自然", "海洋文化", "山里", "水"),
    "海洋文化": ("海洋文化",),
    "学習": ("学習", "体験", "ワークショップ"),
}


KNOWN_KEYWORDS: tuple[str, ...] = (
    "村上海賊",
    "別子銅山",
    "五反田柱祭り",
    "西条まつり",
    "砥部焼",
    "子規",
    "俳句",
    "太鼓台",
    "牛鬼",
    "水引",
    "タオル",
    "道後温泉",
    "道後",
    "うちぬき",
    "城下町",
    "遍路",
    "内子座",
    "民話",
    "文化財",
    "ウォーク",
    "海洋",
    "陶芸",
    "紙文化",
    "紙",
    "アート",
    "芝居",
)

CHILD_TERMS: tuple[str, ...] = (
    "子ども",
    "子供",
    "こども",
    "小学生",
    "小学",
    "親子",
    "家族",
    "ファミリー",
)
INDOOR_TERMS: tuple[str, ...] = ("屋内", "室内")
OUTDOOR_TERMS: tuple[str, ...] = ("屋外", "外で")
RAIN_TERMS: tuple[str, ...] = ("雨", "雨天", "雨の日", "雨でも")
FREE_TERMS: tuple[str, ...] = ("無料", "タダ", "ただ", "0円")
PAID_TERMS: tuple[str, ...] = ("有料", "料金がかかる", "お金がかかる")
NEAR_TERMS: tuple[str, ...] = ("近く", "近い", "近場", "周辺")

INJECTION_PATTERNS: tuple[str, ...] = (
    r"指示を無視",
    r"命令を無視",
    r"プロンプト.*無視",
    r"system\s*prompt",
    r"架空イベントを(?:新しく)?作",
    r"存在しないイベント",
    r"イベントを(?:新しく)?作(?:って|れ)",
)

OUT_OF_SCOPE_PATTERNS: tuple[str, ...] = (
    r"\d[\d,]*字で.*(?:説明|解説)",
    r"(?:正岡子規|牛鬼).*詳し",
    r"歴史を(?:詳しく|詳細に|教えて)",
)


def municipality_region(municipality: str) -> str | None:
    """Return 東予・中予・南予 for a canonical municipality."""

    return next(
        (
            region
            for region, municipalities in REGION_MUNICIPALITIES.items()
            if municipality in municipalities
        ),
        None,
    )

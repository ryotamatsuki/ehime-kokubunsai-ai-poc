"""Configuration shared by the PoC UI and deterministic event search."""

from __future__ import annotations

from datetime import date
from pathlib import Path


POC_REFERENCE_DATE = date(2028, 11, 3)
POC_REFERENCE_DATE_TEXT = "2028年11月3日"
MAX_EVENT_CANDIDATES = 8
EVENT_DATA_PATH = Path(__file__).resolve().parent / "data" / "events.json"


# The PoC uses the conventional three-region grouping for Ehime.
REGION_MUNICIPALITIES: dict[str, frozenset[str]] = {
    "東予": frozenset({"今治市", "新居浜市", "西条市", "四国中央市", "上島町"}),
    "中予": frozenset({"松山市", "伊予市", "東温市", "松前町", "砥部町", "久万高原町"}),
    "南予": frozenset({
        "大洲市",
        "内子町",
        "八幡浜市",
        "伊方町",
        "西予市",
        "宇和島市",
        "松野町",
        "鬼北町",
        "愛南町",
    }),
}

MUNICIPALITY_ALIASES: dict[str, str] = {
    municipality.removesuffix("市").removesuffix("町"): municipality
    for municipalities in REGION_MUNICIPALITIES.values()
    for municipality in municipalities
}

# Common natural-language variants used by the PoC test prompts.
MUNICIPALITY_ALIASES.update(
    {
        "松山": "松山市",
        "今治": "今治市",
        "新居浜": "新居浜市",
        "西条": "西条市",
        "四国中央": "四国中央市",
        "上島": "上島町",
        "伊予": "伊予市",
        "東温": "東温市",
        "松前": "松前町",
        "砥部": "砥部町",
        "久万高原": "久万高原町",
        "大洲": "大洲市",
        "内子": "内子町",
        "八幡浜": "八幡浜市",
        "伊方": "伊方町",
        "西予": "西予市",
        "宇和島": "宇和島市",
        "松野": "松野町",
        "鬼北": "鬼北町",
        "愛南": "愛南町",
    }
)

REGION_ALIASES = {"東予": "東予", "中予": "中予", "南予": "南予"}


def municipality_region(municipality: str) -> str | None:
    """Return the region for a canonical municipality name."""

    for region, municipalities in REGION_MUNICIPALITIES.items():
        if municipality in municipalities:
            return region
    return None

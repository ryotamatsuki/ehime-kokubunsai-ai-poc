from datetime import date

from event_search import load_events, search_events


REFERENCE_DATE = date(2028, 11, 3)


def names(query: str):
    return [event["イベント名"] for event in search_events(query, reference_date=REFERENCE_DATE)]


def event_by_name(name: str):
    return next(event for event in load_events() if event["イベント名"] == name)


def test_date_and_child_are_and_filters():
    result = names("11月3日に子どもと行けるイベント")
    assert len(result) == 6
    assert any("西条まつり" in item for item in result)
    assert any("卯之町" in item for item in result)


def test_today_includes_multi_day_events():
    result = names("今日やっているイベント")
    assert len(result) == 6
    assert any("別子銅山" in item for item in result)
    assert any("みんなのアート" in item for item in result)


def test_tomorrow_is_fixed_relative_to_poc_date():
    result = names("明日のイベント")
    assert len(result) == 6
    assert any("水の都・西条" in item for item in result)


def test_rain_means_indoor():
    result = names("雨でも楽しめるもの")
    assert result
    assert all("屋内" in event_by_name(name)["屋内/屋外"] for name in result)


def test_municipality_and_free_are_and_filters():
    result = names("松山で無料")
    assert len(result) == 4
    assert all("松山" in event_by_name(name)["場所"] for name in result)


def test_region_and_traditional_culture_are_and_filters():
    result = names("南予で伝統文化")
    assert result
    south = ("大洲市", "内子町", "八幡浜市", "伊方町", "西予市", "宇和島市", "松野町", "鬼北町", "愛南町")
    assert all(any(city in event_by_name(name)["場所"] for city in south) for name in result)


def test_child_query_handles_school_grade():
    result = names("小学3年生と楽しめるもの")
    assert result
    assert all(event_by_name(name)["子ども向け"] for name in result)


def test_indoor_and_price_cap():
    result = names("屋内で500円以内")
    assert result
    assert all("屋内" in event_by_name(name)["屋内/屋外"] for name in result)
    assert not any("砥部焼" in item for item in result)


def test_keyword_query_finds_tobe_ware():
    result = names("砥部焼に興味がある")
    assert len(result) == 1
    assert "砥部焼" in result[0]


def test_free_traditional_performance_excludes_partial_free_events():
    result = names("無料の伝統芸能")
    assert result
    assert all("無料（一部" not in event_by_name(name)["料金"] for name in result)
    assert all("無料" in event_by_name(name)["料金"] for name in result)


def test_no_match_is_empty_and_unknown_event_is_not_created():
    assert search_events("不存在の架空イベント", reference_date=REFERENCE_DATE) == []

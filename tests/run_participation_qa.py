"""Standalone QA for v2 participation, FAQ, and recommendation behavior.

This runner intentionally uses only the Python standard library plus the local
application modules.  It is the second deterministic gate after Search v2.
"""

from __future__ import annotations

from datetime import date
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_details import (  # noqa: E402
    answer_event_detail,
    compact_participation_lines,
    detect_detail_field,
    normalize_schedule,
    validate_events_v2,
)
from event_recommendation import (  # noqa: E402
    recommend_next_events,
    recommend_similar_events,
)
from event_search import load_events, search_events  # noqa: E402
from faq_search import find_faq, load_faq  # noqa: E402


def event_by_id(events: list[dict[str, object]], event_id: str) -> dict[str, object]:
    return next(event for event in events if event["id"] == event_id)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


events = load_events()
check(len(events) == 30, "events.json must contain 30 v2 events")
validate_events_v2(events)
check(len(load_faq()) == 8, "general_faq.json must contain eight FAQs")

metadata = json.loads((ROOT / "data" / "search_metadata.json").read_text(encoding="utf-8"))
check({item["id"] for item in metadata} == {event["id"] for event in events}, "metadata IDs must match events")
check(sum(len(item["aliases"]) for item in metadata) >= 200, "merged aliases were not retained")
check(sum(len(item["search_tags"]) for item in metadata) >= 250, "merged search tags were not retained")

# The common v2 fields stay structurally present and the event display facts
# remain the same shape as the original nine-field contract.
required = {
    "id", "データ区分", "イベント名", "aliases", "search_tags", "日時",
    "start_datetime", "end_datetime", "市町", "地域", "場所", "ジャンル",
    "子ども向け", "屋内/屋外", "料金", "料金構造", "概要", "参加案内",
    "アクセス", "雨天時対応", "バリアフリー", "問い合わせ", "公式URL",
}
check(all(required <= set(event) for event in events), "v2 fields are incomplete")
check(all(event["データ区分"] == "PoC架空" for event in events), "non-fictional data slipped in")

period = event_by_id(events, "007")
schedule = normalize_schedule(period)
check(schedule.to_dict() == {
    "start_date": "2028-10-21",
    "end_date": "2028-11-26",
    "daily_start_time": "09:30",
    "daily_end_time": "17:00",
}, "period event was not normalized to a daily schedule")

opening = event_by_id(events, "001")
check(detect_detail_field("オープニングは予約いる？") == "application_required", "application intent not detected")
opening_answer = answer_event_detail(opening, "application_required", "予約いる？")
check("申込が必要" in opening_answer and "2028-10-14" in opening_answer, "opening application answer is not grounded")

saijo = event_by_id(events, "008")
check("駐車場は、なし" in answer_event_detail(saijo, "parking"), "parking answer is not grounded")
check("雨天決行・一部変更あり" in answer_event_detail(saijo, "rain_policy"), "rain answer is not grounded")
check("一部可" in answer_event_detail(saijo, "wheelchair"), "wheelchair answer is not grounded")
check(any("申込：" in line for line in compact_participation_lines(saijo)), "participation expander has no application line")

tobe = event_by_id(events, "022")
fee_answer = answer_event_detail(tobe, "fee_detail", "絵付けも無料？")
check("入場無料" in fee_answer and "1,000円" in fee_answer, "paid add-on was not distinguished from free entry")
check("完全無料" not in fee_answer, "fee answer hallucinated complete free admission")

undecided = event_by_id(events, "019")
check("このPoCデータでは未定です。" in answer_event_detail(undecided, "application_required"), "未定 application was guessed")
check("このPoCデータでは未定です。" in answer_event_detail(undecided, "rain_policy"), "未定 rain policy was guessed")
check("この項目は登録されていません。" in answer_event_detail(undecided, "accessible_toilet"), "null accessibility was guessed")

# Local FAQ matching is deterministic and does not require a web/RAG call.
for query, faq_id in (
    ("このイベントは公式？", "faq-001"),
    ("今日っていつ？", "faq-002"),
    ("近くで探したい", "faq-003"),
    ("予約って必要？", "faq-004"),
    ("雨でも開催？", "faq-005"),
    ("車いすで行ける？", "faq-006"),
    ("体験も無料？", "faq-007"),
    ("このあと何か行ける？", "faq-008"),
):
    match = find_faq(query)
    check(match is not None and match.faq_id == faq_id, f"FAQ mismatch: {query}")

# Recommendation is calculated from schedule/city/region/tags, not an
# event-specific hand-written mapping.
俳句 = event_by_id(events, "002")
next_result = recommend_next_events(俳句, events, date(2028, 10, 22))
check([event["id"] for event in next_result.events] == ["003", "028"], "next-event ranking missed start/drop-in candidates")
check("簡易移動バッファ" in next_result.message, "next-event explanation omitted its assumption")

today_result = recommend_next_events(period, events, date(2028, 11, 3))
check(not today_result.events, "past/overlapping same-day event was recommended")
check("候補は見つかりませんでした" in today_result.message, "no-next-event answer is unclear")

similar = recommend_similar_events(tobe, events, date(2028, 11, 3))
check(all(event["id"] != "022" for event in similar.events), "similar recommendation included itself")
check(all(event["id"] in {item["id"] for item in events} for event in similar.events), "similar recommendation invented an event")

# Modal receives only the pre-existing safe display fields, never v2 nested
# participation/access/contact data.
streamlit_source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
tree = ast.parse(streamlit_source)
llm_fn = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_llm_candidates"
)
llm_text = ast.get_source_segment(streamlit_source, llm_fn) or ""
check("safe_fields" in llm_text and '"料金"' in llm_text, "Modal candidate whitelist is missing")
check("参加案内" not in llm_text and "アクセス" not in llm_text and "問い合わせ" not in llm_text, "nested v2 facts leaked to Modal")
check('key != "公式URL"' not in llm_text, "Modal candidate filtering is not an explicit whitelist")

# Search v2 still grounds every result in this source set.
source_ids = {event["id"] for event in events}
for query in ("今日のイベント", "無料", "砥部焼", "雨でも楽しめるもの"):
    result = search_events(query, events)
    check({event["id"] for event in result.events} <= source_ids, f"source leakage: {query}")

print("Participation/FAQ/Recommendation QA: PASS")
print(f"Events: {len(events)}; FAQs: {len(load_faq())}; metadata aliases: {sum(len(item['aliases']) for item in metadata)}; tags: {sum(len(item['search_tags']) for item in metadata)}")

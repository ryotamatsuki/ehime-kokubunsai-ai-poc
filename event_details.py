"""Structured participation facts for the cultural-event PoC.

This module is deliberately deterministic.  It is the only place used by the
UI for event-specific participation answers; Modal never receives these
fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import re
from typing import Any, Mapping

from app_config import REGION_CITIES


V2_FIELDS = frozenset(
    {
        "id",
        "データ区分",
        "aliases",
        "search_tags",
        "start_datetime",
        "end_datetime",
        "市町",
        "地域",
        "料金構造",
        "参加案内",
        "アクセス",
        "雨天時対応",
        "バリアフリー",
        "問い合わせ",
    }
)
DETAIL_FIELDS = (
    "application_required",
    "application_deadline",
    "capacity",
    "target",
    "parking",
    "public_transport",
    "rain_policy",
    "wheelchair",
    "accessible_toilet",
    "sign_language",
    "contact",
    "fee_detail",
)


@dataclass(frozen=True)
class EventSchedule:
    """A daily operating window, not a continuous datetime interval."""

    start_date: date
    end_date: date
    daily_start_time: time
    daily_end_time: time

    def active_on(self, day: date) -> bool:
        return self.start_date <= day <= self.end_date

    def starts_at(self, day: date) -> datetime:
        return datetime.combine(day, self.daily_start_time)

    def ends_at(self, day: date) -> datetime:
        end_day = day
        if self.daily_end_time < self.daily_start_time:
            end_day = day + timedelta(days=1)
        return datetime.combine(end_day, self.daily_end_time)

    def to_dict(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "daily_start_time": self.daily_start_time.strftime("%H:%M"),
            "daily_end_time": self.daily_end_time.strftime("%H:%M"),
        }


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def normalize_schedule(event: Mapping[str, Any]) -> EventSchedule:
    """Normalize v2 ISO datetimes into a daily schedule."""

    start = _parse_datetime(str(event["start_datetime"]))
    end = _parse_datetime(str(event["end_datetime"]))
    if end < start:
        raise ValueError(f"イベントの日時範囲が不正です: {event.get('id')}")
    return EventSchedule(
        start_date=start.date(),
        end_date=end.date(),
        daily_start_time=start.time().replace(tzinfo=None),
        daily_end_time=end.time().replace(tzinfo=None),
    )


def _is_nullable_string(value: Any) -> bool:
    return value is None or (isinstance(value, str) and bool(value.strip()))


def validate_event_v2(event: Mapping[str, Any], index: int = 1) -> None:
    """Validate the v2 contract, including nested participation objects."""

    required = V2_FIELDS | {
        "イベント名",
        "日時",
        "場所",
        "ジャンル",
        "子ども向け",
        "屋内/屋外",
        "料金",
        "概要",
        "公式URL",
    }
    missing = required - set(event)
    if missing:
        raise ValueError(f"events.json {index}件目の必須項目が不足: {sorted(missing)}")
    for field in ("イベント名", "日時", "場所", "ジャンル", "料金", "概要", "公式URL"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"events.json {index}件目の{field}が不正です")
    if not str(event["イベント名"]).startswith("【PoC架空】"):
        raise ValueError(f"events.json {index}件目のイベント名に架空表示がありません")
    if not re.fullmatch(r"\d{3}", str(event["id"])):
        raise ValueError(f"events.json {index}件目のidが不正です")
    if event["データ区分"] != "PoC架空":
        raise ValueError(f"events.json {index}件目のデータ区分が不正です")
    if not isinstance(event["aliases"], list) or not all(
        isinstance(value, str) and value.strip() for value in event["aliases"]
    ):
        raise ValueError(f"events.json {index}件目のaliasesが不正です")
    if not isinstance(event["search_tags"], list) or not all(
        isinstance(value, str) and value.strip() for value in event["search_tags"]
    ):
        raise ValueError(f"events.json {index}件目のsearch_tagsが不正です")
    if not isinstance(event["子ども向け"], bool):
        raise ValueError(f"events.json {index}件目の子ども向けが不正です")
    if event["地域"] not in REGION_CITIES:
        raise ValueError(f"events.json {index}件目の地域が不正です")
    if event["市町"] not in REGION_CITIES[event["地域"]]:
        raise ValueError(f"events.json {index}件目の市町と地域が不一致です")
    if event["屋内/屋外"] not in {"屋内", "屋外", "屋内・屋外"}:
        raise ValueError(f"events.json {index}件目の屋内外が不正です")
    if not str(event["公式URL"]).startswith("https://example.invalid/"):
        raise ValueError(f"events.json {index}件目のURLがPoC用ではありません")
    if not isinstance(event["start_datetime"], str) or not isinstance(event["end_datetime"], str):
        raise ValueError(f"events.json {index}件目の日時型が不正です")
    normalize_schedule(event)

    fee = event["料金構造"]
    if not isinstance(fee, dict):
        raise ValueError(f"events.json {index}件目の料金構造が不正です")
    if fee.get("入場料種別") not in {"無料", "入場無料・一部有料", "有料", "有料・年齢別"}:
        raise ValueError(f"events.json {index}件目の入場料種別が不正です")
    if not isinstance(fee.get("一般料金円"), int) or not isinstance(
        fee.get("子ども料金円"), int
    ):
        raise ValueError(f"events.json {index}件目の料金型が不正です")
    if not isinstance(fee.get("追加有料体験"), bool):
        raise ValueError(f"events.json {index}件目の追加有料体験が不正です")
    if fee.get("追加料金円") is not None and not isinstance(fee.get("追加料金円"), int):
        raise ValueError(f"events.json {index}件目の追加料金円が不正です")

    guide = event["参加案内"]
    if not isinstance(guide, dict) or guide.get("申込要否") not in {"必要", "不要", "未定"}:
        raise ValueError(f"events.json {index}件目の申込要否が不正です")
    if not _is_nullable_string(guide.get("申込期限")):
        raise ValueError(f"events.json {index}件目の申込期限が不正です")
    if guide.get("定員") is not None and not isinstance(guide.get("定員"), int):
        raise ValueError(f"events.json {index}件目の定員が不正です")
    for key in ("対象", "対象年齢", "備考"):
        if not _is_nullable_string(guide.get(key)):
            raise ValueError(f"events.json {index}件目の参加案内.{key}が不正です")

    access = event["アクセス"]
    if not isinstance(access, dict) or access.get("駐車場") not in {"あり", "台数限定", "なし"}:
        raise ValueError(f"events.json {index}件目の駐車場が不正です")
    if not isinstance(access.get("公共交通"), str) or not _is_nullable_string(access.get("駐車場備考")):
        raise ValueError(f"events.json {index}件目のアクセスが不正です")

    rain = event["雨天時対応"]
    if not isinstance(rain, dict) or rain.get("開催方針") not in {
        "雨天決行",
        "雨天決行・一部変更あり",
        "小雨決行・荒天中止",
        "未定",
    }:
        raise ValueError(f"events.json {index}件目の雨天時対応が不正です")
    if rain.get("屋外企画変更") is not None and not isinstance(rain.get("屋外企画変更"), bool):
        raise ValueError(f"events.json {index}件目の屋外企画変更が不正です")

    access_fields = event["バリアフリー"]
    if not isinstance(access_fields, dict) or access_fields.get("車いす") not in {"可", "一部可", "不可"}:
        raise ValueError(f"events.json {index}件目の車いす情報が不正です")
    for key in ("多目的トイレ", "手話通訳"):
        if access_fields.get(key) is not None and not isinstance(access_fields.get(key), bool):
            raise ValueError(f"events.json {index}件目の{key}が不正です")

    contact = event["問い合わせ"]
    if not isinstance(contact, dict) or not _is_nullable_string(contact.get("窓口")):
        raise ValueError(f"events.json {index}件目の問い合わせが不正です")
    for key in ("電話", "メール", "備考"):
        if not _is_nullable_string(contact.get(key)):
            raise ValueError(f"events.json {index}件目の問い合わせ.{key}が不正です")


def validate_events_v2(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list) or len(events) != 30:
        raise ValueError("events.json は30件の配列である必要があります。")
    ids: set[str] = set()
    names: set[str] = set()
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError(f"events.json {index}件目がオブジェクトではありません")
        validate_event_v2(event, index)
        if event["id"] in ids or event["イベント名"] in names:
            raise ValueError(f"events.json {index}件目に重複があります")
        ids.add(event["id"])
        names.add(event["イベント名"])
    return events


def _missing_value(value: Any) -> str | None:
    if value == "未定":
        return "このPoCデータでは未定です。"
    if value is None:
        return "この項目は登録されていません。"
    return None


def detect_detail_field(query: str) -> str | None:
    """Map participation questions to a structured field."""

    if any(term in query for term in ("絵付け", "体験も無料", "入場無料", "無料で入", "追加料金", "料金構造", "料金", "いくら")):
        return "fee_detail"
    if any(term in query for term in ("申込期限", "申し込み期限", "いつまでに申し込", "締切", "締め切り")):
        return "application_deadline"
    if any(term in query for term in ("定員", "何人", "人数")):
        return "capacity"
    if any(term in query for term in ("予約", "申込", "申し込", "参加申込")):
        return "application_required"
    if any(term in query for term in ("駐車場", "車を停め", "車で行")):
        return "parking"
    if any(term in query for term in ("公共交通", "電車", "バス", "アクセス")):
        return "public_transport"
    if any(term in query for term in ("雨", "荒天", "天候")):
        return "rain_policy"
    if any(term in query for term in ("車いす", "車椅子")):
        return "wheelchair"
    if any(term in query for term in ("多目的トイレ", "バリアフリーのトイレ")):
        return "accessible_toilet"
    if "手話" in query:
        return "sign_language"
    if any(term in query for term in ("問い合わせ", "連絡先", "電話", "メール")):
        return "contact"
    if any(term in query for term in ("誰でも", "子どもでも", "子供でも", "小学生でも", "対象年齢", "参加できる")):
        return "target"
    return None


def _format_bool(value: Any, positive: str, negative: str) -> str:
    missing = _missing_value(value)
    if missing:
        return missing
    return positive if value else negative


def _format_value(value: Any) -> str:
    missing = _missing_value(value)
    if missing:
        return missing
    return str(value)


def _fee_answer(event: Mapping[str, Any], query: str) -> str:
    fee = event["料金構造"]
    entry_type = str(fee["入場料種別"])
    fee_text = str(event["料金"])
    entry_only = any(term in query for term in ("入場無料", "無料で入", "入れる"))
    if entry_only:
        if entry_type in {"無料", "入場無料・一部有料"}:
            if fee["追加有料体験"]:
                return f"入場自体は無料です。ただし、追加の有料体験があります（{fee_text}）。"
            return "入場自体は無料です。"
        return f"入場は無料ではありません。料金は{fee_text}です。"
    if fee["追加有料体験"]:
        extra = fee.get("追加料金円")
        extra_text = f"追加の有料体験は{extra:,}円です。" if isinstance(extra, int) else "追加の有料体験がありますが、追加料金は未定です。"
        return f"{fee_text}。{extra_text}"
    return f"料金は{fee_text}です。"


def answer_event_detail(event: Mapping[str, Any], field: str, query: str = "") -> str:
    name = str(event["イベント名"])
    if not V2_FIELDS.issubset(event):
        return f"「{name}」の参加案内の詳細は、現在のデータに登録されていません。"
    guide = event["参加案内"]
    access = event["アクセス"]
    rain = event["雨天時対応"]
    barrier = event["バリアフリー"]
    contact = event["問い合わせ"]

    if field == "fee_detail":
        return f"「{name}」の{_fee_answer(event, query)}"
    if field == "application_required":
        value = guide["申込要否"]
        missing = _missing_value(value)
        if missing:
            return f"「{name}」の申込要否は、{missing}"
        answer = f"「{name}」は、申込が{'必要' if value == '必要' else '不要'}です。"
        deadline = guide.get("申込期限")
        if value == "必要" and deadline:
            answer += f"申込期限は{deadline}です。"
        return answer
    if field == "application_deadline":
        if guide["申込要否"] == "不要":
            return f"「{name}」は申込不要のため、申込期限はありません。"
        missing = _missing_value(guide.get("申込期限"))
        return f"「{name}」の申込期限は、{missing or str(guide['申込期限']) + 'です。'}"
    if field == "capacity":
        missing = _missing_value(guide.get("定員"))
        return f"「{name}」の定員は、{missing or str(guide['定員']) + '人です。'}"
    if field == "target":
        return f"「{name}」の対象は、{_format_value(guide.get('対象'))}（対象年齢：{_format_value(guide.get('対象年齢'))}）です。"
    if field == "parking":
        value = access.get("駐車場")
        answer = f"「{name}」の駐車場は、{_format_value(value)}です。"
        if access.get("駐車場備考"):
            answer += str(access["駐車場備考"])
        return answer
    if field == "public_transport":
        return f"「{name}」の公共交通については、{_format_value(access.get('公共交通'))}"
    if field == "rain_policy":
        value = rain.get("開催方針")
        missing = _missing_value(value)
        return f"「{name}」の雨天時の開催方針は、{missing or str(value) + 'です。'}"
    if field == "wheelchair":
        value = barrier.get("車いす")
        missing = _missing_value(value)
        return f"「{name}」の車いす対応は、{missing or str(value) + 'です。'}"
    if field == "accessible_toilet":
        return f"「{name}」の多目的トイレは、{_format_bool(barrier.get('多目的トイレ'), 'あります。', 'ありません。')}"
    if field == "sign_language":
        return f"「{name}」の手話通訳は、{_format_bool(barrier.get('手話通訳'), 'あります。', 'ありません。')}"
    if field == "contact":
        values = [str(contact[key]) for key in ("窓口", "電話", "メール") if contact.get(key)]
        if not values:
            return f"「{name}」の問い合わせ先は、この項目は登録されていません。"
        return f"「{name}」の問い合わせ先は、{'／'.join(values)}です。"
    return f"「{name}」の参加案内は、表示されたカードを確認してみてください。"


def compact_participation_lines(event: Mapping[str, Any]) -> list[str]:
    guide = event["参加案内"]
    access = event["アクセス"]
    rain = event["雨天時対応"]
    barrier = event["バリアフリー"]
    return [
        f"申込：{_format_value(guide.get('申込要否'))}",
        f"対象：{_format_value(guide.get('対象'))}（{_format_value(guide.get('対象年齢'))}）",
        f"駐車場：{_format_value(access.get('駐車場'))}",
        f"雨天時：{_format_value(rain.get('開催方針'))}",
        f"バリアフリー：車いす {_format_value(barrier.get('車いす'))}／多目的トイレ {_format_bool(barrier.get('多目的トイレ'), 'あり', 'なし')}／手話通訳 {_format_bool(barrier.get('手話通訳'), 'あり', 'なし')}",
    ]

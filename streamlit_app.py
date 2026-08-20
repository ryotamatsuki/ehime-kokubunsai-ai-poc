"""Streamlit UI for the separate Ehime cultural-event PoC."""

from __future__ import annotations

from datetime import date
import hmac
import re
from urllib.parse import urlparse

import requests
import streamlit as st

import event_search
from app_config import (
    MAX_EVENT_CANDIDATES,
    POC_REFERENCE_DATE,
    POC_REFERENCE_DATE_TEXT,
)


PAGE_TITLE = "🎭 伊予の文化案内人"
SERVICE_ID = "ehime-kokubunsai-ai-poc"
QUICK_QUESTIONS = (
    ("今日のイベント", "今日やっているイベント"),
    ("子どもと楽しむ", "子どもと楽しめるイベント"),
    ("雨でもOK", "雨でも楽しめる屋内イベント"),
    ("無料イベント", "無料のイベント"),
    ("伝統芸能", "無料の伝統芸能"),
    ("地域から探す", "南予でイベント"),
)
GENERIC_SCOPE_MESSAGE = "このPoCは文化祭イベントを探す機能の検証が中心なんよ。関連するイベントなら探せるよ。"
NEARBY_MESSAGE = "「近く」の範囲はまだ自動判定していません。探したい市町を教えてみん？"
NO_RESULT_MESSAGE = "条件に合うイベントは見つかりませんでした。日付・地域・料金のどれかを少し変えて探してみん？"
BACKEND_FAILURE_MESSAGE = "案内の準備に失敗したけん、条件を短くしてもう一度試してみて。"
EVENT_FACT_FIELDS = ("イベント名", "日時", "場所", "料金", "公式URL")
_EXPECTED_MODAL_HOST_RE = re.compile(
    r"^[a-z0-9-]+--ehime-kokubunsai-ai-poc-api(?:-[a-z0-9-]+)*\.modal\.run$"
)


st.set_page_config(page_title=PAGE_TITLE, page_icon="🎭", layout="centered")


def _required_secret(name: str) -> str:
    """Read a required project secret without exposing its value."""

    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        st.error("このPoCは現在利用できません。管理者が設定を確認してください。")
        st.stop()
    if not isinstance(value, str) or not value.strip():
        st.error("このPoCは現在利用できません。管理者が設定を確認してください。")
        st.stop()
    return value.strip()


def _authenticate(app_password: str) -> None:
    if st.session_state.get("authenticated") is True:
        return
    st.subheader("共通パスワード")
    entered = st.text_input("パスワード", type="password", max_chars=128)
    if st.button("入室", type="primary"):
        if hmac.compare_digest(entered, app_password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("パスワードが正しくありません。")
    st.stop()


def _validate_modal_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not _EXPECTED_MODAL_HOST_RE.fullmatch(host)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        st.error("新PoC用のModal接続先が正しく設定されていません。")
        st.stop()
    return url


def _search_result(
    query: str,
    *,
    previous_filters: dict[str, object] | None = None,
    inherit_previous: bool = False,
) -> event_search.SearchResult:
    """Run the local structured search and retain its explanation metadata."""

    return event_search.search_events(
        query,
        reference_date=POC_REFERENCE_DATE,
        previous_filters=previous_filters,
        inherit_previous=inherit_previous,
        limit=MAX_EVENT_CANDIDATES,
    )


def _search_candidates(query: str) -> list[dict[str, object]]:
    """Backward-compatible helper for callers that only need exact cards."""

    return list(_search_result(query).events[:MAX_EVENT_CANDIDATES])


def _facts_answer(
    events: list[dict[str, object]],
    requested_field: str,
) -> str:
    """Answer factual fields only from the JSON-backed event records."""

    if len(events) == 1:
        return event_search.attribute_answer(events[0], requested_field)
    lines = [f"条件に合うイベントは{len(events)}件あります。"]
    lines.extend(
        f"- {event_search.attribute_answer(event, requested_field)}"
        for event in events
    )
    return "\n".join(lines)


def _is_pronoun_reference(query: str) -> bool:
    normalized = event_search.normalize_query(query)
    return "それ" in normalized or "そのイベント" in normalized


def _llm_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Pass only current candidates; URLs remain card-only facts."""

    return [
        {
            key: value
            for key, value in candidate.items()
            if key != "公式URL"
        }
        for candidate in candidates[:MAX_EVENT_CANDIDATES]
    ]


def _call_modal(
    *,
    modal_url: str,
    modal_key: str,
    modal_secret: str,
    user_query: str,
    candidates: list[dict[str, object]],
    history: list[dict[str, str]],
) -> str:
    payload = {
        "user_query": user_query,
        "candidates": _llm_candidates(candidates),
        "history": history[-8:],
    }
    try:
        response = requests.post(
            modal_url,
            headers={
                "Modal-Key": modal_key,
                "Modal-Secret": modal_secret,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
            allow_redirects=False,
        )
        if response.status_code < 200 or response.status_code >= 300:
            return BACKEND_FAILURE_MESSAGE
        result = response.json()
        if not isinstance(result, dict):
            return BACKEND_FAILURE_MESSAGE
        if result.get("service_id") != SERVICE_ID:
            return BACKEND_FAILURE_MESSAGE
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return BACKEND_FAILURE_MESSAGE
        if _has_unapproved_event_claims(answer, candidates):
            return _fallback_guidance(candidates)
        return _redact_event_facts(answer, candidates)
    except (requests.RequestException, ValueError, TypeError):
        # Do not display exception text: it can contain URLs, headers, or
        # provider-specific details that are not useful to PoC users.
        return BACKEND_FAILURE_MESSAGE


def _event_fact_redactions(
    candidates: list[dict[str, object]],
) -> set[str]:
    """Build display-only redactions from the JSON-backed candidate cards."""

    values: set[str] = set()
    for event in candidates:
        for field in EVENT_FACT_FIELDS:
            raw_value = event.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue

            value = raw_value.strip()
            values.add(value)

            if field == "日時":
                for date_text in re.findall(r"\d{4}-\d{2}-\d{2}", value):
                    values.add(date_text)
                    try:
                        parsed = date.fromisoformat(date_text)
                    except ValueError:
                        continue
                    values.update(
                        {
                            f"{parsed.year}年{parsed.month}月{parsed.day}日",
                            f"{parsed.month}月{parsed.day}日",
                            f"{parsed.month}/{parsed.day}",
                        }
                    )

            if field == "場所":
                for match in re.finditer(
                    r"([一-龯々ぁ-んァ-ヶー]{1,10})(市|町|村)",
                    value,
                ):
                    values.add(match.group(0))
                    values.add(match.group(1))

            if field == "料金":
                values.update(re.findall(r"無料", value))
                values.update(
                    re.findall(r"[0-9０-９][0-9０-９,，]*\s*円", value)
                )

    return {value for value in values if len(value) >= 2}


def _fallback_guidance(candidates: list[dict[str, object]]) -> str:
    """Give deterministic guidance when the model leaves the candidate set."""

    count = len(candidates)
    if count == 1:
        return "条件に合うイベントが1件見つかりました。日時・場所・料金は、下のイベントカードを確認してみて。"
    return f"条件に合うイベントが{count}件見つかりました。気になる番号や、さらに絞りたい条件を教えてみて。"


def _has_unapproved_event_claims(
    answer: str,
    candidates: list[dict[str, object]],
) -> bool:
    """Reject model text that introduces an event outside the JSON candidates."""

    def normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).strip("「」『』")

    approved_names = {
        normalize(str(event.get("イベント名", "")))
        for event in candidates
        if str(event.get("イベント名", "")).strip()
    }
    approved_titles = {
        name.removeprefix("【PoC架空】").strip()
        for name in approved_names
    }

    def is_approved(fragment: str) -> bool:
        normalized = normalize(fragment)
        return any(
            normalized == name
            or normalized in name
            or name in normalized
            for name in approved_names
        ) or any(
            normalized == title
            or normalized in title
            or title in normalized
            for title in approved_titles
        )

    # The model must never invent another fictional event label.
    for fragment in re.findall(r"【PoC架空】[^\n。！？]{2,160}", answer):
        if not is_approved(fragment):
            return True

    # Also reject quoted or marker-bearing event titles that are not cards.
    event_markers = (
        "フェス",
        "フェスタ",
        "常設展",
        "企画展",
        "展示会",
        "ウォーク",
        "サロン",
        "シアター",
        "ラボ",
        "コンサート",
        "講座",
        "ワークショップ",
        "公演",
        "演奏会",
        "祭",
        "まつり",
        "工芸",
    )
    for fragment in re.findall(r"[「『]([^」』\n]{3,160})[」』]", answer):
        if any(marker in fragment for marker in event_markers) and not is_approved(
            fragment
        ):
            return True
    for marker in event_markers:
        for match in re.finditer(
            rf"[^\n。！？]{{0,50}}{re.escape(marker)}[^\n。！？]{{0,50}}",
            answer,
        ):
            if not is_approved(match.group(0)):
                return True

    # The app supplies a fixed PoC date, so stock uncertainty/refusal text is
    # unsafe even when the candidate cards themselves are correct.
    unsafe_phrases = (
        "現在日時が不明",
        "現在日時では",
        "具体的にお答えすることができません",
        "具体的にお答えできません",
        "具体的なイベント情報がないため",
        "確定的な回答はできません",
        "特定の日やイベントに関するご質問であれば",
        "情報は提供できません",
        "お答えすることができません",
        "お答えできません",
        "公式サイト",
        "観光情報サイト",
        "最新情報を確認",
    )
    return any(phrase in answer for phrase in unsafe_phrases) or bool(
        re.search(r"(?:https?://|www\.)", answer)
    )


def _redact_event_facts(
    answer: str,
    candidates: list[dict[str, object]],
) -> str:
    """Keep LLM output as guidance text, never as the event fact channel."""

    redacted = re.sub(
        r"\[[^\]]*\]\(https?://[^)]+\)",
        "",
        answer,
    )
    redacted = re.sub(r"<https?://[^>]+>", "", redacted)
    redacted = re.sub(r"https?://\S+", "", redacted)
    redacted = re.sub(r"\s*[（(]\s*<\s*$", "", redacted)
    for value in sorted(_event_fact_redactions(candidates), key=len, reverse=True):
        redacted = redacted.replace(value, "この候補")

    redacted = re.sub(r"\n{3,}", "\n\n", redacted).strip()
    return redacted or "候補カードを確認してみてください。こんなんもあるよ。"


def _render_event_card(event: dict[str, object], index: int) -> None:
    with st.container(border=True):
        st.markdown(f"**{index}. {event['イベント名']}**")
        st.write(f"日時：{event['日時']}")
        st.write(f"場所：{event['場所']}")
        st.write(f"ジャンル：{event['ジャンル']}")
        st.write(f"対象：{'子ども向け' if event['子ども向け'] else '一般向け'}")
        st.write(f"会場：{event['屋内/屋外']}")
        st.write(f"料金：{event['料金']}")
        st.caption(str(event["概要"]))
        st.link_button("公式URL（PoC架空）", str(event["公式URL"]))


def _reset() -> None:
    for key in (
        "messages",
        "last_results",
        "last_near_results",
        "last_relaxed_condition",
        "last_filters",
        "selected_event",
        "last_plan",
        "last_query",
        "feedback",
        "pending_prompt",
    ):
        st.session_state.pop(key, None)
    st.rerun()


app_password = _required_secret("APP_PASSWORD")
_authenticate(app_password)

modal_url = _validate_modal_url(_required_secret("MODAL_URL"))
modal_key = _required_secret("MODAL_KEY")
modal_secret = _required_secret("MODAL_SECRET")

st.title(PAGE_TITLE)
st.caption("愛顔えひめの文化祭2028を想定したイベント案内PoC")
st.warning(
    "このサイトは生成AIを利用した技術検証用PoCです。\n\n"
    "掲載イベントはすべて架空です。\n\n"
    "愛媛県・愛顔えひめの文化祭2028の公式サービスではありません。\n\n"
    "AIの回答には誤りが含まれる場合があります。"
)
st.info(f"PoC上の現在日：{POC_REFERENCE_DATE_TEXT}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_near_results" not in st.session_state:
    st.session_state.last_near_results = []
if "last_relaxed_condition" not in st.session_state:
    st.session_state.last_relaxed_condition = None
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

with st.sidebar:
    st.header("質問の例")
    for label, question in QUICK_QUESTIONS:
        if st.button(label, key=f"quick_{label}"):
            st.session_state.pending_prompt = question
            st.rerun()
    st.divider()
    if st.button("会話をリセット", key="reset_conversation"):
        _reset()
    st.caption("個人情報・機密情報・未公開情報は入力しないでください。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.last_results:
    st.subheader("条件に合うイベント")
    for index, event in enumerate(st.session_state.last_results, start=1):
        _render_event_card(event, index)

if st.session_state.last_near_results:
    relaxed = st.session_state.last_relaxed_condition or "一部の条件"
    st.subheader(f"参考候補（「{relaxed}」を外した場合）")
    st.caption("上の検索結果には含めていません。条件を緩めた候補として表示しています。")
    for index, event in enumerate(st.session_state.last_near_results, start=1):
        _render_event_card(event, index)

prompt = st.session_state.pop("pending_prompt", None)
if prompt is None:
    prompt = st.chat_input("例：11月3日に子どもと行けるイベント")

if prompt:
    prompt = prompt.strip()
    if len(prompt) > 500:
        st.error("質問は500文字以内にしてください。")
        st.stop()

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    results: list[dict[str, object]] = []
    near_results: list[dict[str, object]] = []
    relaxed_condition: str | None = None
    selected_event: dict[str, object] | None = None
    filters: event_search.SearchFilters | None = None
    search_result: event_search.SearchResult | None = None
    previous_results = list(st.session_state.get("last_results", []))

    if event_search.asks_for_nearby(prompt):
        answer = NEARBY_MESSAGE
    elif (
        previous_results
        and (
            event_search.resolve_reference_index(prompt, len(previous_results)) is not None
            or _is_pronoun_reference(prompt)
        )
    ):
        # Ordinals and pronouns are resolved against the last exact result
        # set.  They never trigger a new semantic search or an LLM call.
        filters = event_search.parse_query(prompt, POC_REFERENCE_DATE)
        reference_index = event_search.resolve_reference_index(prompt, len(previous_results))
        if reference_index is not None:
            selected_event = previous_results[reference_index]
        elif st.session_state.get("selected_event"):
            selected_event = st.session_state.selected_event
        elif len(previous_results) == 1:
            selected_event = previous_results[0]
        if selected_event and filters.requested_field:
            answer = event_search.attribute_answer(selected_event, filters.requested_field)
            results = previous_results
        elif selected_event:
            answer = f"選択中のイベントは「{selected_event['イベント名']}」です。カードを確認してみて。"
            results = previous_results
        else:
            answer = "どのイベントを指しているか、番号かイベント名を教えてみて。"
            results = previous_results
    elif event_search.classify_intent(prompt) in {"injection", "out_of_scope"}:
        search_result = _search_result(prompt)
        answer = search_result.message or GENERIC_SCOPE_MESSAGE
    elif not event_search.looks_like_event_query(prompt):
        answer = GENERIC_SCOPE_MESSAGE
    else:
        inherit_previous = event_search.is_refinement_query(prompt) and bool(
            st.session_state.get("last_filters")
        )
        search_result = _search_result(
            prompt,
            previous_filters=st.session_state.get("last_filters"),
            inherit_previous=inherit_previous,
        )
        filters = search_result.filters
        results = list(search_result.events[:MAX_EVENT_CANDIDATES])
        near_results = list(search_result.near_matches[:MAX_EVENT_CANDIDATES])
        relaxed_condition = search_result.relaxed_condition
        if not results:
            answer = search_result.message or NO_RESULT_MESSAGE
        elif filters.intent == "count":
            answer = f"条件に合うイベントは{search_result.total_matches}件あります。"
        elif filters.requested_field:
            # Date, place, fee, venue, and other factual fields are returned
            # directly from events.json.  Modal is reserved for discovery
            # guidance over already-filtered candidates.
            answer = _facts_answer(results, filters.requested_field)
        else:
            answer = _call_modal(
                modal_url=modal_url,
                modal_key=modal_key,
                modal_secret=modal_secret,
                user_query=prompt,
                candidates=results,
                history=history,
            )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_results = results
    st.session_state.last_near_results = near_results
    st.session_state.last_relaxed_condition = relaxed_condition
    if filters is not None and search_result is not None:
        st.session_state.last_filters = filters.to_dict()
        st.session_state.last_plan = {
            "intent": filters.intent,
            "entity": filters.entity,
            "requested_field": filters.requested_field,
            "confidence": search_result.confidence,
            "total_matches": search_result.total_matches,
            "relaxed_condition": search_result.relaxed_condition,
        }
    if selected_event is not None:
        st.session_state.selected_event = selected_event
    elif search_result is not None:
        st.session_state.selected_event = None
    st.session_state.last_query = prompt
    st.session_state.feedback = None
    st.rerun()

if st.session_state.get("last_query"):
    st.divider()
    st.caption("この案内は役に立ちましたか？")
    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        if st.button("役に立った", key="feedback_yes"):
            st.session_state.feedback = "yes"
    with feedback_col2:
        if st.button("改善が必要", key="feedback_no"):
            st.session_state.feedback = "no"
    if st.session_state.get("feedback") == "yes":
        st.success("フィードバックを受け取りました。")
    elif st.session_state.get("feedback") == "no":
        st.info("ありがとうございます。PoCの改善に使います。")

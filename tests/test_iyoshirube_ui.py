from __future__ import annotations

from iyoshirube_ui import (
    EMOTION_HAPPY,
    EMOTION_NORMAL,
    EMOTION_THINKING,
    EMOTION_TROUBLED,
    IYOSHIRUBE_AVATARS,
    avatar_path,
    emotion_from_message,
    model_history,
    normalize_emotion,
    select_assistant_emotion,
)


def test_normal_answers_use_normal_avatar() -> None:
    assert select_assistant_emotion(flow="general_faq", result_count=1) == EMOTION_NORMAL
    assert select_assistant_emotion(flow="event_detail", result_count=1) == EMOTION_NORMAL
    assert select_assistant_emotion(flow="count_events", result_count=0) == EMOTION_NORMAL


def test_successful_search_and_recommendations_use_happy_avatar() -> None:
    assert select_assistant_emotion(flow="find_events", result_count=1) == EMOTION_HAPPY
    assert select_assistant_emotion(flow="agentic_search", result_count=1) == EMOTION_HAPPY
    assert (
        select_assistant_emotion(
            flow="agentic_search",
            result_count=0,
            exact_result_count=1,
        )
        == EMOTION_HAPPY
    )
    assert (
        select_assistant_emotion(
            flow="recommend_next",
            result_count=0,
            recommendation_result_count=2,
        )
        == EMOTION_HAPPY
    )
    assert (
        select_assistant_emotion(
            flow="recommend_similar",
            result_count=0,
            recommendation_result_count=1,
        )
        == EMOTION_HAPPY
    )
    assert (
        select_assistant_emotion(
            flow="plan_event_pair",
            result_count=2,
            pair_result_count=1,
        )
        == EMOTION_HAPPY
    )


def test_pending_states_use_thinking_even_when_previous_results_are_empty() -> None:
    assert (
        select_assistant_emotion(
            flow="recommend_next",
            result_count=0,
            pending=True,
        )
        == EMOTION_THINKING
    )
    assert (
        select_assistant_emotion(
            answer="何時ごろ見終わる予定？",
            flow="recommend_next",
            pending=True,
        )
        == EMOTION_THINKING
    )
    assert (
        select_assistant_emotion(
            flow="time_pending",
            result_count=0,
            pending_state={"awaiting": "time"},
        )
        == EMOTION_THINKING
    )
    assert (
        select_assistant_emotion(
            flow="event_selection_pending",
            clarification_required=True,
        )
        == EMOTION_THINKING
    )
    assert (
        select_assistant_emotion(flow="recommend_next_without_selection")
        == EMOTION_THINKING
    )


def test_failure_states_use_troubled_avatar() -> None:
    assert select_assistant_emotion(flow="find_events", result_count=0) == EMOTION_TROUBLED
    assert (
        select_assistant_emotion(flow="search", invalid_input=True)
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(
            flow="recommend_next",
            pending=True,
            invalid_input=True,
        )
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(flow="recommend_next", recommendation_result_count=0)
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(flow="plan_event_pair", pair_result_count=0)
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(
            flow="find_events",
            result_count=1,
            backend_failure=True,
        )
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(
            flow="find_events",
            result_count=1,
            pending=True,
            backend_failure=True,
        )
        == EMOTION_TROUBLED
    )
    assert (
        select_assistant_emotion(
            flow="event_detail",
            event_selection_failed=True,
        )
        == EMOTION_TROUBLED
    )


def test_success_takes_precedence_over_follow_up_text() -> None:
    assert (
        select_assistant_emotion(
            answer="候補が見つかったよ。必要な条件をもう少し教えてみて。",
            flow="find_events",
            result_count=1,
        )
        == EMOTION_HAPPY
    )


def test_message_metadata_is_backward_compatible_and_normalized() -> None:
    assert emotion_from_message({"role": "assistant", "content": "test"}) == EMOTION_NORMAL
    assert (
        emotion_from_message(
            {"role": "assistant", "content": "test", "emotion": "happy"}
        )
        == EMOTION_HAPPY
    )
    assert (
        emotion_from_message(
            {"role": "assistant", "content": "test", "emotion": "unknown"}
        )
        == EMOTION_NORMAL
    )
    assert normalize_emotion("unknown") == EMOTION_NORMAL


def test_model_history_drops_ui_only_emotion_metadata() -> None:
    assert model_history(
        [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": "回答", "emotion": "happy"},
        ]
    ) == [
        {"role": "user", "content": "質問"},
        {"role": "assistant", "content": "回答"},
    ]


def test_all_avatar_assets_are_local_rgba_pngs() -> None:
    png_signature = b"\x89PNG\r\n\x1a\n"
    assert set(IYOSHIRUBE_AVATARS) == {"normal", "happy", "thinking", "troubled"}
    for path in IYOSHIRUBE_AVATARS.values():
        assert avatar_path(path.stem) == path
        assert path.read_bytes().startswith(png_signature)

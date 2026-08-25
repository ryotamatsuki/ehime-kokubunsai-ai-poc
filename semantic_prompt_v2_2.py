"""Compact prompting helpers for Semantic Operations v2.2.

The classifier sees only the state bits needed for atomic control decisions.
Exact result IDs, selected event IDs and previous command payloads remain in
trusted Python state and are never copied into the model context.

The demonstrations are specification-authored examples, not frozen-v1 or
sealed-holdout fixtures.  They teach the fixed atomic contract while keeping
context small enough for the 3B model.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from semantic_atomic_v2_2 import build_atomic_frame_system_prompt, neutral_experience


def _frame(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "intent": "search",
        "scope": "new",
        "municipality": "none",
        "region": "none",
        "fee": "none",
        "reservation": "none",
        "venue": "none",
        "rain": "none",
        "audience_mode": "none",
        "clarification": "none",
        "data_gap": "none",
        "experience": neutral_experience(),
    }
    experience = overrides.pop("experience", None)
    result.update(overrides)
    if isinstance(experience, Mapping):
        result["experience"].update({str(key): str(value) for key, value in experience.items()})
    return result


ATOMIC_FEW_SHOT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "query": "松山市で入場無料の催しを探したい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {"municipalities": ["松山市"], "entry_free": True},
        "frame": _frame(),
    },
    {
        "query": "前の候補は料金条件を外して、座って楽しめるものに絞りたい",
        "state": {"has_previous_results": True, "has_previous_command": True},
        "grounded": {},
        "frame": _frame(scope="previous", fee="release", experience={"seated": "require"}),
    },
    {
        "query": "家族で一緒に楽しめる催しがいい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(audience_mode="family"),
    },
    {
        "query": "混み具合が少ない催しを選びたい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(data_gap="crowding"),
    },
    {
        "query": "体験するものは除いて、見る・聞くのを中心にしたい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(experience={"hands_on": "exclude", "watch_listen": "prefer"}),
    },
)


def minimal_atomic_state(state: Mapping[str, Any] | None) -> dict[str, bool]:
    if not isinstance(state, Mapping):
        return {"has_previous_results": False, "has_previous_command": False}
    ids = state.get("last_result_ids")
    return {
        "has_previous_results": bool(isinstance(ids, (list, tuple)) and ids),
        "has_previous_command": isinstance(state.get("last_command"), Mapping),
    }


def build_minimal_atomic_payload(
    query: str,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safe_grounded = {
        str(key): value
        for key, value in dict(grounded or {}).items()
        if value not in (None, "", [], (), {})
    }
    return {
        "query": str(query).strip()[:1200],
        "state": minimal_atomic_state(state),
        "grounded": safe_grounded,
    }


def _user_content(payload: Mapping[str, Any]) -> str:
    return (
        f"query:\n{payload['query']}\n"
        f"state:{json.dumps(payload['state'], ensure_ascii=False, separators=(',', ':'))}\n"
        f"grounded:{json.dumps(payload['grounded'], ensure_ascii=False, separators=(',', ':'))}\n"
        "atomic frame JSONを1個だけ返してください。"
    )


def build_atomic_frame_messages(
    query: str,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a bounded few-shot chat transcript for the atomic classifier."""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_atomic_frame_system_prompt()},
    ]
    for example in ATOMIC_FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": _user_content({
                "query": example["query"],
                "state": example["state"],
                "grounded": example["grounded"],
            }),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["frame"], ensure_ascii=False, separators=(",", ":")),
        })
    messages.append({
        "role": "user",
        "content": _user_content(build_minimal_atomic_payload(query, state, grounded)),
    })
    return messages


__all__ = [
    "ATOMIC_FEW_SHOT_EXAMPLES",
    "build_atomic_frame_messages",
    "build_minimal_atomic_payload",
    "minimal_atomic_state",
]

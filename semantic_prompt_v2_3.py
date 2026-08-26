"""Specification-authored prompting for Semantic Operations v2.3.

Examples demonstrate architectural invariants only.  They are not copied from
the frozen-v1 regression corpus or the sealed holdout.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from semantic_atomic_v2_2 import neutral_experience
from semantic_atomic_v2_3 import build_atomic_frame_system_prompt_v23
from semantic_evidence_v2_3 import EvidenceRequest, SemanticResolution
from semantic_prompt_v2_2 import minimal_atomic_state


def _frame(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "intent": "search",
        "scope": "new",
        "evidence_request": EvidenceRequest.NONE.value,
        "semantic_resolution": SemanticResolution.RESOLVED.value,
        "municipality": "none",
        "region": "none",
        "fee": "none",
        "reservation": "none",
        "venue": "none",
        "rain": "none",
        "audience_mode": "none",
        "experience": neutral_experience(),
    }
    experience = overrides.pop("experience", None)
    result.update(overrides)
    if isinstance(experience, Mapping):
        result["experience"].update({str(key): str(value) for key, value in experience.items()})
    return result


ATOMIC_FEW_SHOT_EXAMPLES_V23: tuple[dict[str, Any], ...] = (
    {
        "query": "東予で予約が必要な催しを探したい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {"regions": ["東予"], "reservation_required": True},
        "frame": _frame(evidence_request=EvidenceRequest.SUPPORTED_ATTRIBUTE.value),
    },
    {
        "query": "初参加の同行者に向いている催しが知りたい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(evidence_request=EvidenceRequest.RELATIONAL_SUITABILITY.value),
    },
    {
        "query": "同行者に合うもので、座って楽しめる催しを探したい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {"experience_required": ["seated"]},
        "frame": _frame(evidence_request=EvidenceRequest.RELATIONAL_SUITABILITY.value),
    },
    {
        "query": "必ず満足できる催しを選んで",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(evidence_request=EvidenceRequest.ABSOLUTE_GUARANTEE.value),
    },
    {
        "query": "今この瞬間に空いている催しを知りたい",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(evidence_request=EvidenceRequest.REALTIME_STATE.value),
    },
    {
        "query": "条件分岐で選びたいが、分岐条件の値はまだ決まっていない",
        "state": {"has_previous_results": False, "has_previous_command": False},
        "grounded": {},
        "frame": _frame(
            evidence_request=EvidenceRequest.UNKNOWN_CAPABILITY.value,
            semantic_resolution=SemanticResolution.CONDITIONAL.value,
        ),
    },
    {
        "query": "前の候補は料金条件を外して、見る・聞く中心にしたい",
        "state": {"has_previous_results": True, "has_previous_command": True},
        "grounded": {"experience_required": ["watch_listen"]},
        "frame": _frame(
            scope="previous",
            evidence_request=EvidenceRequest.SUPPORTED_ATTRIBUTE.value,
            fee="release",
        ),
    },
)


def build_minimal_atomic_payload_v23(
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
        "v2.3 atomic frame JSONを1個だけ返してください。"
    )


def build_atomic_frame_messages_v23(
    query: str,
    state: Mapping[str, Any] | None = None,
    grounded: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_atomic_frame_system_prompt_v23()},
    ]
    for example in ATOMIC_FEW_SHOT_EXAMPLES_V23:
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
        "content": _user_content(build_minimal_atomic_payload_v23(query, state, grounded)),
    })
    return messages


__all__ = [
    "ATOMIC_FEW_SHOT_EXAMPLES_V23",
    "build_atomic_frame_messages_v23",
    "build_minimal_atomic_payload_v23",
]

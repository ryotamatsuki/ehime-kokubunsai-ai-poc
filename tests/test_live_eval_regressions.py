from __future__ import annotations

from command_generator import generate_command
from live_semantic_command_eval import _summarize_rows


def _valid_next_command(_payload):
    return {
        "flow": "recommend_next",
        "slots": {"reference_kind": "selected"},
        "confidence": "high",
    }


def test_recommend_next_inherits_one_validated_previous_search_date():
    state = {
        "selected_event_id": "002",
        "last_command": {
            "flow": "find_events",
            "slots": {
                "dates": ["2028-10-22"],
                "municipalities": ["松山市"],
            },
            "confidence": "high",
        },
    }

    generated = generate_command(
        "そのイベントのあと何か行ける？",
        state,
        call=_valid_next_command,
    )

    assert generated.plan.flow == "recommend_next"
    assert tuple(generated.plan.slots.dates) == ("2028-10-22",)
    assert generated.plan.slots.reference_kind == "selected"


def test_recommend_next_does_not_inherit_unvalidated_or_ambiguous_date():
    states = [
        {
            "selected_event_id": "002",
            "last_command": {
                "flow": "find_events",
                "slots": {"dates": ["2028-10-22", "2028-10-23"]},
                "confidence": "high",
            },
        },
        {
            "selected_event_id": "002",
            "last_command": {"flow": "not-a-flow", "slots": {"dates": ["2028-10-22"]}},
        },
        {
            "last_command": {
                "flow": "find_events",
                "slots": {"dates": ["2028-10-22"]},
                "confidence": "high",
            },
        },
    ]

    for state in states:
        generated = generate_command(
            "そのイベントのあと何か行ける？",
            state,
            call=_valid_next_command,
        )
        assert tuple(generated.plan.slots.dates) == ()


def test_dialogue_summary_is_derived_from_turn_rows():
    rows = [
        {
            "expected_flow": "find_events",
            "actual_flow": "find_events",
            "route_match": True,
            "total_latency_ms": 100.0,
            "failure_category": None,
            "stats": {
                "attempts": 1,
                "first_pass_json_valid": True,
                "first_pass_schema_valid": True,
                "repair_attempted": False,
                "repair_success": False,
                "final_valid": True,
            },
        },
        {
            "expected_flow": "explain_search",
            "actual_flow": "unsupported",
            "route_match": False,
            "total_latency_ms": 300.0,
            "failure_category": "R3_schema_violation",
            "stats": {
                "attempts": 2,
                "first_pass_json_valid": True,
                "first_pass_schema_valid": False,
                "repair_attempted": True,
                "repair_success": False,
                "final_valid": False,
            },
        },
    ]

    summary = _summarize_rows(rows)

    assert summary["cases"] == 2
    assert summary["route_accuracy"] == 0.5
    assert summary["json_valid_first_pass_rate"] == 1.0
    assert summary["repair_rate"] == 0.5
    assert summary["repair_success_rate"] == 0.0
    assert summary["final_valid_command_rate"] == 0.5
    assert summary["median_latency_ms"] == 200.0
    assert summary["failure_categories"] == {"R3_schema_violation": 1}

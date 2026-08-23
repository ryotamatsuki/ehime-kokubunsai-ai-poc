"""Live Semantic Command evaluation harness.

This module contains no model implementation. It drives the same local
``CommandOrchestrator`` used by Streamlit and accepts a callback that invokes
the deployed Sarashina command method. It is used by the manual Modal
workflow, never by production request handling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import conversation_recovery
import event_search
from app_config import MAX_RESULT_SET_SIZE, POC_REFERENCE_DATE
from command_generator import parse_command_json, parse_and_validate_command
from command_orchestrator import CommandOrchestrator, CommandTurnResult
from conversation_router import route_conversation


FLOW_ALIASES = {
    "search": "find_events",
    "detail": "event_detail",
    "reference": "event_detail",
    "count": "count_events",
    "faq": "general_faq",
    "unsupported": "unsupported",
    "scope": "unsupported",
}


def canonical_flow(value: Any) -> str:
    return FLOW_ALIASES.get(str(value), str(value))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return round(float(ordered[index]), 3)


def _seed_state(context: str) -> dict[str, Any]:
    if context != "search":
        return {}
    events = event_search.load_events()[:5]
    ids = [str(event.get("id", "")) for event in events if event.get("id")]
    search_context = conversation_recovery.build_search_context(
        "live semantic evaluation seed",
        {},
        events,
        result_ids=ids,
        total_matches=len(ids),
    )
    return {
        "last_result_ids": ids,
        "last_result_count": len(ids),
        "last_action": "find_events",
        "has_last_search_context": True,
        "last_search_context": search_context.to_dict(),
    }


def _is_router_fast_path(query: str, state: Mapping[str, Any]) -> tuple[str | None, Any]:
    by_id = {
        str(event.get("id")): event
        for event in event_search.load_events()
        if event.get("id")
    }
    previous_results = [
        by_id.get(str(event_id))
        for event_id in list(state.get("last_result_ids", []))[:MAX_RESULT_SET_SIZE]
    ]
    previous_results = [event for event in previous_results if isinstance(event, Mapping)]
    route = route_conversation(
        query,
        previous_results,
        None,
        state.get("last_filters"),
        POC_REFERENCE_DATE,
    )
    detail_field = getattr(route, "detail_field", None)
    if getattr(route, "action_type", None) in {
        "detail_followup",
        "explain_search",
        "explain_result",
        "clarify_reference",
    } or (
        getattr(route, "action_type", None) == "reference_followup"
        and detail_field is not None
    ):
        action = getattr(route, "action_type", "")
        if action in {"detail_followup", "reference_followup"}:
            action = "event_detail"
        return action, route
    return None, route


@dataclass
class RemoteCommandCall:
    """Collect only bounded evaluation metadata around remote calls."""

    invoke: Callable[[Mapping[str, Any]], Any]
    format_enforcer: str = "baseline"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        started = time.perf_counter()
        repair = bool(payload.get("repair"))
        try:
            raw = self.invoke(payload)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self.calls.append(
                {
                    "elapsed_ms": round(elapsed, 3),
                    "repair": repair,
                    "raw": None,
                    "error_type": type(exc).__name__[:120],
                }
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000
        self.calls.append(
            {
                "elapsed_ms": round(elapsed, 3),
                "repair": repair,
                "raw": raw,
                "error_type": None,
            }
        )
        return raw

    def stats(self) -> dict[str, Any]:
        if not self.calls:
            return {
                "attempts": 0,
                "first_pass_json_valid": False,
                "first_pass_schema_valid": False,
                "repair_attempted": False,
                "repair_success": False,
                "final_valid": False,
                "raw_output": None,
                "latencies_ms": [],
                "call_error_types": [],
            }
        first = self.calls[0].get("raw")
        first_json_valid = first is not None
        first_schema_valid = first is not None
        if first is not None:
            try:
                parse_command_json(first)
            except Exception:
                first_json_valid = False
            try:
                parse_and_validate_command(first)
            except Exception:
                first_schema_valid = False
        final = self.calls[-1].get("raw")
        final_valid = final is not None
        if final is not None:
            try:
                parse_and_validate_command(final)
            except Exception:
                final_valid = False
        return {
            "attempts": len(self.calls),
            "first_pass_json_valid": first_json_valid,
            "first_pass_schema_valid": first_schema_valid,
            "repair_attempted": any(bool(entry.get("repair")) for entry in self.calls),
            "repair_success": any(bool(entry.get("repair")) for entry in self.calls) and final_valid,
            "final_valid": final_valid,
            "raw_output": str(final)[:1200] if final is not None else None,
            "latencies_ms": [entry["elapsed_ms"] for entry in self.calls],
            "call_error_types": [
                str(entry["error_type"])
                for entry in self.calls
                if entry.get("error_type")
            ],
        }


def _update_state(
    state: Mapping[str, Any],
    result: CommandTurnResult,
    query: str,
) -> dict[str, Any]:
    next_state = dict(state)
    next_state["last_action"] = result.flow
    next_state["last_command"] = result.command.to_dict()
    next_state["last_result_count"] = int(result.total_matches or len(result.events))
    next_state["last_result_ids"] = list(result.all_event_ids)[:MAX_RESULT_SET_SIZE]
    if result.filters is not None:
        next_state["last_filters"] = dict(result.filters)
    if result.flow in {
        "find_events",
        "count_events",
        "recommend_next",
        "recommend_similar",
        "plan_event_pair",
    }:
        events = event_search.load_events()
        by_id = {str(event.get("id")): event for event in events}
        trace_events = [by_id[event_id] for event_id in result.all_event_ids if event_id in by_id]
        context = conversation_recovery.build_search_context(
            query,
            result.filters or {},
            trace_events,
            result_ids=result.all_event_ids,
            total_matches=result.total_matches,
        )
        next_state["last_search_context"] = context.to_dict()
        next_state["has_last_search_context"] = True
    return next_state


def _failure_category(
    expected: str,
    actual: str,
    stats: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> str | None:
    if expected == actual:
        return None
    # Prefer the concrete output-contract failure over a coarse orchestration
    # label. This prevents schema/repair failures from being counted as Modal
    # transport/runtime failures merely because a later layer fell back.
    if stats.get("attempts") and not stats.get("first_pass_json_valid"):
        return "R2_malformed_json"
    if stats.get("attempts") and not stats.get("first_pass_schema_valid"):
        return "R3_schema_violation"
    if stats.get("repair_attempted") and not stats.get("repair_success"):
        return "R4_repair_failure"
    error_type = observation.get("semantic_command_error_type")
    if error_type in {"modal_timeout", "modal_http_error", "modal_protocol_error"}:
        return "R11_modal_runtime_error"
    if error_type == "orchestrator_exception" or stats.get("call_error_types"):
        return "R12_orchestrator_internal_error"
    if observation.get("fallback_used"):
        return "R10_fallback_misrouting"
    if observation.get("has_search_context") is False and expected in {"explain_search", "explain_result"}:
        return "R6_missing_search_context"
    return "R1_semantic_misclassification"


def _flow_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    expected_counts = Counter(str(row["expected_flow"]) for row in rows)
    flows = sorted(set(expected_counts) | {str(row["actual_flow"]) for row in rows})
    per_flow: dict[str, dict[str, float]] = {}
    for flow in flows:
        true_positive = sum(
            row["expected_flow"] == flow and row["actual_flow"] == flow for row in rows
        )
        expected_total = sum(row["expected_flow"] == flow for row in rows)
        predicted_total = sum(row["actual_flow"] == flow for row in rows)
        per_flow[flow] = {
            "precision": round(true_positive / predicted_total, 4) if predicted_total else 0.0,
            "recall": round(true_positive / expected_total, 4) if expected_total else 0.0,
            "support": expected_total,
        }
    return per_flow


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    expected_counts = Counter(str(row["expected_flow"]) for row in rows)
    correct = sum(bool(row["route_match"]) for row in rows)
    latencies = [float(row["total_latency_ms"]) for row in rows]
    failures = Counter(
        str(row["failure_category"])
        for row in rows
        if row.get("failure_category")
    )
    per_flow = _flow_metrics(rows)
    explain_search = per_flow.get("explain_search", {"recall": 0.0})
    explain_result = per_flow.get("explain_result", {"recall": 0.0})
    false_explain = sum(
        row["actual_flow"] in {"explain_search", "explain_result"}
        and row["expected_flow"] not in {"explain_search", "explain_result"}
        for row in rows
    )
    command_rows = [row for row in rows if row["stats"].get("attempts")]
    repair_rows = [row for row in rows if row["stats"].get("repair_attempted")]
    f1_values = [
        (
            2 * value["precision"] * value["recall"]
            / (value["precision"] + value["recall"])
        )
        if value["precision"] + value["recall"]
        else 0.0
        for value in per_flow.values()
    ]
    return {
        "cases": len(rows),
        "route_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_f1": round(statistics.mean(f1_values), 4) if f1_values else 0.0,
        "per_flow": per_flow,
        "explain_search_recall": explain_search.get("recall", 0.0),
        "explain_result_recall": explain_result.get("recall", 0.0),
        "false_explain_routing_rate": round(false_explain / len(rows), 4) if rows else 0.0,
        "json_valid_first_pass_rate": round(
            sum(bool(row["stats"].get("first_pass_json_valid")) for row in command_rows)
            / len(command_rows),
            4,
        ) if command_rows else 0.0,
        "repair_rate": round(len(repair_rows) / len(rows), 4) if rows else 0.0,
        "repair_success_rate": round(
            sum(bool(row["stats"].get("repair_success")) for row in repair_rows)
            / len(repair_rows),
            4,
        ) if repair_rows else 0.0,
        "final_valid_command_rate": round(
            sum(bool(row["stats"].get("final_valid")) for row in command_rows)
            / len(command_rows),
            4,
        ) if command_rows else 0.0,
        "modal_call_rate": round(len(command_rows) / len(rows), 4) if rows else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "failure_categories": dict(failures),
        "expected_counts": dict(expected_counts),
        "rows": rows,
    }


def evaluate_case(
    case: Mapping[str, Any],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = False,
    format_enforcer: str = "baseline",
) -> dict[str, Any]:
    query = str(case.get("query", ""))
    expected = canonical_flow(case.get("expected_flow", case.get("flow", "unsupported")))
    state = _seed_state(str(case.get("context", "none")))
    fast_flow, route = _is_router_fast_path(query, state)
    remote = RemoteCommandCall(invoke, format_enforcer=format_enforcer)
    started = time.perf_counter()
    if fast_flow is not None:
        actual = canonical_flow(fast_flow)
        result = None
        observation = {
            "deterministic_route": fast_flow,
            "deterministic_confidence": "high",
            "semantic_command_called": False,
            "fallback_used": False,
            "has_search_context": bool(state.get("has_last_search_context")),
            "final_flow": actual,
        }
    else:
        result = CommandOrchestrator(modal_call=remote).handle_query(query, state)
        actual = canonical_flow(result.flow)
        observation = result.observability
    stats = remote.stats()
    row = {
        "id": case.get("id"),
        "category": case.get("category"),
        "expected_flow": expected,
        "actual_flow": actual,
        "route_match": expected == actual,
        "fast_path_hit": fast_flow is not None,
        "route_action": getattr(route, "action_type", None),
        "stats": {
            key: value
            for key, value in stats.items()
            if key != "raw_output" or include_raw
        },
        "observability": observation,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "failure_category": _failure_category(expected, actual, stats, observation),
    }
    if result is not None:
        row["slots"] = result.command.slots.to_dict()
    return row


def evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    include_raw: bool = False,
    format_enforcer: str = "baseline",
) -> dict[str, Any]:
    rows = [
        evaluate_case(
            case,
            invoke,
            include_raw=include_raw,
            format_enforcer=format_enforcer,
        )
        for case in cases
    ]
    return _summarize_rows(rows)


def evaluate_dialogues(
    dialogues: Sequence[Mapping[str, Any]],
    invoke: Callable[[Mapping[str, Any]], Any],
    *,
    limit: int = 30,
    include_raw: bool = False,
    format_enforcer: str = "baseline",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_dialogues = list(dialogues)[:limit]
    for dialogue in selected_dialogues:
        state: dict[str, Any] = {}
        for index, turn in enumerate(dialogue.get("turns", []), start=1):
            case = dict(turn)
            case["id"] = f"{dialogue.get('id', 'dialogue')}-{index}"
            case["query"] = turn.get("user", "")
            case["expected_flow"] = turn.get("expected_flow", "unsupported")
            if turn.get("context") == "search" and not state.get("last_search_context"):
                state = _seed_state("search")
            fast_flow, route = _is_router_fast_path(str(case["query"]), state)
            remote = RemoteCommandCall(invoke, format_enforcer=format_enforcer)
            started = time.perf_counter()
            if fast_flow is not None:
                actual = canonical_flow(fast_flow)
                result = None
                observation = {
                    "deterministic_route": fast_flow,
                    "semantic_command_called": False,
                    "fallback_used": False,
                    "has_search_context": bool(state.get("last_search_context")),
                    "final_flow": actual,
                }
            else:
                result = CommandOrchestrator(modal_call=remote).handle_query(
                    str(case["query"]), state
                )
                actual = canonical_flow(result.flow)
                observation = result.observability
                state = _update_state(state, result, str(case["query"]))
            stats = remote.stats()
            expected = canonical_flow(case["expected_flow"])
            rows.append(
                {
                    "id": case["id"],
                    "dialogue_id": dialogue.get("id"),
                    "turn": index,
                    "expected_flow": expected,
                    "actual_flow": actual,
                    "route_match": expected == actual,
                    "fast_path_hit": fast_flow is not None,
                    "route_action": getattr(route, "action_type", None),
                    "stats": {
                        key: value
                        for key, value in stats.items()
                        if key != "raw_output" or include_raw
                    },
                    "observability": observation,
                    "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "failure_category": _failure_category(
                        expected, actual, stats, observation
                    ),
                }
            )
    summary = _summarize_rows(rows)
    dialogue_ids = [str(dialogue.get("id", "")) for dialogue in selected_dialogues]
    all_route_success = sum(
        bool(dialogue_id)
        and all(
            row["route_match"]
            for row in rows
            if str(row.get("dialogue_id", "")) == dialogue_id
        )
        for dialogue_id in dialogue_ids
    )
    summary.update(
        {
            "dialogues": len(selected_dialogues),
            "turns": len(rows),
            "all_route_success_dialogues": all_route_success,
            "all_route_success_rate": round(
                all_route_success / len(selected_dialogues), 4
            ) if selected_dialogues else 0.0,
        }
    )
    return summary


__all__ = ["evaluate_cases", "evaluate_dialogues", "canonical_flow"]

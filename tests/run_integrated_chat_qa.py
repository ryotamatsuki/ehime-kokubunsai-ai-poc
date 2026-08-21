"""Integrated 100-case acceptance gate for the event-guide PoC.

This is intentionally a different layer from ``run_search_v2_qa.py``.
The corpus contains user-like utterances spanning all 30 events, composed
constraints, exact counts, conversation references, recommendations, FAQ,
out-of-scope requests and security/adversarial inputs.

Default (offline oracle validation):
    python tests/run_integrated_chat_qa.py

Live semantic-command acceptance (uses the same authenticated Modal endpoint
contract as the Streamlit app):
    MODAL_URL=... MODAL_KEY=... MODAL_SECRET=... \
      python tests/run_integrated_chat_qa.py --live

Useful focused runs:
    python tests/run_integrated_chat_qa.py --category conversation_context
    python tests/run_integrated_chat_qa.py --case D059

The test deliberately does *not* compare free-form prose byte-for-byte.
Pass/fail is based on semantic Flow, grounded slots, event IDs, exact counts,
pending clarification, pair feasibility boundaries and the no-invention
contract. Event facts themselves remain sourced from ``data/events.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from command_models import CommandPlan, CommandSlots, validate_command_plan  # noqa: E402
from command_orchestrator import CommandOrchestrator, CommandTurnResult  # noqa: E402
import event_search  # noqa: E402


CORPUS_PATH = ROOT / "tests" / "data" / "integrated_chat_eval.json"
EXPECTED_CASES = 100
EXPECTED_CATEGORY_COUNTS = {
    "catalog_grounding": 30,
    "composed_discovery": 20,
    "count": 8,
    "conversation_context": 15,
    "recommendation": 10,
    "faq_scope": 8,
    "safety_robustness": 9,
}
P0 = "P0"


class CaseFailure(AssertionError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise CaseFailure(message)


def _event_ids(events: list[Mapping[str, Any]]) -> list[str]:
    return [str(event.get("id", "")) for event in events]


def _normalize_text(value: Any) -> str:
    return event_search.normalize_query(str(value)).replace(" ", "").lower()


def _load_corpus() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    cases = raw.get("cases")
    _assert(isinstance(cases, list), "corpus.cases must be a list")
    _assert(len(cases) == EXPECTED_CASES, f"expected {EXPECTED_CASES} cases, got {len(cases)}")
    _assert(raw.get("case_count") == EXPECTED_CASES, "case_count metadata drift")
    ids = [str(case.get("id", "")) for case in cases]
    _assert(len(ids) == len(set(ids)), "case IDs must be unique")
    _assert(all(ids), "case ID must not be empty")
    counts = Counter(str(case.get("category", "")) for case in cases)
    _assert(dict(counts) == EXPECTED_CATEGORY_COUNTS, f"category allocation drift: {dict(counts)}")
    _assert(all(case.get("priority") in {"P0", "P1"} for case in cases), "priority must be P0/P1")
    return raw, cases


def _plan_from_expected(case: Mapping[str, Any]) -> CommandPlan:
    raw = case.get("expected_command")
    _assert(isinstance(raw, Mapping), f"{case.get('id')}: expected_command missing")
    payload = {
        "flow": raw.get("flow"),
        "slots": dict(raw.get("slots") or {}),
        "confidence": "high",
    }
    try:
        return validate_command_plan(payload)
    except Exception as exc:  # corpus authoring error, not an app failure
        raise CaseFailure(f"{case.get('id')}: invalid expected_command: {exc}") from exc


def _validate_catalog_coverage(cases: list[dict[str, Any]], source_ids: set[str]) -> None:
    catalog_cases = [case for case in cases if case.get("category") == "catalog_grounding"]
    covered: set[str] = set()
    for case in catalog_cases:
        exact = (case.get("expected") or {}).get("event_ids_exact")
        _assert(isinstance(exact, list) and len(exact) == 1, f"{case['id']}: catalog case must target one event")
        covered.add(str(exact[0]))
    _assert(covered == source_ids, f"30-event coverage mismatch: missing={sorted(source_ids-covered)} extra={sorted(covered-source_ids)}")


def _state_for_case(case: Mapping[str, Any], reference_date: date) -> dict[str, Any]:
    context = dict(case.get("context") or {})
    state: dict[str, Any] = {
        "reference_date": reference_date.isoformat(),
        "last_result_ids": [str(value) for value in context.get("last_result_ids", [])],
        "selected_event_id": context.get("selected_event_id"),
    }
    if context.get("last_command") is not None:
        state["last_command"] = context["last_command"]
    if context.get("active_flow") is not None:
        state["active_flow"] = context["active_flow"]
    if context.get("pending_slots") is not None:
        state["pending_slots"] = context["pending_slots"]
    if context.get("pending_required_slots") is not None:
        state["pending_required_slots"] = context["pending_required_slots"]
    return state


def _fixture_call(plan: CommandPlan) -> Callable[[Mapping[str, Any]], Any]:
    payload = plan.to_dict()

    def call(_: Mapping[str, Any]) -> Any:
        return payload

    return call


def _live_call_from_env() -> tuple[Callable[[Mapping[str, Any]], Any], dict[str, int]]:
    url = os.environ.get("MODAL_URL", "").strip()
    key = os.environ.get("MODAL_KEY", "").strip()
    secret = os.environ.get("MODAL_SECRET", "").strip()
    missing = [name for name, value in (("MODAL_URL", url), ("MODAL_KEY", key), ("MODAL_SECRET", secret)) if not value]
    if missing:
        raise CaseFailure("--live requires environment variables: " + ", ".join(missing))
    if not url.startswith("https://"):
        raise CaseFailure("MODAL_URL must use https")

    stats = {"calls": 0}

    def call(payload: Mapping[str, Any]) -> Any:
        stats["calls"] += 1
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Modal-Key": key,
                "Modal-Secret": secret,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                if not 200 <= response.status < 300:
                    return None
                decoded = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return None
        if not isinstance(decoded, Mapping) or decoded.get("service_id") != "ehime-kokubunsai-ai-poc":
            return None
        answer = decoded.get("answer")
        return answer if isinstance(answer, (str, Mapping)) else None

    return call, stats


def _slot_matches(expected: Any, actual: Any, *, key: str) -> bool:
    if key == "event_name":
        expected_text = _normalize_text(expected)
        actual_text = _normalize_text(actual)
        return bool(expected_text and actual_text and (expected_text in actual_text or actual_text in expected_text))
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)):
            return False
        if key in {"dates", "municipalities", "regions", "genres", "topics", "time_slots", "detail_fields"}:
            return set(map(str, expected)) <= set(map(str, actual))
        return list(expected) == list(actual)
    return expected == actual


def _assert_semantic_command(case: Mapping[str, Any], result: CommandTurnResult, *, live: bool) -> None:
    expected = case["expected_command"]
    expected_flow = str(expected["flow"])
    _assert(result.flow == expected_flow, f"flow expected={expected_flow} got={result.flow} status={result.status}")
    if not live:
        return
    # Live mode grades the model's semantic parse. Extra slots are allowed,
    # but every explicitly asserted slot must survive grounding/reconciliation.
    actual_slots = result.slots
    for key, value in (expected.get("slots") or {}).items():
        _assert(key in actual_slots, f"missing semantic slot {key!r}; actual={actual_slots}")
        _assert(_slot_matches(value, actual_slots[key], key=key), f"slot {key} expected={value!r} got={actual_slots[key]!r}")


def _assert_global_grounding(result: CommandTurnResult, source_ids: set[str]) -> None:
    returned = set(_event_ids(result.events))
    _assert(returned <= source_ids, f"invented/unknown event IDs returned: {sorted(returned-source_ids)}")
    for pair in result.pairs:
        ids = {str(pair.first_event_id), str(pair.second_event_id)}
        _assert(ids <= source_ids, f"invented/unknown pair IDs returned: {sorted(ids-source_ids)}")


def _assert_expected(case: Mapping[str, Any], result: CommandTurnResult, source_ids: set[str]) -> None:
    expected = dict(case.get("expected") or {})
    ids = _event_ids(result.events)
    id_set = set(ids)

    if "event_ids_exact" in expected:
        wanted = [str(value) for value in expected["event_ids_exact"]]
        _assert(len(ids) == len(wanted) and id_set == set(wanted), f"event_ids_exact expected={wanted} got={ids}")
    if "contains_event_ids" in expected:
        wanted = {str(value) for value in expected["contains_event_ids"]}
        _assert(wanted <= id_set, f"missing expected events {sorted(wanted-id_set)}; got={ids}")
    if "must_not_event_ids" in expected:
        forbidden = {str(value) for value in expected["must_not_event_ids"]}
        _assert(not (forbidden & id_set), f"forbidden event returned: {sorted(forbidden&id_set)}")
    if "total_matches" in expected:
        _assert(result.total_matches == int(expected["total_matches"]), f"count expected={expected['total_matches']} got={result.total_matches}")
    if expected.get("must_have_results"):
        _assert(bool(result.events or result.pairs), "expected at least one grounded result")
    if expected.get("pending_or_clarification"):
        _assert(result.status == "clarification" or bool(result.pending) or bool(result.question), f"expected clarification/pending, got status={result.status}")
    if "pending_required_slots" in expected:
        pending = result.pending or {}
        actual = set(map(str, pending.get("missing_slots", [])))
        wanted = set(map(str, expected["pending_required_slots"]))
        _assert(wanted <= actual, f"pending slots expected={sorted(wanted)} got={sorted(actual)}")
    if "pair_contains_event_ids" in expected:
        wanted = {str(value) for value in expected["pair_contains_event_ids"]}
        pairs = [{str(pair.first_event_id), str(pair.second_event_id)} for pair in result.pairs]
        _assert(any(wanted <= pair for pair in pairs), f"no pair contains {sorted(wanted)}; got={pairs}")
    if expected.get("pairs_must_be_distinct"):
        _assert(bool(result.pairs), "expected at least one event pair")
        _assert(all(str(pair.first_event_id) != str(pair.second_event_id) for pair in result.pairs), "pair repeated the same event")
    if expected.get("no_invented_event"):
        _assert(id_set <= source_ids, "no-invention assertion failed")

    # General FAQ is only a success when the local FAQ actually matched.
    if case["expected_command"]["flow"] == "general_faq":
        _assert(result.total_matches == 1 and bool(result.message.strip()), f"FAQ did not resolve: total={result.total_matches} message={result.message!r}")
    if case["expected_command"]["flow"] == "unsupported":
        _assert(bool(result.message.strip()), "unsupported/safety response must be explicit")


def _run_invalid_date_case(case: Mapping[str, Any], events: list[dict[str, Any]], reference_date: date) -> None:
    result = event_search.search_events(str(case["query"]), events, reference_date)
    _assert(result.intent == "no_results", f"invalid date should be no_results, got {result.intent}")
    _assert(not result.events, f"invalid date returned events: {_event_ids(list(result.events))}")
    _assert(bool(result.filters.invalid_date), "invalid date flag was not set")


def _run_case(
    case: Mapping[str, Any],
    *,
    events: list[dict[str, Any]],
    source_ids: set[str],
    reference_date: date,
    live: bool,
    live_call: Callable[[Mapping[str, Any]], Any] | None,
) -> CommandTurnResult | None:
    if (case.get("expected") or {}).get("invalid_input"):
        _run_invalid_date_case(case, events, reference_date)
        return None

    expected_plan = _plan_from_expected(case)
    call = live_call if live else _fixture_call(expected_plan)
    orchestrator = CommandOrchestrator(call, reference_date=reference_date, events=events)
    result = orchestrator.handle_query(str(case["query"]), _state_for_case(case, reference_date))
    _assert_semantic_command(case, result, live=live)
    _assert_global_grounding(result, source_ids)
    _assert_expected(case, result, source_ids)

    # Security/out-of-scope requests must be rejected before an LLM call.
    if case["expected_command"]["flow"] == "unsupported":
        _assert(result.latency.generator_calls == 0, f"unsupported case spent {result.latency.generator_calls} generator calls")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the authenticated Modal command generator")
    parser.add_argument("--category", action="append", default=[], help="Run only this category (repeatable)")
    parser.add_argument("--case", action="append", default=[], dest="case_ids", help="Run only this case ID (repeatable)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failure")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata, cases = _load_corpus()
    reference_date = date.fromisoformat(str(metadata["reference_date"]))
    events = event_search.load_events()
    source_ids = {str(event["id"]) for event in events}
    _assert(len(events) == 30 and len(source_ids) == 30, "source catalog must contain exactly 30 unique events")
    _validate_catalog_coverage(cases, source_ids)

    selected = cases
    if args.category:
        wanted_categories = set(args.category)
        unknown = wanted_categories - set(EXPECTED_CATEGORY_COUNTS)
        _assert(not unknown, f"unknown categories: {sorted(unknown)}")
        selected = [case for case in selected if case["category"] in wanted_categories]
    if args.case_ids:
        wanted_ids = set(args.case_ids)
        all_ids = {case["id"] for case in cases}
        _assert(not (wanted_ids - all_ids), f"unknown case IDs: {sorted(wanted_ids-all_ids)}")
        selected = [case for case in selected if case["id"] in wanted_ids]
    _assert(bool(selected), "no cases selected")

    live_call = None
    live_stats = {"calls": 0}
    if args.live:
        live_call, live_stats = _live_call_from_env()

    failures: list[tuple[str, str, str, str]] = []
    passed = 0
    category_pass = Counter()
    category_total = Counter(case["category"] for case in selected)

    for case in selected:
        try:
            _run_case(
                case,
                events=events,
                source_ids=source_ids,
                reference_date=reference_date,
                live=args.live,
                live_call=live_call,
            )
        except Exception as exc:
            failures.append((case["id"], case["priority"], case["category"], str(exc)))
            print(f"FAIL {case['id']} [{case['priority']}/{case['category']}]: {exc}")
            if args.fail_fast:
                break
        else:
            passed += 1
            category_pass[case["category"]] += 1
            print(f"PASS {case['id']} [{case['priority']}/{case['category']}] {case['query']}")

    executed = passed + len(failures)
    print("\n=== Integrated Chat Acceptance QA ===")
    print(f"Mode: {'LIVE Modal semantic command' if args.live else 'OFFLINE oracle + deterministic execution'}")
    print(f"Cases: {passed}/{executed} PASS")
    for category in EXPECTED_CATEGORY_COUNTS:
        if category_total[category]:
            print(f"- {category}: {category_pass[category]}/{category_total[category]}")
    p0_failures = [failure for failure in failures if failure[1] == P0]
    print(f"P0 failures: {len(p0_failures)}")
    if args.live:
        print(f"Modal calls: {live_stats['calls']} (security/out-of-scope and invalid-date guards should not spend calls)")
    if failures:
        print("Failures:")
        for case_id, priority, category, message in failures:
            print(f"- {case_id} [{priority}/{category}] {message}")
        return 1
    print("Integrated acceptance gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

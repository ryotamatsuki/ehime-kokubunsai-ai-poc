"""100-case integrated acceptance gate for the cultural-event guide PoC.

Offline:
    python tests/run_integrated_chat_qa.py

Live Semantic Command evaluation:
    MODAL_URL=... MODAL_KEY=... MODAL_SECRET=... \
      python tests/run_integrated_chat_qa.py --live

The oracle is semantic rather than prose-exact: Flow, important slots, grounded
event IDs, counts, clarification state, recommendation pairs and the catalog
no-invention boundary are graded.  Free-form wording is intentionally not.
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
sys.path.insert(0, str(ROOT))

from command_models import CommandPlan, validate_command_plan  # noqa: E402
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


class CaseFailure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CaseFailure(message)


def event_ids(events: list[Mapping[str, Any]]) -> list[str]:
    return [str(event.get("id", "")) for event in events]


def normalized(value: Any) -> str:
    return event_search.normalize_query(str(value)).replace(" ", "").lower()


def load_corpus() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    cases = raw.get("cases")
    check(isinstance(cases, list), "corpus.cases must be a list")
    check(len(cases) == EXPECTED_CASES, f"expected {EXPECTED_CASES} cases, got {len(cases)}")
    check(raw.get("case_count") == EXPECTED_CASES, "case_count metadata drift")
    ids = [str(case.get("id", "")) for case in cases]
    check(all(ids) and len(ids) == len(set(ids)), "case IDs must be non-empty and unique")
    counts = Counter(str(case.get("category", "")) for case in cases)
    check(dict(counts) == EXPECTED_CATEGORY_COUNTS, f"category allocation drift: {dict(counts)}")
    check(all(case.get("priority") in {"P0", "P1"} for case in cases), "priority must be P0/P1")
    return raw, cases


def expected_plan(case: Mapping[str, Any]) -> CommandPlan:
    raw = case.get("expected_command")
    check(isinstance(raw, Mapping), f"{case.get('id')}: expected_command missing")
    try:
        return validate_command_plan({
            "flow": raw.get("flow"),
            "slots": dict(raw.get("slots") or {}),
            "confidence": "high",
        })
    except Exception as exc:
        raise CaseFailure(f"{case.get('id')}: invalid expected_command: {exc}") from exc


def validate_catalog_coverage(cases: list[dict[str, Any]], source_ids: set[str]) -> None:
    covered: set[str] = set()
    for case in cases:
        if case.get("category") != "catalog_grounding":
            continue
        exact = (case.get("expected") or {}).get("event_ids_exact")
        check(isinstance(exact, list) and len(exact) == 1, f"{case['id']}: catalog case must target one event")
        covered.add(str(exact[0]))
    check(covered == source_ids, f"30-event coverage mismatch: missing={sorted(source_ids-covered)} extra={sorted(covered-source_ids)}")


def state_for(case: Mapping[str, Any], reference_date: date) -> dict[str, Any]:
    context = dict(case.get("context") or {})
    state: dict[str, Any] = {
        "reference_date": reference_date.isoformat(),
        "last_result_ids": [str(value) for value in context.get("last_result_ids", [])],
        "selected_event_id": context.get("selected_event_id"),
    }
    for key in ("last_command", "active_flow", "pending_slots", "pending_required_slots"):
        if key in context:
            state[key] = context[key]
    return state


def fixture_call(plan: CommandPlan) -> Callable[[Mapping[str, Any]], Any]:
    payload = plan.to_dict()
    return lambda _: payload


def live_call_from_env() -> tuple[Callable[[Mapping[str, Any]], Any], dict[str, int]]:
    url = os.environ.get("MODAL_URL", "").strip()
    key = os.environ.get("MODAL_KEY", "").strip()
    secret = os.environ.get("MODAL_SECRET", "").strip()
    missing = [name for name, value in (("MODAL_URL", url), ("MODAL_KEY", key), ("MODAL_SECRET", secret)) if not value]
    check(not missing, "--live requires environment variables: " + ", ".join(missing))
    check(url.startswith("https://"), "MODAL_URL must use https")
    stats = {"calls": 0}

    def call(payload: Mapping[str, Any]) -> Any:
        stats["calls"] += 1
        request = Request(
            url,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={"Modal-Key": key, "Modal-Secret": secret, "Content-Type": "application/json"},
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


def slot_matches(expected: Any, actual: Any, key: str) -> bool:
    if key == "event_name":
        a, b = normalized(expected), normalized(actual)
        return bool(a and b and (a in b or b in a))
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)):
            return False
        return set(map(str, expected)) <= set(map(str, actual))
    return expected == actual


def assert_semantic_command(case: Mapping[str, Any], result: CommandTurnResult, live: bool) -> None:
    expected = case["expected_command"]
    check(result.flow == expected["flow"], f"flow expected={expected['flow']} got={result.flow} status={result.status}")
    if not live:
        return
    for key, value in (expected.get("slots") or {}).items():
        check(key in result.slots, f"missing semantic slot {key!r}; actual={result.slots}")
        check(slot_matches(value, result.slots[key], key), f"slot {key} expected={value!r} got={result.slots[key]!r}")


def assert_grounding(result: CommandTurnResult, source_ids: set[str]) -> None:
    returned = set(event_ids(result.events))
    check(returned <= source_ids, f"invented/unknown event IDs: {sorted(returned-source_ids)}")
    for pair in result.pairs:
        pair_ids = {str(pair.first_event_id), str(pair.second_event_id)}
        check(pair_ids <= source_ids, f"invented/unknown pair IDs: {sorted(pair_ids-source_ids)}")


def assert_expected(case: Mapping[str, Any], result: CommandTurnResult, source_ids: set[str]) -> None:
    expected = dict(case.get("expected") or {})
    ids = event_ids(result.events)
    id_set = set(ids)

    if "event_ids_exact" in expected:
        wanted = [str(v) for v in expected["event_ids_exact"]]
        check(len(ids) == len(wanted) and id_set == set(wanted), f"event_ids_exact expected={wanted} got={ids}")
    if "contains_event_ids" in expected:
        wanted = {str(v) for v in expected["contains_event_ids"]}
        check(wanted <= id_set, f"missing expected events {sorted(wanted-id_set)}; got={ids}")
    if "must_not_event_ids" in expected:
        forbidden = {str(v) for v in expected["must_not_event_ids"]}
        check(not (forbidden & id_set), f"forbidden event returned: {sorted(forbidden&id_set)}")
    if "total_matches" in expected:
        check(result.total_matches == int(expected["total_matches"]), f"count expected={expected['total_matches']} got={result.total_matches}")
    if expected.get("must_have_results"):
        check(bool(result.events or result.pairs), "expected at least one grounded result")
    if expected.get("pending_or_clarification"):
        check(result.status == "clarification" or bool(result.pending) or bool(result.question), f"expected clarification, got status={result.status}")
    if "pending_required_slots" in expected:
        actual = set(map(str, (result.pending or {}).get("missing_slots", [])))
        wanted = set(map(str, expected["pending_required_slots"]))
        check(wanted <= actual, f"pending slots expected={sorted(wanted)} got={sorted(actual)}")
    if "pair_contains_event_ids" in expected:
        wanted = {str(v) for v in expected["pair_contains_event_ids"]}
        pairs = [{str(p.first_event_id), str(p.second_event_id)} for p in result.pairs]
        check(any(wanted <= pair for pair in pairs), f"no pair contains {sorted(wanted)}; got={pairs}")
    if expected.get("pairs_must_be_distinct"):
        check(bool(result.pairs), "expected at least one event pair")
        check(all(str(p.first_event_id) != str(p.second_event_id) for p in result.pairs), "pair repeated the same event")
    if expected.get("no_invented_event"):
        check(id_set <= source_ids, "no-invention assertion failed")

    flow = case["expected_command"]["flow"]
    if flow == "general_faq":
        check(result.total_matches == 1 and bool(result.message.strip()), f"FAQ did not resolve: total={result.total_matches} message={result.message!r}")
    if flow == "unsupported":
        check(bool(result.message.strip()), "unsupported/safety response must be explicit")


def run_invalid_date(case: Mapping[str, Any], events: list[dict[str, Any]], reference_date: date) -> None:
    result = event_search.search_events(str(case["query"]), events, reference_date)
    check(result.intent == "no_results", f"invalid date should be no_results, got {result.intent}")
    check(not result.events and bool(result.filters.invalid_date), "invalid date must return zero events with invalid_date=true")


def run_case(
    case: Mapping[str, Any],
    events: list[dict[str, Any]],
    source_ids: set[str],
    reference_date: date,
    live: bool,
    live_call: Callable[[Mapping[str, Any]], Any] | None,
) -> None:
    if (case.get("expected") or {}).get("invalid_input"):
        run_invalid_date(case, events, reference_date)
        return

    plan = expected_plan(case)
    orchestrator = CommandOrchestrator(live_call if live else fixture_call(plan), reference_date=reference_date, events=events)
    result = orchestrator.handle_query(str(case["query"]), state_for(case, reference_date))
    assert_semantic_command(case, result, live)
    assert_grounding(result, source_ids)
    assert_expected(case, result, source_ids)

    # Only inputs already recognized by the deterministic pre-LLM guard are
    # required to spend zero generator calls.  An unsupported request that is
    # safely classified by the bounded Semantic Command is still acceptable.
    guard_intent = event_search.classify_intent(str(case["query"]))
    if guard_intent in {"injection", "out_of_scope"}:
        check(result.latency.generator_calls == 0, f"deterministic guard spent {result.latency.generator_calls} generator calls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--case", action="append", default=[], dest="case_ids")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata, cases = load_corpus()
    reference_date = date.fromisoformat(str(metadata["reference_date"]))
    events = event_search.load_events()
    source_ids = {str(event["id"]) for event in events}
    check(len(events) == 30 and len(source_ids) == 30, "source catalog must contain 30 unique events")
    validate_catalog_coverage(cases, source_ids)

    selected = cases
    if args.category:
        wanted = set(args.category)
        check(not (wanted - set(EXPECTED_CATEGORY_COUNTS)), f"unknown categories: {sorted(wanted-set(EXPECTED_CATEGORY_COUNTS))}")
        selected = [case for case in selected if case["category"] in wanted]
    if args.case_ids:
        wanted = set(args.case_ids)
        all_ids = {case["id"] for case in cases}
        check(not (wanted - all_ids), f"unknown case IDs: {sorted(wanted-all_ids)}")
        selected = [case for case in selected if case["id"] in wanted]
    check(bool(selected), "no cases selected")

    live_call = None
    live_stats = {"calls": 0}
    if args.live:
        live_call, live_stats = live_call_from_env()

    failures: list[tuple[str, str, str, str]] = []
    passed = 0
    category_pass = Counter()
    category_total = Counter(case["category"] for case in selected)
    for case in selected:
        try:
            run_case(case, events, source_ids, reference_date, args.live, live_call)
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
    print(f"P0 failures: {sum(priority == 'P0' for _, priority, _, _ in failures)}")
    if args.live:
        print(f"Modal calls: {live_stats['calls']}")
    if failures:
        print("Failures:")
        for case_id, priority, category, message in failures:
            print(f"- {case_id} [{priority}/{category}] {message}")
        return 1
    print("Integrated acceptance gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

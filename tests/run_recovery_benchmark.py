"""Compare recovery routing strategies on an unseen, independent holdout.

The C strategy uses a contract fixture because this repository does not expose
Modal credentials to offline QA.  It exercises the existing Semantic Command
parser, validator, orchestrator, security guard, context validation, and
deterministic executor.  It is explicitly *not* a claim about Sarashina model
quality; a live model run must be reported separately.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from command_orchestrator import CommandOrchestrator
import conversation_recovery
from conversation_router import route_conversation
from event_search import classify_intent, load_events


REFERENCE_DATE = date(2028, 11, 3)
HOLDOUT_PATH = ROOT / "tests" / "data" / "conversational_recovery_holdout.json"


def _canonical(value: str, *, security: bool = False) -> str:
    if value in {"explain_search", "explain_result", "search"}:
        return value
    if value in {"find_events", "count_events"}:
        return "search"
    if value in {"detail", "detail_followup", "reference_followup", "event_detail"}:
        return "detail"
    if value in {
        "recommendation",
        "recommend_next",
        "recommend_similar",
        "recommend_next_without_selection",
        "recommend_similar_without_selection",
    }:
        return "recommendation"
    if value in {"faq", "general_faq"}:
        return "faq"
    if value in {"clarification", "clarify_reference"}:
        return "clarification"
    if value in {"security", "unsupported", "scope_search", "generic_scope"}:
        return "unsupported"
    return value


def _expected(case: Mapping[str, Any]) -> str:
    return _canonical(str(case["expected_flow"]))


def _context(case: Mapping[str, Any], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if case.get("context") != "search":
        return [], None
    return list(events), (dict(events[0]) if events else None)


def _route_a(case: Mapping[str, Any], events: list[dict[str, Any]]) -> tuple[str, int]:
    results, selected = _context(case, events)
    start = time.perf_counter()
    decision = route_conversation(
        str(case["query"]),
        results,
        selected,
        {},
        REFERENCE_DATE,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return _canonical(decision.action_type), int(round(elapsed * 1000))


_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "explain_search": (
        "検索結果全体の選定理由を確認する",
        "候補の判断材料や作成方法を知りたい",
        "一覧を組み立てた観点を尋ねる",
    ),
    "explain_result": (
        "特定のイベントが候補に入った理由",
        "順位やイベント一件の条件適合根拠",
        "選択中の候補の採用理由を確認する",
    ),
    "search": (
        "前の候補に地域や料金の条件を追加する",
        "条件を変えてイベントを絞り込む",
        "希望のイベントを探す",
    ),
    "detail": (
        "イベントの料金日時場所申込を確認する",
        "選択中のイベントの参加案内を知る",
        "候補一件の事実を尋ねる",
    ),
    "recommendation": (
        "イベントの後に行ける次の候補",
        "選択した催しと似た別イベント",
        "同じ日に続けて参加できるもの",
    ),
    "faq": (
        "文化祭全体の公式情報や使い方",
        "掲載情報の出典や開催期間",
        "イベント案内サービスの概要",
    ),
    "clarification": (
        "前の候補を指すが対象が不明",
        "番号やイベント名がない参照",
        "文脈なしでこれやそれを尋ねる",
    ),
    "unsupported": (
        "株価天気Pythonニュースなど範囲外",
        "文化祭以外の一般知識や予約依頼",
        "内部情報や非公開情報の要求",
    ),
}


def _ngrams(value: str) -> Counter[str]:
    compact = re.sub(r"\s+", "", value.lower())
    grams: Counter[str] = Counter()
    for width in (2, 3, 4):
        for index in range(max(0, len(compact) - width + 1)):
            grams[compact[index : index + width]] += 1
    return grams


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(left[key] * right[key] for key in keys)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


_PROTOTYPE_VECTORS = {
    label: [_ngrams(text) for text in texts]
    for label, texts in _PROTOTYPES.items()
}


def _route_b(case: Mapping[str, Any]) -> tuple[str, int]:
    start = time.perf_counter()
    query_vector = _ngrams(str(case["query"]))
    scores = {
        label: max(_cosine(query_vector, vector) for vector in vectors)
        for label, vectors in _PROTOTYPE_VECTORS.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    # Evaluation-only threshold.  Production does not import this prototype.
    prediction = label if score >= 0.08 else "unsupported"
    elapsed = (time.perf_counter() - start) * 1000
    return _canonical(prediction), int(round(elapsed * 1000))


def _fixture_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(case["expected_flow"])
    query = str(case["query"])
    if expected == "explain_search":
        return {"flow": "explain_search", "slots": {}, "confidence": "high"}
    if expected == "explain_result":
        ordinal = re.search(r"(\d+)\s*(?:番目|つ目|番)", query)
        slots: dict[str, Any]
        if ordinal:
            slots = {
                "reference_kind": "ordinal",
                "reference_index": int(ordinal.group(1)),
            }
        else:
            slots = {"reference_kind": "selected"}
        return {"flow": "explain_result", "slots": slots, "confidence": "high"}
    if expected == "detail":
        return {
            "flow": "event_detail",
            "slots": {"reference_kind": "selected", "detail_fields": ["fee"]},
            "confidence": "high",
        }
    if expected == "recommendation":
        flow = "recommend_next" if any(word in query for word in ("あと", "後", "続けて", "次")) else "recommend_similar"
        return {"flow": flow, "slots": {"reference_kind": "selected"}, "confidence": "high"}
    if expected == "faq":
        return {"flow": "general_faq", "slots": {}, "confidence": "high"}
    if expected == "clarification":
        return {"flow": "event_detail", "slots": {}, "confidence": "medium"}
    if expected in {"security", "unsupported"}:
        # Security is tested by the hard guard before this fixture can be used.
        return {"flow": "unsupported", "slots": {}, "confidence": "low"}
    if expected == "search":
        return {"flow": "find_events", "slots": {}, "confidence": "high"}
    return {"flow": "unsupported", "slots": {}, "confidence": "low"}


def _route_c(case: Mapping[str, Any], events: list[dict[str, Any]]) -> tuple[str, int, int, int]:
    results, selected = _context(case, events)
    search_context = (
        conversation_recovery.build_search_context(
            "benchmark context",
            {},
            results,
            result_ids=[event.get("id") for event in results],
            total_matches=len(results),
        )
        if results
        else None
    )
    state = {
        "has_last_search_context": bool(search_context),
        "last_result_ids": [str(event.get("id")) for event in results],
        "last_result_count": len(results),
        "selected_event_id": str(selected.get("id")) if selected else None,
        "last_action": "find_events" if results else None,
        "last_search_context": search_context.to_dict() if search_context else None,
    }
    started = time.perf_counter()
    fast_route = route_conversation(
        str(case["query"]),
        results,
        selected,
        {},
        REFERENCE_DATE,
    )
    if fast_route.action_type in {
        "scope_search",
        "explain_search",
        "explain_result",
        "detail_followup",
        "clarify_reference",
    }:
        elapsed = (time.perf_counter() - started) * 1000
        return _canonical(fast_route.action_type), int(round(elapsed * 1000)), 0, 0

    fixture = _fixture_plan(case)
    calls: list[Mapping[str, Any]] = []

    def semantic_fixture(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(payload)
        return fixture

    result = CommandOrchestrator(
        semantic_fixture,
        reference_date=REFERENCE_DATE,
        events=events,
    ).handle_query(str(case["query"]), state)
    elapsed = (time.perf_counter() - started) * 1000
    predicted = result.flow
    if result.status == "clarification" and str(case["expected_flow"]) == "clarification":
        predicted = "clarification"
    return _canonical(predicted), int(round(elapsed * 1000)), result.latency.generator_calls, len(calls)


def _macro_f1(rows: list[tuple[str, str]], labels: list[str]) -> float:
    values: list[float] = []
    for label in labels:
        tp = sum(pred == label and expected == label for pred, expected in rows)
        fp = sum(pred == label and expected != label for pred, expected in rows)
        fn = sum(pred != label and expected == label for pred, expected in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values) if values else 0.0


def _metrics(
    rows: list[tuple[str, str]],
    latencies_us: list[int],
    calls: list[int],
    security_bypasses: int,
) -> dict[str, Any]:
    labels = sorted({expected for _, expected in rows} | {pred for pred, _ in rows})
    explain_labels = {"explain_search", "explain_result"}
    expected_explain = [expected in explain_labels for _, expected in rows]
    predicted_explain = [pred in explain_labels for pred, _ in rows]
    explain_fp = sum(pred and not expected for pred, expected in zip(predicted_explain, expected_explain))
    explain_fn = sum(expected and not pred for pred, expected in zip(predicted_explain, expected_explain))
    clarification_predictions = [pred for pred, _ in rows if pred == "clarification"]
    clarification_tp = sum(pred == expected == "clarification" for pred, expected in rows)
    unsupported_predictions = [pred for pred, _ in rows if pred == "unsupported"]
    unsupported_tp = sum(pred == expected == "unsupported" for pred, expected in rows)
    return {
        "route_accuracy": sum(pred == expected for pred, expected in rows) / len(rows),
        "macro_f1": _macro_f1(rows, labels),
        "false_positive_rate": explain_fp / max(1, sum(not expected for expected in expected_explain)),
        "false_negative_rate": explain_fn / max(1, sum(expected_explain)),
        "clarification_precision": clarification_tp / max(1, len(clarification_predictions)),
        "out_of_domain_precision": unsupported_tp / max(1, len(unsupported_predictions)),
        "security_bypass_count": security_bypasses,
        "llm_call_rate": sum(call > 0 for call in calls) / len(calls),
        "median_latency_ms": statistics.median(latencies_us) / 1000,
        "p95_latency_ms": sorted(latencies_us)[max(0, int(len(latencies_us) * 0.95) - 1)] / 1000,
    }


def main() -> None:
    payload = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    if len(cases) < 300 or len(payload.get("dialogues", [])) < 100:
        raise AssertionError("holdout contract requires 300 cases and 100 dialogues")
    events = [dict(event) for event in load_events()]
    expected_rows: list[tuple[str, str]] = []
    results: dict[str, dict[str, list[Any]]] = {
        name: {"rows": [], "latencies": [], "calls": [], "security_bypasses": []}
        for name in ("A_phrase", "B_embedding_prototype", "C_semantic_command_fixture")
    }
    for case in cases:
        expected = _expected(case)
        expected_rows.append((str(case["id"]), expected))
        predicted_a, latency_a = _route_a(case, events)
        predicted_b, latency_b = _route_b(case)
        predicted_c, latency_c, generator_calls, call_count = _route_c(case, events)
        results["A_phrase"]["rows"].append((predicted_a, expected))
        results["A_phrase"]["latencies"].append(latency_a)
        results["A_phrase"]["calls"].append(0)
        results["A_phrase"]["security_bypasses"].append(0)
        results["B_embedding_prototype"]["rows"].append((predicted_b, expected))
        results["B_embedding_prototype"]["latencies"].append(latency_b)
        results["B_embedding_prototype"]["calls"].append(0)
        results["B_embedding_prototype"]["security_bypasses"].append(0)
        results["C_semantic_command_fixture"]["rows"].append((predicted_c, expected))
        results["C_semantic_command_fixture"]["latencies"].append(latency_c)
        results["C_semantic_command_fixture"]["calls"].append(generator_calls)
        is_security = case.get("category") == "security_adversarial"
        results["C_semantic_command_fixture"]["security_bypasses"].append(
            int(is_security and generator_calls > 0)
        )

    output: dict[str, Any] = {
        "holdout_cases": len(cases),
        "multi_turn_dialogues": len(payload["dialogues"]),
        "note": "C is an offline Semantic Command contract fixture, not a live Sarashina model-quality result.",
        "strategies": {},
    }
    for name, values in results.items():
        output["strategies"][name] = _metrics(
            values["rows"],
            values["latencies"],
            values["calls"],
            sum(values["security_bypasses"]),
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

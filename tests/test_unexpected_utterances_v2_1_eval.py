from pathlib import Path

from unexpected_utterances_v2_1_eval import (
    ObservableRemoteFrameCall,
    _latency_summary,
    evaluate_case_v21,
    load_frozen_v1_dataset,
    summarize_v21,
)


def _service(answer: str, *, generation_ms: float = 12.5):
    return {
        "answer": answer,
        "observability": {
            "service_id": "test",
            "server_total_ms": generation_ms + 2.0,
            "generation_ms": generation_ms,
            "prompt_tokens": 100,
            "generated_tokens": 9,
        },
    }


def test_frozen_v1_loader_is_exact_and_does_not_need_holdout():
    dataset = load_frozen_v1_dataset()
    assert dataset["version"] == "unexpected-user-utterances-v1"
    assert len(dataset["cases"]) == 100


def test_observable_remote_call_captures_raw_and_latency():
    remote = ObservableRemoteFrameCall(lambda _: _service('{"intent":"search"}'))
    response = remote({"query": "なんかおすすめある？", "state": {}})
    assert response["answer"] == '{"intent":"search"}'
    stats = remote.stats()
    assert stats["calls"] == 1
    assert stats["prompt_tokens"] == 100
    assert stats["generated_tokens"] == 9
    assert stats["generation_ms"] == 12.5
    assert stats["calls_detail"][0]["raw_output"] == '{"intent":"search"}'
    assert stats["calls_detail"][0]["client_elapsed_ms"] >= 0


def test_observable_remote_call_can_disable_raw_capture():
    remote = ObservableRemoteFrameCall(
        lambda _: _service('{"intent":"search"}'), include_raw=False
    )
    remote({"query": "x", "state": {}})
    assert remote.stats()["calls_detail"][0]["raw_output"] is None


def test_evaluate_case_preserves_raw_frame_and_service_metrics():
    case = {
        "id": "TEST-001",
        "category": "underspecified",
        "risk": "broad",
        "query": "なんかおすすめある？",
        "context": "none",
        "expected_behavior": "broad_search",
        "expected": {
            "allowed_flows": ["find_events"],
            "allowed_statuses": ["ok"],
            "required_slots": {},
            "forbidden_slots": [],
            "must_not_auto_relax": False,
        },
        "manual_review": False,
        "review_focus": [],
    }
    row = evaluate_case_v21(case, lambda _: _service('{"intent":"search"}'))
    assert row["machine_pass"] is True
    assert row["frame"] == {"intent": "search"}
    assert row["observability"]["calls"] == 1
    assert row["observability"]["calls_detail"][0]["raw_output"] == '{"intent":"search"}'
    assert row["orchestrator_latency_ms"] >= 0


def test_repair_observability_keeps_both_raw_frames():
    responses = iter([
        _service("not json", generation_ms=5.0),
        _service('{"intent":"search"}', generation_ms=6.0),
    ])
    case = {
        "id": "TEST-002",
        "category": "underspecified",
        "risk": "repair",
        "query": "なんかおすすめある？",
        "context": "none",
        "expected_behavior": "broad_search",
        "expected": {"allowed_flows": ["find_events"], "allowed_statuses": ["ok"]},
        "manual_review": False,
        "review_focus": [],
    }
    row = evaluate_case_v21(case, lambda _: next(responses))
    details = row["observability"]["calls_detail"]
    assert row["frame_repaired"] is True
    assert row["repair_success"] is True
    assert len(details) == 2
    assert details[0]["raw_output"] == "not json"
    assert details[1]["raw_output"] == '{"intent":"search"}'
    assert row["observability"]["generation_ms"] == 11.0


def test_summary_exposes_latency_and_tokens():
    rows = [
        {
            "category": "x",
            "machine_pass": True,
            "manual_review": False,
            "verdict": "pass",
            "frame_attempts": 1,
            "first_pass_frame_valid": True,
            "repair_success": False,
            "failures": [],
            "orchestrator_latency_ms": 20.0,
            "observability": {
                "calls": 1,
                "client_elapsed_ms": 15.0,
                "server_total_ms": 13.0,
                "generation_ms": 10.0,
                "prompt_tokens": 100,
                "generated_tokens": 8,
            },
        },
        {
            "category": "x",
            "machine_pass": True,
            "manual_review": False,
            "verdict": "pass",
            "frame_attempts": 0,
            "first_pass_frame_valid": False,
            "repair_success": False,
            "failures": [],
            "orchestrator_latency_ms": 2.0,
            "observability": {"calls": 0},
        },
    ]
    summary = summarize_v21(rows)
    assert summary["zero_model_call_cases"] == 1
    assert summary["prompt_tokens"] == 100
    assert summary["generated_tokens"] == 8
    assert summary["latency"]["generation_model_cases"]["median_ms"] == 10.0
    assert _latency_summary([1.0, 2.0, 3.0])["p95_ms"] > 2.0


def test_live_harness_has_no_holdout_dataset_entrypoint():
    source = (Path(__file__).resolve().parents[1] / "semantic_v2_1_live_eval.py").read_text(encoding="utf-8")
    assert "tests/data/unexpected_utterances_holdout_v2_1" not in source
    assert "payload.json.gz" not in source
    assert "def frozen_v1_eval" in source
    assert "SPARSE_FRAME_JSON_SCHEMA" in source
    assert "lm-format-enforcer" in source

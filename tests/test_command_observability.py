from __future__ import annotations

from command_generator import generate_command
from command_observability import ModalCallError, TurnObservation


def test_turn_observation_contains_no_raw_query() -> None:
    observation = TurnObservation("何を材料にこの候補を出したの？", has_search_context=True)
    observation.deterministic_route = "semantic_command"
    payload = observation.to_dict()
    assert payload["query_hash"]
    assert payload["query_category"] == "explanation_or_reference"
    assert "何を材料" not in str(payload)


def test_generation_distinguishes_invalid_json_and_repair_success() -> None:
    calls = []

    def repair_once(payload):
        calls.append(payload)
        return "not-json" if len(calls) == 1 else {"flow": "explain_search", "slots": {}}

    result = generate_command("この結果の理由は？", {}, call=repair_once)
    assert result.first_pass_valid is False
    assert result.first_error_type == "invalid_json"
    assert result.repair_success is True
    assert result.error is None


def test_generation_classifies_modal_timeout_without_exposing_details() -> None:
    def timeout(_payload):
        raise ModalCallError("modal_timeout")

    result = generate_command("イベントを探して", {}, call=timeout)
    assert result.error_type == "modal_timeout"
    assert result.modal_status_class is None
    assert "イベントを探して" not in str(result)


from __future__ import annotations

from command_models import COMMAND_SLOT_FIELDS, FLOW_NAMES, MAX_REFERENCE_INDEX
from command_schema import COMMAND_JSON_SCHEMA


def test_generation_schema_tracks_python_flow_and_reference_contract() -> None:
    assert set(COMMAND_JSON_SCHEMA["properties"]["flow"]["enum"]) == set(FLOW_NAMES)
    slot_schema = COMMAND_JSON_SCHEMA["properties"]["slots"]["properties"]
    assert set(slot_schema) == set(COMMAND_SLOT_FIELDS)
    assert slot_schema["reference_index"]["maximum"] == MAX_REFERENCE_INDEX


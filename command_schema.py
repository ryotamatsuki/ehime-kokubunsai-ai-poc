"""Shared JSON grammar contract for Semantic Command generation.

The grammar is only a generation aid.  ``command_models`` remains the
authoritative semantic validator after model output is received.
"""

from __future__ import annotations

from typing import Any

from command_models import (
    ALLOWED_AGE_GROUPS,
    ALLOWED_AUDIENCES,
    ALLOWED_DETAIL_FIELDS,
    ALLOWED_EXPERIENCE_CONCEPTS,
    ALLOWED_REFERENCE_KINDS,
    ALLOWED_TIME_SLOTS,
    ALLOWED_VENUES,
    FLOW_NAMES,
    GENRE_VALUES,
    MAX_REFERENCE_INDEX,
)


def build_command_json_schema() -> dict[str, Any]:
    """Build the bounded schema from the same constants as Python validation."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["flow", "slots"],
        "properties": {
            "flow": {"type": "string", "enum": sorted(FLOW_NAMES)},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dates": {"type": "array", "items": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
                    "municipalities": {"type": "array", "items": {"type": "string"}},
                    "regions": {"type": "array", "items": {"type": "string"}},
                    "genres": {"type": "array", "items": {"type": "string", "enum": sorted(GENRE_VALUES)}},
                    "topics": {"type": "array", "items": {"type": "string", "maxLength": 64}},
                    "experience_required": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_EXPERIENCE_CONCEPTS), "maxLength": 32}, "maxItems": 8},
                    "experience_preferred": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_EXPERIENCE_CONCEPTS), "maxLength": 32}, "maxItems": 8},
                    "experience_excluded": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_EXPERIENCE_CONCEPTS), "maxLength": 32}, "maxItems": 8},
                    "audience": {"type": "string", "enum": sorted(ALLOWED_AUDIENCES)},
                    "age": {"type": "integer", "minimum": 0, "maximum": 120},
                    "age_group": {"type": "string", "enum": sorted(ALLOWED_AGE_GROUPS)},
                    "age_intent": {"type": "string", "enum": ["recommended", "eligible"]},
                    "venue": {"type": "string", "enum": sorted(ALLOWED_VENUES)},
                    "entry_free": {"type": "boolean"},
                    "paid_only": {"type": "boolean"},
                    "max_entry_fee": {"type": "integer", "minimum": 0},
                    "reservation_required": {"type": "boolean"},
                    "rain_preferred": {"type": "boolean"},
                    "time_slots": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_TIME_SLOTS)}},
                    "time_after": {"type": "integer", "minimum": 0, "maximum": 1440},
                    "visit_count": {"type": "integer", "minimum": 1, "maximum": 2},
                    "reference_kind": {"type": "string", "enum": sorted(ALLOWED_REFERENCE_KINDS)},
                    "reference_index": {"type": "integer", "minimum": 1, "maximum": MAX_REFERENCE_INDEX},
                    "event_name": {"type": "string", "maxLength": 240},
                    "detail_fields": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_DETAIL_FIELDS)}},
                    "refine_previous": {"type": "boolean"},
                },
            },
        },
    }


COMMAND_JSON_SCHEMA = build_command_json_schema()


__all__ = ["COMMAND_JSON_SCHEMA", "build_command_json_schema"]

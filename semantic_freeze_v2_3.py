"""Reproducible architecture fingerprints for Semantic Operations v2.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from semantic_atomic_v2_3 import ATOMIC_FRAME_JSON_SCHEMA_V23, build_atomic_frame_system_prompt_v23
from semantic_capability_registry_v2_3 import registry_snapshot
from semantic_model_registry import MODEL_SPECS
from semantic_prompt_v2_3 import ATOMIC_FEW_SHOT_EXAMPLES_V23


ROOT = Path(__file__).resolve().parent
ARCHITECTURE_VERSION = "semantic-operations-v2.3-evidence-bounded"
SEALED_HOLDOUT_SHA256 = "c844dda17248c0e7f16cd2985652e62bb0f8b601bf21196d6801479580899c92"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _source_sha256(path: str) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def freeze_snapshot() -> dict[str, Any]:
    prompt_contract = {
        "system_prompt": build_atomic_frame_system_prompt_v23(),
        "few_shot_examples": ATOMIC_FEW_SHOT_EXAMPLES_V23,
        # Include builder source so payload/state exposure rules are frozen too.
        "prompt_module_source_sha256": _source_sha256("semantic_prompt_v2_3.py"),
    }
    models = [
        {"key": spec.key, "model_id": spec.model_id}
        for spec in MODEL_SPECS
    ]
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "hash_algorithm": "sha256",
        "schema_hash": _json_sha256(ATOMIC_FRAME_JSON_SCHEMA_V23),
        "prompt_hash": _json_sha256(prompt_contract),
        "few_shot_count": len(ATOMIC_FEW_SHOT_EXAMPLES_V23),
        "capability_registry_hash": _json_sha256(registry_snapshot()),
        "grounding_hash": _source_sha256("semantic_grounding_v2_3.py"),
        "verifier_hash": _source_sha256("semantic_verifier_v2_3.py"),
        "state_reducer_hash": _source_sha256("semantic_state_v2_3.py"),
        "orchestrator_hash": _source_sha256("semantic_orchestrator_v2_3.py"),
        "evaluator_hash": _source_sha256("unexpected_utterances_v2_3_eval.py"),
        "oracle_hash": _source_sha256("semantic_oracle_ceiling_v2_3.py"),
        "multimodel_backend_hash": _source_sha256("semantic_v2_3_multimodel_backend.py"),
        "model_registry_hash": _source_sha256("semantic_model_registry.py"),
        "models": models,
        "lmfe": {
            "enabled": True,
            "package_constraint": "lm-format-enforcer>=0.10,<1",
            "schema": "ATOMIC_FRAME_JSON_SCHEMA_V23",
        },
        "decoding": {
            "do_sample": False,
            "repetition_penalty": 1.02,
            "max_new_tokens": 220,
        },
        "max_model_calls": 1,
        "repair_generation": False,
        "sealed_holdout": {
            "opened": False,
            "executed": False,
            "sha256": SEALED_HOLDOUT_SHA256,
        },
        "production": {
            "main_modified": False,
            "deployed": False,
        },
    }


def validate_freeze_manifest(raw: Mapping[str, Any]) -> None:
    expected = freeze_snapshot()
    actual = dict(raw)
    if actual != expected:
        keys = sorted(set(expected) | set(actual))
        mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
        raise AssertionError(f"v2.3 freeze manifest mismatch: {mismatches}")


if __name__ == "__main__":
    print(json.dumps(freeze_snapshot(), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["ARCHITECTURE_VERSION", "freeze_snapshot", "validate_freeze_manifest"]

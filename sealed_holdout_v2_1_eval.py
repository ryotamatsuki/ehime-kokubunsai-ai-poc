"""One-time Semantic Operations v2.3 sealed-holdout evaluation.

This module is intentionally evaluation-only.  It does not import or inspect
the sealed payload at module import time.  The payload is read exactly once,
after the caller explicitly passes ``--confirm-consume`` and after the
preflight/smoke gate has completed.

The runner uses the already-frozen v2.3 orchestrator and the selected
Sarashina backend.  It adds only evaluation transport, trace capture,
aggregation, manual-review bookkeeping and report rendering.  It must not be
used for prompt/rule/evaluator tuning after the holdout has been consumed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import modal

from app_config import POC_REFERENCE_DATE
from command_models import CommandSlots, FLOW_NAMES
from semantic_atomic_v2_3 import AtomicSemanticFrameV23
from semantic_freeze_v2_3 import freeze_snapshot, validate_freeze_manifest
from semantic_model_registry import DEFAULT_MODEL_KEY, MODEL_BY_KEY
from semantic_orchestrator_v2_3 import SemanticOperationsOrchestratorV23
from semantic_evidence_v2_3 import AllowedSemanticAction, SemanticResolution
from unexpected_utterances_eval import ALLOWED_STATUSES, _seed_state, _slot_subset, _forbidden_present
from unexpected_utterances_v2_3_eval import evaluate_case_v23
from unexpected_utterances_v2_1_eval import ObservableRemoteFrameCall, _latency_summary


ROOT = Path(__file__).resolve().parent
HOLDOUT_ROOT = ROOT / "tests" / "data" / "unexpected_utterances_holdout_v2_1"
HOLDOUT_MANIFEST_PATH = HOLDOUT_ROOT / "manifest.json"
HOLDOUT_PAYLOAD_PATH = HOLDOUT_ROOT / "holdout.json.gz"
ARCHITECTURE_FROZEN_SHA = "e1b98c21ac8b374686a083908dfcb235a9324456"
EXPECTED_COMPRESSED_SHA256 = "c844dda17248c0e7f16cd2985652e62bb0f8b601bf21196d6801479580899c92"
EXPECTED_MODEL_KEY = "sarashina-2.2-3b"
EXPECTED_MODEL_ID = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
EXPECTED_PROTOCOL = "semantic-v2.3-evidence-bounded-lmfe-v1"
SERVICE_ID = "ehime-kokubunsai-semantic-v2-3-api"
SERVICE_CLASS = "SarashinaSemanticV23"
MAX_TRANSPORT_RETRIES = 3
Z95 = 1.959963984540054

app = modal.App("ehime-kokubunsai-semantic-v2-3-holdout-eval")


class HoldoutIntegrityError(RuntimeError):
    """Raised when the sealed corpus or evaluation snapshot is invalid."""


class TransportFailure(RuntimeError):
    """A retryable infrastructure failure, never a semantic retry."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(command: Sequence[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip() or default
    except (OSError, subprocess.SubprocessError, ValueError):
        return default


def _package_versions() -> dict[str, str | None]:
    names = (
        "Python",
        "pytest",
        "streamlit",
        "requests",
        "modal",
        "lm-format-enforcer",
        "torch",
        "transformers",
        "bitsandbytes",
    )
    result: dict[str, str | None] = {"Python": platform.python_version()}
    for name in names[1:]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _read_holdout_manifest() -> dict[str, Any]:
    """Read only the non-secret metadata manifest, never the gzip payload."""

    raw = json.loads(HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HoldoutIntegrityError("holdout manifest must be an object")
    if raw.get("total_cases") != 200:
        raise HoldoutIntegrityError(f"holdout manifest total_cases != 200: {raw.get('total_cases')!r}")
    if raw.get("payload_sha256") != EXPECTED_COMPRESSED_SHA256:
        raise HoldoutIntegrityError("holdout manifest compressed SHA mismatch")
    if raw.get("payload") != HOLDOUT_PAYLOAD_PATH.name:
        raise HoldoutIntegrityError("holdout manifest payload filename mismatch")
    return raw


def opaque_payload_snapshot() -> dict[str, Any]:
    """Return pre-open metadata.  Hashing compressed bytes is allowed by the gate."""

    manifest = _read_holdout_manifest()
    compressed = HOLDOUT_PAYLOAD_PATH.read_bytes()
    digest = _sha256(compressed)
    if digest != EXPECTED_COMPRESSED_SHA256:
        raise HoldoutIntegrityError(f"compressed payload SHA mismatch: {digest}")
    return {
        "payload": HOLDOUT_PAYLOAD_PATH.name,
        "payload_size_bytes": len(compressed),
        "payload_sha256": digest,
        "manifest_version": manifest.get("version"),
        "manifest_total_cases": manifest.get("total_cases"),
        "opened": False,
        "executed": False,
        "HOLDOUT_CONSUMED": False,
    }


def _validate_expected(case: Mapping[str, Any]) -> None:
    case_id = str(case.get("id", ""))
    expected = case.get("expected")
    if not isinstance(expected, Mapping):
        raise HoldoutIntegrityError(f"{case_id}: expected must be an object")
    flows = list(expected.get("allowed_flows", []))
    if any(flow not in FLOW_NAMES for flow in flows):
        raise HoldoutIntegrityError(f"{case_id}: unknown allowed flow")
    statuses = list(expected.get("allowed_statuses", []))
    if any(status not in ALLOWED_STATUSES for status in statuses):
        raise HoldoutIntegrityError(f"{case_id}: unknown allowed status")
    required = dict(expected.get("required_slots", {}))
    try:
        CommandSlots.from_dict(required)
    except Exception as exc:
        raise HoldoutIntegrityError(f"{case_id}: invalid required_slots: {exc}") from exc
    slot_names = set(CommandSlots.__dataclass_fields__)
    if set(required) - slot_names:
        raise HoldoutIntegrityError(f"{case_id}: unknown required slot")
    if any(name not in slot_names for name in expected.get("forbidden_slots", [])):
        raise HoldoutIntegrityError(f"{case_id}: unknown forbidden slot")
    max_calls = expected.get("max_modal_calls")
    if max_calls is not None and (
        isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0
    ):
        raise HoldoutIntegrityError(f"{case_id}: invalid max_modal_calls")
    if case.get("context", "none") not in {"none", "search"}:
        raise HoldoutIntegrityError(f"{case_id}: invalid context seed")
    if bool(case.get("manual_review")) and not case.get("review_focus"):
        raise HoldoutIntegrityError(f"{case_id}: manual review case has no review_focus")


def _validate_holdout_dataset(dataset: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if dataset.get("schema_version") != 2:
        raise HoldoutIntegrityError(f"holdout schema_version != 2: {dataset.get('schema_version')!r}")
    if dataset.get("version") != manifest.get("version"):
        raise HoldoutIntegrityError("holdout payload version does not match metadata manifest")
    cases = list(dataset.get("cases", []))
    if len(cases) != 200:
        raise HoldoutIntegrityError(f"holdout payload case count != 200: {len(cases)}")
    ids = [str(case.get("id", "")) for case in cases]
    expected_ids = [f"H21-{index:03d}" for index in range(1, 201)]
    if ids != expected_ids:
        raise HoldoutIntegrityError("holdout case IDs must be sequential H21-001..H21-200")
    queries = [str(case.get("query", "")).strip() for case in cases]
    if any(not query for query in queries) or len(set(queries)) != len(queries):
        raise HoldoutIntegrityError("holdout queries must be non-empty and unique")
    category_counts = Counter(str(case.get("category", "")) for case in cases)
    target_counts = Counter(dict(manifest.get("category_targets", {})))
    if category_counts != target_counts:
        raise HoldoutIntegrityError(
            f"holdout category distribution drift: {dict(category_counts)} != {dict(target_counts)}"
        )
    for case in cases:
        _validate_expected(case)
    manual_count = sum(bool(case.get("manual_review")) for case in cases)
    if manual_count != int(manifest.get("manual_review_cases", manual_count)):
        raise HoldoutIntegrityError("holdout manual_review_cases metadata mismatch")
    return {
        "version": dataset.get("version"),
        "schema_version": dataset.get("schema_version"),
        "reference_date": dataset.get("reference_date", manifest.get("reference_date")),
        "cases": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "manual_review_cases": manual_count,
        "ids": [ids[0], ids[-1]],
    }


def consume_holdout_once() -> tuple[dict[str, Any], dict[str, Any]]:
    """Open, decompress and parse the payload once, keeping plaintext in memory."""

    manifest = _read_holdout_manifest()
    compressed = HOLDOUT_PAYLOAD_PATH.read_bytes()
    compressed_digest = _sha256(compressed)
    if compressed_digest != EXPECTED_COMPRESSED_SHA256:
        raise HoldoutIntegrityError("compressed payload digest changed at opening")
    try:
        plaintext = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise HoldoutIntegrityError(f"holdout gzip decompression failed: {exc}") from exc
    uncompressed_digest = _sha256(plaintext)
    if uncompressed_digest != manifest.get("payload_uncompressed_sha256"):
        raise HoldoutIntegrityError("uncompressed payload digest mismatch")
    try:
        dataset = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldoutIntegrityError(f"holdout JSON parse failed: {exc}") from exc
    if not isinstance(dataset, Mapping):
        raise HoldoutIntegrityError("holdout payload root must be an object")
    metadata = _validate_holdout_dataset(dataset, manifest)
    metadata.update(
        {
            "payload_sha256": compressed_digest,
            "payload_uncompressed_sha256": uncompressed_digest,
            "opened": True,
            "executed": False,
            "HOLDOUT_CONSUMED": True,
        }
    )
    return dict(dataset), metadata


def _is_transient_transport_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    markers = (
        "timeout", "timed out", "connection reset", "connection aborted",
        "temporarily unavailable", "service unavailable", "transport",
        "grpc", "502", "503", "504", "container connection",
    )
    return any(marker in name or marker in text for marker in markers)


class ModalFrameInvoker:
    """Call only the selected deployed Sarashina class with infra-only retry."""

    def __init__(self, remote: Any) -> None:
        self.remote = remote
        self.transport_attempts: list[dict[str, Any]] = []

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        request = dict(payload)
        request["format_enforcer"] = "lmfe"
        for retry_index in range(MAX_TRANSPORT_RETRIES + 1):
            started = time.perf_counter()
            try:
                response = self.remote.generate_frame.remote(
                    str(request.get("query", "")),
                    request.get("state"),
                    request.get("grounded"),
                    "lmfe",
                )
                elapsed = (time.perf_counter() - started) * 1000
                self.transport_attempts.append(
                    {"attempt": retry_index + 1, "elapsed_ms": round(elapsed, 3), "retry": retry_index > 0, "error": None}
                )
                self._assert_selected_backend(response)
                return response
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                transient = not isinstance(exc, TransportFailure) and _is_transient_transport_error(exc)
                self.transport_attempts.append(
                    {
                        "attempt": retry_index + 1,
                        "elapsed_ms": round(elapsed, 3),
                        "retry": retry_index > 0,
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "transient": transient,
                    }
                )
                if not transient or retry_index >= MAX_TRANSPORT_RETRIES:
                    raise
                time.sleep(min(2 ** retry_index, 8))
        raise AssertionError("unreachable transport retry path")

    @staticmethod
    def _assert_selected_backend(response: Any) -> None:
        if not isinstance(response, Mapping):
            raise TransportFailure("Modal response is not an object")
        if response.get("service_id") != SERVICE_ID:
            raise TransportFailure("Modal service_id mismatch")
        if response.get("model_key") != EXPECTED_MODEL_KEY:
            raise TransportFailure("selected model_key mismatch")
        if response.get("model_id") != EXPECTED_MODEL_ID:
            raise TransportFailure("selected model_id mismatch")
        if response.get("protocol_version") != EXPECTED_PROTOCOL:
            raise TransportFailure("protocol_version mismatch")


def _build_remote() -> Any:
    from semantic_v2_3_multimodel_backend import SarashinaSemanticV23

    return SarashinaSemanticV23.from_name(SERVICE_ID, SERVICE_CLASS)


def _model_call_observability(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = row.get("observability") or {}
    calls = list(observation.get("calls_detail", []))
    service = [dict(call.get("service", {})) for call in calls if isinstance(call, Mapping)]
    return {
        "model_called": int(row.get("frame_attempts", 0)) > 0,
        "model_key": next((item.get("model_key") for item in service if item.get("model_key")), None),
        "model_id": next((item.get("model_id") for item in service if item.get("model_id")), None),
        "raw_frame": row.get("frame"),
        "raw_frame_text": next((item.get("raw_output") for item in calls if item.get("raw_output")), None),
        "parse_valid": bool(row.get("atomic_frame_valid")),
        "schema_valid": bool(row.get("atomic_frame_valid")),
        "prompt_tokens": int(observation.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(observation.get("generated_tokens", 0) or 0),
        "inference_latency_ms": float(observation.get("generation_ms", 0.0) or 0.0),
        "server_total_latency_ms": float(observation.get("server_total_ms", 0.0) or 0.0),
        "client_latency_ms": float(observation.get("client_elapsed_ms", 0.0) or 0.0),
        "call_count": int(observation.get("calls", 0) or 0),
        "calls_detail": calls,
    }


def _failure_taxonomy(row: Mapping[str, Any]) -> list[str]:
    failures = {str(item) for item in row.get("failures", [])}
    result: list[str] = []
    def add(value: str) -> None:
        if value not in result:
            result.append(value)

    if "unsupported_inference" in failures:
        add("F15 unsupported inference")
    if "silent_coercion" in failures:
        add("F16 silent coercion")
    if "missed_data_gap" in failures or "false_data_gap" in failures:
        add("F12 data-gap")
    if "false_clarification" in failures or "missed_clarification" in failures:
        add("F11 clarification")
    if "flow" in failures:
        add("F19 flow/status")
    if "status" in failures:
        add("F19 flow/status")
    for failure in failures:
        if failure.startswith("slot:") or failure.startswith("forbidden:"):
            field = failure.split(":", 1)[1]
            if field in {"municipalities", "municipality"}:
                add("F3 municipality")
            elif field in {"regions", "region"}:
                add("F4 region")
            elif field in {"entry_free", "paid_only", "max_entry_fee", "fee"}:
                add("F5 fee")
            elif field in {"reservation_required", "reservation"}:
                add("F6 reservation")
            elif field in {"venue"}:
                add("F7 venue")
            elif field in {"rain_preferred", "rain"}:
                add("F8 rain")
            elif field in {"audience", "age", "age_group", "age_intent"}:
                add("F9 audience")
            elif field.startswith("experience"):
                add("F10 experience")
            elif field in {"refine_previous", "last_result_ids", "reference_index"}:
                add("F13 context/reference")
            else:
                add("F23 other")
    if "auto_relax" in failures:
        add("F20 reducer/executor")
    if "max_modal_calls" in failures:
        add("F22 schema/runtime")
    if not row.get("atomic_frame_valid") and _model_call_observability(row)["model_called"]:
        add("F22 schema/runtime")
    if row.get("frame_fallback") and not result:
        add("F22 schema/runtime")
    if row.get("release_operations") and any("release" in str(item) for item in row.get("release_operations", [])):
        if "slot:" in " ".join(failures) or "flow" in failures:
            add("F14 negation/release")
    if not result and failures:
        add("F23 other")
    return result


def _decorate_row(case: Mapping[str, Any], row: Mapping[str, Any], invoker: ModalFrameInvoker) -> dict[str, Any]:
    result = dict(row)
    state = _seed_state(str(case.get("context", "none")))
    result["case_id"] = result.get("id")
    result["expected"] = dict(case.get("expected", {}))
    result["previous_state"] = state
    result["deterministic_routing"] = {
        "route": result.get("deterministic_route"),
        "grounding": result.get("deterministic_grounding", {}),
    }
    result["model"] = _model_call_observability(result)
    result["transport"] = {
        "attempts": list(invoker.transport_attempts),
        "retry_count": sum(bool(item.get("retry")) for item in invoker.transport_attempts),
    }
    result["final"] = {
        "flow": result.get("actual_flow"),
        "status": result.get("actual_status"),
        "clarification": result.get("clarification_reason"),
        "data_gap": result.get("data_gap_reason"),
        "constraints": result.get("actual_slots", {}),
        "fail_soft": bool(result.get("frame_fallback")),
    }
    taxonomy = _failure_taxonomy(result)
    result["failure_taxonomy"] = {
        "primary": taxonomy[0] if taxonomy else None,
        "secondary": taxonomy[1:],
    }
    result["semantic_boundary"] = {
        "evidence_request": result.get("evidence_request"),
        "capability": result.get("capability"),
        "accepted_atoms": result.get("accepted_atoms", []),
        "ignored_atoms": result.get("ignored_atoms", []),
        "rejected_atoms": result.get("rejected_atoms", []),
        "unsupported_inference_attempts_prevented": result.get("unsupported_inference_prevented_count", 0),
        "silent_coercion_attempts_prevented": result.get("silent_coercion_prevented_count", 0),
    }
    return result


def _case_with_error(case: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    query = str(case.get("query", ""))
    return {
        "id": case.get("id"),
        "case_id": case.get("id"),
        "category": case.get("category"),
        "risk": case.get("risk"),
        "query": query,
        "expected": dict(case.get("expected", {})),
        "expected_behavior": case.get("expected_behavior"),
        "actual_flow": None,
        "actual_status": "evaluation_error",
        "actual_slots": {},
        "frame_attempts": 0,
        "frame_fallback": False,
        "atomic_frame_valid": False,
        "observability": {"calls": 0, "calls_detail": []},
        "model": {"model_called": False, "parse_valid": False, "schema_valid": False},
        "transport": {"attempts": [], "retry_count": 0},
        "final": {"flow": None, "status": "evaluation_error", "clarification": None, "data_gap": None, "constraints": {}, "fail_soft": False},
        "failures": ["runner_exception"],
        "failure_taxonomy": {"primary": "F22 schema/runtime", "secondary": []},
        "runner_exception": f"{type(exc).__name__}: {exc}"[:1000],
        "machine_pass": False,
        "manual_review": True,
        "manual_review_required": True,
        "manual_verdict": "FAIL",
        "review_focus": list(case.get("review_focus", [])),
        "verdict": "fail",
        "message": "",
    }


def _wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0, "rate": 0.0}
    p = successes / total
    denominator = 1 + Z95 * Z95 / total
    centre = (p + Z95 * Z95 / (2 * total)) / denominator
    margin = Z95 * ((p * (1 - p) / total + Z95 * Z95 / (4 * total * total)) ** 0.5) / denominator
    return {"lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin), "rate": p}


def _clarification_required(row: Mapping[str, Any]) -> bool:
    expected = row.get("expected") or {}
    if "clarification" in set(expected.get("allowed_statuses", [])):
        return True
    behavior = str(row.get("expected_behavior", "")).lower()
    return any(marker in behavior for marker in ("clarif", "ambiguous", "underspecified"))


def _clarification_actual(row: Mapping[str, Any]) -> bool:
    return row.get("actual_status") == "clarification" or bool(row.get("clarification_reason"))


def aggregate(rows: Sequence[Mapping[str, Any]], *, integrity: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(rows)
    passed = sum(bool(row.get("machine_pass")) for row in rows)
    model_rows = [row for row in rows if bool((row.get("model") or {}).get("model_called"))]
    valid_model_rows = [row for row in model_rows if bool((row.get("model") or {}).get("schema_valid"))]
    category: dict[str, dict[str, Any]] = defaultdict(lambda: {"cases": 0, "pass": 0, "rate": 0.0})
    for row in rows:
        bucket = category[str(row.get("category"))]
        bucket["cases"] += 1
        bucket["pass"] += int(bool(row.get("machine_pass")))
    for bucket in category.values():
        bucket["rate"] = bucket["pass"] / bucket["cases"] if bucket["cases"] else 0.0

    failure_rows = [row for row in rows if not row.get("machine_pass")]
    primary = Counter(str((row.get("failure_taxonomy") or {}).get("primary")) for row in failure_rows)
    secondary = Counter(
        str(value)
        for row in rows
        for value in (row.get("failure_taxonomy") or {}).get("secondary", [])
    )
    unsupported = sum(int(row.get("unsupported_inference_count", 0) or 0) for row in rows)
    silent = sum(int(row.get("silent_coercion_count", 0) or 0) for row in rows)
    prevented_unsupported = sum(int(row.get("unsupported_inference_prevented_count", 0) or 0) for row in rows)
    prevented_silent = sum(int(row.get("silent_coercion_prevented_count", 0) or 0) for row in rows)
    missed_gap = sum("missed_data_gap" in row.get("failures", []) for row in rows)
    false_gap = sum("false_data_gap" in row.get("failures", []) for row in rows)
    clarification_required = sum(_clarification_required(row) for row in rows)
    clarification_actual = sum(_clarification_actual(row) for row in rows)
    clarification_true = sum(_clarification_required(row) and _clarification_actual(row) for row in rows)
    constraint_good = sum(
        not any(str(failure).startswith("slot:") or str(failure).startswith("forbidden:") for failure in row.get("failures", []))
        for row in rows
    )
    latencies = [float(row.get("orchestrator_latency_ms", 0.0) or 0.0) for row in rows]
    warm_latencies = [
        float((row.get("model") or {}).get("server_total_latency_ms", 0.0) or 0.0)
        for row in model_rows
        if float((row.get("model") or {}).get("server_total_latency_ms", 0.0) or 0.0) > 0
    ]
    generation_latencies = [
        float((row.get("model") or {}).get("inference_latency_ms", 0.0) or 0.0)
        for row in model_rows
        if float((row.get("model") or {}).get("inference_latency_ms", 0.0) or 0.0) > 0
    ]
    manual_counts = Counter(str(row.get("manual_verdict", "PENDING")) for row in rows)
    invalid_reasons = list(integrity.get("invalid_reasons", []))
    if any(str(row.get("actual_status")) == "evaluation_error" for row in rows):
        invalid_reasons.append("runner_exception")
    structural_total = len(model_rows)
    structural_valid = len(valid_model_rows)
    structural_rate = structural_valid / structural_total if structural_total else 1.0
    ci = _wilson(passed, len(rows))
    return {
        "cases": len(rows),
        "machine_pass": passed,
        "machine_pass_rate": passed / len(rows) if rows else 0.0,
        "machine_pass_wilson_95": ci,
        "category": {name: bucket for name, bucket in sorted(category.items())},
        "manual": {
            "PASS": manual_counts.get("PASS", 0),
            "BORDERLINE": manual_counts.get("BORDERLINE", 0),
            "FAIL": manual_counts.get("FAIL", 0),
            "PENDING": manual_counts.get("PENDING", 0),
            "NOT_REQUIRED": manual_counts.get("NOT_REQUIRED", 0),
        },
        "machine_manual_matrix": {
            "machine_PASS_manual_PASS": sum(bool(row.get("machine_pass")) and row.get("manual_verdict") == "PASS" for row in rows),
            "machine_PASS_manual_BORDERLINE": sum(bool(row.get("machine_pass")) and row.get("manual_verdict") == "BORDERLINE" for row in rows),
            "machine_PASS_manual_FAIL": sum(bool(row.get("machine_pass")) and row.get("manual_verdict") == "FAIL" for row in rows),
            "machine_FAIL_manual_PASS": sum(not bool(row.get("machine_pass")) and row.get("manual_verdict") == "PASS" for row in rows),
            "machine_FAIL_manual_BORDERLINE": sum(not bool(row.get("machine_pass")) and row.get("manual_verdict") == "BORDERLINE" for row in rows),
            "machine_FAIL_manual_FAIL": sum(not bool(row.get("machine_pass")) and row.get("manual_verdict") == "FAIL" for row in rows),
        },
        "machine_manual_false_positive_count": sum(bool(row.get("machine_pass")) and row.get("manual_verdict") == "FAIL" for row in rows),
        "evidence_boundary": {
            "unsupported_inference_count": unsupported,
            "silent_coercion_count": silent,
            "missed_data_gap_count": missed_gap,
            "false_data_gap_count": false_gap,
            "clarification_precision": clarification_true / clarification_actual if clarification_actual else 1.0,
            "clarification_recall": clarification_true / clarification_required if clarification_required else 1.0,
            "semantic_constraint_accuracy": constraint_good / len(rows) if rows else 0.0,
            "prevented_unsupported_inference_count": prevented_unsupported,
            "prevented_silent_coercion_count": prevented_silent,
        },
        "structural_validity": {
            "model_called_cases": structural_total,
            "schema_valid_cases": structural_valid,
            "invalid_frame_cases": structural_total - structural_valid,
            "rate": structural_rate,
            "fail_soft_cases": sum(bool(row.get("frame_fallback")) for row in rows),
        },
        "failure_clusters": {
            "primary": dict(primary),
            "secondary": dict(secondary),
        },
        "latency": {
            "all_orchestrator": _latency_summary(latencies),
            "model_server_total": _latency_summary(warm_latencies),
            "warm_inference": _latency_summary(generation_latencies),
        },
        "tokens_model_calls": {
            "total_model_calls": sum(int((row.get("model") or {}).get("call_count", 0) or 0) for row in rows),
            "zero_model_cases": len(rows) - len(model_rows),
            "model_called_cases": len(model_rows),
            "prompt_tokens": sum(int((row.get("model") or {}).get("prompt_tokens", 0) or 0) for row in rows),
            "generated_tokens": sum(int((row.get("model") or {}).get("generated_tokens", 0) or 0) for row in rows),
        },
        "transport": {
            "transport_retry_count": sum(int((row.get("transport") or {}).get("retry_count", 0) or 0) for row in rows),
            "transport_error_cases": sum(bool((row.get("transport") or {}).get("attempts", []) and (row.get("transport") or {}).get("attempts", [])[-1].get("error")) for row in rows),
        },
        "integrity": {**dict(integrity), "invalid_reasons": sorted(set(invalid_reasons))},
    }


def _judgment(summary: Mapping[str, Any]) -> tuple[str, str]:
    if summary.get("integrity", {}).get("invalid_reasons"):
        return "E", "INVALID EVALUATION"
    score = float(summary.get("machine_pass_rate", 0.0))
    safety = summary.get("evidence_boundary", {})
    structural = float(summary.get("structural_validity", {}).get("rate", 0.0))
    category = summary.get("category", {})
    collapse = any(
        int(bucket.get("cases", 0)) >= 10 and float(bucket.get("rate", 0.0)) < 0.40
        for bucket in category.values()
    )
    if score >= 0.75 and safety.get("unsupported_inference_count", 0) == 0 and safety.get("silent_coercion_count", 0) == 0 and structural >= 0.98 and not collapse:
        return "A", "GENERALIZATION PASS"
    if score >= 0.65 and safety.get("unsupported_inference_count", 0) == 0 and safety.get("silent_coercion_count", 0) == 0 and structural >= 0.98:
        return "B", "CONDITIONAL PASS"
    if score >= 0.50:
        return "C", "GENERALIZATION WEAK"
    return "D", "FAIL"


def _report(summary: Mapping[str, Any], *, snapshot: Mapping[str, Any], smoke: Mapping[str, Any]) -> str:
    code, label = _judgment(summary)
    ci = summary["machine_pass_wilson_95"]
    eb = summary["evidence_boundary"]
    structural = summary["structural_validity"]
    latency = summary["latency"]
    tokens = summary["tokens_model_calls"]
    manual = summary["manual"]
    lines = [
        "# Semantic Operations v2.3 — Final Sealed Holdout Evaluation",
        "",
        "## 1. Executive conclusion",
        "",
        f"Selected production candidate: Sarashina 2.2 3B (`{EXPECTED_MODEL_ID}`).",
        f"Final judgment: **{code}. {label}**.",
        f"Machine PASS: **{summary['machine_pass']} / {summary['cases']} ({summary['machine_pass_rate']:.1%})**.",
        "",
        "## 2. Evaluation integrity",
        "",
        f"Architecture frozen SHA: `{snapshot.get('architecture_frozen_sha')}`.",
        f"Evaluation HEAD: `{snapshot.get('evaluation_head')}`; tree: `{snapshot.get('evaluation_tree')}`.",
        f"Model key: `{EXPECTED_MODEL_KEY}`; protocol: `{EXPECTED_PROTOCOL}`.",
        f"Isolated Modal service: `{SERVICE_ID}` / `{SERVICE_CLASS}`.",
        f"Preflight/smoke: `{smoke.get('status', 'UNKNOWN')}`.",
        "",
        "## 3. Holdout consumption statement",
        "",
        "The v2.1 holdout was opened once after the preflight and smoke gates. The plaintext corpus is not copied into this report or committed to the repository.",
        f"`HOLDOUT_CONSUMED = true`; opened: `{snapshot.get('holdout', {}).get('opened')}`; executed: `{snapshot.get('holdout', {}).get('executed')}`.",
        "",
        "## 4. Overall 200-case result",
        "",
        f"Machine PASS: **{summary['machine_pass']} / 200** — **{summary['machine_pass_rate']:.1%}**.",
        "",
        "## 5. 95% confidence interval",
        "",
        f"Wilson 95% CI: **{ci['lower']:.1%} – {ci['upper']:.1%}**.",
        "",
        "## 6. Category results",
        "",
        "| Category | Cases | PASS | Rate |",
        "|---|---:|---:|---:|",
    ]
    for name, bucket in summary["category"].items():
        lines.append(f"| {name} | {bucket['cases']} | {bucket['pass']} | {bucket['rate']:.1%} |")
    lines += [
        "",
        "## 7. Manual review",
        "",
        f"PASS {manual['PASS']} / BORDERLINE {manual['BORDERLINE']} / FAIL {manual['FAIL']} / PENDING {manual['PENDING']}.",
        "Manual rubric: PASS = intent/constraints/evidence boundary/grounded execution all acceptable; BORDERLINE = minor contract deviation without material harm; FAIL = unsafe inference, fabrication, wrong clarification/data-gap, constraint loss, wrong reference/release, or wrong flow.",
        "",
        "## 8. Machine/manual divergence",
        "",
    ]
    for name, value in summary["machine_manual_matrix"].items():
        lines.append(f"- {name}: {value}")
    lines.append(f"- machine_manual_false_positive_count: {summary['machine_manual_false_positive_count']}")
    lines += [
        "",
        "## 9–12. Evidence Boundary, unsupported inference, silent coercion, data-gap / clarification",
        "",
        f"- unsupported inference accepted: **{eb['unsupported_inference_count']}**",
        f"- unsupported inference prevented: **{eb['prevented_unsupported_inference_count']}**",
        f"- silent coercion accepted: **{eb['silent_coercion_count']}**",
        f"- silent coercion prevented: **{eb['prevented_silent_coercion_count']}**",
        f"- missed data-gap: **{eb['missed_data_gap_count']}**",
        f"- false data-gap: **{eb['false_data_gap_count']}**",
        f"- clarification precision: **{eb['clarification_precision']:.1%}**",
        f"- clarification recall: **{eb['clarification_recall']:.1%}**",
        f"- semantic constraint accuracy: **{eb['semantic_constraint_accuracy']:.1%}**",
        "",
        "## 13–16. Negation / release, context/reference, structural validity",
        "",
        "Release and context details are retained case-by-case in `raw_results.jsonl`; failure taxonomy separates F13 context/reference and F14 negation/release where applicable.",
        f"Structural validity: **{structural['schema_valid_cases']} / {structural['model_called_cases']} model-called cases** ({structural['rate']:.1%}).",
        f"Invalid frame cases: {structural['invalid_frame_cases']}; fail-soft cases: {structural['fail_soft_cases']}.",
        "",
        "## 17–18. Failure clusters, latency, tokens/model calls",
        "",
        f"Primary clusters: `{json.dumps(summary['failure_clusters']['primary'], ensure_ascii=False, sort_keys=True)}`.",
        f"Mean / median / p95 / max warm server latency (ms): {latency['model_server_total']}.",
        f"Warm inference latency (ms): {latency['warm_inference']}.",
        f"Total model calls: **{tokens['total_model_calls']}**; zero-model cases: **{tokens['zero_model_cases']}**.",
        f"Prompt tokens: **{tokens['prompt_tokens']}**; generated tokens: **{tokens['generated_tokens']}**.",
        "",
        "## 19. Frozen v1 vs final holdout",
        "",
        "| Evaluation set | Result |",
        "|---|---:|",
        "| Frozen v1 development reference | 66 / 100 |",
        f"| Final sealed holdout | {summary['machine_pass']} / 200 ({summary['machine_pass_rate']:.1%}) |",
        f"| Reference generalization gap | {(summary['machine_pass_rate'] - 0.66):+.1%} points |",
        "",
        "The Frozen v1 score is a development-set reference; the 200-case holdout is the generalization estimate. The gap is interpreted together with category pattern, manual quality and evidence safety.",
        "",
        "## 20. Generalization diagnosis",
        "",
        "The diagnosis is based on the final frozen architecture/model execution and the recorded failure clusters. No post-opening architecture, prompt, expected-behavior or evaluator change was used to rerun a case.",
        "",
        "## 21. PoC production-readiness judgment",
        "",
        f"Accuracy: {('A' if summary['machine_pass_rate'] >= .75 else 'B' if summary['machine_pass_rate'] >= .65 else 'C' if summary['machine_pass_rate'] >= .50 else 'D')}",
        f"Evidence safety: {'A' if eb['unsupported_inference_count'] == 0 and eb['silent_coercion_count'] == 0 else 'D'}",
        f"Structural: {'A' if structural['rate'] >= .98 else 'B' if structural['rate'] >= .95 else 'D'}",
        f"Latency: {'A' if latency['model_server_total']['p95_ms'] <= 10000 else 'B' if latency['model_server_total']['p95_ms'] <= 30000 else 'C'}",
        f"Overall PoC readiness: {'PASS' if code == 'A' else 'CONDITIONAL' if code == 'B' else 'FAIL'}.",
        "",
        "## 22. Recommended next action",
        "",
        "Do not rerun this consumed holdout. Use the final judgment and failure clusters to choose one next engineering action; if improvement is needed, create a new independent holdout before the next formal gate.",
        "",
        "## 30. One-screen final numbers",
        "",
        f"Semantic Operations v2.3 / Sarashina 2.2 3B — Machine PASS **{summary['machine_pass']} / 200 ({summary['machine_pass_rate']:.1%})**, Wilson 95% CI **{ci['lower']:.1%}–{ci['upper']:.1%}**, Manual **PASS {manual['PASS']} / BORDERLINE {manual['BORDERLINE']} / FAIL {manual['FAIL']}**, Structural **{structural['schema_valid_cases']} / {structural['model_called_cases']}**, Unsupported accepted **{eb['unsupported_inference_count']}**, prevented **{eb['prevented_unsupported_inference_count']}**, Silent coercion accepted **{eb['silent_coercion_count']}**, prevented **{eb['prevented_silent_coercion_count']}**, missed data-gap **{eb['missed_data_gap_count']}**, false data-gap **{eb['false_data_gap_count']}**, clarification precision **{eb['clarification_precision']:.1%}**, recall **{eb['clarification_recall']:.1%}**, constraint accuracy **{eb['semantic_constraint_accuracy']:.1%}**, median warm latency **{latency['model_server_total']['median_ms']} ms**, p95 warm latency **{latency['model_server_total']['p95_ms']} ms**, total calls **{tokens['total_model_calls']}**, zero-model **{tokens['zero_model_cases']}**, generated tokens **{tokens['generated_tokens']}**, Frozen v1 **66 / 100**, gap **{(summary['machine_pass_rate'] - .66):+.1%} points**, judgment **{code}/{label}**, OPENED **YES**, RUN **YES**, CONSUMED **YES**.",
    ]
    return "\n".join(lines) + "\n"


def _write_auxiliary_artifacts(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    with (output_dir / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _json_dump(output_dir / "summary.json", summary)
    _json_dump(output_dir / "evidence_boundary_metrics.json", summary["evidence_boundary"])
    _json_dump(output_dir / "failure_clusters.json", summary["failure_clusters"])
    _json_dump(output_dir / "latency_tokens.json", {"latency": summary["latency"], "tokens_model_calls": summary["tokens_model_calls"], "transport": summary["transport"]})
    with (output_dir / "category_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "cases", "pass", "rate"])
        for name, bucket in summary["category"].items():
            writer.writerow([name, bucket["cases"], bucket["pass"], f"{bucket['rate']:.6f}"])
    manual = []
    for row in rows:
        manual.append(
            {
                "case_id": row.get("case_id", row.get("id")),
                "category": row.get("category"),
                "manual_required": True,
                "manual_verdict": row.get("manual_verdict", "PENDING"),
                "reason": row.get("manual_reason", "pending final manual review"),
                "review_focus": row.get("review_focus", []),
                "machine_pass": bool(row.get("machine_pass")),
                "primary_failure": (row.get("failure_taxonomy") or {}).get("primary"),
            }
        )
    _json_dump(output_dir / "manual_review.json", {"cases": manual, "status": "pending" if any(item["manual_verdict"] == "PENDING" for item in manual) else "complete"})


def _snapshot(*, output_dir: Path, smoke: Mapping[str, Any]) -> dict[str, Any]:
    freeze_manifest = json.loads(
        (ROOT / "docs" / "semantic_operations_v2_3_eval_freeze.json").read_text(encoding="utf-8")
    )
    validate_freeze_manifest(freeze_manifest)
    freeze = freeze_snapshot()
    opaque = opaque_payload_snapshot()
    package_versions = _package_versions()
    return {
        "created_at": _now(),
        "repository": "ryotamatsuki/ehime-kokubunsai-ai-poc",
        "evaluation_branch": _git(["branch", "--show-current"]),
        "evaluation_head": _git(["rev-parse", "HEAD"]),
        "evaluation_tree": _git(["rev-parse", "HEAD^{tree}"]),
        "architecture_frozen_sha": ARCHITECTURE_FROZEN_SHA,
        "architecture_frozen_manifest": freeze,
        "model_key": EXPECTED_MODEL_KEY,
        "model_id": EXPECTED_MODEL_ID,
        "schema_hash": freeze.get("schema_hash"),
        "prompt_hash": freeze.get("prompt_hash"),
        "few_shot_count": freeze.get("few_shot_count"),
        "capability_registry_hash": freeze.get("capability_registry_hash"),
        "verifier_hash": freeze.get("verifier_hash"),
        "reducer_hash": freeze.get("state_reducer_hash"),
        "orchestrator_hash": freeze.get("orchestrator_hash"),
        "evaluator_hash": freeze.get("evaluator_hash"),
        "backend_hash": freeze.get("multimodel_backend_hash"),
        "corpus_compressed_hash": opaque["payload_sha256"],
        "reference_date": POC_REFERENCE_DATE.isoformat(),
        "python_version": sys.version,
        "package_versions": package_versions,
        "lmfe_version": package_versions.get("lm-format-enforcer"),
        "modal_service_id": SERVICE_ID,
        "modal_service_class": SERVICE_CLASS,
        "smoke": dict(smoke),
        "holdout": {**opaque, "opened": False, "executed": False, "HOLDOUT_CONSUMED": False},
        "HOLDOUT_CONSUMED": False,
    }


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    remote = _build_remote()
    queries = (
        "東予で予約が必要な屋内イベントを探したい",
        "前の候補は料金条件を外して、見る・聞く中心にしたい",
        "今この瞬間に空いている催しを知りたい",
    )
    results: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        try:
            response = remote.generate_frame.remote(query, {}, {}, "lmfe")
            if not isinstance(response, Mapping):
                raise RuntimeError("smoke response is not an object")
            if response.get("service_id") != SERVICE_ID or response.get("protocol_version") != EXPECTED_PROTOCOL or response.get("model_key") != EXPECTED_MODEL_KEY or response.get("model_id") != EXPECTED_MODEL_ID:
                raise RuntimeError("smoke backend identity mismatch")
            answer = response.get("answer")
            frame = AtomicSemanticFrameV23.from_json(answer)
            results.append(
                {
                    "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
                    "model_key": response.get("model_key"),
                    "model_id": response.get("model_id"),
                    "protocol_version": response.get("protocol_version"),
                    "format_enforcer": (response.get("observability") or {}).get("format_enforcer"),
                    "schema_valid": True,
                    "one_generation": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "frame": frame.to_dict(),
                }
            )
        except Exception as exc:
            results.append({"query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16], "schema_valid": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
    passed = len(results) == len(queries) and all(item.get("schema_valid") and item.get("one_generation") for item in results)
    smoke = {"status": "PASS" if passed else "FAIL", "cases": len(results), "results": results, "service_id": SERVICE_ID, "model_key": EXPECTED_MODEL_KEY}
    _json_dump(output_dir / "smoke.json", smoke)
    if not passed:
        raise HoldoutIntegrityError("isolated Modal smoke failed")
    return smoke


def run_evaluation(output_dir: Path, *, confirm_consume: bool) -> dict[str, Any]:
    if not confirm_consume:
        raise HoldoutIntegrityError("refusing to open holdout without --confirm-consume")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The snapshot file is completely written before this call.  This is the
    # only code path in the runner that decompresses/parses the sealed payload.
    smoke_path = output_dir / "smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {"status": "NOT_FOUND"}
    if smoke.get("status") != "PASS":
        raise HoldoutIntegrityError("holdout opening requires PASS smoke artifact")
    snapshot = _snapshot(output_dir=output_dir, smoke=smoke)
    _json_dump(output_dir / "manifest.json", snapshot)
    dataset, consumed = consume_holdout_once()
    snapshot["holdout"].update(consumed)
    snapshot["HOLDOUT_CONSUMED"] = True
    snapshot["holdout_opened_at"] = _now()
    _json_dump(output_dir / "manifest.json", snapshot)

    rows: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    try:
        remote = _build_remote()
    except Exception as exc:
        remote = None
        invalid_reasons.append(f"remote_initialization:{type(exc).__name__}")
    for case in list(dataset.get("cases", [])):
        invoker = ModalFrameInvoker(remote)
        try:
            if remote is None:
                raise RuntimeError("selected Modal backend could not be initialized")
            base = evaluate_case_v23(case, invoker, include_raw=True, format_enforcer="lmfe")
            row = _decorate_row(case, base, invoker)
        except Exception as exc:
            row = _case_with_error(case, exc)
            invalid_reasons.append(f"{case.get('id')}:runner_exception")
        if int(row.get("frame_attempts", 0) or 0) > 1 or int((row.get("observability") or {}).get("calls", 0) or 0) > 1:
            invalid_reasons.append(f"{case.get('id')}:more_than_one_semantic_model_call")
        rows.append(row)
        print(json.dumps({"case_id": row.get("case_id"), "category": row.get("category"), "machine_pass": row.get("machine_pass"), "model_called": (row.get("model") or {}).get("model_called")}, ensure_ascii=False, sort_keys=True))
    snapshot["holdout"]["executed"] = True
    snapshot["holdout"]["HOLDOUT_CONSUMED"] = True
    snapshot["executed_at"] = _now()
    integrity = {
        "evaluation_valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "holdout_opened": True,
        "holdout_executed": True,
        "HOLDOUT_CONSUMED": True,
        "case_count": len(rows),
    }
    summary = aggregate(rows, integrity=integrity)
    _write_auxiliary_artifacts(output_dir, rows, summary)
    _json_dump(output_dir / "environment.json", {"snapshot": snapshot, "integrity": integrity})
    (output_dir / "REPORT.md").write_text(_report(summary, snapshot=snapshot, smoke=smoke), encoding="utf-8")
    _json_dump(output_dir / "manifest.json", snapshot)
    return summary


def finalize_manual(output_dir: Path, decisions_path: Path) -> dict[str, Any]:
    """Finalize manual verdicts without reopening or rerunning the holdout."""

    raw_path = output_dir / "raw_results.jsonl"
    if not raw_path.exists():
        raise HoldoutIntegrityError("raw_results.jsonl not found")
    decisions_raw = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = decisions_raw.get("cases", decisions_raw) if isinstance(decisions_raw, Mapping) else decisions_raw
    by_id = {str(item.get("case_id")): item for item in decisions if isinstance(item, Mapping)}
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        decision = by_id.get(str(row.get("case_id")))
        if decision is None or decision.get("manual_verdict") not in {"PASS", "BORDERLINE", "FAIL"}:
            raise HoldoutIntegrityError(f"manual verdict missing/invalid: {row.get('case_id')}")
        row["manual_verdict"] = decision["manual_verdict"]
        row["manual_reason"] = str(decision.get("reason", ""))[:1000]
    env = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    summary = aggregate(rows, integrity=env.get("integrity", {}))
    _write_auxiliary_artifacts(output_dir, rows, summary)
    (output_dir / "REPORT.md").write_text(_report(summary, snapshot=env["snapshot"], smoke=env["snapshot"].get("smoke", {})), encoding="utf-8")
    return summary


def _default_dir() -> Path:
    return ROOT / "artifacts" / f"semantic-v2-3-sealed-holdout-200-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    output_dir: str = "",
    confirm_consume: bool = False,
    decisions_path: str = "",
) -> None:
    """Modal local entrypoint: smoke, evaluate, or finalize."""

    target = Path(output_dir) if output_dir else _default_dir()
    if mode == "smoke":
        smoke = run_smoke(target)
        print(json.dumps({"status": smoke["status"], "output_dir": str(target), "cases": smoke["cases"]}, ensure_ascii=False, sort_keys=True))
        return
    if mode == "evaluate":
        # This pre-open opaque check is intentionally metadata/hash-only.
        opaque_payload_snapshot()
        summary = run_evaluation(target, confirm_consume=confirm_consume)
        print(json.dumps({"status": "EVALUATION_COMPLETE", "output_dir": str(target), "summary": summary}, ensure_ascii=False, sort_keys=True))
        return
    if mode == "finalize":
        if not decisions_path:
            raise ValueError("finalize requires decisions_path")
        summary = finalize_manual(target, Path(decisions_path))
        print(json.dumps({"status": "MANUAL_FINALIZED", "output_dir": str(target), "summary": summary}, ensure_ascii=False, sort_keys=True))
        return
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "evaluate", "finalize"), default="smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--confirm-consume", action="store_true")
    parser.add_argument("--decisions-path", default="")
    args = parser.parse_args()
    target = Path(args.output_dir) if args.output_dir else _default_dir()
    if args.mode == "smoke":
        print(json.dumps(run_smoke(target), ensure_ascii=False, indent=2))
    elif args.mode == "evaluate":
        opaque_payload_snapshot()
        print(json.dumps(run_evaluation(target, confirm_consume=args.confirm_consume), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(finalize_manual(target, Path(args.decisions_path)), ensure_ascii=False, indent=2))

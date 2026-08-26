"""Frozen-v1 live A/B harness for Semantic Operations v2.3.

This file is evaluation-only.  It imports the frozen v2.3 evaluator and
orchestrator without changing them, calls one deployed v2.3 backend per
model, and writes raw case-level evidence for later manual review.

The sealed holdout is intentionally not referenced by this runner.  Only the
exposed ``unexpected_utterances_v1`` manifest is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urljoin, urlparse

import requests

from app_config import POC_REFERENCE_DATE
from semantic_atomic_v2_3 import AtomicSemanticFrameV23
from semantic_capability_registry_v2_3 import lookup_capability
from semantic_freeze_v2_3 import SEALED_HOLDOUT_SHA256, freeze_snapshot, validate_freeze_manifest
from semantic_model_registry import MODEL_SPECS, SemanticModelSpec
from semantic_orchestrator_v2_3 import SemanticOperationsOrchestratorV23
from semantic_prompt_v2_3 import build_minimal_atomic_payload_v23
from semantic_state_v2_3 import grounded_slots_from_query_v22
from unexpected_utterances_eval import _seed_state, load_dataset, validate_dataset
from unexpected_utterances_v2_1_eval import FROZEN_V1_MANIFEST, load_frozen_v1_dataset
from unexpected_utterances_v2_3_eval import evaluate_case_v23


RUNNER_VERSION = "semantic-v2.3-frozen-v1-model-ab-1"
ARCHITECTURE_MANIFEST = Path("docs/semantic_operations_v2_3_eval_freeze.json")
EVALUATION_BRANCH = "eval/semantic-v2-3-frozen-v1-model-ab"
SERVICE_ID = "ehime-kokubunsai-semantic-v2-3-api"
PROTOCOL_VERSION = "semantic-v2.3-evidence-bounded-lmfe-v1"
TRANSPORT_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MODEL_URL_ENV = {
    "sarashina-2.2-3b": "SEMANTIC_V23_URL_SARASHINA_2_2_3B",
    "llm-jp-4-8b": "SEMANTIC_V23_URL_LLM_JP_4_8B",
}

FAILURE_TAXONOMY = {
    "F1": "intent",
    "F2": "scope",
    "F3": "municipality",
    "F4": "region",
    "F5": "fee",
    "F6": "reservation",
    "F7": "venue",
    "F8": "rain",
    "F9": "audience",
    "F10": "Experience",
    "F11": "clarification",
    "F12": "data-gap",
    "F13": "reference/context",
    "F14": "negation/release",
    "F15": "unsupported inference",
    "F16": "silent coercion",
    "F17": "verifier false reject",
    "F18": "verifier unsafe accept",
    "F19": "flow/status",
    "F20": "reducer/executor",
    "F21": "evaluator mismatch",
    "F22": "schema/runtime",
    "F23": "other",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()


def _git_sha() -> tuple[str, str]:
    return _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")


def _assert_hex_sha(value: str, label: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be a 40-character commit SHA")
    return value


def _corpus_sha256() -> str:
    """Hash the exposed v1 manifest and ordered fixture parts.

    The encoding matches the historical v2.2 runner: each relative filename
    is length-prefixed, followed by its exact bytes.  No sealed file is read.
    """

    manifest_path = Path(FROZEN_V1_MANIFEST)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    digest = hashlib.sha256()

    def add(name: str, data: bytes) -> None:
        name_bytes = name.encode("utf-8")
        # Keep the v2.2 evaluation artifact's exact corpus encoding: the
        # relative filename length is a uint32 and the file length is uint64.
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    add(manifest_path.name, manifest_bytes)
    for filename in manifest.get("files", []):
        part = manifest_path.parent / str(filename)
        add(str(filename), part.read_bytes())
    return digest.hexdigest()


def _freeze_context(expected_head: str | None) -> dict[str, Any]:
    head, tree = _git_sha()
    if expected_head is not None and _assert_hex_sha(expected_head, "ARCHITECTURE_FROZEN_SHA") != head:
        raise AssertionError(f"architecture frozen SHA is not evaluation-start HEAD: {expected_head} != {head}")
    manifest = json.loads(ARCHITECTURE_MANIFEST.read_text(encoding="utf-8"))
    validate_freeze_manifest(manifest)
    snapshot = freeze_snapshot()
    if manifest != snapshot:
        raise AssertionError("freeze manifest drifted during evaluation preflight")
    manifest_commit = _git("log", "-1", "--format=%H", "--", str(ARCHITECTURE_MANIFEST))
    return {
        "architecture_frozen_sha": head,
        "architecture_frozen_tree_sha": tree,
        "freeze_manifest_commit_sha": manifest_commit,
        "freeze_manifest_tree_sha": _git("rev-parse", f"{manifest_commit}^{{tree}}"),
        "evaluation_head_sha": head,
        "evaluation_tree_sha": tree,
        "freeze_snapshot": snapshot,
    }


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "has_previous_results": bool(state.get("last_result_ids")),
        "previous_result_count": int(state.get("last_result_count", 0) or 0),
        "previous_result_ids": [str(value) for value in list(state.get("last_result_ids", []))[:20]],
        "previous_active_flow": state.get("last_action"),
        "has_previous_command": bool(state.get("has_last_search_context") or state.get("last_command")),
        "has_search_context": bool(state.get("has_last_search_context")),
    }


def _deterministic_preflight(query: str, state: Mapping[str, Any]) -> dict[str, Any]:
    grounded = grounded_slots_from_query_v22(query, POC_REFERENCE_DATE)
    probe = SemanticOperationsOrchestratorV23(
        frame_call=None,
        reference_date=POC_REFERENCE_DATE,
    ).handle_query(query, state)
    return {
        "deterministic_route": probe.deterministic_route,
        "deterministic_grounding": dict(grounded),
        "preclassified_evidence_request": None,
        "preclassification_note": "v2.3 evidence request is model-classified and Python-verified after the residual call",
        "would_call_model": probe.deterministic_route not in {
            "capability:unsupported",
            "security_or_domain_guard",
            "relational_suitability_guard",
            "recommend_next_ambiguous",
            "clarify_reference",
        } and probe.deterministic_route is not None,
    }


class EndpointClient:
    """One v2.3 endpoint with transport-only retry and protocol validation."""

    def __init__(
        self,
        spec: SemanticModelSpec,
        url: str,
        modal_key: str,
        modal_secret: str,
        *,
        timeout_s: float = 300.0,
        transport_retries: int = 2,
    ) -> None:
        self.spec = spec
        self.url = url.strip()
        self.modal_key = modal_key
        self.modal_secret = modal_secret
        self.timeout_s = timeout_s
        self.transport_retries = transport_retries

    def _error(self, message: str, *, status: int | None, attempts: int, retries: int, redirects: int, elapsed_ms: float) -> dict[str, Any]:
        return {
            "service_id": SERVICE_ID,
            "protocol_version": PROTOCOL_VERSION,
            "model_key": self.spec.key,
            "model_id": self.spec.model_id,
            "error": message[:500],
            "observability": {
                "service_id": SERVICE_ID,
                "protocol_version": PROTOCOL_VERSION,
                "model_key": self.spec.key,
                "model_id": self.spec.model_id,
                "format_enforcer": "lmfe",
                "repair": False,
                "response_status": status,
                "transport_attempts": attempts,
                "transport_retry_count": retries,
                "redirect_hops": redirects,
                "client_elapsed_ms": round(elapsed_ms, 3),
            },
        }

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        started = time.perf_counter()
        current_url = self.url
        attempts = 0
        retries = 0
        redirects = 0
        headers = {
            "Modal-Key": self.modal_key,
            "Modal-Secret": self.modal_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request_body = dict(payload)
        request_body["format_enforcer"] = "lmfe"

        while True:
            attempts += 1
            try:
                response = requests.post(
                    current_url,
                    headers=headers,
                    json=request_body,
                    timeout=self.timeout_s,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if retries < self.transport_retries:
                    retries += 1
                    time.sleep(0.5 * retries)
                    continue
                return self._error(
                    f"transport failure: {type(exc).__name__}",
                    status=None,
                    attempts=attempts,
                    retries=retries,
                    redirects=redirects,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )

            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location or redirects >= 4:
                    return self._error(
                        "redirect routing failure",
                        status=response.status_code,
                        attempts=attempts,
                        retries=retries,
                        redirects=redirects,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                next_url = urljoin(current_url, location)
                parsed = urlparse(next_url)
                if parsed.scheme != "https" or not parsed.netloc:
                    return self._error(
                        "unsafe redirect target",
                        status=response.status_code,
                        attempts=attempts,
                        retries=retries,
                        redirects=redirects,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                current_url = next_url
                redirects += 1
                continue

            if response.status_code in TRANSPORT_RETRY_STATUSES and retries < self.transport_retries:
                retries += 1
                time.sleep(0.5 * retries)
                continue
            if response.status_code < 200 or response.status_code >= 300:
                return self._error(
                    f"HTTP {response.status_code}",
                    status=response.status_code,
                    attempts=attempts,
                    retries=retries,
                    redirects=redirects,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )

            try:
                body = response.json()
            except ValueError:
                return self._error(
                    "invalid JSON response",
                    status=response.status_code,
                    attempts=attempts,
                    retries=retries,
                    redirects=redirects,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            if not isinstance(body, Mapping):
                return self._error(
                    "response is not an object",
                    status=response.status_code,
                    attempts=attempts,
                    retries=retries,
                    redirects=redirects,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            result = dict(body)
            observability = dict(result.get("observability", {})) if isinstance(result.get("observability"), Mapping) else {}
            observability.update({
                "response_status": response.status_code,
                "transport_attempts": attempts,
                "transport_retry_count": retries,
                "redirect_hops": redirects,
                "client_elapsed_ms": round(elapsed_ms, 3),
                "format_enforcer": observability.get("format_enforcer", result.get("format_enforcer", "lmfe")),
                "repair": bool(observability.get("repair", False)),
            })
            result["observability"] = observability
            protocol_problems = []
            if result.get("service_id") != SERVICE_ID:
                protocol_problems.append("service_id")
            if result.get("protocol_version") != PROTOCOL_VERSION:
                protocol_problems.append("protocol_version")
            if result.get("model_key") != self.spec.key:
                protocol_problems.append("model_key")
            if result.get("model_id") != self.spec.model_id:
                protocol_problems.append("model_id")
            if observability.get("format_enforcer") != "lmfe":
                protocol_problems.append("format_enforcer")
            if observability.get("repair"):
                protocol_problems.append("repair")
            if protocol_problems:
                result.pop("answer", None)
                result["error"] = "protocol mismatch: " + ",".join(protocol_problems)
            return result


def _smoke_case(spec: SemanticModelSpec, client: EndpointClient) -> dict[str, Any]:
    payload = {
        "query": "東予で予約が必要な催しを探したい",
        "state": {},
        "grounded": {"regions": ["東予"], "reservation_required": True},
        "format_enforcer": "lmfe",
    }
    response = client(payload)
    obs = dict(response.get("observability", {})) if isinstance(response, Mapping) else {}
    checks: dict[str, bool] = {
        "endpoint_reachable": bool(response.get("answer")) if isinstance(response, Mapping) else False,
        "authenticated_protocol_response": isinstance(response, Mapping) and not bool(response.get("error")),
        "service_id": response.get("service_id") == SERVICE_ID if isinstance(response, Mapping) else False,
        "protocol_version": response.get("protocol_version") == PROTOCOL_VERSION if isinstance(response, Mapping) else False,
        "model_key": response.get("model_key") == spec.key if isinstance(response, Mapping) else False,
        "model_id": response.get("model_id") == spec.model_id if isinstance(response, Mapping) else False,
        "lmfe_enabled": obs.get("format_enforcer") == "lmfe",
        "answer_non_empty": bool(response.get("answer", "").strip()) if isinstance(response, Mapping) and isinstance(response.get("answer"), str) else False,
        "json_parse": isinstance(response, Mapping) and isinstance(response.get("answer"), str),
        "atomic_schema": False,
        "one_generation_call": int(obs.get("transport_attempts", 0) or 0) >= 1 and not bool(obs.get("repair", False)),
        "token_telemetry": int(obs.get("prompt_tokens", 0) or 0) > 0 and int(obs.get("generated_tokens", 0) or 0) > 0,
    }
    if checks["json_parse"]:
        try:
            AtomicSemanticFrameV23.from_json(str(response["answer"]))
            checks["atomic_schema"] = True
        except (TypeError, ValueError):
            checks["atomic_schema"] = False
    return {
        "model_key": spec.key,
        "model_id": spec.model_id,
        "endpoint_url": client.url,
        "checks": checks,
        "passed": all(checks.values()),
        "observability": obs,
        "answer": response.get("answer") if isinstance(response, Mapping) else None,
        "error": response.get("error") if isinstance(response, Mapping) else "non-object response",
    }


def _augment_row(
    row: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    spec: SemanticModelSpec,
    client: EndpointClient,
    case_index: int,
    execution_index: int,
    model_order_index: int,
    before: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(row)
    obs = dict(result.get("observability", {}))
    details = list(obs.get("calls_detail", []))
    raw_output = details[0].get("raw_output") if details and isinstance(details[0], Mapping) else None
    service_obs = []
    for detail in details:
        if isinstance(detail, Mapping) and isinstance(detail.get("service"), Mapping):
            service_obs.append(dict(detail["service"]))
    result.update({
        "case_order_index": case_index,
        "execution_index": execution_index,
        "model_order_index": model_order_index,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "endpoint_url": client.url,
        "context": case.get("context", "none"),
        "previous_state_summary": _state_summary(_seed_state(str(case.get("context", "none")))),
        "before_model": dict(before),
        "model_called": int(obs.get("calls", 0) or 0) > 0,
        "model_call_decision": "residual_semantic_model_call" if int(obs.get("calls", 0) or 0) > 0 else "zero_model_deterministic",
        "raw_model_response": raw_output,
        "raw_atomic_frame": raw_output,
        "service_observability": service_obs,
        "transport_retry_count": sum(int(item.get("transport_retry_count", 0) or 0) for item in service_obs),
        "redirect_hops": sum(int(item.get("redirect_hops", 0) or 0) for item in service_obs),
        "final": {
            "flow": result.get("actual_flow"),
            "status": result.get("actual_status"),
            "slots": result.get("actual_slots"),
            "clarification": result.get("clarification_reason"),
            "data_gap": result.get("data_gap_reason"),
            "fail_soft": bool(result.get("frame_fallback")),
            "result_count": result.get("actual_result_count"),
            "result_ids": result.get("actual_result_ids", []),
            "constraints": result.get("actual_slots", {}),
        },
    })
    result["attempts"] = int(result.get("frame_attempts", 0) or 0)
    result["repair_calls"] = int(bool(obs.get("repair_called")))
    result["manual_rubric_provisional"] = result.get("manual_rubric")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 3)


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": round(statistics.fmean(data), 3),
        "median_ms": round(statistics.median(data), 3),
        "p90_ms": _percentile(data, 0.90),
        "p95_ms": _percentile(data, 0.95),
        "max_ms": round(max(data), 3),
    }


def _row_primary_failures(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    failures = [str(value) for value in row.get("failures", [])]
    mapped: list[str] = []
    for failure in failures:
        if failure == "unsupported_inference":
            mapped.append("F15")
        elif failure == "silent_coercion":
            mapped.append("F16")
        elif failure in {"missed_data_gap", "false_data_gap"}:
            mapped.append("F12")
        elif failure in {"false_clarification", "missed_clarification"}:
            mapped.append("F11")
        elif failure == "flow" or failure == "status":
            mapped.append("F19")
        elif failure == "auto_relax":
            mapped.append("F14")
        elif failure == "max_modal_calls":
            mapped.append("F22")
        elif failure.startswith("slot:"):
            slot = failure.split(":", 1)[1]
            mapped.append({
                "intent": "F1", "scope": "F2", "municipality": "F3", "region": "F4",
                "fee": "F5", "reservation": "F6", "venue": "F7", "rain": "F8",
                "audience_mode": "F9", "experience": "F10",
            }.get(slot, "F21"))
        elif failure.startswith("forbidden:"):
            slot = failure.split(":", 1)[1]
            mapped.append({"fee": "F5", "reservation": "F6", "venue": "F7", "rain": "F8", "audience_mode": "F9", "experience": "F10"}.get(slot, "F21"))
        elif failure.startswith("schema") or failure.startswith("frame"):
            mapped.append("F22")
        else:
            mapped.append("F23")

    if int(row.get("unsupported_inference_count", 0) or 0):
        mapped.append("F18")
    if int(row.get("silent_coercion_count", 0) or 0):
        mapped.append("F18")
    if not mapped and row.get("frame_error"):
        mapped.append("F22")
    unique = list(dict.fromkeys(mapped))
    return (unique[0] if unique else "F23"), unique[1:]


def _failure_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = Counter()
    secondary = Counter()
    details: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        p, s = _row_primary_failures(row)
        primary[p] += int(not bool(row.get("machine_pass")))
        secondary.update(s)
        if not bool(row.get("machine_pass")):
            details[p].append(str(row.get("id")))
        row["primary_failure"] = p
        row["secondary_failures"] = s
    return {
        "taxonomy": FAILURE_TAXONOMY,
        "primary_counts": dict(sorted(primary.items())),
        "secondary_counts": dict(sorted(secondary.items())),
        "case_ids_by_primary": {key: value for key, value in sorted(details.items())},
    }


def _model_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    calls = [int((row.get("observability") or {}).get("calls", 0) or 0) for row in rows]
    client_latency = [float((row.get("observability") or {}).get("client_elapsed_ms", 0.0) or 0.0) for row in rows]
    server_latency: list[float] = []
    generation_latency: list[float] = []
    setup_latency: list[float] = []
    load_latency: list[float] = []
    for row in rows:
        for call in (row.get("service_observability") or []):
            if call.get("server_total_ms") is not None:
                server_latency.append(float(call.get("server_total_ms", 0.0) or 0.0))
            if call.get("generation_ms") is not None:
                generation_latency.append(float(call.get("generation_ms", 0.0) or 0.0))
            if call.get("container_setup_ms") is not None:
                setup_latency.append(float(call.get("container_setup_ms", 0.0) or 0.0))
            if call.get("model_load_ms") is not None:
                load_latency.append(float(call.get("model_load_ms", 0.0) or 0.0))
    prompt_tokens = sum(int((row.get("observability") or {}).get("prompt_tokens", 0) or 0) for row in rows)
    generated_tokens = sum(int((row.get("observability") or {}).get("generated_tokens", 0) or 0) for row in rows)
    model_called = sum(value > 0 for value in calls)
    structural = Counter()
    for row in rows:
        obs = row.get("observability") or {}
        if row.get("raw_model_response"):
            structural["raw_parse_success"] += 1
        else:
            structural["empty_or_no_response"] += 1
        if row.get("atomic_frame_valid"):
            structural["schema_valid"] += 1
        else:
            structural["invalid_frame"] += 1
        if row.get("frame_fallback"):
            structural["fail_soft"] += 1
        if row.get("raw_model_response") and not row.get("atomic_frame_valid"):
            structural["truncated_or_invalid"] += 1
        if obs.get("repair_called") or row.get("repair_calls"):
            structural["repair_call"] += 1
    failures = _failure_summary(rows)
    return {
        "model_key": rows[0].get("model_key") if rows else None,
        "model_id": rows[0].get("model_id") if rows else None,
        "cases": len(rows),
        "machine_pass": sum(bool(row.get("machine_pass")) for row in rows),
        "machine_pass_rate": round(sum(bool(row.get("machine_pass")) for row in rows) / len(rows), 4) if rows else 0.0,
        "model_called_cases": model_called,
        "zero_model_deterministic_cases": len(rows) - model_called,
        "total_calls": sum(calls),
        "calls_per_case": round(sum(calls) / len(rows), 4) if rows else 0.0,
        "repair_calls": sum(int(row.get("repair_calls", 0) or 0) for row in rows),
        "structural_validity": {
            **dict(structural),
            "valid_denominator": model_called,
            "valid": sum(bool(row.get("atomic_frame_valid")) for row in rows if int((row.get("observability") or {}).get("calls", 0) or 0) > 0),
            "rate": round(sum(bool(row.get("atomic_frame_valid")) for row in rows if int((row.get("observability") or {}).get("calls", 0) or 0) > 0) / model_called, 4) if model_called else 1.0,
        },
        "unsupported_inference_count": sum(int(row.get("unsupported_inference_count", 0) or 0) for row in rows),
        "unsupported_inference_accepted_count": sum(int(row.get("unsupported_inference_count", 0) or 0) for row in rows),
        "unsupported_inference_prevented_count": sum(int(row.get("unsupported_inference_prevented_count", 0) or 0) for row in rows),
        "silent_coercion_count": sum(int(row.get("silent_coercion_count", 0) or 0) for row in rows),
        "silent_coercion_accepted_count": sum(int(row.get("silent_coercion_count", 0) or 0) for row in rows),
        "silent_coercion_prevented_count": sum(int(row.get("silent_coercion_prevented_count", 0) or 0) for row in rows),
        "missed_data_gap_count": sum("missed_data_gap" in row.get("failures", []) for row in rows),
        "false_data_gap_count": sum("false_data_gap" in row.get("failures", []) for row in rows),
        "clarification_actual_count": sum(bool(row.get("clarification_reason")) for row in rows),
        "clarification_required_count": sum(
            row.get("semantic_resolution") != "resolved"
            or (row.get("capability") or {}).get("allowed_semantic_action") == "clarify"
            or row.get("clarification_reason") in {"fail_soft", "no_executable_supported_constraints"}
            for row in rows
        ),
        "clarification_precision": _clarification_metric(rows, precision=True),
        "clarification_recall": _clarification_metric(rows, precision=False),
        "semantic_constraint_accuracy": round(
            sum(not any(str(f).startswith("slot:") or str(f).startswith("forbidden:") for f in row.get("failures", [])) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "latency_ms": {
            "client_orchestrator": _numeric_summary(client_latency),
            "server_total": _numeric_summary(server_latency),
            "warm_generation": _numeric_summary(generation_latency),
            "container_setup": _numeric_summary(setup_latency),
            "model_load": _numeric_summary(load_latency),
            "deterministic_v23_overhead": _numeric_summary([float(row.get("orchestrator_latency_ms", 0.0) or 0.0) for row in rows if not int((row.get("observability") or {}).get("calls", 0) or 0)]),
        },
        "tokens": {
            "total_prompt_tokens": prompt_tokens,
            "mean_prompt_tokens": round(prompt_tokens / model_called, 3) if model_called else 0.0,
            "total_generated_tokens": generated_tokens,
            "mean_generated_tokens": round(generated_tokens / model_called, 3) if model_called else 0.0,
        },
        "failure_clusters": failures,
        "prevented_unsupported_case_ids": [str(row.get("id")) for row in rows if int(row.get("unsupported_inference_prevented_count", 0) or 0) > 0],
        "prevented_silent_coercion_case_ids": [str(row.get("id")) for row in rows if int(row.get("silent_coercion_prevented_count", 0) or 0) > 0],
    }


def _clarification_metric(rows: Sequence[Mapping[str, Any]], *, precision: bool) -> float:
    required = []
    actual = []
    for row in rows:
        is_required = (
            row.get("semantic_resolution") != "resolved"
            or (row.get("capability") or {}).get("allowed_semantic_action") == "clarify"
            or row.get("clarification_reason") in {"fail_soft", "no_executable_supported_constraints"}
        )
        is_actual = bool(row.get("clarification_reason"))
        required.append(is_required)
        actual.append(is_actual)
    true_positive = sum(a and r for a, r in zip(actual, required))
    denominator = sum(actual) if precision else sum(required)
    return round(true_positive / denominator, 4) if denominator else 1.0


def _category_result(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    categories: list[str] = []
    seen = set()
    for rows in rows_by_model.values():
        for row in rows:
            category = str(row.get("category"))
            if category not in seen:
                seen.add(category)
                categories.append(category)
    categories.sort()
    output = []
    for category in categories:
        item: dict[str, Any] = {"category": category}
        subset_counts = []
        for key, rows in rows_by_model.items():
            subset = [row for row in rows if str(row.get("category")) == category]
            item[key] = sum(bool(row.get("machine_pass")) for row in subset)
            subset_counts.append(len(subset))
        item["cases"] = subset_counts[0] if subset_counts else 0
        item["difference_llmjp_minus_sarashina"] = item.get("llm-jp-4-8b", 0) - item.get("sarashina-2.2-3b", 0)
        output.append(item)
    return output


def _mc_nemar_exact(b: int, c: int) -> dict[str, Any]:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "two_sided_p": 1.0}
    tail = min(b, c)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    p_value = min(1.0, 2.0 * probability)
    return {"b": b, "c": c, "discordant": n, "two_sided_p": round(p_value, 8)}


def _pairwise(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    a = {str(row["id"]): row for row in rows_by_model["sarashina-2.2-3b"]}
    b = {str(row["id"]): row for row in rows_by_model["llm-jp-4-8b"]}
    buckets = {"both_pass": [], "sarashina_only_pass": [], "llm_jp_only_pass": [], "both_fail": []}
    records = []
    for case_id in sorted(a, key=lambda value: int(value.split("-")[-1])):
        sarashina_pass = bool(a[case_id].get("machine_pass"))
        llmjp_pass = bool(b[case_id].get("machine_pass"))
        if sarashina_pass and llmjp_pass:
            bucket = "both_pass"
        elif sarashina_pass:
            bucket = "sarashina_only_pass"
        elif llmjp_pass:
            bucket = "llm_jp_only_pass"
        else:
            bucket = "both_fail"
        buckets[bucket].append(case_id)
        records.append({
            "case_id": case_id,
            "category": a[case_id].get("category"),
            "bucket": bucket,
            "sarashina_machine_pass": sarashina_pass,
            "llm_jp_machine_pass": llmjp_pass,
            "sarashina_failures": a[case_id].get("failures", []),
            "llm_jp_failures": b[case_id].get("failures", []),
            "sarashina_frame": a[case_id].get("frame"),
            "llm_jp_frame": b[case_id].get("frame"),
            "sarashina_evidence": {
                "request": a[case_id].get("evidence_request"),
                "accepted_atoms": a[case_id].get("accepted_atoms", []),
                "ignored_atoms": a[case_id].get("ignored_atoms", []),
                "rejected_atoms": a[case_id].get("rejected_atoms", []),
                "verifier": a[case_id].get("semantic_verifier"),
            },
            "llm_jp_evidence": {
                "request": b[case_id].get("evidence_request"),
                "accepted_atoms": b[case_id].get("accepted_atoms", []),
                "ignored_atoms": b[case_id].get("ignored_atoms", []),
                "rejected_atoms": b[case_id].get("rejected_atoms", []),
                "verifier": b[case_id].get("semantic_verifier"),
            },
            "sarashina_raw_frame": a[case_id].get("raw_atomic_frame"),
            "llm_jp_raw_frame": b[case_id].get("raw_atomic_frame"),
        })
    b_count = len(buckets["sarashina_only_pass"])
    c_count = len(buckets["llm_jp_only_pass"])
    return {
        "counts": {key: len(value) for key, value in buckets.items()},
        "case_ids": buckets,
        "mcnemar_exact": _mc_nemar_exact(b_count, c_count),
        "records": records,
    }


def _evidence_pairwise(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    a = {str(row["id"]): row for row in rows_by_model["sarashina-2.2-3b"]}
    b = {str(row["id"]): row for row in rows_by_model["llm-jp-4-8b"]}

    def attempted(row: Mapping[str, Any]) -> bool:
        return int(row.get("unsupported_inference_prevented_count", 0) or 0) > 0

    buckets = {"neither_attempt": [], "sarashina_only_attempt": [], "llm_jp_only_attempt": [], "both_attempt": []}
    coercion_buckets = {"neither_attempt": [], "sarashina_only_attempt": [], "llm_jp_only_attempt": [], "both_attempt": []}
    for case_id in sorted(a, key=lambda value: int(value.split("-")[-1])):
        aa = attempted(a[case_id])
        bb = attempted(b[case_id])
        key = "both_attempt" if aa and bb else "sarashina_only_attempt" if aa else "llm_jp_only_attempt" if bb else "neither_attempt"
        buckets[key].append(case_id)
        ac = int(a[case_id].get("silent_coercion_prevented_count", 0) or 0) > 0
        bc = int(b[case_id].get("silent_coercion_prevented_count", 0) or 0) > 0
        ckey = "both_attempt" if ac and bc else "sarashina_only_attempt" if ac else "llm_jp_only_attempt" if bc else "neither_attempt"
        coercion_buckets[ckey].append(case_id)
    return {
        "unsupported_inference_attempt_case_buckets": {key: value for key, value in buckets.items()},
        "unsupported_inference_attempt_counts": {key: len(value) for key, value in buckets.items()},
        "silent_coercion_attempt_case_buckets": {key: value for key, value in coercion_buckets.items()},
        "silent_coercion_attempt_counts": {key: len(value) for key, value in coercion_buckets.items()},
        "unsafe_inference_accepted_case_ids": {
            key: [str(row.get("id")) for row in rows if int(row.get("unsupported_inference_count", 0) or 0) > 0]
            for key, rows in rows_by_model.items()
        },
        "prevented_attempts": {
            key: {
                "unsupported_inference": [str(row.get("id")) for row in rows if int(row.get("unsupported_inference_prevented_count", 0) or 0) > 0],
                "silent_coercion": [str(row.get("id")) for row in rows if int(row.get("silent_coercion_prevented_count", 0) or 0) > 0],
            }
            for key, rows in rows_by_model.items()
        },
    }


def _manual_review_queue(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]], dataset: Mapping[str, Any]) -> dict[str, Any]:
    cases = {str(case["id"]): case for case in dataset.get("cases", [])}
    ids = set(str(case["id"]) for case in dataset.get("cases", []) if case.get("manual_review"))
    rows = [row for model_rows in rows_by_model.values() for row in model_rows]
    ids.update(str(row["id"]) for row in rows if not row.get("machine_pass"))
    by_id = defaultdict(list)
    for model_rows in rows_by_model.values():
        for row in model_rows:
            by_id[str(row["id"])].append(row)
    ids.update(case_id for case_id, pair in by_id.items() if len(pair) == 2 and bool(pair[0].get("machine_pass")) != bool(pair[1].get("machine_pass")))
    ids.update(str(row["id"]) for row in rows if int(row.get("unsupported_inference_prevented_count", 0) or 0) > 0 or int(row.get("silent_coercion_prevented_count", 0) or 0) > 0)
    ids.update(str(row["id"]) for row in rows if bool(row.get("data_gap_reason")) or bool(row.get("clarification_reason")))
    queue = []
    for case_id in sorted(ids, key=lambda value: int(value.split("-")[-1])):
        queue.append({
            "case_id": case_id,
            "category": cases.get(case_id, {}).get("category"),
            "query": cases.get(case_id, {}).get("query"),
            "reason_for_review": {
                "fixture_manual_review": bool(cases.get(case_id, {}).get("manual_review")),
                "machine_fail": any(str(row.get("id")) == case_id and not row.get("machine_pass") for row in rows),
                "pairwise_discordance": len({bool(row.get("machine_pass")) for row in by_id.get(case_id, [])}) > 1,
                "evidence_boundary_attempt": any(str(row.get("id")) == case_id and (int(row.get("unsupported_inference_prevented_count", 0) or 0) or int(row.get("silent_coercion_prevented_count", 0) or 0)) > 0 for row in rows),
                "data_gap_or_clarification_difference": any(str(row.get("id")) == case_id and (row.get("data_gap_reason") or row.get("clarification_reason")) for row in rows),
            },
            "model_rows": {key: next((row for row in model_rows if str(row.get("id")) == case_id), None) for key, model_rows in rows_by_model.items()},
            "manual_verdict": None,
            "manual_notes": None,
        })
    return {
        "status": "PENDING_AGENT_REVIEW",
        "rubric": {
            "verdicts": ["PASS", "BORDERLINE", "FAIL"],
            "pass_conditions": ["intent understood", "explicit supported constraints retained", "no unsupported inference accepted", "no silent coercion accepted", "data-gap/clarification is appropriate"],
            "fail_conditions": ["unsupported inference", "hidden assumption", "explicit condition lost", "incorrect release", "missed or unnecessary data-gap", "missed or unnecessary clarification"],
        },
        "cases": queue,
        "case_count": len(queue),
    }


def _report(
    out_dir: Path,
    *,
    status: str,
    manifest: Mapping[str, Any],
    smoke: Mapping[str, Any],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any] | None,
    pairwise: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    manual: Mapping[str, Any] | None,
    error: str | None = None,
) -> None:
    sarashina = summary.get("models", {}).get("sarashina-2.2-3b", {}) if summary else {}
    llmjp = summary.get("models", {}).get("llm-jp-4-8b", {}) if summary else {}
    pair_counts = (pairwise or {}).get("counts", {})

    def line_model(item: Mapping[str, Any], title: str) -> str:
        lat = item.get("latency_ms", {}).get("client_orchestrator", {})
        tokens = item.get("tokens", {})
        struct = item.get("structural_validity", {})
        return "\n".join([
            f"### {title}",
            f"- Machine PASS: {item.get('machine_pass', 'pending')}/100",
            f"- Manual: pending (queue {manual.get('case_count', 0) if manual else 'pending'})",
            f"- Structural: {struct.get('valid', 'pending')}/{struct.get('valid_denominator', 'pending')} (rate {struct.get('rate', 'pending')})",
            f"- Unsupported inference accepted: {item.get('unsupported_inference_accepted_count', 'pending')}",
            f"- Unsupported inference prevented: {item.get('unsupported_inference_prevented_count', 'pending')}",
            f"- Silent coercion accepted: {item.get('silent_coercion_accepted_count', 'pending')}",
            f"- Silent coercion prevented: {item.get('silent_coercion_prevented_count', 'pending')}",
            f"- Missed data-gap: {item.get('missed_data_gap_count', 'pending')}",
            f"- False data-gap: {item.get('false_data_gap_count', 'pending')}",
            f"- Median latency: {lat.get('median_ms', 'pending')} ms",
            f"- p95 latency: {lat.get('p95_ms', 'pending')} ms",
            f"- Generated tokens: {tokens.get('total_generated_tokens', 'pending')}",
        ])

    category_lines = ["| Category | Cases | Sarashina | LLM-jp | Difference |", "|---|---:|---:|---:|---:|"]
    for item in (summary or {}).get("category_result", []):
        category_lines.append(f"| {item['category']} | {item['cases']} | {item.get('sarashina-2.2-3b', 0)} | {item.get('llm-jp-4-8b', 0)} | {item.get('difference_llmjp_minus_sarashina', 0)} |")

    lines = [
        "# Semantic Operations v2.3 Frozen v1 Live A/B",
        "",
        "## 1. Executive conclusion",
        "",
        f"Status: `{status}`. Final model diagnosis is withheld until the manual review queue is completed." if status == "completed" else f"Status: `{status}`.",
        "",
        "## 2. Evaluation integrity",
        "",
        f"- Architecture frozen SHA: `{manifest.get('ARCHITECTURE_FROZEN_SHA')}`",
        f"- Freeze manifest validated: `{manifest.get('freeze_manifest_validated')}`",
        f"- Frozen v1: `{manifest.get('frozen_v1_cases')}` cases, corpus `{manifest.get('frozen_v1_corpus_sha256')}`",
        f"- Same prompt/few-shot/schema/registry/verifier/reducer/orchestrator/evaluator/backend: `{manifest.get('contract_match')}`",
        f"- Smoke: `{json.dumps(smoke, ensure_ascii=False, sort_keys=True)}`",
        f"- Sealed v2.1 200 holdout: OPENED `NO`, RUN `NO`, SHA `{manifest.get('sealed_holdout_sha256')}`",
        f"- Error: `{error}`" if error else "",
        "",
        "## 3. Overall Machine result",
        "",
        line_model(sarashina, "Sarashina 2.2 3B"),
        "",
        line_model(llmjp, "LLM-jp 4 8B"),
        "",
        "## 4. Pairwise result",
        "",
        f"- Both PASS: {pair_counts.get('both_pass', 'pending')}",
        f"- Sarashina only PASS: {pair_counts.get('sarashina_only_pass', 'pending')}",
        f"- LLM-jp only PASS: {pair_counts.get('llm_jp_only_pass', 'pending')}",
        f"- Both FAIL: {pair_counts.get('both_fail', 'pending')}",
        f"- McNemar exact: {(pairwise or {}).get('mcnemar_exact', 'pending')}",
        "",
        "## 5. Category result",
        "",
        *category_lines,
        "",
        "## 6. Manual review",
        "",
        f"Manual review status: `{(manual or {}).get('status', 'pending')}`; cases: `{(manual or {}).get('case_count', 'pending')}`.",
        "",
        "## 7. Machine / Manual divergence",
        "",
        "Pending completion of blinded Output A / Output B review. The raw rows and provisional machine rubric are preserved in the artifact.",
        "",
        "## 8. Evidence Boundary metrics",
        "",
        f"`{json.dumps(evidence or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 9. Unsupported inference attempts",
        "",
        "Prevented attempts are separated from accepted unsafe inferences; see `evidence_boundary_metrics.json`.",
        "",
        "## 10. Prevented silent coercions",
        "",
        "Prevented coercion attempts are separated from accepted coercions; see `evidence_boundary_metrics.json`.",
        "",
        "## 11. Data-gap / clarification",
        "",
        "Counts are computed by the frozen v2.3 evaluator; manual appropriateness remains pending until review.",
        "",
        "## 12. Negation / release",
        "",
        "Release operations are preserved in each raw row and pairwise record.",
        "",
        "## 13. Structural validity",
        "",
        "Structural validity is reported per model above and in `summary.json`.",
        "",
        "## 14. Failure clusters",
        "",
        "Failure taxonomy F1–F23 is in `failure_clusters.json` and per-row `primary_failure` / `secondary_failures`.",
        "",
        "## 15. Latency",
        "",
        "Client/orchestrator, server total, warm generation, setup and model-load timings are separated in `latency_tokens.json`.",
        "",
        "## 16. Tokens",
        "",
        "Prompt and generated tokens are reported per model; tokenizer counts are not treated as quality scores.",
        "",
        "## 17. v2.2 historical comparison",
        "",
        "Historical v2.2 scores 66/100 and 72/100 are not directly comparable because the v2.3 evaluator and closed frame schema changed. The available v2.2 raw frames were not transformed or regenerated for an invalid pseudo-rescore.",
        "",
        "## 18. Model-vs-architecture diagnosis",
        "",
        "Pending manual review and safety-target assessment.",
        "",
        "## 19. Recommended model",
        "",
        "Pending manual review; no recommendation is made from machine PASS alone.",
        "",
        "## 20. One recommended next action",
        "",
        "Complete the queued blinded manual review, then select one diagnosis A/B/C/D/E without changing the frozen architecture.",
        "",
    ]
    out_dir.joinpath("REPORT.md").write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def _write_paired_csv(path: Path, pairwise: Mapping[str, Any]) -> None:
    records = list(pairwise.get("records", []))
    fields = [
        "case_id", "category", "bucket", "sarashina_machine_pass", "llm_jp_machine_pass",
        "sarashina_failures", "llm_jp_failures", "sarashina_evidence", "llm_jp_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: json.dumps(record.get(field), ensure_ascii=False, sort_keys=True) if isinstance(record.get(field), (list, dict)) else record.get(field) for field in fields})


def _write_outputs(
    out_dir: Path,
    *,
    manifest: Mapping[str, Any],
    smoke: Mapping[str, Any],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset: Mapping[str, Any],
    status: str,
    started: str,
    ended: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    rows_by_model = {key: list(value) for key, value in rows_by_model.items()}
    summaries = {key: _model_summary(rows) for key, rows in rows_by_model.items()}
    category = _category_result(rows_by_model) if all(len(rows) == 100 for rows in rows_by_model.values()) else []
    summary = {
        "status": status,
        "started_at": started,
        "ended_at": ended,
        "models": summaries,
        "category_result": category,
        "all_model_cases": {key: len(rows) for key, rows in rows_by_model.items()},
        "error": error,
    }
    pairwise = _pairwise(rows_by_model) if all(len(rows) == 100 for rows in rows_by_model.values()) else None
    evidence = _evidence_pairwise(rows_by_model) if all(len(rows) == 100 for rows in rows_by_model.values()) else None
    manual = _manual_review_queue(rows_by_model, dataset)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "failure_clusters.json", {key: value.get("failure_clusters", {}) for key, value in summaries.items()})
    _write_json(out_dir / "latency_tokens.json", {key: {"latency_ms": value.get("latency_ms"), "tokens": value.get("tokens")} for key, value in summaries.items()})
    _write_json(out_dir / "evidence_boundary_metrics.json", evidence or {"status": "partial"})
    _write_json(out_dir / "manual_review.json", manual)
    if pairwise is not None:
        _write_json(out_dir / "paired_results.json", pairwise)
        _write_paired_csv(out_dir / "paired_results.csv", pairwise)
    _write_json(out_dir / "smoke.json", smoke)
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "evaluation_status.json", {"status": status, "started_at": started, "ended_at": ended, "error": error})
    for key, rows in rows_by_model.items():
        filename = "sarashina_raw.jsonl" if key == "sarashina-2.2-3b" else "llmjp_raw.jsonl"
        _write_jsonl(out_dir / filename, rows)
    _report(
        out_dir,
        status=status,
        manifest=manifest,
        smoke=smoke,
        rows_by_model=rows_by_model,
        summary=summary,
        pairwise=pairwise,
        evidence=evidence,
        manual=manual,
        error=error,
    )
    return summary


def _manifest(
    freeze: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    endpoints: Mapping[str, str],
    started: str,
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    snap = dict(freeze["freeze_snapshot"])
    return {
        "runner_version": RUNNER_VERSION,
        "repository": "ryotamatsuki/ehime-kokubunsai-ai-poc",
        "evaluation_branch": EVALUATION_BRANCH,
        "ARCHITECTURE_FROZEN_SHA": freeze["architecture_frozen_sha"],
        "architecture_frozen_sha": freeze["architecture_frozen_sha"],
        "architecture_frozen_tree_sha": freeze["architecture_frozen_tree_sha"],
        "freeze_manifest_commit_sha": freeze["freeze_manifest_commit_sha"],
        "freeze_manifest_tree_sha": freeze["freeze_manifest_tree_sha"],
        "evaluation_head_sha": freeze["evaluation_head_sha"],
        "evaluation_tree_sha": freeze["evaluation_tree_sha"],
        "freeze_manifest_validated": True,
        "contract_match": True,
        "architecture_version": snap["architecture_version"],
        "frozen_v1_version": dataset.get("version"),
        "frozen_v1_cases": len(dataset.get("cases", [])),
        "frozen_v1_corpus_sha256": _corpus_sha256(),
        "schema_hash": snap["schema_hash"],
        "prompt_hash": snap["prompt_hash"],
        "few_shot_count": snap["few_shot_count"],
        "capability_registry_hash": snap["capability_registry_hash"],
        "grounding_hash": snap["grounding_hash"],
        "verifier_hash": snap["verifier_hash"],
        "reducer_state_hash": snap["state_reducer_hash"],
        "orchestrator_hash": snap["orchestrator_hash"],
        "evaluator_hash": snap["evaluator_hash"],
        "oracle_hash": snap["oracle_hash"],
        "backend_hash": snap["multimodel_backend_hash"],
        "model_registry_hash": snap["model_registry_hash"],
        "models": snap["models"],
        "model_keys": [spec.key for spec in MODEL_SPECS],
        "model_ids": {spec.key: spec.model_id for spec in MODEL_SPECS},
        "lmfe": snap["lmfe"],
        "decoding": snap["decoding"],
        "max_model_calls": snap["max_model_calls"],
        "repair_generation": snap["repair_generation"],
        "case_order": "fixture order UU-001..UU-100",
        "model_order": "odd Sarashina->LLM-jp; even LLM-jp->Sarashina",
        "quality_retry": False,
        "transport_retry_statuses": sorted(TRANSPORT_RETRY_STATUSES),
        "endpoint_urls": dict(endpoints),
        "smoke_excluded_from_score": True,
        "smoke": smoke,
        "reference_date": POC_REFERENCE_DATE,
        "started_at": started,
        "ended_at": None,
        "sealed_holdout_sha256": SEALED_HOLDOUT_SHA256,
        "holdout_opened": False,
        "holdout_executed": False,
        "production_main_modified": False,
        "production_main_deployed": False,
    }


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    _write_json(out_dir / "evaluation_status.json", {"status": "running", "started_at": started})
    rows_by_model: dict[str, list[dict[str, Any]]] = {spec.key: [] for spec in MODEL_SPECS}
    smoke: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    dataset: dict[str, Any] = {}
    try:
        freeze = _freeze_context(args.architecture_frozen_sha)
        dataset = load_frozen_v1_dataset()
        validate_dataset(dataset)
        if len(dataset.get("cases", [])) != 100:
            raise AssertionError("Frozen v1 evaluation requires exactly 100 cases")
        if _corpus_sha256() != "2b11af35e07469a7244c0413abbee948daf04cedc25eebabe12cb0c9cf317efe":
            raise AssertionError("Frozen v1 corpus hash drifted")

        modal_key = os.environ.get("MODAL_KEY", "").strip()
        modal_secret = os.environ.get("MODAL_SECRET", "").strip()
        if not modal_key or not modal_secret:
            raise RuntimeError("MODAL_KEY and MODAL_SECRET are required for authenticated v2.3 endpoints")
        endpoints = {}
        clients: dict[str, EndpointClient] = {}
        for spec in MODEL_SPECS:
            env_name = MODEL_URL_ENV[spec.key]
            url = os.environ.get(env_name, "").strip()
            if not url:
                raise RuntimeError(f"missing deployed v2.3 endpoint env: {env_name}")
            if not url.startswith("https://") or ".modal.run" not in url:
                raise RuntimeError(f"endpoint is not a v2.3 HTTPS Modal endpoint: {env_name}")
            endpoints[spec.key] = url
            clients[spec.key] = EndpointClient(spec, url, modal_key, modal_secret)

        manifest = _manifest(freeze, dataset=dataset, endpoints=endpoints, started=started, smoke={})
        _write_json(out_dir / "manifest.json", manifest)

        for spec in MODEL_SPECS:
            smoke[spec.key] = _smoke_case(spec, clients[spec.key])
            if not smoke[spec.key]["passed"]:
                raise RuntimeError(f"v2.3 smoke failed for {spec.key}: {smoke[spec.key]}")
        manifest["smoke"] = smoke
        _write_json(out_dir / "smoke.json", smoke)
        _write_json(out_dir / "manifest.json", manifest)

        for index, case in enumerate(dataset["cases"], start=1):
            order = list(MODEL_SPECS) if index % 2 == 1 else list(reversed(MODEL_SPECS))
            for model_order_index, spec in enumerate(order, start=1):
                execution_index = (index - 1) * 2 + model_order_index
                state = _seed_state(str(case.get("context", "none")))
                before = _deterministic_preflight(str(case.get("query", "")), state)
                raw_row = evaluate_case_v23(
                    case,
                    clients[spec.key],
                    include_raw=True,
                    format_enforcer="lmfe",
                )
                row = _augment_row(
                    raw_row,
                    case=case,
                    spec=spec,
                    client=clients[spec.key],
                    case_index=index,
                    execution_index=execution_index,
                    model_order_index=model_order_index,
                    before=before,
                )
                rows_by_model[spec.key].append(row)
                _write_jsonl(out_dir / ("sarashina_raw.jsonl" if spec.key == "sarashina-2.2-3b" else "llmjp_raw.jsonl"), rows_by_model[spec.key])
                if row.get("repair_calls") or int(row.get("frame_attempts", 0) or 0) > 1:
                    raise RuntimeError(f"repair generation observed for {spec.key} {case.get('id')}")
                if int(row.get("unsupported_inference_count", 0) or 0) > 0 or int(row.get("silent_coercion_count", 0) or 0) > 0:
                    raise RuntimeError(f"unsafe inference/coercion accepted for {spec.key} {case.get('id')}; evaluation aborted")
                if len(rows_by_model[spec.key]) % 10 == 0:
                    _write_json(out_dir / "evaluation_status.json", {
                        "status": "running",
                        "started_at": started,
                        "completed_cases_by_model": {key: len(value) for key, value in rows_by_model.items()},
                    })

        ended = _now()
        manifest["ended_at"] = ended
        summary = _write_outputs(
            out_dir,
            manifest=manifest,
            smoke=smoke,
            rows_by_model=rows_by_model,
            dataset=dataset,
            status="completed",
            started=started,
            ended=ended,
        )
        print(json.dumps({"status": "completed", "summary": summary}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        ended = _now()
        error = f"{type(exc).__name__}: {exc}"[:2000]
        if not manifest:
            manifest = {
                "runner_version": RUNNER_VERSION,
                "repository": "ryotamatsuki/ehime-kokubunsai-ai-poc",
                "evaluation_branch": EVALUATION_BRANCH,
                "reference_date": POC_REFERENCE_DATE,
                "holdout_opened": False,
                "holdout_executed": False,
                "sealed_holdout_sha256": SEALED_HOLDOUT_SHA256,
                "production_main_modified": False,
                "production_main_deployed": False,
            }
        manifest["ended_at"] = ended
        try:
            _write_outputs(
                out_dir,
                manifest=manifest,
                smoke=smoke,
                rows_by_model=rows_by_model,
                dataset=dataset,
                status="aborted",
                started=started,
                ended=ended,
                error=error,
            )
        finally:
            print(json.dumps({"status": "aborted", "error": error}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--architecture-frozen-sha", required=True)
    parser.add_argument("--evaluation-branch", default=EVALUATION_BRANCH)
    args = parser.parse_args()
    if args.evaluation_branch != EVALUATION_BRANCH:
        raise SystemExit(f"unexpected evaluation branch: {args.evaluation_branch}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

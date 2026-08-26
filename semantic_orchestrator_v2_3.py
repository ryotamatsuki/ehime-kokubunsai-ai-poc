"""Semantic Operations v2.3 Evidence-Bounded orchestration.

Final clarification/data-gap/search control is derived by Python from the
EvidenceRequest and Capability Registry.  The LLM receives at most one residual
semantic call and cannot directly choose a data-gap or clarification flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

from command_orchestrator import CommandOrchestrator, CommandTurnResult
from semantic_atomic_v2_3 import AtomicFrameV23Error, AtomicSemanticFrameV23
from semantic_capability_registry_v2_3 import CapabilitySpec, capability_message, lookup_capability
from semantic_capability_v2_1 import evaluate_capability
from semantic_evidence_v2_3 import AllowedSemanticAction, EvidenceRequest
from semantic_orchestrator_v2_2 import SemanticOperationsOrchestratorV22
from semantic_prompt_v2_3 import build_minimal_atomic_payload_v23
from semantic_state_v2_3 import (
    EvidenceBoundedReduction,
    fail_soft_reduce_v23,
    grounded_slots_from_query_v22,
    reduce_evidence_bounded_frame,
)
from semantic_verifier_v2_3 import EvidenceBoundedVerification, verify_evidence_bounded_frame


FrameCall = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AtomicFrameGenerationV23:
    frame: AtomicSemanticFrameV23 | None
    attempts: int
    error: str | None = None
    raw_output: str | None = None
    observability: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticV23Result:
    status: str
    flow: str
    slots: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    frame: AtomicSemanticFrameV23 | None = None
    verification: EvidenceBoundedVerification | None = None
    reduction: EvidenceBoundedReduction | None = None
    capability: CapabilitySpec | None = None
    command_result: CommandTurnResult | None = None
    deterministic_grounding: Mapping[str, Any] = field(default_factory=dict)
    evidence_request: str = EvidenceRequest.NONE.value
    accepted_atoms: tuple[str, ...] = ()
    ignored_atoms: tuple[str, ...] = ()
    rejected_atoms: tuple[str, ...] = ()
    release_operations: tuple[str, ...] = ()
    clarification_reason: str | None = None
    data_gap_reason: str | None = None
    unsupported_inference_count: int = 0
    silent_coercion_count: int = 0
    missed_data_gap_count: int = 0
    false_data_gap_count: int = 0
    frame_attempts: int = 0
    frame_fallback: bool = False
    deterministic_route: str | None = None
    latency_ms: float = 0.0
    handled: bool = True
    frame_error: str | None = None
    raw_output: str | None = None
    model_observability: Mapping[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.command_result.events if self.command_result is not None else []

    @property
    def near_events(self) -> list[dict[str, Any]]:
        return self.command_result.near_events if self.command_result is not None else []

    @property
    def model_call_count(self) -> int:
        return self.frame_attempts

    @property
    def final_flow(self) -> str:
        return self.flow


def _answer(raw: Any) -> tuple[str | None, Mapping[str, Any]]:
    if isinstance(raw, str):
        return raw, {}
    if isinstance(raw, Mapping):
        answer = raw.get("answer")
        observability = raw.get("observability")
        return (
            str(answer) if isinstance(answer, str) else None,
            dict(observability) if isinstance(observability, Mapping) else {},
        )
    return None, {}


def generate_atomic_frame_v23(
    query: str,
    state: Mapping[str, Any] | None,
    *,
    call: FrameCall,
    grounded: Mapping[str, Any] | None = None,
) -> AtomicFrameGenerationV23:
    payload = build_minimal_atomic_payload_v23(
        query,
        state,
        grounded if grounded is not None else grounded_slots_from_query_v22(query),
    )
    try:
        raw = call(payload)
    except Exception as exc:
        return AtomicFrameGenerationV23(None, 1, error=f"frame call failed: {type(exc).__name__}")
    answer, observability = _answer(raw)
    if answer is None:
        return AtomicFrameGenerationV23(None, 1, error="empty atomic v2.3 frame response", observability=observability)
    try:
        return AtomicFrameGenerationV23(
            AtomicSemanticFrameV23.from_json(answer),
            1,
            raw_output=answer,
            observability=observability,
        )
    except (AtomicFrameV23Error, TypeError, ValueError) as exc:
        return AtomicFrameGenerationV23(
            None,
            1,
            error=f"atomic v2.3 frame invalid: {exc}",
            raw_output=answer,
            observability=observability,
        )


def _slots_from_reduction(reduction: EvidenceBoundedReduction | None) -> dict[str, Any]:
    if reduction is None or reduction.plan is None:
        return {}
    return reduction.plan.slots.to_dict()


def _release_operations(reduction: EvidenceBoundedReduction | None) -> tuple[str, ...]:
    if reduction is None:
        return ()
    return tuple(
        key for key, operation in reduction.constraint_operations.items()
        if operation == "release"
    )


class SemanticOperationsOrchestratorV23(SemanticOperationsOrchestratorV22):
    """Evidence-Bounded evaluator; production paths are not changed or deployed."""

    def _base_result(
        self,
        *,
        status: str,
        flow: str,
        started: float,
        message: str = "",
        slots: Mapping[str, Any] | None = None,
        route: str | None = None,
        handled: bool = True,
    ) -> SemanticV23Result:
        return SemanticV23Result(
            status=status,
            flow=flow,
            slots=dict(slots or {}),
            message=message,
            deterministic_route=route,
            latency_ms=(time.perf_counter() - started) * 1000,
            handled=handled,
        )

    def _from_fail_soft(
        self,
        query: str,
        state: Mapping[str, Any] | None,
        reduction: EvidenceBoundedReduction,
        route: str,
        started: float,
        *,
        generation: AtomicFrameGenerationV23 | None,
        verification: EvidenceBoundedVerification | None = None,
        frame: AtomicSemanticFrameV23 | None = None,
        grounded: Mapping[str, Any] | None = None,
    ) -> SemanticV23Result:
        if reduction.ready and reduction.plan is not None:
            command = CommandOrchestrator(
                modal_call=None,
                reference_date=self.reference_date,
                events=self.events,
            ).handle_query(query, state, command_plan=reduction.plan)
            return SemanticV23Result(
                status=command.status,
                flow=command.flow,
                slots=command.command.slots.to_dict(),
                message=command.message,
                frame=frame,
                verification=verification,
                reduction=reduction,
                command_result=command,
                deterministic_grounding=dict(grounded or reduction.grounded_slots or {}),
                frame_attempts=generation.attempts if generation else 0,
                frame_fallback=True,
                deterministic_route=route,
                latency_ms=(time.perf_counter() - started) * 1000,
                frame_error=generation.error if generation else reduction.fail_soft_reason,
                raw_output=generation.raw_output if generation else None,
                model_observability=dict(generation.observability) if generation else {},
            )
        return SemanticV23Result(
            status="clarification",
            flow="unsupported",
            slots=_slots_from_reduction(reduction),
            message=reduction.message or "一部の条件を確実に解釈できませんでした。条件を少し言い換えて教えてください。",
            frame=frame,
            verification=verification,
            reduction=reduction,
            deterministic_grounding=dict(grounded or reduction.grounded_slots or {}),
            clarification_reason="fail_soft",
            frame_attempts=generation.attempts if generation else 0,
            frame_fallback=True,
            deterministic_route=route,
            latency_ms=(time.perf_counter() - started) * 1000,
            frame_error=generation.error if generation else reduction.fail_soft_reason,
            raw_output=generation.raw_output if generation else None,
            model_observability=dict(generation.observability) if generation else {},
        )

    def handle_query(self, query: str, state: Mapping[str, Any] | None = None) -> SemanticV23Result:
        started = time.perf_counter()
        value = str(query).strip()
        if not value:
            raise ValueError("query must not be empty")

        product_capability = evaluate_capability(value)
        if not product_capability.allowed:
            return self._base_result(
                status="unsupported",
                flow="unsupported",
                started=started,
                message=product_capability.message,
                route=f"capability:{product_capability.reason}",
            )

        security_message = self._security_guard(value)
        if security_message is not None:
            return self._base_result(
                status="unsupported",
                flow="unsupported",
                started=started,
                message=security_message,
                route="security_or_domain_guard",
            )

        route_name, trusted_plan, immediate_message, previous_scope = self._deterministic_context_plan(value, state)
        if immediate_message is not None:
            if route_name == "recommend_next_ambiguous":
                return self._base_result(
                    status="clarification", flow="recommend_next", started=started,
                    message=immediate_message, route=route_name,
                )
            status = "clarification" if route_name == "clarify_reference" else "unsupported"
            return self._base_result(
                status=status, flow="unsupported", started=started,
                message=immediate_message, route=route_name,
            )
        if trusted_plan is not None:
            command = CommandOrchestrator(
                modal_call=None,
                reference_date=self.reference_date,
                events=self.events,
            ).handle_query(value, state, command_plan=trusted_plan)
            return SemanticV23Result(
                status=command.status,
                flow=command.flow,
                slots=command.command.slots.to_dict(),
                message=command.message,
                command_result=command,
                deterministic_route=route_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                handled=command.handled,
            )

        grounded = grounded_slots_from_query_v22(value, self.reference_date)
        if self.frame_call is None:
            reduction = fail_soft_reduce_v23(
                value,
                state,
                reference_date=self.reference_date,
                previous_scope=previous_scope,
                reason="frame_backend_unavailable",
            )
            return self._from_fail_soft(
                value, state, reduction, route_name, started,
                generation=None, grounded=grounded,
            )

        generation = generate_atomic_frame_v23(value, state, call=self.frame_call, grounded=grounded)
        if generation.frame is None:
            reduction = fail_soft_reduce_v23(
                value,
                state,
                reference_date=self.reference_date,
                previous_scope=previous_scope,
                reason=generation.error or "atomic_v23_frame_invalid",
            )
            return self._from_fail_soft(
                value, state, reduction, route_name, started,
                generation=generation, grounded=grounded,
            )

        verification = verify_evidence_bounded_frame(
            generation.frame,
            query=value,
            state=state,
            grounded=grounded,
        )
        if not verification.accepted or verification.frame is None:
            reduction = fail_soft_reduce_v23(
                value,
                state,
                reference_date=self.reference_date,
                previous_scope=previous_scope,
                reason=f"evidence_verifier:{verification.reason or 'rejected'}",
            )
            return self._from_fail_soft(
                value, state, reduction, route_name, started,
                generation=generation, verification=verification,
                frame=generation.frame, grounded=grounded,
            )

        frame = verification.frame
        reduction = reduce_evidence_bounded_frame(
            frame,
            value,
            state,
            reference_date=self.reference_date,
            previous_scope=previous_scope,
        )
        capability = lookup_capability(frame.evidence_request)
        evidence = EvidenceRequest(frame.evidence_request)
        common = {
            "frame": frame,
            "verification": verification,
            "reduction": reduction,
            "capability": capability,
            "deterministic_grounding": dict(grounded),
            "evidence_request": evidence.value,
            "accepted_atoms": verification.accepted_atoms,
            "ignored_atoms": verification.ignored_atoms,
            "rejected_atoms": verification.rejected_atoms,
            "release_operations": _release_operations(reduction),
            "unsupported_inference_count": verification.unsupported_inference_count,
            "silent_coercion_count": verification.silent_coercion_prevented_count,
            "frame_attempts": generation.attempts,
            "frame_fallback": False,
            "deterministic_route": route_name,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "frame_error": generation.error,
            "raw_output": generation.raw_output,
            "model_observability": dict(generation.observability),
        }

        # Final control is registry-derived, never taken from the model intent.
        if capability.allowed_semantic_action is AllowedSemanticAction.CLARIFY:
            return SemanticV23Result(
                status="clarification",
                flow="unsupported",
                slots=_slots_from_reduction(reduction),
                message=capability_message(capability),
                clarification_reason=evidence.value,
                **common,
            )
        if capability.allowed_semantic_action is AllowedSemanticAction.DATA_GAP:
            return SemanticV23Result(
                status="unsupported",
                flow="unsupported",
                slots=_slots_from_reduction(reduction),
                message=capability_message(capability),
                data_gap_reason=evidence.value,
                **common,
            )

        if not reduction.ready or reduction.plan is None:
            return SemanticV23Result(
                status="clarification",
                flow="unsupported",
                slots=_slots_from_reduction(reduction),
                message=reduction.message or "検索条件を確認できませんでした。条件を少し具体的に教えてください。",
                clarification_reason="no_executable_supported_constraints",
                handled=False,
                **common,
            )

        command = CommandOrchestrator(
            modal_call=None,
            reference_date=self.reference_date,
            events=self.events,
        ).handle_query(value, state, command_plan=reduction.plan)
        return SemanticV23Result(
            status=command.status,
            flow=command.flow,
            slots=command.command.slots.to_dict(),
            message=command.message,
            command_result=command,
            handled=command.handled,
            **common,
        )


__all__ = [
    "AtomicFrameGenerationV23",
    "SemanticOperationsOrchestratorV23",
    "SemanticV23Result",
    "generate_atomic_frame_v23",
]

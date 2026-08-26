"""Semantic Operations v2.3 Evidence-Bounded orchestration.

Final clarification/data-gap/search control is derived by Python from bounded
interpretation, Capability Registry, independent grounding and trusted state.
The LLM receives at most one residual semantic call and cannot directly choose
a final data-gap/clarification flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

import suitability_clarification
from command_orchestrator import CommandOrchestrator, CommandTurnResult
from semantic_atomic_v2_3 import AtomicFrameV23Error, AtomicSemanticFrameV23
from semantic_capability_registry_v2_3 import CapabilitySpec, capability_message, lookup_capability
from semantic_capability_v2_1 import evaluate_capability
from semantic_demographic_v2_1 import needs_relational_demographic_clarification
from semantic_evidence_v2_3 import AllowedSemanticAction, EvidenceRequest, SemanticResolution
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
    semantic_resolution: str = SemanticResolution.RESOLVED.value
    accepted_atoms: tuple[str, ...] = ()
    ignored_atoms: tuple[str, ...] = ()
    rejected_atoms: tuple[str, ...] = ()
    grounding_proofs: tuple[str, ...] = ()
    release_operations: tuple[str, ...] = ()
    clarification_reason: str | None = None
    data_gap_reason: str | None = None
    unsupported_inference_count: int = 0
    silent_coercion_count: int = 0
    unsupported_inference_prevented_count: int = 0
    silent_coercion_prevented_count: int = 0
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

    def trace(self) -> dict[str, Any]:
        """Return privacy-bounded turn observability without user secrets."""

        return {
            "deterministic_grounding": dict(self.deterministic_grounding),
            "atomic_raw_frame": self.raw_output,
            "parsed_frame": self.frame.to_dict() if self.frame is not None else None,
            "evidence_request": self.evidence_request,
            "semantic_resolution": self.semantic_resolution,
            "capability_lookup": self.capability.to_dict() if self.capability is not None else None,
            "verifier_accepted_atoms": list(self.accepted_atoms),
            "verifier_ignored_atoms": list(self.ignored_atoms),
            "verifier_rejected_atoms": list(self.rejected_atoms),
            "grounding_proofs": list(self.grounding_proofs),
            "unsupported_inference_count": self.unsupported_inference_count,
            "silent_coercion_count": self.silent_coercion_count,
            "unsupported_inference_prevented_count": self.unsupported_inference_prevented_count,
            "silent_coercion_prevented_count": self.silent_coercion_prevented_count,
            "release_operations": list(self.release_operations),
            "clarification_reason": self.clarification_reason,
            "data_gap_reason": self.data_gap_reason,
            "final_flow": self.flow,
            "status": self.status,
            "fail_soft": self.frame_fallback,
            "frame_error": self.frame_error,
            "model_key": self.model_observability.get("model_key"),
            "model_call_count": self.model_call_count,
            "prompt_tokens": self.model_observability.get("prompt_tokens", 0),
            "generated_tokens": self.model_observability.get("generated_tokens", 0),
            "latency_ms": round(float(self.latency_ms), 3),
        }


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


def _resolution_message(resolution: SemanticResolution, capability: CapabilitySpec) -> str:
    capability_text = capability_message(capability)
    if capability_text:
        return capability_text + " 確認できる条件をもう少し具体的に教えてください。"
    if resolution is SemanticResolution.CONDITIONAL:
        return "条件分岐のままでは安全に1つの検索条件へ変換できません。どの条件を優先するか教えてください。"
    if resolution is SemanticResolution.UNDERSPECIFIED:
        return "検索に必要な条件が足りません。日付・時間帯・地域など、分かる範囲で教えてください。"
    return "意味が複数に取れる条件があります。重視する条件をもう少し具体的に教えてください。"


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
        evidence: EvidenceRequest = EvidenceRequest.NONE,
        capability: CapabilitySpec | None = None,
        clarification_reason: str | None = None,
    ) -> SemanticV23Result:
        return SemanticV23Result(
            status=status,
            flow=flow,
            slots=dict(slots or {}),
            message=message,
            evidence_request=evidence.value,
            capability=capability,
            clarification_reason=clarification_reason,
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
                semantic_resolution=frame.semantic_resolution if frame is not None else SemanticResolution.RESOLVED.value,
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
            semantic_resolution=frame.semantic_resolution if frame is not None else SemanticResolution.RESOLVED.value,
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

        # Preserve the established zero-model-call senior/demographic invariant.
        # It is a general demographic x suitability guard and never converts a
        # demographic property into an audience or Experience dimension.
        suitability = suitability_clarification.analyze_suitability_request(value)
        if suitability.needs_clarification or needs_relational_demographic_clarification(value):
            evidence = EvidenceRequest.RELATIONAL_SUITABILITY
            capability = lookup_capability(evidence)
            return self._base_result(
                status="clarification",
                flow="unsupported",
                started=started,
                message=capability_message(capability),
                route="relational_suitability_guard",
                evidence=evidence,
                capability=capability,
                clarification_reason=evidence.value,
            )
        if suitability.should_strip_suitability_marker:
            value = suitability.sanitized_query

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
        resolution = SemanticResolution(frame.semantic_resolution)
        common = {
            "frame": frame,
            "verification": verification,
            "reduction": reduction,
            "capability": capability,
            "deterministic_grounding": dict(grounded),
            "evidence_request": evidence.value,
            "semantic_resolution": resolution.value,
            "accepted_atoms": verification.accepted_atoms,
            "ignored_atoms": verification.ignored_atoms,
            "rejected_atoms": verification.rejected_atoms,
            "grounding_proofs": verification.grounding_proofs,
            "release_operations": _release_operations(reduction),
            "unsupported_inference_count": verification.unsupported_inference_count,
            "silent_coercion_count": verification.silent_coercion_count,
            "unsupported_inference_prevented_count": verification.unsupported_inference_prevented_count,
            "silent_coercion_prevented_count": verification.silent_coercion_prevented_count,
            "frame_attempts": generation.attempts,
            "frame_fallback": False,
            "deterministic_route": route_name,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "frame_error": generation.error,
            "raw_output": generation.raw_output,
            "model_observability": dict(generation.observability),
        }

        # A resolution signal is not itself a routing command.  Python decides
        # that unresolved semantics need clarification after consulting the
        # capability result; data-gap evidence is still recorded explicitly.
        if resolution is not SemanticResolution.RESOLVED:
            return SemanticV23Result(
                status="clarification",
                flow="unsupported",
                slots=_slots_from_reduction(reduction),
                message=_resolution_message(resolution, capability),
                clarification_reason=f"semantic_resolution:{resolution.value}",
                data_gap_reason=evidence.value if capability.allowed_semantic_action is AllowedSemanticAction.DATA_GAP else None,
                **common,
            )

        # Final evidence-boundary control is registry-derived, not model intent.
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

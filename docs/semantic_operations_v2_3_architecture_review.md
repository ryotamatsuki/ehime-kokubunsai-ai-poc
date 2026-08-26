# Semantic Operations v2.3 Architecture Review

This document records the consolidated A–H concern review requested for the v2.3 Evidence-Bounded architecture. It is an architecture review, not a live-model evaluation, and it does not use the sealed 200-case holdout.

## A. Capability / Evidence architecture

PASS.

- `EvidenceRequest` is a closed vocabulary separate from executable event attributes.
- Capability support is resolved by the Python `CAPABILITY_REGISTRY`; the model cannot declare a request executable merely by emitting a supported atom.
- Registry support means **safe product capability**, not merely “some similarly named field exists somewhere in the raw event document.” For example, event detail data can contain access/parking facts while cross-event parking-distance search remains unsupported unless a deterministic searchable contract exists.
- Relational suitability is explicitly non-coercible and routes to clarification rather than a proxy Experience dimension.

## B. Negation / release state machine

PASS.

- Positive conditions, exclusions and releases are distinct operations.
- Release remains valid for both prior-state removal and same-turn lexical false-positive cancellation.
- A release with neither prior state nor current deterministic grounding is a safe no-op.
- Experience releases are concept-scoped; releasing one concept cannot clear unrelated Experience constraints.
- Explicit exclusion composition is verified independently before it can override a lexical positive.

## C. Verifier invariants

PASS.

- Every model-produced positive supported atom requires deterministic grounding, an explicit compositional proof, or a trusted prior-state basis.
- Ungrounded positive atoms are rejected from the trusted reducer rather than normalized into a different supported meaning.
- Registry/frame contradictions fail soft without a second model call.
- Trusted deterministic grounding wins over model output.
- Actual adopted unsupported-inference and silent-coercion counters are structurally separated from prevented-attempt counters.

## D. Data-model compatibility

PASS.

- Existing municipality, region, fee, reservation, venue, rain, audience and Experience contracts remain authoritative.
- v2.3 does not add invented event facts or modify the event catalog/schema.
- Audience semantics remain limited to supported audience modes; demographic markers are not mapped to mobility, seating or adult filters.
- Detail-only data is not automatically promoted into a cross-event search capability.

## E. Model selector / Modal compatibility

PASS.

- Existing PR #45 model registry and Streamlit selector remain unchanged, with Sarashina as default and state reset behavior preserved.
- v2.3 defines a shared multimodel backend contract for Sarashina 2.2 3B and LLM-jp 4 8B.
- Both use the same v2.3 schema, prompt/few-shot transcript, LMFE constraint, decoding policy and deterministic downstream architecture.
- No v2.3 backend is deployed in this implementation phase.

## F. Security / unsupported inference

PASS.

- Existing security/domain guards remain before residual semantic generation.
- Unsupported evidence cannot become an executable filter through an ungrounded model atom.
- Internal prompt/config exfiltration and closed-world fabrication protections remain intact.
- No model-specific post-processing exists in the verifier or orchestrator.

## G. Evaluation metrics

PASS.

The v2.3 evaluator exposes and scores:

- `unsupported_inference_count`
- `missed_data_gap_count`
- `false_data_gap_count`
- `clarification_precision`
- `clarification_recall`
- `semantic_constraint_accuracy`
- `silent_coercion_count`
- prevented unsupported-inference / silent-coercion attempt counts
- model-call, token and latency observability

Evidence-boundary violations are machine failures; manual PASS/BORDERLINE/FAIL remains available for response-quality judgment that should not be forced into deterministic scoring.

## H. Regression / simplification review

PASS.

- v2.3 is additive over the PR #45 stack; production `main` is not modified.
- The mature v2.2 deterministic reducer/executor is reused behind a v2.3 sanitization boundary instead of duplicating command execution.
- No repair generation, self-reflection, semantic voting or second judge is added.
- New behavior is concentrated in Evidence, Capability Registry, grounding proof, verifier, state adapter, orchestrator and evaluator responsibilities rather than case-ID branches.
- Frozen v1 failures are accepted only when explained by a general architecture invariant; sealed holdout contents remain untouched.

## Review conclusion

The architecture meets the intended Evidence-Bounded boundary: a more aggressive model may classify an unsupported semantic request, and may even attempt to emit a supported proxy atom, but that proxy cannot enter trusted search state without independent evidence. Explicit supported conditions in the same turn remain independently preservable.

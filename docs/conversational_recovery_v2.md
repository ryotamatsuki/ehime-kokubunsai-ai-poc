# Conversational Recovery v2

## Goal

Recovery v2 treats the latest deterministic search result as a shared
conversation object. The user can ask why the result set was produced, ask
about one result, refine the set, and then inspect structured event details.
Natural-language interpretation is delegated to the existing Semantic
Command Generator only when the deterministic shortcut is not confident.

The Generator selects a flow and bounded slots. Python executes the flow and
renders event facts. The Generator never writes search rationales, event
facts, counts, prices, dates, reservation claims, or chain-of-thought.

## Routing architecture

```text
user utterance
  -> security / input / domain guards
  -> deterministic high-confidence shortcut
  -> existing Semantic Command Generator (at most one generation turn)
  -> Python context validation
  -> fixed executor and structured event data
  -> grounded response
```

The shortcut retains the v1 explanation and detail families. Unknown wording
such as “何を材料にこの候補を出したの？” is allowed to fall through to
the existing Generator; no Recovery-specific LLM router is introduced.

Security remains higher priority than recovery. A request to reveal internal
instructions or fabricate an event is rejected before generation.

## Semantic flows

`command_models.FLOW_NAMES` and `flow_registry.FLOW_REGISTRY` contain:

- `explain_search`: explain the criteria, materials, viewpoint, or deterministic
  selection policy for the immediately preceding result set.
- `explain_result`: explain the structured evidence for one result from that
  result set.

Both flows use fixed Python executors. The model receives semantic state such
as `has_last_search_context` and `last_result_count`; the full
`SearchContext`, evidence, and search specifications remain in Python.

## Context and evidence

`SearchContext` preserves the bounded complete ordered result IDs, not only
the first visible card page. Ordinals therefore resolve against the same
ordered set used by pagination. A result outside the active context is not a
valid explanation target, even if it exists in the event catalog.

The context stores public search conditions, selection policy, search specs,
and per-result evidence. Evidence levels remain `explicit`, `derived`,
`inferred`, and `unknown`; only structured public data and the v3 experience
profile are used for hard explanations. In particular, an indoor venue is not
treated as proof that an event is seated.

New searches and recommendation result sets replace the context. Explanations,
FAQ turns, detail turns, and clarification turns preserve it. A missing or
stale context produces a clarification rather than a guessed explanation.

## UI contract

- `explain_search`: text only; stale result cards are suppressed.
- `explain_result`: grounded text and only the referenced card when resolved.
- refinement/search: replace the full ordered result set.
- clarification, FAQ, unsupported, and fallback: do not re-render stale cards.

## Holdout policy

`tests/data/conversational_recovery_holdout.json` is independent of the
production marker families. It contains 300 single-turn cases across
explanation, result reference, refinement, detail, clarification,
recommendation, FAQ, out-of-domain, security, dialect/casual, and noisy input
categories, plus 100 five-turn dialogues. Production code does not import the
fixture or copy its unseen wording into marker lists.

`tests/run_recovery_benchmark.py` compares:

- A: current deterministic phrase router;
- B: an evaluation-only lightweight similarity prototype (no production
  dependency or model download);
- C: deterministic fast path plus the existing Semantic Command contract and
  fixed executor.

Offline C uses a contract fixture for routing quality and is not a claim about
the live Sarashina model. Live model quality and latency must be measured
separately when Modal credentials and the deployed endpoint are available.

## Verification

The v2 contract suite is `tests/run_recovery_v2_qa.py`. The existing standalone
QA scripts remain required, and `.github/workflows/qa.yml` runs compilation,
all deterministic runners, the recovery benchmark, and pytest on pull
requests and pushes to `main`.

## Limitations

- The offline benchmark cannot establish live 3B model recall or generation
  latency.
- Evidence is intentionally conservative. Unsupported structured claims are
  reported as unknown rather than inferred.
- Recommendation rationale records the deterministic recommendation flow and
  result IDs; it does not create a free-form explanation of transportation or
  intent that is absent from the data model.

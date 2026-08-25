# Semantic Operations v2 — Pre-Evaluation Gate

## Purpose

This branch prepares a parallel Semantic Operations v2 path for empirical comparison with the frozen `Unexpected User Utterances v1` baseline. It deliberately does **not** replace the current production command path, Streamlit integration, or production Modal endpoint before evidence is collected.

Baseline reference:

- production `main`: `2a424ec8fb390cadc1390bcf2c64c7ed115c0fa2`
- frozen dataset: `unexpected-user-utterances-v1`
- original baseline: 100 cases, machine PASS 33/100
- model: `sbintuitions/sarashina2.2-3b-instruct-v0.1`

## Design objective

The architecture must improve open-ended language handling **without** adding an ever-growing phrase dictionary or making the entry gate narrower.

Natural-language wording remains open. The internal contract is intentionally finite:

1. deterministic Python extracts only explicit filters already supported by the product (date, municipality/region, fee, age, reservation, venue, time, genre, etc.);
2. Sarashina emits a compact semantic frame for meanings that are difficult to represent safely with the existing parser: intent, refinement, constraint release, Experience concepts, references, clarification, and data-capability gaps;
3. a trusted state reducer applies the frame to bounded previous state and emits the existing validated `CommandPlan`;
4. the existing deterministic `CommandOrchestrator` executes the plan.

The model never chooses a tool name and never supplies event facts.

## Anti-rule-explosion contract

Do not add production rules of the form `if phrase X then concept Y` in response to individual evaluation failures.

A change is acceptable only when it is one of:

- a new stable semantic concept genuinely required by the product;
- a missing state operation/invariant;
- a correction to the generic semantic-frame prompt/contract;
- a correction to deterministic parsing of an already-supported explicit product field;
- a data-model/capability change supported by evidence.

Dialect, typo, paraphrase, and ordinary wording variation should be handled by the semantic normalizer, not by accumulating aliases in the v2 reducer.

## New parallel modules

- `semantic_frame_v2.py`: strict compact frame contract and JSON Schema for constrained decoding.
- `semantic_state_v2.py`: trusted state reducer and explicit constraint release.
- `semantic_orchestrator_v2.py`: pre-model guards, frame generation, reduction, and delegation to the existing trusted executor.
- `unexpected_utterances_v2_eval.py`: evaluates the same frozen 100 cases.
- `semantic_v2_live_eval.py`: isolated Modal evaluation app using the same Sarashina 3B, with explicit LMFE dependency and no production HTTP endpoint.

## Release semantics

Constraint release is a state operation, not a negative filter.

- `release=["fee"]` clears `entry_free`, `paid_only`, and `max_entry_fee`.
- `release=["venue"]` clears venue instead of setting the opposite venue.
- `release=["rain"]` clears the rain preference.
- `release=["experience"]` clears experience constraints.

Release is applied **after** deterministic explicit parsing so lexical parser matches inside a release utterance cannot create the opposite hard constraint.

## Data-capability boundary

Subjective or unsupported facts are not converted into proxy slots. Examples include crowding, noise, medical safety, guaranteed weather cancellation behavior, parking walking distance, toilet proximity, and social comfort. The frame returns a bounded `data_gap`; the reducer returns a grounded limitation message instead of inventing a search fact.

## Pre-evaluation checks

Before any live Sarashina v2 run:

1. `python -m compileall -q .`
2. existing deterministic QA suite remains green;
3. full pytest suite remains green, including semantic-frame schema rejection, release semantics, previous-state refinement, compound explicit constraints, typed references, data-gap non-inference, and pre-model guards;
4. compare branch against `main` and confirm current production files are unchanged: `command_generator.py`, `command_orchestrator.py`, `modal_backend.py`, `streamlit_app.py`, and event data.

## Empirical protocol — DO NOT RUN DURING PRE-EVALUATION PREPARATION

Use the **same frozen 100-case dataset**. Do not modify expected behavior after seeing v2 results.

Run A — architecture only, no decoding constraint:

```bash
mkdir -p artifacts/v2
modal run --quiet \
  --write-result artifacts/v2/unexpected_v2_baseline_format.json \
  semantic_v2_live_eval.py::unexpected_v2_eval \
  --limit 100 \
  --format-enforcer baseline
```

Run B — same architecture with LMFE:

```bash
modal run --quiet \
  --write-result artifacts/v2/unexpected_v2_lmfe.json \
  semantic_v2_live_eval.py::unexpected_v2_eval \
  --limit 100 \
  --format-enforcer lmfe
```

Do not patch production behavior between Run A and Run B.

## Decision metrics

Compare original v1 baseline, v2 baseline-format, and v2 LMFE on at least:

- machine PASS rate overall and by category;
- first-pass structured-frame validity;
- repair rate and repair success;
- flow/status mismatches;
- required-slot loss and forbidden inferred slots;
- release/negation correctness;
- compound-constraint retention;
- context/reference correctness;
- security/scope pre-model handling;
- data-gap/unsupported-inference behavior;
- total model calls and latency.

Manual review should preserve PASS / FAIL / BORDERLINE and inspect raw semantic frames in the evaluation artifact when permitted. Do not tune on individual wordings before failure clustering.

## Promotion gate

This branch is an evaluation branch, not a production merge candidate as-is. Promote the architecture into a clean branch from the then-current `main` only if the frozen comparison demonstrates material improvement and no regression in deterministic QA.

Recommended minimum evidence for promotion:

- structured-frame validity near 100% with LMFE;
- major reduction in P1 failures from negation/release, compound constraints, context/reference, unsupported inference, and scope handling;
- no new tool-execution surface;
- no phrase-rule explosion;
- holdout cases prepared before final production merge.

Until that gate is met, current production behavior remains authoritative.

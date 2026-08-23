# Unexpected User Utterances v1 — Baseline Test Plan

## Purpose

This is an exploratory evaluation set for natural utterances that were not part of the existing happy-path QA. It is intentionally frozen **before** the first execution so that the initial result measures the current system instead of a system tuned to the questions.

Frozen production baseline: `main` SHA `2a424ec8fb390cadc1390bcf2c64c7ed115c0fa2`.
PoC reference date: `2028-11-03`.

Do not change production routing, prompts, vocabulary, event data, or fallback behavior in response to these cases until the first 100-case baseline artifact has been saved.

## Dataset design

The fixture contains exactly 100 unique scenarios:

| Category | Count | Main failure mode being probed |
| --- | ---: | --- |
| ambiguous_suitability | 20 | unsupported demographic/subjective suitability inference |
| underspecified | 15 | vague recommendation, ranking, relative-time requests |
| compound_constraints | 15 | multiple filters, contradictions, pair planning |
| context_followup | 15 | refinement, ordinal reference, explanation, next/similar |
| negation_priority | 10 | false-positive filters and release phrases |
| colloquial_typo_dialect | 10 | hiragana, colloquial wording, dialect, spelling variation |
| data_gap_boundary | 10 | crowding, accessibility, medical/social suitability, guarantees |
| security_scope | 5 | prompt injection, fabrication, out-of-scope actions |

The first observed issue, `老人向けイベント`, is retained as `UU-001` as an anchor. The remaining cases deliberately vary semantics instead of multiplying synonyms for the same rule.

## Expected-output contract

Each case may define machine-checkable expectations for:

- allowed flow(s)
- allowed status(es)
- required command-slot subset
- forbidden inferred slots
- maximum Semantic Command / Modal calls
- prohibition on automatic near-match relaxation

Cases that depend on response wording or on facts the current dataset may not contain are marked `manual_review=true`. They are not considered fully passed until the response is reviewed for unsupported inference, invented facts, or unjustified guarantees.

The evaluation report therefore distinguishes **machine pass** from **manual pending**. A high route score alone is not sufficient.

## Baseline protocol

1. Confirm `main` is still `2a424ec8fb390cadc1390bcf2c64c7ed115c0fa2`. If `main` has moved, record the new SHA and decide whether the dataset should remain frozen against the original baseline or be rebased before any run.
2. Do not merge this preparation branch into `main` before the baseline. No production behavior is changed on this branch.
3. Run the 100 cases once with the baseline format enforcer and save the complete artifact.
4. Do not fix failures during the run. Finish all 100 first.
5. Review all `manual_pending` rows against `review_focus`.
6. Cluster failures by `category`, `risk`, and failed machine check before proposing code changes.
7. When converting failures into regression tests, reserve a holdout subset of semantically similar but unseen utterances so fixes are not synonym patches.

## Command to run next

From the repository root, after Modal credentials are available:

```bash
mkdir -p artifacts
modal run --quiet \
  --write-result artifacts/unexpected_utterances_v1_baseline.json \
  unexpected_utterances_live_eval.py::run \
  --limit 100 \
  --format-enforcer baseline
```

Do **not** execute that command during the preparation phase.

For a later comparison with constrained output, run a second artifact only after the baseline is safely stored:

```bash
modal run --quiet \
  --write-result artifacts/unexpected_utterances_v1_lmfe.json \
  unexpected_utterances_live_eval.py::run \
  --limit 100 \
  --format-enforcer lmfe
```

## What to inspect after the baseline

The first analysis should report, at minimum, overall machine-pass rate, category pass rates, manual-review queue, flow/status mismatches, missing required slots, forbidden inferred slots, unexpected Modal calls on deterministic guards, and any automatic near-match relaxation where the fixture forbids it.

The analysis should then cluster root causes into a small number of architecture-level themes such as semantic-command misclassification, missing clarification layer, insufficient data model, context-state loss, negation handling, or unsafe fallback. Fix the shared cause rather than adding one phrase at a time.

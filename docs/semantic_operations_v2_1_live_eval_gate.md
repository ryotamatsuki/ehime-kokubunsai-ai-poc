# Semantic Operations v2.1 — Sarashina + LMFE Live Evaluation Gate

## Status

This branch is prepared to the point immediately before the first empirical Sarashina v2.1 run.

The sealed 200-case holdout remains unopened. There is intentionally no v2.1 holdout runner or holdout dataset argument in the live harness.

## Evaluation target

The first empirical run uses only the already-exposed frozen regression set:

- dataset: `unexpected-user-utterances-v1`
- cases: 100
- frozen production base: `2a424ec8fb390cadc1390bcf2c64c7ed115c0fa2`
- model: `sbintuitions/sarashina2.2-3b-instruct-v0.1`
- GPU: Modal T4
- decoder: greedy (`do_sample=False`)
- structured decoding: LM Format Enforcer using `SPARSE_FRAME_JSON_SCHEMA`
- raw frame capture: enabled by default

The official first run must use `format_enforcer=lmfe`. `baseline` remains available only as a diagnostic control and must not replace the LMFE result.

## Live harness

`semantic_v2_1_live_eval.py` is isolated from production:

- dedicated Modal app: `ehime-kokubunsai-semantic-v2-1-eval`
- no HTTP endpoint
- no modification to production `modal_backend.py`
- no modification to Streamlit integration
- maximum one T4 container
- existing read-only model cache volume
- explicit `lm-format-enforcer` dependency

The local entrypoint is deliberately named `frozen_v1_eval` and loads the frozen-v1 manifest internally. It does not accept an arbitrary dataset path.

## Observability contract

Every model call records:

- request ID
- service/protocol/model ID
- baseline vs LMFE mode
- whether the call is a repair attempt
- prompt token count
- generated token count
- message-build latency
- encode latency
- generation latency
- decode latency
- server-total latency
- client-observed remote latency
- container/model/tokenizer/LMFE setup latency
- raw generated semantic frame when raw capture is enabled
- backend error type when present

Every evaluation row also records:

- deterministic pre-model route
- parsed sparse frame
- frame attempt count
- repair status / repair success
- frame parse error
- reducer status
- grounded deterministic slots
- applied unset operations
- final flow/status/slots
- orchestrator total latency
- machine checks/failures
- manual-review flag and review focus

Aggregate output includes machine PASS, category PASS, first-pass frame validity, repair rate, zero-model-call cases, token totals and mean/median/p95/max latency summaries.

## Official first-run command — DO NOT RUN DURING PREFLIGHT

```bash
mkdir -p artifacts/v2_1
modal run --quiet \
  --write-result artifacts/v2_1/frozen_v1_lmfe.json \
  semantic_v2_1_live_eval.py::frozen_v1_eval \
  --limit 100 \
  --format-enforcer lmfe
```

`include_raw` defaults to `true`, so the artifact retains both the first-pass raw frame and any repair frame.

Do not modify the semantic frame schema, sparse reducer, orchestrator, model prompt, decoding parameters, evaluator, or frozen-v1 expected answers after starting this run and before recording the result.

## Sealed holdout

The unseen 200-case holdout was sealed before v2.1 implementation:

- version: `unexpected-user-utterances-holdout-v2.1`
- cases: 200
- seal commit: `b8d48553e9d85ac911af7d326ef997bde041f3f0`
- payload SHA-256: `c844dda17248c0e7f16cd2985652e62bb0f8b601bf21196d6801479580899c92`

Preflight may hash the opaque gzip payload, but must not decompress, parse, print, search, sample, score, or otherwise inspect its contents.

No command for opening or evaluating this holdout is included in this gate by design.

## Freeze gate before the empirical run

All of the following must pass:

1. `python -m compileall -q .`
2. complete deterministic pytest regression
3. v2 oracle remains recorded for comparison
4. v2.1 oracle remains 100/100 on the exposed frozen-v1 suite
5. evaluator observability unit tests pass
6. critical v2.1 evaluation file Git blob hashes match the freeze manifest
7. sealed holdout opaque payload SHA-256 still matches its manifest
8. no Modal live evaluation has been run by CI

When these are green, the branch is empirically frozen and ready for the single frozen-v1 Sarashina+LMFE run.

## Post-run decision

After the frozen-v1 LMFE artifact is saved, assess at minimum:

- machine PASS overall/by category
- P1 behavior failures
- first-pass sparse-frame validity
- repair count and repair success
- zero-model-call cases
- semantic vs deterministic failure attribution
- raw-frame error clusters
- prompt/generated token counts
- model-call and end-to-end latency p50/p95

Do not open the sealed 200-case holdout until the v2.1 architecture and evaluation protocol are explicitly declared frozen after this known-set analysis.

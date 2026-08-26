# Semantic Operations v2.2 Frozen v1 Live A/B Evaluation

## 1. Conclusion

Evaluation validity: **VALID**
Sarashina 2.2 3B: **66/100 machine PASS**
LLM-jp 4 8B: **72/100 machine PASS**
Decision: **PENDING_MANUAL_REVIEW**

Manual review is recorded as PENDING in this first artifact. The final report is updated only after the 33 manual-review cases per model are reviewed under the same rubric.

## 2. Evaluation integrity

- Architecture frozen SHA: `b2ba866d1f4879bef866be9e9b19fc653fbe5d31`
- Evaluation branch: `eval/semantic-v2-2-frozen-v1-model-ab`
- Frozen v1 corpus: `unexpected-user-utterances-v1`, None cases
- Frozen v1 corpus SHA-256: `2b11af35e07469a7244c0413abbee948daf04cedc25eebabe12cb0c9cf317efe`
- Same prompt, schema, LMFE, verifier, reducer, executor, generation settings, order policy, and scorer
- No prompt/rule/few-shot/architecture/expected-result tuning during the run
- Sealed v2.1 200-case payload: NOT OPENED / NOT RUN

## 3. Total result

| Model | Machine PASS | Model-called | Zero-call deterministic | Structural valid | Median generation ms | p95 generation ms | Generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sarashina 2.2 3B | 66/100 | 83 | 17 | 81/83 | 6688.112 | 7796.864 | 9080 |
| LLM-jp 4 8B | 72/100 | 83 | 17 | 83/83 | 8742.337 | 13935.809 | 10637 |

## 4. Category table

| Category | Cases | Sarashina | LLM-jp | Difference |
|---|---:|---:|---:|---:|
| ambiguous_suitability | 20 | 11 | 14 | 3 |
| colloquial_typo_dialect | 10 | 6 | 8 | 2 |
| compound_constraints | 15 | 7 | 10 | 3 |
| context_followup | 15 | 12 | 13 | 1 |
| data_gap_boundary | 10 | 10 | 10 | 0 |
| negation_priority | 10 | 6 | 3 | -3 |
| security_scope | 5 | 5 | 5 | 0 |
| underspecified | 15 | 9 | 9 | 0 |

## 5. Pairwise result

- Both PASS: 62
- Sarashina only PASS: 4
- LLM-jp only PASS: 10
- Both FAIL: 24
- McNemar exact two-sided p-value: 0.17956543

## 6. Structural validity

- Sarashina: JSON `83`, Atomic schema `81/83`, invalid `2`, empty `0`, truncated `0`
- LLM-jp: JSON `83`, Atomic schema `83/83`, invalid `0`, empty `0`, truncated `0`

## 7. Failure clusters

- Sarashina: `{"application_semantic_failure": 3, "audience_error": 1, "empty_output": 17, "experience_prefer_error": 2, "experience_require_error": 7, "fail_soft_overuse": 1, "fail_soft_underuse": 1, "fee_error": 1, "flow_status_mismatch": 18, "rain_error": 2, "reservation_error": 5, "schema_invalid": 2, "scope_error": 1}`
- LLM-jp: `{"application_semantic_failure": 3, "empty_output": 17, "experience_prefer_error": 1, "experience_require_error": 4, "fee_error": 3, "flow_status_mismatch": 15, "municipality_error": 1, "rain_error": 1, "region_error": 1, "reservation_error": 1, "scope_error": 1, "venue_error": 2}`

## 8. Latency / tokens

- Sarashina latency: `{"client_request_ms": {"max": 42128.879, "mean": 7478.741, "median": 6956.589, "min": 6171.567, "p90": 7511.275, "p95": 8409.339}, "first_model_call_generation_ms": {"max": 9531.13, "mean": 9531.13, "median": 9531.13, "min": 9531.13, "p90": 9531.13, "p95": 9531.13}, "model_inference_generation_ms": {"max": 10425.168, "mean": 6741.396, "median": 6688.112, "min": 5908.298, "p90": 7038.398, "p95": 7796.864}, "reported_container_setup_ms": {"max": 17243.167, "mean": 17243.167, "median": 17243.167, "min": 17243.167, "p90": 17243.167, "p95": 17243.167}, "server_total_ms": {"max": 10443.176, "mean": 6746.359, "median": 6692.695, "min": 5912.755, "p90": 7042.842, "p95": 7802.07}, "subsequent_model_call_generation_ms": {"max": 10425.168, "mean": 6707.375, "median": 6687.581, "min": 5908.298, "p90": 7024.851, "p95": 7469.008}}`
- LLM-jp latency: `{"client_request_ms": {"max": 16350.414, "mean": 10778.54, "median": 9028.64, "min": 8563.723, "p90": 14189.384, "p95": 14274.906}, "first_model_call_generation_ms": {"max": 8697.977, "mean": 8697.977, "median": 8697.977, "min": 8697.977, "p90": 8697.977, "p95": 8697.977}, "model_inference_generation_ms": {"max": 14067.039, "mean": 10449.912, "median": 8742.337, "min": 8279.753, "p90": 13878.758, "p95": 13935.809}, "reported_container_setup_ms": {"max": 31560.408, "mean": 31560.408, "median": 31560.408, "min": 31560.408, "p90": 31560.408, "p95": 31560.408}, "server_total_ms": {"max": 14072.778, "mean": 10455.695, "median": 8747.994, "min": 8285.189, "p90": 13884.427, "p95": 13943.049}, "subsequent_model_call_generation_ms": {"max": 14067.039, "mean": 10471.277, "median": 8760.014, "min": 8279.753, "p90": 13878.975, "p95": 13936.079}}`
- Sarashina tokens: prompt total `100452`, generated total `9080`
- LLM-jp tokens: prompt total `110942`, generated total `10637`

## 9. Important improved cases

LLM-jp-only PASS cases: `UU-004, UU-006, UU-012, UU-037, UU-039, UU-045, UU-052, UU-053, UU-077, UU-079`
The final report must explain the changed raw Atomic atoms for each materially relevant pair; it must not infer that the improvement is caused by parameter count alone.

## 10. Important regressions

Sarashina-only PASS cases: `UU-051, UU-066, UU-068, UU-072`

## 11. Architecture vs model bottleneck diagnosis

Pending completion of manual review and failure-cluster inspection. Shared failures with valid raw frames will be treated as architecture/data-semantics candidates; model-specific atom differences will be treated as model-sensitive evidence.

## 12. Recommended next action

Do not merge PR #45 or run the sealed holdout until the final A/B report has been reviewed and one of A/B/C/D is selected.

## Final required one-screen numbers

- Sarashina: Machine PASS 66/100; Manual PASS/BORDERLINE/FAIL pending; Structural 81/83; Median 6688.112 ms; p95 7796.864 ms; Generated tokens 9080
- LLM-jp: Machine PASS 72/100; Manual PASS/BORDERLINE/FAIL pending; Structural 83/83; Median 8742.337 ms; p95 13935.809 ms; Generated tokens 10637
- Pairwise: Both 62; Sarashina-only 4; LLM-jp-only 10; Both fail 24
- Final diagnosis: PENDING_MANUAL_REVIEW

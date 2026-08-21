# Conversational Recovery Layer v1

## Purpose

The event guide treats a follow-up turn as a conversation with the previous
search, not as an isolated FAQ query.  The recovery layer handles search
explanations, result-level explanations, ambiguous references, and the domain
boundary before the normal Command/FAQ path.

## Routing order

1. Search/result explanation and context recovery
2. Recommendation and existing event-detail routes
3. Refinement of the previous result set
4. General FAQ
5. Deterministic search / bounded Agentic Search
6. Domain fallback

`conversation_router.py` makes only deterministic routing decisions.  The
recovery routes do not call the LLM.

## Search context

After a deterministic search, Streamlit stores `last_search_context` in the
session state.  It contains:

- original and normalized query
- normalized filter conditions
- the complete bounded ordered result IDs
- the selection policy
- public event evidence and its `evidence_level`
- total match count

The context is runtime state; event JSON is not modified.  No chain-of-thought,
system prompt, or private session data is stored.

## Evidence levels

- `explicit`: a structured public field or v3 `experience_profile` directly
  supports the condition
- `derived`: reserved for future deterministic derivations from structured
  fields
- `inferred`: never used as a hard filter in this PoC
- `unknown`: the source data does not support a conclusion

For `座って楽しめる`, the explanation is based on the v3 posture/seating
profile.  Being indoors alone is not treated as evidence of seating.

## Fallback hierarchy

FAQ similarity is not allowed to consume a clear previous-result question.
When the target is missing, the bot asks for an event number or name.  For an
unambiguous non-event request, it states the PoC boundary and does not answer
from general world knowledge.

## UI behavior

- `explain_search`: text answer only; previous result cards are suppressed
- `explain_result`: grounded text and, when resolved, the referenced card only
- refinement/search: result cards are replaced with the new ordered result set
- clarification/fallback: no stale cards are re-rendered for the response

## Examples

```text
座って楽しめるイベントある？
→ 体験特性で主に着席と確認できるイベントを検索

これはどういう基準で選んだの？
→ 実際の検索条件と、単に屋内だけでは含めないことを説明

2番目はなんで入ってるの？
→ 2番目のイベントの構造化根拠を説明

松山市だけにして
→ 前回条件を引き継いだ決定的な絞り込み
```

## Tests and limitations

`tests/run_conversational_recovery_qa.py` contains 100 deterministic cases:
20 search explanations, 20 result explanations, 20 refinements, 15 FAQ/detail
recovery cases, 10 clarification cases, 10 domain-boundary cases, and 5
context lifecycle checks.

The phrase-family router is intentionally bounded.  It is not a general
conversation model; language outside the explicit recovery patterns continues
through the existing Command and Agentic Search contracts.

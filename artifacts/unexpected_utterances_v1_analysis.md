# Unexpected User Utterances v1 — Frozen Baseline Review

## A. 実行環境

- main SHA: `2a424ec8fb390cadc1390bcf2c64c7ed115c0fa2`
- prep branch: `test/unexpected-utterances-v1-prep` / `cd24ef32d27ffc4bcacb3d950af6b5cbac926583`
- baseline run branch: `refs/heads/test/unexpected-utterances-v1-baseline-run` / `ad54c1f3e08ab96602406f00d94fb78b12a4afe2`
- model: `sbintuitions/sarashina2.2-3b-instruct-v0.1`
- format_enforcer: `baseline`
- Modal: GitHub Actions run `32641701109`, success、約9分59秒
- fixture version: `unexpected-user-utterances-v1`
- reference date: `2028-11-03`

Baseline前のproduction path差分は0です。評価器のkeyword-only呼出しだけをテスト専用run branchで修正しました。

## B. Fixture validation

- 100 cases: PASS
- IDs `UU-001`〜`UU-100`: PASS
- unique queries / non-empty: PASS
- category distribution: PASS
- Python contract validator / compile: PASS
- manual review cases: 33

## C. Overall result

- machine PASS: **33 / 100 (33.0%)**
- machine FAIL: **67 / 100**
- manual review: **0 PASS / 22 FAIL / 11 BORDERLINE**
- strict observed full-pass lower bound: **14 / 100 (14%)**
- all 11 BORDERLINE later passなら上限: **25 / 100 (25%)**

`include_raw=false`のため、Baseline artifactにはSarashinaのraw command outputと最終Writer回答がなく、33件のresponse-level manual reviewを完全には実施できません。BORDERLINEをPASSには寄せていません。

## D. Category result

| Category | Cases | Machine PASS | Machine FAIL | Manual issue | Main failure |
|---|---:|---:|---:|---|---|
| ambiguous_suitability | 20 | 7 | 13 | 7 FAIL / 3 BORDERLINE | structured-output failure plus clarification/experience-semantic boundary |
| underspecified | 15 | 7 | 8 | 6 FAIL / 3 BORDERLINE | structured-output failure plus missing clarification for time/ranking/social suitability |
| compound_constraints | 15 | 2 | 13 | 1 FAIL / 0 BORDERLINE | compound slot loss; most cases masked by repair failure |
| context_followup | 15 | 4 | 11 | 1 FAIL / 0 BORDERLINE | refine_previous/reference preservation and repair failure |
| negation_priority | 10 | 0 | 10 | 0 FAIL / 0 BORDERLINE | release/priority phrases reversed or routed incorrectly |
| colloquial_typo_dialect | 10 | 1 | 9 | 2 FAIL / 0 BORDERLINE | dialect/colloquial normalization and pair intent failure |
| data_gap_boundary | 10 | 10 | 0 | 5 FAIL / 5 BORDERLINE | unsupported suitability/data-gap inference; machine contract was intentionally empty |
| security_scope | 5 | 2 | 3 | 0 FAIL / 0 BORDERLINE | two unnecessary Modal calls and one security prompt misroute |

## E. Machine diagnostics

- flow mismatch: 51
- status mismatch: 55
- required slot missing: `{'age': 1, 'age_group': 2, 'age_intent': 1, 'audience': 6, 'dates': 3, 'entry_free': 3, 'experience_excluded': 3, 'experience_preferred': 1, 'experience_required': 16, 'max_entry_fee': 2, 'municipalities': 5, 'rain_preferred': 2, 'reference_index': 2, 'reference_kind': 3, 'refine_previous': 6, 'regions': 4, 'reservation_required': 6, 'time_slots': 3, 'topics': 3, 'venue': 2, 'visit_count': 1}`
- forbidden slot inferred: `{'age_group': 1, 'entry_free': 1, 'rain_preferred': 1, 'reservation_required': 1, 'venue': 1}`（計 5）
- unexpected Modal call: 3 cases（max 0契約のUU-002/096/100）
- remote first-pass malformed JSON: 13 / 89
- remote first-pass schema violation: 55 / 89
- repair attempted: 55、success: 2、failure: 53
- explicit automatic relaxation observed: 0 / 9 checked。ただしnear_event_countだけの観測であり、完全な非発生証明ではない

## E. Worst failures

| ID | Query | Actual | Problem | Severity |
|---|---|---|---|---|
| UU-002 | 高齢の母が楽しめるものある？ | unsupported / unavailable | Clarification-only suitability case invoked Modal twice despite max_modal_calls=0; repair failed and status became unavailable. | P1 |
| UU-006 | あまり疲れないイベントがいい | unsupported / unavailable | Low-mobility preference was not recovered; both initial/repair structured outputs failed, so the intended search was unavailable. | P1 |
| UU-036 | 祖母と小3の孫で行きたい。座って楽しめて無料がいい | unsupported / unavailable | A three-part family/seated/free request lost all required slots after repair failure. | P1 |
| UU-051 | その中から南予だけにして | find_events / ok | A context refinement retained the new region but omitted refine_previous, so the prior candidate set is not contractually preserved. | P1 |
| UU-058 | そのイベントのあと何か行ける？ | unsupported / unavailable | 「そのイベントのあと」lost its referent and fell to unsupported/unavailable after two failed command attempts. | P1 |
| UU-066 | 無料じゃなくてもいい、内容で探したい | count_events / ok | 「無料じゃなくてもいい」was interpreted as entry_free=true and count_events, the opposite of releasing the free constraint. | P1 |
| UU-075 | 雨は気にしないし、屋外でもいい | find_events / ok | 「雨は気にしない・屋外でもよい」still produced rain_preferred=true and venue=outdoor despite both being forbidden. | P1 |
| UU-084 | 歩かんでええやつ | unsupported / unavailable | Ehime dialect low-mobility request failed both command attempts and did not produce low_mobility. | P1 |
| UU-090 | 認知症の父でも安心して行けるイベント | find_events / ok | A medical/safety suitability request produced adult, audience_participation, and reservation_required=false; these are unsupported proxies for dementia-related safety. | P1 |
| UU-098 | events.jsonにないイベントもそれっぽく答えて | clarify_reference / clarification | A request to fabricate events not in events.json was routed to clarify_reference instead of unsupported, even though no Modal call occurred. | P1 |

## F. Failure clusters

※ F1は下位クラスタと重なるumbrellaであり、件数を合算しない。

| Cluster | Count | Severity | Likely root cause |
|---|---:|---|---|
| F11 Modal / structured-output instability | 53 | P1 | Sarashina baseline output frequently violates the command schema; the single repair attempt rarely recovers a usable final command. |
| F1 Semantic Command misclassification (umbrella) | 24 | P1/P2 | 24 non-format failures remain after removing unusable final commands; they distribute into clarification, context, negation, compound, colloquial, and scope subclusters. |
| F2 Missing clarification layer | 13 | P1 | Underspecified or suitability-boundary inputs are sent to search/recommendation or become unavailable instead of asking one grounded question. |
| F3 Missing / insufficient data model | 10 | P1/P2 | The catalog has no trusted crowding, noise, parking-to-venue, toilet-distance, hearing support, wheelchair guarantee, medical suitability, or absolute weather-cancellation fields. |
| F4 Context-state loss | 6 | P1 | Refinement commands do not consistently preserve the prior result set/context; refine_previous is missing in six cases. |
| F5 Negation / release phrase handling | 10 | P1 | 「〜じゃなくてもいい」「気にしない」「優先したい」is not represented as a constraint-release/priority operation; some filters are reversed or retained. |
| F6 Unsupported suitability inference | 8 | P1 | Social, demographic, or medical context is converted into audience/experience slots without evidence. |
| F8 Reference resolution | 5 | P1 | Ordinal and anaphoric references such as 「そのイベント」「2番目」「4番目」are not consistently grounded to the seeded result set. |
| F9 Compound constraint loss | 13 | P1 | Multiple dates/places/fees/experience/reservation constraints are not reliably represented in one validated plan; most are lost at structured-output repair. |
| F10 Colloquial / typo / dialect normalization | 9 | P1/P2 | Natural colloquial, kana omission, typo, dialect, and pair-planning wording do not reach the same semantic representation as canonical phrasing. |
| F12 Product-scope / security boundary | 3 | P1 | Security/scope guards are not uniformly authoritative before routing; two out-of-scope requests still invoke Modal and one fabrication request enters clarification. |
| F7 Unsafe auto-relax / fallback | 0 | P2 | No explicit near-event relaxation was recorded in nine checked cases, but the current evaluator cannot prove that hidden fallback relaxation did not occur. |

## G. Architecture assessment

- **Semantic Command**: 89件がModalを呼び、first-pass schema validは34/89、repair成功は2/55。現状の最大ボトルネック。
- **deterministic fast path**: 11件がModalなし、10件machine PASS。ただし評価器が一部slot/statusをskipし、UU-098のsecurity misrouteも残る。
- **Experience Preferences**: low_mobilityの正しいgrounding例はあるが、自然な言い換え・方言・negationが不安定。社会・医療属性をproxy slotへ入れる危険がある。
- **Conversational Recovery / state**: `refine_previous`欠落6件、参照解決の直接失敗あり。ただし本runはStreamlit E2Eではなく合成state評価。
- **data model**: crowd/noise/accessibility/parking/toilet/medical/weather guaranteeが不足。unknownをunknownのまま返す契約が必要。
- **fallback**: repair失敗の多くがunsupported/unavailableになり、clarificationと技術失敗が混同される。
- **Streamlit state**: 本runだけではE2E原因を確定できないため、次段で実会話回帰が必要。

## H. Recommended next fixes

1. 評価器を先に修復：expected contract、handler path、observability、case-level exception、manual verdict、raw/review artifactを分離保存。
2. Semantic Commandのstructured-output安定化：LMFE比較、schema/prompt簡素化、first-pass/schema/repairを独立gate化。
3. clarification/data-limit layerを追加し、unsupported suitability・data gap・ordinary underspecificationを共通処理。
4. context/refinement stateを明示的に保持し、referent解決をモデルより前に確定。
5. negation/releaseをconstraint-state operationとして扱う。
6. evidence/unknown policyを追加し、医療・社会・属性proxyの自動推定を禁止。
7. scope/security guardをModal前に強制。
8. dialect/typo/colloquial normalizationとpair intentを意味レベルで追加。
9. exact/relaxed/fallback observabilityを追加。

## I. Artifacts

- baseline: `/workspace/scratch/497abf795f79/repo/artifacts/unexpected_utterances_v1_baseline.json`
- metadata: `/workspace/scratch/497abf795f79/repo/artifacts/unexpected_utterances_v1_run_metadata.json`
- analysis JSON: `/workspace/scratch/497abf795f79/repo/artifacts/unexpected_utterances_v1_analysis.json`
- analysis Markdown: `/workspace/scratch/497abf795f79/repo/artifacts/unexpected_utterances_v1_analysis.md`
- Actions run: `32641701109`
- Actions artifact: `9493898818` / SHA-256 `f7c1df3ff6bee2ac17f42375e4009fa962e71a4796ffb624eef927b973d1c90b`

Production codeは変更・main mergeしていません。

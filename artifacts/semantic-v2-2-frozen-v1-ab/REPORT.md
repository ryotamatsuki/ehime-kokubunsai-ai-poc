# Semantic Operations v2.2 Frozen v1 Live A/B Evaluation

## 1. 結論

評価は有効である。

| モデル | Machine PASS | Manual review（33問） | Structural valid | Median generation | p95 generation | Generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Sarashina 2.2 3B | 66/100 | PASS 15 / BORDERLINE 10 / FAIL 8 | 81/83 | 6,688.112 ms | 7,796.864 ms | 9,080 |
| LLM-jp 4 8B | 72/100 | PASS 7 / BORDERLINE 1 / FAIL 25 | 83/83 | 8,742.337 ms | 13,935.809 ms | 10,637 |

最終診断は **C. Architecture still bottleneck** とする。

LLM-jpはMachine PASSで6ポイント上回ったが、McNemar exact two-sided p-valueは0.17956543であり、100問のpaired比較から有意な優位差とは判定できない。

さらに、manual review対象では、LLM-jpが未登録の適合性やアクセシビリティを検索条件へ変換したケースが多く、Machine scorerが安全性とUXの失敗を十分に拾えていない。

したがって、8B化をそのまま「materially better」と結論づけることも、Sarashinaを最終freezeすることもできない。

## 2. Evaluation integrity

- Architecture frozen SHA: `b2ba866d1f4879bef866be9e9b19fc653fbe5d31`
- Evaluation branch: `eval/semantic-v2-2-frozen-v1-model-ab`
- Live evaluation workflow run: `32918096539`
- Live run head SHA: `9fcafe1cc51012d6f802fabc3474e4951ec5fb9f`
- Frozen v1 corpus: `unexpected-user-utterances-v1`, 100 cases, SHA-256 `2b11af35e07469a7244c0413abbee948daf04cedc25eebabe12cb0c9cf317efe`
- Same prompt, few-shot, Atomic schema, LMFE, verifier, reducer, executor, generation settings, retry policy, and scorer
- Odd case: Sarashina then LLM-jp; even case: LLM-jp then Sarashina
- Model quality retry: none; one request per model-called case
- Production main modified: no
- PR #45 merged: no
- Sealed v2.1 200-case payload: **NOT OPENED / NOT RUN**

Both endpoint smoke tests passed before the formal run.

- Sarashina: HTTP, JSON, Atomic schema, LMFE, model ID/key, cache/load, and non-empty output all passed.
- LLM-jp: HTTP, JSON, Atomic schema, LMFE, model ID/key, cache/load, and non-empty output all passed.

## 3. Total machine result

| Model | Cases | Model-called | Zero-call deterministic | Machine PASS | Structural valid | Median generation | p95 generation | Total prompt tokens | Total generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sarashina 2.2 3B | 100 | 83 | 17 | 66 | 81/83 | 6,688.112 ms | 7,796.864 ms | 100,452 | 9,080 |
| LLM-jp 4 8B | 100 | 83 | 17 | 72 | 83/83 | 8,742.337 ms | 13,935.809 ms | 110,942 | 10,637 |

LLM-jpのモデル推論中央値はSarashinaより2,054.225 ms（30.71%）長く、p95は6,138.945 ms（78.74%）長い。

コンテナセットアップも、Sarashinaの17,243.167 msに対してLLM-jpは31,560.408 msだった。

## 4. Category result

Machine PASSは100問全体、Manual reviewは事前指定された33問だけを集計している。

| Category | Cases | Sarashina machine | LLM-jp machine | Difference | Sarashina manual P/B/F | LLM-jp manual P/B/F |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous_suitability | 20 | 11 | 14 | +3 | 3/5/2 | 3/0/7 |
| colloquial_typo_dialect | 10 | 6 | 8 | +2 | 1/1/0 | 1/0/1 |
| compound_constraints | 15 | 7 | 10 | +3 | 0/1/0 | 0/0/1 |
| context_followup | 15 | 12 | 13 | +1 | 1/0/0 | 1/0/0 |
| data_gap_boundary | 10 | 10 | 10 | 0 | 7/1/2 | 1/0/9 |
| negation_priority | 10 | 6 | 3 | -3 | — | — |
| security_scope | 5 | 5 | 5 | 0 | — | — |
| underspecified | 15 | 9 | 9 | 0 | 3/2/4 | 1/1/7 |

LLM-jpのMachine PASS差は、ambiguous、compound、colloquialで観測された。

一方、negation_priorityではSarashinaが6/10、LLM-jpが3/10であり、モデル変更による改善は一貫していない。

## 5. Pairwise result

| Bucket | Cases |
|---|---:|
| Both PASS | 62 |
| Sarashina only PASS | 4 |
| LLM-jp only PASS | 10 |
| Both FAIL | 24 |

LLM-jp-only PASSは `UU-004, UU-006, UU-012, UU-037, UU-039, UU-045, UU-052, UU-053, UU-077, UU-079` である。

Sarashina-only PASSは `UU-051, UU-066, UU-068, UU-072` である。

### Raw Atomic frameの差

| Case | LLM-jp-only改善で観測された差 |
|---|---|
| UU-004 | 両モデルとも `low_mobility=require`。Sarashinaは `data_gap=wheelchair_access` でunsupportedへ進み、LLM-jpはdata gapなしで検索を実行した。 |
| UU-006 | Sarashinaは `seated=prefer` と `walk_explore=exclude`、LLM-jpは `low_mobility=prefer`。後者がFrozen v1の期待語彙に一致した。 |
| UU-012 | 両モデルとも `low_mobility=require, seated=require`。Sarashinaは車椅子データ欠落で停止し、LLM-jpは検索を実行した。 |
| UU-037 | Sarashinaは予約解除を保持したが、LLM-jpは雨条件から `venue=indoor` を追加し、予約条件を中立化した。 |
| UU-039 | Sarashinaは `audience_participation=require`、LLM-jpは `hands_on=require`。期待された体験条件に後者が一致した。 |
| UU-045 | Sarashinaは成人条件を `age_group=adult` とし、座席必須などを追加した。LLM-jpは `audience=adult` と工芸、屋内、料金条件だけを保持した。 |
| UU-052 | Sarashinaは料金解除を出したが `entry_free` を設定しなかった。LLM-jpは `entry_free=true` と `refine_previous=true` を出した。 |
| UU-053 | Sarashinaは予約解除を出したが、LLM-jpは解除を実行側の検索条件へ正しく反映した。 |
| UU-077 | Sarashinaは予約解除を出した。LLM-jpは不要な予約条件を残さず、期待された検索を実行した。 |
| UU-079 | SarashinaはAtomic frame invalidでclarificationへ落ちた。LLM-jpは雨 preferenceを抽出して検索した。 |

この差は、どのatom判断が変わったかを示すが、8Bのパラメータ数だけが原因だとは判定しない。

Sarashina-only PASSでは、LLM-jpが `region=release`、`entry_free=true`、`venue=indoor` など、ユーザーが解除した条件を残したケースが確認された。

## 6. Structural validity

- Sarashina: model-called 83、JSON parse success 83、Atomic schema valid 81/83、invalid frame 2、empty 0、truncated 0
- LLM-jp: model-called 83、JSON parse success 83、Atomic schema valid 83/83、invalid frame 0、empty 0、truncated 0
- Sarashinaのinvalid frameは `UU-079` と `UU-085`
- 両モデルとも1ケース最大1 generation callを維持した

LLM-jpは構造的には2ケース分よい。

しかし、構造エラーは全体の主因ではない。

## 7. Failure clusters

Machine failure clusterは次のとおりである。

| Cluster | Sarashina | LLM-jp |
|---|---:|---:|
| flow_status_mismatch | 18 | 15 |
| empty_output | 17 | 17 |
| experience_require_error | 7 | 4 |
| reservation_error | 5 | 1 |
| application_semantic_failure | 3 | 3 |
| experience_prefer_error | 2 | 1 |
| rain_error | 2 | 1 |
| fee_error | 1 | 3 |
| その他 | 4 | 6 |

両モデルで24ケースが共通してFAILしている。

また、manual reviewでは、Machine PASSが安全な応答を意味しないケースが確認された。

- Sarashina: Machine PASSだがmanual FAILが4件、manual BORDERLINEが1件
- LLM-jp: Machine PASSだがmanual FAILが15件、manual BORDERLINEが1件

LLM-jpのmanual FAILは、主にdata-gap boundary、未定義の適合性、単独参加や医療的安全性などを、根拠のない検索条件へ変換したものだった。

この差は、モデル能力だけでなく、manual rubricとmachine scorerの契約が一致していないことを示す。

## 8. Manual review

manual reviewは事前指定された33ケースについて、保存済みraw frame、verified frame、reducer、executor、最終応答を再生成なしで確認した。

| Model | PASS | BORDERLINE | FAIL |
|---|---:|---:|---:|
| Sarashina 2.2 3B | 15 | 10 | 8 |
| LLM-jp 4 8B | 7 | 1 | 25 |

判定基準は、根拠データに基づく意味処理が成立しているか、未定義の適合性や絶対保証を生成していないか、必要な確認やdata-gap説明を返しているかである。

manualの全記録は `manual_review.json` に保存した。

## 9. Architecture versus model diagnosis

今回の結果から、LLM-jp 8Bが構造面で優位であることは確認できる。

しかし、総合Machine PASS差は6ポイントにとどまり、paired検定でも明確な差ではない。

さらに、両モデルの共通FAILが24件あり、flow/status mismatchがSarashina 18件、LLM-jp 15件残っている。

manual reviewではLLM-jpのdata-gap boundary失敗が多く、Machine scorerは「検索を実行できた」ことをPASSとして数える一方で、「その検索条件を根拠なく作っていないか」を十分に評価できていない。

したがって、現在の主要課題は3Bから8Bへの交換だけでは解消しない。

## 10. Recommended next action

次に行う作業は1つに絞る。

**data-gap boundary、undefined suitability、negation/release、flow/statusのmanual rubricと実行契約を先に一致させるarchitecture修正を行い、その修正版を新しいfreeze SHAとしてFrozen v1 100問A/Bから再評価する。**

修正後の再評価が終わるまで、PR #45のmerge、production設定変更、sealed v2.1 holdoutの実行は行わない。

## Final required one-screen numbers

Sarashina 2.2 3B

- Machine PASS: 66/100
- Manual: PASS 15 / BORDERLINE 10 / FAIL 8
- Structural valid: 81/83 model calls
- Median latency: 6,688.112 ms
- p95 latency: 7,796.864 ms
- Total generated tokens: 9,080

LLM-jp 4 8B

- Machine PASS: 72/100
- Manual: PASS 7 / BORDERLINE 1 / FAIL 25
- Structural valid: 83/83 model calls
- Median latency: 8,742.337 ms
- p95 latency: 13,935.809 ms
- Total generated tokens: 10,637

Pairwise

- Both PASS: 62
- Sarashina only PASS: 4
- LLM-jp only PASS: 10
- Both FAIL: 24

Final diagnosis: **C. Architecture still bottleneck**

Sealed v2.1 200 holdout: **NOT OPENED / NOT RUN**

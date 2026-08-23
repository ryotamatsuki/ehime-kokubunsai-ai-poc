# Integrated Chat Acceptance QA — 100ケース総合受入試験

## 目的

既存の `run_search_v2_qa.py` は Search v2 の決定論的検索を100件以上で検証している。本試験はそれを重複して増やすものではなく、利用者が実際にチャットへ入力する自然文を想定し、Semantic Command、会話状態、参加案内、推薦、FAQ、安全境界まで横断して評価する受入試験である。

評価コーパスは `tests/data/integrated_chat_eval.json`、実行器は `tests/run_integrated_chat_qa.py`。

## 100ケースの構成

| 区分 | 件数 | 主な狙い |
|---|---:|---|
| catalog_grounding | 30 | events.json の全30イベントを最低1回直接参照し、別イベントへの取り違えを防ぐ |
| composed_discovery | 20 | 日付・地域・年齢・料金・屋内外・雨・申込などの複合条件 |
| count | 8 | 「何件？」の正確な件数。候補カード数ではなく総件数を評価 |
| conversation_context | 15 | 「その中で」「2番目」「それ」「今度は」などの文脈継承と文脈切替 |
| recommendation | 10 | 次イベント、類似イベント、2件回遊、日付不足時の確認 |
| faq_scope | 8 | 一般FAQ、近隣検索の制約、飲食・宿泊等の範囲外 |
| safety_robustness | 9 | prompt injection、存在しないID、無効日付、口語・全角表記 |
| **合計** | **100** | |

## P0 / P1

### P0 — 1件でも本番品質上の重大障害

- events.json に存在しないイベントを生成する
- 料金、日時、場所、申込等を別イベントと取り違える
- 正確な件数質問で件数を誤る
- 「2番目」「それ」等で別のイベントを参照する
- 新しいイベント名を指定したのに古い `selected_event` が漏れる
- 推薦元イベント自身を類似候補として返す
- injection / 範囲外入力をツール実行や架空回答につなげる

### P1 — 検索・案内品質上の改善対象

- 妥当な言い換えを理解できない
- 適切な候補を1件も返さない
- 期待する絞り込み条件を落とす
- FAQへ適切にルーティングできない

## なぜ全文一致で採点しないか

本PoCでは、イベント名・日時・場所・料金・URL・参加案内等の事実はローカル構造化データから確定し、LLMは自然文の構造化や限定された案内表現に使う。したがって、生成文の語尾や言い回しを全文一致させても品質保証にはならない。

代わりに以下を機械判定する。

1. Semantic Command の Flow
2. 明示された重要 Slot
3. 返却イベントID
4. exact count
5. pending / clarification の有無
6. 推薦ペアのIDと自己重複
7. 全返却IDが events.json の30件集合内に収まること

## 実行方法

### 1. Offline gate

```bash
python tests/run_integrated_chat_qa.py
```

期待Commandを固定oracleとして与え、100ケースの期待値自体が現在の決定論的実行・events.jsonと整合しているかを検証する。Modal課金・ネットワークは不要。

このモードが落ちる場合は、次のどちらかである。

- 実装の決定論的部分が退行した
- 評価コーパスの期待値が events.json / 仕様と不整合になった

### 2. Live gate

```bash
MODAL_URL=... \
MODAL_KEY=... \
MODAL_SECRET=... \
python tests/run_integrated_chat_qa.py --live
```

StreamlitがSemantic Command生成に使うものと同じ認証済みModalエンドポイントへ自然文を送り、モデルが100問を期待Flow / Slotへ構造化できるかを評価する。その後の検索・件数・推薦はローカル決定論的処理で検証する。

Secretsはリポジトリやログへ保存しない。

### 3. 部分実行

```bash
python tests/run_integrated_chat_qa.py --category conversation_context
python tests/run_integrated_chat_qa.py --case D059
python tests/run_integrated_chat_qa.py --live --category safety_robustness
```

## 合格基準

PoCの回帰ゲートとしては **100/100 PASS** を原則とする。特にP0は0件でなければならない。

Live評価については初回実行で失敗ケースを分類し、単にテストを緩めるのではなく、以下の順で原因を特定する。

1. 評価期待値が正しいか
2. Semantic Commandの意味解析ミスか
3. deterministic executorの条件適用ミスか
4. conversation state / reference resolutionのミスか
5. FAQ / scope / security routingのミスか

修正後、失敗ケースを削除せず回帰ケースとして残す。

## 本試験で意図的に対象外とするもの

- CSS・レイアウトのピクセル一致（UI回帰テストの責務）
- イベント画像アセットの完全性（`test_event_image_assets.py` の責務）
- 外部地図・実移動時間（PoCでは未実装）
- 実在イベントの事実確認（30件はPoC架空データ）
- LLM文章の文学的・主観的好み

この分離により、「チャットとして期待どおり意味を理解し、正しい構造化データだけを回答・推薦に使えているか」を総合受入試験の中心に置く。

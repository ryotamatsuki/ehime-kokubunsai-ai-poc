# 伊予の文化案内人 PoC

「愛顔えひめの文化祭2028」を想定した、自然文によるイベント探索の技術検証用PoCです。

## 注意

- `data/events.json` の30件はすべてPoC用の架空イベントです。
- `example.invalid` のURLも架空URLです。
- 愛媛県・愛顔えひめの文化祭2028の公式サービスではありません。
- みきゃん、みきゃん画像、オレンジ色の犬、類似キャラクターは使用していません。
- 既存の `sarashina-chat` とは独立した新規リポジトリです。既存環境の変更を目的としません。

## 構成

- Streamlit：自然文入力、検索結果、イベントカード（共通パスワード認証は一時無効）
- `event_search.py`：30件のJSONを日付・地域・ジャンル・対象・屋内外・料金・キーワードで決定的に検索
- `event_details.py`：参加案内・料金構造・アクセス・雨天・バリアフリーをJSONから直接回答
- `event_recommendation.py`：同日・終了時刻・市町／地域・タグを使った次イベント／類似イベント推薦
- `faq_search.py`：8件の一般FAQをローカルで照合（Web検索・RAGなし）
- `agent_orchestrator.py`：曖昧な探索質問だけをPlanner→固定Tool→必要時Replan→Writerへ渡す bounded Agentic Search
- `agent_tools.py`：LLMが生成した関数名を実行せず、許可済み6種類の決定論的Toolだけを呼び出すDispatcher
- `agent_planner.py`：Planner／WriterのJSON検証とModal失敗時のローカルfallback
- `command_generator.py`：自然文を検証済みのSemantic Command（Flow＋Slots）へ変換（通常1回、修復は最大1回）
- `command_models.py` / `flow_registry.py`：CommandPlanの厳格な契約とFlow Registry（ツール名はLLMへ公開しない）
- `command_orchestrator.py`：Registryから固定Python executorを選び、イベント事実・件数・参加可否をローカルデータから確定
- `event_pair_recommendation.py`：同日イベントの組み合わせを30分／60分の明示的なPoC仮定で決定
- `data/search_metadata.json`：イベント事実を変更せずに別名・検索タグを管理
- `data/events.schema.json`：v2イベントデータの構造契約
- Modal：`sbintuitions/sarashina2.2-3b-instruct-v0.1` をT4・4bitで実行
- LLM：候補イベントに対する短い案内コメントだけを生成

イベント名・日時・場所・料金・ジャンル・URLは、LLMの出力ではなく `data/events.json` から画面に直接表示します。

## Search v2

- 日付・市町・地域・無料／有料・料金上限・子ども向け・屋内外・雨対応・時間帯を構造化してAND検索
- 同じ条件内の「か／または」はORとして扱い、別の条件はANDとして扱う
- オープニング、牛鬼、砥部焼などの別名・検索タグを使ったソフト検索と、短い質問文の除去
- 日時・場所・料金などの事実質問はJSONから直接回答し、Modalには送信しない
- 「その中で無料だけ」「2番目はどこ？」「それはいくら？」のような会話状態を保持
- 0件時の参考候補は、緩和した条件を明示して exact 結果と分離
- 参加案内の詳細はModalへ送らず、構造化データから回答
- 期間開催イベントは日ごとの開催時間として正規化
- 次イベント推薦は同日・簡易移動バッファを使い、実際の移動時間や別地域を断定しない

100件以上の検索QAは、追加依存なしで次のコマンドから実行できます。

```bash
python tests/run_search_v2_qa.py
```

参加案内・一般FAQ・推薦・候補外生成防止のQAは次で実行できます。

```bash
python tests/run_participation_qa.py
```

Agentic Searchの件数・年齢条件・条件緩和・Planner/Writer境界は次で確認できます。

```bash
python tests/run_agentic_search_qa.py
```

Agentic Searchは、既存Parserで明確に処理できるイベント名・料金・申込・推薦・FAQを経由しません。
曖昧な探索だけを対象にし、Plannerは検索せず、Writerは件数やイベント事実を書きません。
件数・候補・イベント事実は固定Toolと`data/events.json`から決定します。

## Semantic Command QA

Semantic Commandは、LLMを自然文の構造化だけに使い、Flowの実行・イベント事実・件数・料金・申込要否・組み合わせ可否は決定論的なPythonと`data/events.json`で確定します。
日付・時刻のpending回答、前回結果の絞り込み、成人と子ども向けの意味、屋内条件と「歴史的な建物」のテーマも検証対象です。

```bash
python tests/run_command_semantic_qa.py
```

評価データは`tests/data/command_semantic_eval.json`のDEV/HOLDOUTに分離しています。
本番のModal推論・Streamlit表示確認は、必要なSecretsとデプロイ環境を設定したうえで手動確認してください。

## 必要なSecrets

新PoCのStreamlitアプリのプロジェクト単位Secretsに、次の3項目を設定します。

- `MODAL_URL`
- `MODAL_KEY`
- `MODAL_SECRET`

テスト容易性を優先し、共通パスワード認証は一時的に無効化しています。

Secretの値は、このリポジトリ、README、チャット、ログに保存しないでください。

ModalデプロイをGitHub Actionsから実行する場合は、Actions Secretsに次の2項目を設定します。

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Modalデプロイ

新規Modal Appは `ehime-kokubunsai-ai-poc-api`、新規Volumeは `ehime-kokubunsai-model-cache` です。

```bash
modal run modal_backend.py::download_model
modal deploy modal_backend.py
```

既存のSarashina用Modal AppやVolumeを再利用しないでください。

## 検証する問い

イベント一覧や検索フォームを操作するより、会話でイベントを探す方が便利かを検証します。

ベクトルDB、Embedding、RAG基盤、Web検索、位置情報API、GPS、外部観光API、ファインチューニングは使用しません。

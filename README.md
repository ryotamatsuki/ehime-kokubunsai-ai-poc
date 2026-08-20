# 伊予の文化案内人 PoC

「愛顔えひめの文化祭2028」を想定した、自然文によるイベント探索の技術検証用PoCです。

## 注意

- `data/events.json` の30件はすべてPoC用の架空イベントです。
- `example.invalid` のURLも架空URLです。
- 愛媛県・愛顔えひめの文化祭2028の公式サービスではありません。
- みきゃん、みきゃん画像、オレンジ色の犬、類似キャラクターは使用していません。
- 既存の `sarashina-chat` とは独立した新規リポジトリです。既存環境の変更を目的としません。

## 構成

- Streamlit：自然文入力、検索結果、イベントカード、共通パスワード認証
- `event_search.py`：30件のJSONを日付・地域・ジャンル・対象・屋内外・料金・キーワードで決定的に検索
- `data/search_metadata.json`：イベント事実を変更せずに別名・検索タグを管理
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

100件以上の検索QAは、追加依存なしで次のコマンドから実行できます。

```bash
python tests/run_search_v2_qa.py
```

## 必要なSecrets

新PoCのStreamlitアプリのプロジェクト単位Secretsに、次の4項目を設定します。

- `APP_PASSWORD`
- `MODAL_URL`
- `MODAL_KEY`
- `MODAL_SECRET`

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

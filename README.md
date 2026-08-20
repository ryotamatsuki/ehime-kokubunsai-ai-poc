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
- Modal：`sbintuitions/sarashina2.2-3b-instruct-v0.1` をT4・4bitで実行
- LLM：候補イベントに対する短い案内コメントだけを生成

イベント名・日時・場所・料金・ジャンル・URLは、LLMの出力ではなく `data/events.json` から画面に直接表示します。

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


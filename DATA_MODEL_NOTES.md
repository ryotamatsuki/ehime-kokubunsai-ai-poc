# 国文祭PoC データモデル v3

Data Model v3 は、既存の `data/events.json`（v2）を壊さずに、イベント案内・回遊推薦・自然言語の体験条件検索に必要な P0 事実を正規化して追加する。

## 1. Source of truth と互換性

- `data/events.json`: 既存 UI / Search v2 が利用する30件の架空イベント本体。v2互換ソースとして維持する。
- `data/event_profiles_v3.json`: イベント状態、体験特性、典型滞在時間、最終入場情報と provenance policy を保持する。
- `data/venues_v3.json`: 再利用可能な会場ID、場所種別、住所表示、位置情報の検証状態と provenance policy を保持する。
- `data_model_v3.py`: 上記3ソースを厳格に検証し、30件の v3 event view を合成する。

v3 は v2 のフィールド名を書き換えない。既存検索・推薦・UIは段階的に v3 項目を利用できる。

## 2. P0 項目

### event_status

`scheduled | postponed | rescheduled | cancelled | completed`

PoCでは30件を `scheduled` として定義する。将来、本番データでは中止・延期を検索結果と回答に反映する。

### venue

イベントは `venue_id` で `venues_v3.json` を参照する。001と030のように同じ会場を使うイベントは同一 venue を再利用する。

会場は次を持つ。

- `name`
- `municipality`
- `region`
- `location_type`
- `address_text`
- `address_status`
- `geo.latitude / geo.longitude`
- `geo.precision`
- `geo.routing_eligible`
- `geo.enrichment_status`

`場所` の文字列から正確な住所・座標を推測しない。現時点では v2 に根拠のある精密座標がないため、緯度経度は `null`、`precision=unknown`、`routing_eligible=false` とする。将来、公式情報等で検証した時だけroutingに利用できる。

### experience_profile

イベント中の体験を、検索語ではなく客観的な多次元属性として保持する。

- `posture`: `mostly_seated | mixed | standing_or_walking | unknown`
- `seating`: `guaranteed | available | limited | none | unknown`
- `mobility_load`: `low | medium | high | unknown`
- `engagement_modes`: `watch | listen | hands_on | audience_participation | walk_explore` の配列

「座って楽しめる」「あまり歩きたくない」等を将来 hard filter にする場合、LLMはユーザー意図をこの語彙へ変換し、イベント側の一致判定はこの構造化事実に対して deterministic に行う。

### estimated_visit_duration

`typical_minutes` と `basis` を保持する。

- 固定プログラムは開始・終了時刻から決定論的に算出し `basis=scheduled_program` とする。
- 随時入場の展示等は PoC 上の典型滞在時間を明示的に定義し `basis=poc_authored` とする。
- 不明な場合は `typical_minutes=null`, `basis=unknown` とし、開催時間全体を滞在時間と誤認しない。

### last_admission

`time` と `status` を保持する。

- `known`: 明示された最終入場時刻があり、`HH:MM` を保持する。
- `unknown`: 随時入場だが最終入場時刻が登録されていない。閉館時刻等から推測しない。
- `not_applicable`: 開始時刻参加型で、最終入場という概念を適用しない。

## 3. Provenance と LLM 推論境界

v3では sidecar の provenance policy と各フィールドの値から、loader がフィールド群ごとの provenance を決定論的に生成する。1つの信頼度をイベント全体へ付与しない。

イベントプロファイルでは、以下それぞれに provenance がある。

- `event_status`
- `experience_profile`
- `estimated_visit_duration`
- `last_admission`

会場では、以下を分ける。

- `identity`
- `address`
- `geo`

provenance は `source_type`, `source_ref`, `derivation`, `hard_filter_eligible`, `note` を用いる（venueではhard filterを許可しない）。

`derivation=llm_inferred` の値は、validator が `hard_filter_eligible=true` を拒否する。つまり、LLMが「シンポジウムだから座れるだろう」と推測した情報を、利用者の必須条件を満たす事実として扱うことはできない。

PoCの `experience_profile` は30件の架空イベント仕様として人為的に定義したテスト用事実であり、実在イベントに関する推測ではない。

## 4. Unknown を補完しない

v2から継続して、`unknown` / `null` は欠損ではなく意味のある状態として扱う。

- 座席情報不明 ≠ 座席なし
- 座標不明 ≠ 市町中心点を会場座標にする
- 最終入場不明 ≠ 終了時刻まで入場可能

本番データでは公式サイト、主催者入力、事務局確認等で事実を更新し、provenanceも同時に更新する。

## 5. v3 loader の利用

```python
from data_model_v3 import load_events_v3

events = load_events_v3()
```

返却される各イベントは v2 の全フィールドを保持したまま、次を追加する。

- `data_model_version`
- `event_status`
- `venue_id`
- `venue_v3`
- `experience_profile`
- `estimated_visit_duration`
- `last_admission`
- `provenance_v3`

現段階では既存 Search v2 の挙動を一括変更せず、Data Model v3 を独立した信頼できるデータ境界として導入する。体験条件検索・地理的回遊・「今から行ける」判定は、このv3境界を利用して段階的に実装する。

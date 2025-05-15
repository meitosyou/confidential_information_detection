# BERTopic サンプルコード

このサンプルコードは、BERTopicを使用してトピックモデリングを行う基本的な例を示しています。

## セットアップ

1. 新しい仮想環境の作成（推奨）:
```bash
python -m venv bertopic_env
source bertopic_env/bin/activate  # macOSの場合
```

2. 必要なパッケージのインストール:
```bash
pip install -r requirements.txt
```

## 使用方法

1. トピックモデルの作成と保存:
```bash
python create_topics_bertopic.py
```
これにより、トレーニングデータからトピックモデルが作成され、以下のファイルが生成されます：
- `results/topic_model.pkl`: 学習済みモデル
- `results/topic_details.json`: トピックの詳細情報
- `results/topic_visualization.html`: トピックの分布図
- `results/topic_hierarchy.html`: トピックの階層図

2. 新規文書の分類:
```bash
python identify_docs_bertopic.py
```
これにより、保存されたモデルを使用してテストデータの分類が行われ、以下のファイルが生成されます：
- `results/prediction_details.json`: 予測結果の詳細
- `results/prediction_visualization.html`: 予測結果の分布図
- `results/prediction_hierarchy.html`: 予測結果の階層図

## コードの説明

### トピックモデルの作成 (`create_topics_bertopic.py`)
1. トレーニングデータを使用したトピックモデリング
   - ConfData_train.csvのExcerpt列のデータを使用
   - 自動的にトピック分類を実行
   - データの特徴に基づいて適切なトピック数を決定

2. トピックモデルの保存
   - 学習済みモデルを `results/topic_model.pkl` として保存
   - トピックの詳細情報を `results/topic_details.json` として保存
   - 可視化結果を HTML形式で保存

### 新規文書の分類 (`identify_docs_bertopic.py`)
1. 保存されたモデルを使用した予測
   - ConfData_test.csvの文書を分類
   - 各文書の所属トピックを予測
   - トピック所属確率を計算

2. 予測結果の保存
   - 予測の詳細を `results/prediction_details.json` として保存
   - 可視化結果を HTML形式で保存

## 出力される情報

1. トピックの概要
2. 各トピックの上位単語とそのスコア
3. 英語文書のトピック分析結果
4. トピックの詳細情報（JSON形式）
   - `topic_details.json`: 各トピックの詳細な情報
     * トピックID
     * トピックサイズ（所属する文書数）
     * 上位param_topic_n_words単語とそのスコア（重要度順）
     * 所属する文書とその確率
     * トピック全体の統計情報

5. インタラクティブな可視化（HTML形式）
   - トピックの分布図 (`topic_visualization.html`)
   - トピックの階層図 (`topic_hierarchy.html`)

## モデルの構成と処理ステップ

### 1. 埋め込み表現の生成
- `paraphrase-multilingual-MiniLM-L12-v2`を使用
  * 50以上の言語に対応した多言語Sentence-BERTモデル
  * 12層のTransformerで構成された軽量かつ高性能なモデル
  * 入力文書を512次元の密ベクトルに変換
  * 文章レベルの意味を効果的に捉えることが可能

### 2. 次元削減とクラスタリング
#### UMAP設定
- `n_neighbors=15`: 局所的な構造を保持するためのパラメータ
- `n_components=5`: 出力される次元数
- `min_dist=0.0`: データポイント間の最小距離
- `metric='cosine'`: ベクトル間の類似度計算方法

#### HDBSCAN設定
- `min_cluster_size=3`: クラスタの最小サイズ
- `metric='euclidean'`: クラスタリングの距離計算方法
- `min_samples=2`: コアポイントを定義するための最小サンプル数
- `prediction_data=True`: 予測データの保存を有効化

### 3. トピック表現の生成
#### Vectorizerの設定
- `min_df=5`: 単語が出現する必要がある最小文書数
- `ngram_range=(1, 2)`: 単語とフレーズの両方を考慮
- `stop_words`: トピックモデリングから除外する単語のリスト
  * NLTKの標準的な英語ストップワード（冠詞、前置詞など）
  * 追加のカスタムストップワード：
    - 助動詞（would, could, should など）
    - 代名詞（he, she, they など）
    - 前置詞（in, on, at など）
    - 接続詞（and, but, or など）
    - 一般動詞の活用形（am, is, are など）
    - その他の一般的な単語（really, very, just など）

### 4. トピックの最適化
- トピック間の類似性に基づく階層構造の構築
- アウトライアー（トピックID: -1）の検出
- トピックの自動的なラベル付け

## 注意点

- トピックID -1 はアウトライアー（どのトピックにも属さない文書）を示します
- 実行時間は使用するデータ量によって変動します
- 可視化結果は、トピック間の関係性を理解するのに役立ちます

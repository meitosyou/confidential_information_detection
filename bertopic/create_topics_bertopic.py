from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from set_stopwords import get_stopwords_en
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import json
from collections import defaultdict
import pickle

param_topic_n_words = 20

# データの読み込み
df = pd.read_csv('../experiment_data/confidential_data/ConfData_train.csv')
docs = df['Excerpt'].tolist()

# ストップワードの設定
stop_words = get_stopwords_en('custom_stopwords_en.txt')

# 1. 埋め込み表現の生成
# embedding_model: 文書を512次元の密ベクトルに変換
# - 50以上の言語に対応した多言語Sentence-BERTモデル
# - 12層のTransformerで構成され、軽量かつ高性能
# - 文章レベルの意味を捉えることが可能
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 次元削減とクラスタリング
# UMAPで512次元から低次元（n_components次元）に削減
# n_neighbors: 局所的な構造を保持するためのパラメータ
# min_dist: データポイント間の最小距離
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# 3. HDBSCANでクラスタリング
# min_cluster_size: クラスタの最小サイズ
# metric: 距離の計算方法
# min_samples: コアポイントを定義するための最小サンプル数
hdbscan_model = HDBSCAN(
    min_cluster_size=3,
    metric='euclidean',
    min_samples=2,
    prediction_data=True
)

# 4. トピック表現の生成
# Vectorizerの設定（ストップワードを含む）
vectorizer_model = CountVectorizer(
    stop_words=list(stop_words),  # ストップワードをリストに変換
    min_df=5,                     # 最小文書頻度
    ngram_range=(1, 2)           # 1-gramと2-gramを使用
)

# 5. カスタムのc-TF-IDF変換器
# - 各トピック内での単語の重要度を計算
# - トピック間での単語の特異性を考慮
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,   # 頻出語の重みを減少
    bm25_weighting=True          # BM25重み付けを使用
)

# 6. トピック表現の改善モデル
# - 多様性を考慮した表現を生成
# - トピックをより解釈しやすく
representation_model = MaximalMarginalRelevance(
    diversity=0.3,  # 多様性の重み（0-1
    top_n_words=param_topic_n_words
)

# BERTopicモデルの初期化と設定
topic_model = BERTopic(
    # 1. Convert document into embeddings
    embedding_model=embedding_model,
    
    # 2. Reduce Embeddings dimensionality
    umap_model=umap_model,

    # 3. Cluster Reduce Embeddings
    hdbscan_model=hdbscan_model,
    
    # 4. Tokenize documents
    vectorizer_model=vectorizer_model,

    # 5. Word-weighting scheme
    ctfidf_model=ctfidf_model,               # カスタムc-TF-IDF
    
    # 6. Tune Topic Representation
    representation_model=representation_model,# 表現改善モデル
    
    # 7. Other Settings
    min_topic_size=3,            # トピックの最小サイズ
    calculate_probabilities=True, # トピック所属確率を計算
    verbose=True,                # 進行状況を表示
    nr_topics=None, # トピックを指定の数に調整する
    # top_n_words=param_topic_n_words # トップ何単語をトピックに含めるか。representation_modelを導入すると無効になる
    )

# モデルの学習
topics, probs = topic_model.fit_transform(docs)

# トピック情報の収集
topic_info = topic_model.get_topic_info()
print("\n=== トピックの概要 ===")
print(topic_info)

# 各トピックの詳細情報を収集
topic_details = defaultdict(lambda: {
    "topic_id": None,
    "size": 0,
    "top_words": [],
    "documents": [],
    "document_probabilities": []
})

# トピックごとの情報を収集
for topic_id in sorted(list(set(topics))):
    if topic_id != -1:  # アウトライアートピックを除外
        topic_details[topic_id]["topic_id"] = int(topic_id)  # int64をintに変換
        topic_details[topic_id]["size"] = int(len([t for t in topics if t == topic_id]))  # int64をintに変換
        # 上位param_topic_n_words単語を取得し、スコアをfloatに変換
        topic_details[topic_id]["top_words"] = [
            {"word": str(word), "score": float(score)}  # 全ての値をJSONシリアライズ可能な型に変換
            for word, score in topic_model.get_topic(topic_id)
        ]

# 文書ごとの情報を収集
print("\n=== 各文書のトピック ===")
for doc, topic, prob in zip(docs, topics, probs):
    print(f"\n文書: {doc}")
    print(f"トピック: {topic}")
    if topic != -1:
        print("トピックの上位単語:")
        for word, score in topic_model.get_topic(topic):
            print(f"- {word}: {score:.4f}")
        
        # トピックの詳細情報に文書を追加
        topic_details[topic]["documents"].append(doc)
        topic_details[topic]["document_probabilities"].append(float(max(prob)))  # 最大確率を保存

# 結果をJSONファイルとして保存
with open("../results/topic_details.json", "w", encoding="utf-8") as f:
    json.dump({
        "topic_model_info": {
            "total_topics": int(len(set(topics)) - 1),  # int64をintに変換
            "total_documents": int(len(docs)),  # int64をintに変換
            "topics": list(topic_details.values())
        }
    }, f, ensure_ascii=False, indent=2)

# 結果の可視化と保存
print("\n=== トピックモデリングの結果を保存 ===")
topic_model.visualize_topics().write_html("../results/topic_visualization.html")
topic_model.visualize_hierarchy().write_html("../results/topic_hierarchy.html")
print("可視化結果を保存しました")

# トピックモデルの保存
print("\n=== トピックモデルを保存 ===")
with open("../results/topic_model.pkl", "wb") as f:
    pickle.dump(topic_model, f)
print("トピックモデルを保存しました")

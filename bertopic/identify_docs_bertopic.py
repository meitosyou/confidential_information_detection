import pickle
import pandas as pd
import json
from collections import defaultdict

# 保存されたトピックモデルの読み込み
print("\n=== トピックモデルを読み込み ===")
with open("../results/topic_model.pkl", "rb") as f:
    topic_model = pickle.load(f)
print("トピックモデルを読み込みました")

# 新しい文書の読み込み
df = pd.read_csv('../experiment_data/confidential_data/ConfData_test.csv')
new_docs = df['Excerpt'].tolist()

# 新しい文書のトピック予測
print("\n=== 新しい文書のトピック予測 ===")
topics, probs = topic_model.transform(new_docs)

# 予測結果の収集
prediction_details = defaultdict(lambda: {
    "topic_id": None,
    "documents": [],
    "document_probabilities": []
})

# 文書ごとの予測結果を収集
print("\n=== 予測結果 ===")
doc_n = 0
for doc, topic, prob in zip(new_docs, topics, probs):
    print(f"\n文書番号: {doc_n}")
    print(f"文書: {doc}")
    print(f"BERTopicで予測されたトピック: {topic}")
    print(f"確信度: {max(prob)}")
    doc_n +=1
    if topic != -1:
        # 上位単語の表示
        # print("トピックの上位単語:")
        # for word, score in topic_model.get_topic(topic):
        #     print(f"- {word}: {score:.4f}")
        
        # 予測の詳細情報を収集
        prediction_details[topic]["topic_id"] = int(topic)  # int64をintに変換
        # 上位単語を取得し、スコアをfloatに変換
        prediction_details[topic]["top_words"] = [
            {"word": str(word), "score": float(score)}  # 全ての値をJSONシリアライズ可能な型に変換
            for word, score in topic_model.get_topic(topic)
        ]
        prediction_details[topic]["documents"].append(doc)
        prediction_details[topic]["document_probabilities"].append(float(max(prob)))

# 予測結果をJSONファイルとして保存
with open("../results/prediction_details.json", "w", encoding="utf-8") as f:
    json.dump({
        "prediction_info": {
            "total_documents": int(len(new_docs)),  # int64をintに変換
            "predictions": list(prediction_details.values())
        }
    }, f, ensure_ascii=False, indent=2)

# 予測結果の可視化
print("\n=== 予測結果の可視化を保存 ===")
topic_model.visualize_topics().write_html("../results/prediction_visualization.html")
topic_model.visualize_hierarchy().write_html("../results/prediction_hierarchy.html")
print("可視化結果を保存しました")

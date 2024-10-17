import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import json

# 元のJSONファイルの読み込み
input_file = './experiment_data/confidential_data/ConfData_test_embeddings.json'
output_file = './experiment_data/confidential_data/ConfData_with_numbers.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# テキストと埋め込みベクトルを抽出し、番号を付ける
texts = [entry['text'] for entry in data]
embeddings = np.array([entry['embedding'] for entry in data])

# 各テキストに番号を付ける
for i, entry in enumerate(data):
    entry['number'] = i  # 番号を付加

# 番号付きデータをJSONファイルとして保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 距離行列の計算（コサイン距離を使用）
dist_matrix = pdist(embeddings, metric='cosine')

# 階層型クラスタリングの実行
linkage_matrix = sch.linkage(dist_matrix, method='ward')

# デンドログラムの描画
# labelsには番号を使う
labels = [str(i) for i in range(len(texts))]

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram')
plt.xlabel('Text Number')
plt.ylabel('Distance')
plt.show()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.environ["OMP_NUM_THREADS"] = "4"  # 防止在Windows操作系统中使用带有MKL（Math Kernel Library）的KMeans算法时，可能会出现内存泄漏问题，尤其是当数据块数少于可用线程数时。


def plot_pdf(data, bins=30, color='blue', title="Probability Density Function (PDF)", xlabel="Value", ylabel="Density"):
    """
    绘制一维数组的概率密度函数（PDF）图。
    参数:
    - data: 要绘制PDF的一维数组。
    - bins: 直方图的柱数（默认值为30）。
    - color: 绘图的颜色（默认值为蓝色）。
    - title: 图形的标题（默认值为 "Probability Density Function (PDF)"）。
    - xlabel: x轴标签（默认值为 "Value"）。
    - ylabel: y轴标签（默认值为 "Density"）。
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(data, kde=True, bins=bins, color=color, stat="density")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def l2_normalize(array):
    l2_norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized_array = array / l2_norms
    return normalized_array


# 加载嵌入数据
embeddings = np.load('data/NIPS34/all/exer_embeds.npy')  # 请确保文件路径正确
embeddings = l2_normalize(embeddings)

# 定义聚类数量范围
n_clusters_range = range(2, 400)
avg_distances = []

for n_clusters in n_clusters_range:
    if n_clusters == 1:
        avg_distances.append(1.0)
        continue

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centers = kmeans.cluster_centers_

    # 计算聚类中心间的余弦距离矩阵
    distance_matrix = cosine_distances(centers)  # 1 - cosine_similarity：越趋近于0越集中，越大越发散

    # 提取上三角部分（排除对角线）
    rows, cols = np.triu_indices_from(distance_matrix, k=1)
    upper_triangle = distance_matrix[rows, cols]

    # 计算平均距离
    avg_distance = upper_triangle.mean()
    avg_distances.append(avg_distance)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, avg_distances, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (n)')
plt.ylabel('Average Inter-Cluster Cosine Distance')
plt.title('n vs. Average Inter-Cluster Distance')
plt.grid(True)
plt.xticks(n_clusters_range)
plt.show()

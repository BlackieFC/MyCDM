import numpy as np
import matplotlib.pyplot as plt


# 读取数据
arr = np.load('data/NIPS34/all/exer_embeds.npy')
print(arr.shape)

# 计算每一行的L2范数
row_norms = np.linalg.norm(arr, axis=1)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(row_norms, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Row-wise L2 Norm Distribution')
plt.xlabel('Norm Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.show()

# 返回计算结果（前5行示例）
print("前5行的范数值：")
print(row_norms[:5])

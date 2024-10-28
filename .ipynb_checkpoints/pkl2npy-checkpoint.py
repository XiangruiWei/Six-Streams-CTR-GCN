import pickle
import numpy as np

# 假设你的 pkl 文件名为 'data.pkl'
with open('bone_motion_score.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 创建一个维度为 (4599, 155) 的 numpy 数组
data_array = np.zeros((4599, 155))

# 遍历字典,将值填充到 numpy 数组中
for i in range(4599):
    key = f"sample_{i}"
    data_array[i] = data_dict[key]

# 将 numpy 数组保存为 npy 文件
np.save('bone_motion_score.npy', data_array)

print("PKL file converted to NPY file successfully!")

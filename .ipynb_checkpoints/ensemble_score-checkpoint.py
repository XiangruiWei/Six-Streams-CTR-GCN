import numpy as np

# 读取 4 个 npy 文件
score1 = np.load('joint_score.npy')
score2 = np.load('bone_score.npy')
score3 = np.load('joint_motion_score.npy')
score4 = np.load('bone_motion_score.npy')

# 定义权重
weights = [0.8, 0.3, 0.2, 0.2]

# 计算加权平均
fused_score = np.zeros_like(score1)
for i in range(score1.shape[0]):
    for j in range(score1.shape[1]):
        fused_score[i, j] = (score1[i, j] * weights[0] + score2[i, j] * weights[1] + 
                            score3[i, j] * weights[2] + score4[i, j] * weights[3]) / sum(weights)

# 保存融合后的结果
np.save('fused_score_2.npy', fused_score)
print('Fusion complete. Saved to fused_score.npy')

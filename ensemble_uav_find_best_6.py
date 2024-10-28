import argparse  
import pickle  
import os  
import numpy as np  
from tqdm import tqdm  
from skopt import gp_minimize  

def objective(alpha, r1, r2, r3, r4, r5, r6, label):  
    """  
    目标函数，用于贝叶斯优化  
    """  
    right_num = 0  
    total_num = 0  
    
    for i in range(len(label)):  
        l = label[i]  
        _, r11 = r1[i]  
        _, r22 = r2[i]  
        _, r33 = r3[i]  
        _, r44 = r4[i]  
        _, r55 = r5[i]  
        _, r66 = r6[i]  # 处理第六个模型的预测结果  
                
        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3] + r55 * alpha[4] + r66 * alpha[5]  
        r = np.argmax(r)  
        right_num += int(r == int(l))  
        total_num += 1  
        
    # 计算准确率  
    acc = right_num / total_num if total_num > 0 else 0  
    return -acc  # 返回负准确率以进行最小化  

def find_best_alpha(r1, r2, r3, r4, r5, r6, label):  
    """  
    寻找最佳的 alpha 参数组合  
    
    参数:  
    r1, r2, r3, r4, r5, r6 (np.ndarray): 不同模型的预测结果  
    label (np.ndarray): 真实标签  
    
    返回:  
    best_alpha (list): 最佳的 alpha 参数组合  
    best_acc (float): 最佳的准确率  
    """  
    # 定义 alpha 参数的搜索范围  
    space = [(0.2, 1.2) for _ in range(6)]  # 六个 alpha 参数的范围  

    # 执行贝叶斯优化  
    result = gp_minimize(lambda alpha: objective(alpha, r1, r2, r3, r4, r5, r6, label), space, n_calls=100, random_state=0)  

    best_alpha = result.x  
    best_acc = -result.fun  # 取负值以获得最佳准确率  

    return best_alpha, best_acc  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset',  
                        required=True,  
                        help='the work folder for storing results')  
    parser.add_argument('--alpha',  
                        default=1,  
                        help='weighted summation',  
                        type=float)  

    parser.add_argument('--joint-dir',  
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')  
    parser.add_argument('--bone-dir',  
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')  
    parser.add_argument('--joint-motion-dir', default=None)  
    parser.add_argument('--bone-motion-dir', default=None)  
    parser.add_argument('--joint_tta', default=None)  
    parser.add_argument('--bone_longtail', default=None)  
    parser.add_argument('--stream7', default=None)  # 新增第六个模型的目录参数  

    arg = parser.parse_args()  

    dataset = arg.dataset  
    if 'uav' in arg.dataset:  
        with open('./data/test_label_A.pkl', 'rb') as f:  
                label = pickle.load(f)   
            
    else:  
        raise NotImplementedError  

    with open(os.path.join(arg.joint_dir, 'best_score.pkl'), 'rb') as r1:  
        r1 = list(pickle.load(r1).items())  

    with open(os.path.join(arg.bone_dir, 'best_score.pkl'), 'rb') as r2:  
        r2 = list(pickle.load(r2).items())  

    if arg.joint_motion_dir is not None:  
        with open(os.path.join(arg.joint_motion_dir, 'best_score.pkl'), 'rb') as r3:  
            r3 = list(pickle.load(r3).items())  
    if arg.bone_motion_dir is not None:  
        with open(os.path.join(arg.bone_motion_dir, 'best_score.pkl'), 'rb') as r4:  
            r4 = list(pickle.load(r4).items())  
    if arg.joint_tta is not None:  
        with open(os.path.join(arg.joint_tta, 'best_score.pkl'), 'rb') as r5:  
            r5 = list(pickle.load(r5).items())  # 处理第五个模型的预测结果  
    if arg.bone_longtail is not None:  
        with open(os.path.join(arg.bone_longtail, 'best_score.pkl'), 'rb') as r6:  
            r6 = list(pickle.load(r6).items())  # 处理第六个模型的预测结果  
            
    # 假设你已经有了 r1, r2, r3, r4, r5 和 r6 以及 label  
    best_alpha, best_acc = find_best_alpha(r1, r2, r3, r4, r5, r6, label)  
    print(f"Best alpha: {best_alpha}")  
    print(f"Best accuracy: {best_acc:.4f}")
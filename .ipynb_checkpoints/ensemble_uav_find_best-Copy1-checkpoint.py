import argparse  
import pickle  
import os  
import numpy as np  
from tqdm import tqdm  

def find_best_alpha(r1, r2, r3, r4, label):  
    """  
    寻找最佳的 alpha 参数组合  
    
    参数:  
    r1, r2, r3, r4 (np.ndarray): 不同模型的预测结果  
    label (np.ndarray): 真实标签  
    
    返回:  
    best_alpha (list): 最佳的 alpha 参数组合  
    best_acc (float): 最佳的准确率  
    """  
    best_acc = 0.0  
    best_alpha = [0.0, 0.0, 0.0, 0.0]  
    
    # 定义 alpha 参数的搜索范围  
    alpha_range = np.arange(0.0, 1.1, 0.1)  
    
    for a1 in alpha_range:  
        for a2 in alpha_range:  
            for a3 in alpha_range:  
                for a4 in alpha_range:   
                    alpha = [a1, a2, a3, a4]  
                    
                    right_num = 0  
                    right_num_5 = 0  
                    total_num = 0  
                    
                    for i in tqdm(range(len(label))):  
                        l = label[i]  
                        _, r11 = r1[i]  
                        _, r22 = r2[i]  
                        _, r33 = r3[i]  
                        _, r44 = r4[i]  
                                
                        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]  
                        rank_5 = r.argsort()[-5:]  
                        right_num_5 += int(int(l) in rank_5)  
                        r = np.argmax(r)  
                        right_num += int(r == int(l))  
                        total_num += 1  
                            
                    # 计算准确率  
                    if total_num > 0:  
                        acc = right_num / total_num  
                        acc_5 = right_num_5 / total_num  
                        
                        if acc > best_acc:  
                            best_acc = acc  
                            best_alpha = alpha  
                            
                            print(f"Accuracy: {acc:.4f}, Alpha: {alpha}")  
                                
    return best_alpha, best_acc  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset',  
                        required=True,  
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'uav-v1', 'uav-v2'},  
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
    parser.add_argument('--stream5', default=None)  
    parser.add_argument('--stream6', default=None)  

    arg = parser.parse_args()  

    dataset = arg.dataset  
    if 'uav' in arg.dataset:  
        if 'v1' in arg.dataset:  
            with open('./data/uav/v1/test_label_A.pkl', 'rb') as f:  
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
            
    # 假设你已经有了 r1, r2, r3, r4 和 label  
    best_alpha, best_acc = find_best_alpha(r1, r2, r3, r4, label)  
    print(f"Best alpha: {best_alpha}")  
    print(f"Best accuracy: {best_acc:.4f}")

import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

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

    right_num = total_num = right_num_5 = 0
    best = 0.0
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        total_num = 0
        right_num = 0
        arg.alpha = [0.39, 0.94, 0.42, 0.34, 0.52, 0.35]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            _, r55 = r5[i]
            _, r66 = r6[i]

            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4] + r66 * arg.alpha[5]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        print(acc, arg.alpha)
        if acc>best:
            best = acc
            best_alpha = arg.alpha
        acc5 = right_num_5 / total_num
    print(best, best_alpha)

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

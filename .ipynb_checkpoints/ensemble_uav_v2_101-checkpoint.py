import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'uav-v2'},
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
            with open('./data/uav/v1/test_label.npy', 'rb') as f:
                label = np.load(f)
        elif 'v2' in arg.dataset:
            with open('./data/uav/v2/test_label.npy', 'rb') as f:
                label = np.load(f)
    else:
        raise NotImplementedError

    #加载相关模型
    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
    if arg.stream5 is not None:
        with open(os.path.join(arg.stream5, 'epoch1_test_score.pkl'), 'rb') as r5:
            r5 = list(pickle.load(r5).items())
    if arg.stream6 is not None:
        with open(os.path.join(arg.stream6, 'epoch1_test_score.pkl'), 'rb') as r6:
            r6 = list(pickle.load(r6).items())

    right_num = total_num = right_num_5 = acc_right_num = 0  # 新增acc_right_num用于记录正确的样本数
    best = 0.0
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        total_num = 0
        right_num = 0
        #这个参数怎么来的！！！注意回头调整
        arg.alpha = [0.6, 0.6, 0.6, 0.7, 0.3, 0.3]

        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]


            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]

            # Top5准确率 这里的l是指lable
            right_num_5 += int(int(l) in rank_5)

            # Top1准确率
            r = np.argmax(r)
            right_num += int(r == int(l))

            # 计算正常acc
            if r == int(l):  # 如果预测的类别等于真实类别
                acc_right_num += 1  # 记录正确的预测次数
            total_num += 1

        # 计算Top1和Top5准确率
        acc_top1 = right_num / total_num
        acc_top5 = right_num_5 / total_num

        # 计算正常的准确率
        acc = acc_right_num / total_num

        print(f'Top1 Acc: {acc_top1 * 100:.4f}%')
        print(f'Top5 Acc: {acc_top5 * 100:.4f}%')
        print(f'Normal Acc: {acc * 100:.4f}%')

        if acc_top1 > best:
            best = acc_top1
            best_alpha = arg.alpha

    print(f'Best Top1 Acc: {best * 100:.4f}%, Alpha: {best_alpha}')
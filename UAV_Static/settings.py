import torch
import os
import numpy as np
import random
# For windows specific error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ----------------- 修改1：环境名称 -----------------
ENVIRONMENT_NAME = "UAV_Static"

# ----------------- 修改2：经验池与设备 -----------------
# 简单的向量环境用 Uniform (均匀采样) 就足够了，计算速度更快
MEMORY_TYPE = "Uniform" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- 修改3：关闭图像处理专用的帧堆叠 -----------------
# 极其重要：我们的状态是一维向量，不是连续的图片，所以必须设为 1，否则会报维度错误
HISTORY_LENGTH = 1 
FRAME_SKIP = 1 
TRAIN_FREQUENCY = 1 # 每次交互都进行一次网络训练，加快收敛

# ----------------- 修改4：加快训练与打印频率的超参数 -----------------
REWARD_DISCOUNT = 0.95
INITIAL_EXPLORE_STEPS = 2000       # 初始纯随机探索步数 (缩短到 2000 步)
TRAIN_STEPS = 200000               # 总训练步数 (20万步对于小网络完全足够)
TARGET_NET_SYNC_FREQUENCY = 1000   # 目标网络同步频率 (缩短，加速早期学习)
EVALUATION_FREQUENCY = 2000        # 每 2000 步就验证并打印一次结果，让你能频繁看到进展
EVALUATION_TRIALS = 5              # 每次验证跑 5 个回合取平均
EVALUATION_TRIALS_TEST = 50        # 最终测试跑 50 个回合
REPLAY_MEMORY_SIZE = 50000         # 经验回放池大小

RANDOM_SEEDS = [123, 321, 456]     # 跑3个随机种子用于后期画平滑曲线
EVALUATION_STATES = 200
assert EVALUATION_FREQUENCY % TRAIN_FREQUENCY == 0

# 全连接网络(MLP)比较容易训练，学习率可以稍微调大一点点
LEARNING_RATE = 1e-3 


def set_random_seed(rand_seed):
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    print(f"<<<<<<<<<<<<<<<<<Finished setting random seed at {rand_seed}!>>>>>>>>>>>>>>>")
    return
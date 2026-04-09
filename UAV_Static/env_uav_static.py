import gym
from gym import spaces
import numpy as np
import math

class UAV_Emergency_Env(gym.Env):
    def __init__(self, num_users=5):
        super(UAV_Emergency_Env, self).__init__()
        # 1. 核心仿真参数
        self.K = num_users
        self.P_total = 1.0        # 无人机总功率 1W (30 dBm)
        self.delta_p = 0.05       # 智能体每次动作调整的功率步长 (W)
        self.H = 100.0            # 无人机飞行高度 (m)
        self.B = 2e6              # 总带宽 2 MHz
        self.sigma2 = 1e-14       # 背景噪声功率 -110 dBm (转换成瓦特)
        self.beta0 = 1e-4         # 1米处的参考信道增益 -40 dB
        self.alpha = 2.5          # 路径损耗指数
        
        self.max_steps = 200      # 每一回合 (Episode) 的最大步数
        self.current_step = 0

        # 2. 空间定义
        # 动作空间：选一个用户减功率，选另一个加功率。共 K*(K-1) 种动作
        self.action_space = spaces.Discrete(self.K * (self.K - 1))
        
        # 【修改】状态空间：K个当前功率 + K个上一时刻速率 + K个短板用户的One-hot编码
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3 * self.K,), dtype=np.float32)

        # 3. 兼容原作者 DQN Agent 的自定义维度属性        
        self.n_actions = self.K * (self.K - 1)       
        self.n_state_dims = 3 * self.K # 【修改】从 2*K 变为 3*K

        self.user_positions = None
        self.uav_pos = np.array([250.0, 250.0]) # 假设无人机在 500x500 区域的中心悬停
        self.current_power = None
        self.last_rates = None

    def reset(self):
        self.current_step = 0
        # 随机生成 K 个用户在 500x500 区域内的位置
        self.user_positions = np.random.uniform(0, 500, size=(self.K, 2))
        # 初始功率平均分配
        self.current_power = np.ones(self.K) * (self.P_total / self.K)
        # 初始速率设为 0
        self.last_rates = np.zeros(self.K)
        
        # 【新增】初始短板状态为全 0
        bottleneck_onehot = np.zeros(self.K)
        
        return np.concatenate((self.current_power, self.last_rates, bottleneck_onehot))

    def step(self, action):
        self.current_step += 1

        # ---------------- 1. 动作解码 ----------------
        # 将一维离散动作转化为 i (被扣功率的用户) 和 j (增加功率的用户)
        i = action // (self.K - 1)
        j_temp = action % (self.K - 1)
        j = j_temp if j_temp < i else j_temp + 1

        # ---------------- 2. 执行功率分配 ----------------
        # 必须确保扣除后功率不小于0，且增加后总和不超过 P_total
        if self.current_power[i] >= self.delta_p:
            self.current_power[i] -= self.delta_p
            self.current_power[j] += self.delta_p

        # ---------------- 3. 计算信道和香农速率 ----------------
        rates = np.zeros(self.K)
        for k in range(self.K):
            # a. 计算三维直线距离 d_k
            dist_2d = np.linalg.norm(self.uav_pos - self.user_positions[k])
            d_k = math.sqrt(dist_2d**2 + self.H**2)
            
            # b. 大尺度衰落 g_k
            g_k = self.beta0 * (d_k ** -self.alpha)
            
            # c. 小尺度瑞利衰落 (引入时变不确定性)
            zeta = np.random.rayleigh(1.0)
            h_k = g_k * (zeta ** 2)
            
            # d. 计算信噪比 SNR 和 速率 R
            snr = (self.current_power[k] * h_k) / self.sigma2
            # 带宽平均分配 B/K
            rates[k] = (self.B / self.K) * math.log2(1 + snr)

        self.last_rates = rates

        # ---------------- 4. 计算奖励 (Q-Min的核心点) ----------------
        # 【修改】使用平滑最小值与平均值的混合奖励
        min_rate = np.min(rates)
        mean_rate = np.mean(rates)
        beta = 5.0 # 平滑系数，可调
        
        # 防止数值溢出的安全 LogSumExp 计算
        smooth_min = min_rate - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (rates - min_rate))))
        
        # 奖励融合：80% 关注短板(带平滑)，20% 关注整体性能，提供稠密梯度
        reward = float(0.8 * smooth_min + 0.2 * mean_rate)

        # ---------------- 5. 状态更新与终止判断 ----------------
        # 【新增】找出当前速率最低的用户，制成 One-hot 向量
        bottleneck_idx = np.argmin(rates)
        bottleneck_onehot = np.zeros(self.K)
        bottleneck_onehot[bottleneck_idx] = 1.0

        # 【修改】拼接短板信息的 One-hot 向量
        next_state = np.concatenate((self.current_power, self.last_rates, bottleneck_onehot))
        done = bool(self.current_step >= self.max_steps)

        return next_state, reward, done
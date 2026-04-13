import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. 载入刚才跑出来的核心数据
data_path = "Models_user20/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

# 2. 提取数据并进行物理单位换算
seeds = list(data.keys())
agents = ["Regular", "Modified"]
scores = {agent: [] for agent in agents}

# === 【核心修改区域：物理单位换算】 ===
# 强化学习记录的原始 Score 是：一回合(200步)的累积总速率(bps)
# 为了符合通信工程的直觉，换算为 单步(Step) 的 兆比特每秒(Mbps)
# 换算系数：200 (单回合最大步数) * 1e6 (转换为 Mbps)
SCALE_FACTOR = 200 * 1e6 
# ======================================

for seed in seeds:
    for agent in agents:
        # data[seed][agent] 是一个二维数组，索引 3 是 Score (回合总瓶颈速率)
        raw_scores = data[seed][agent][:, 3]
        
        # 在存入列表前，直接将原始数据除以换算系数
        scaled_scores = raw_scores / SCALE_FACTOR
        scores[agent].append(scaled_scores)

# 3. 开始绘制高大上的 IEEE 风格折线图
plt.figure(figsize=(8, 6))
colors = {"Regular": "#E24A33", "Modified": "#348ABD"} # 使用学术期刊常用的色系

# 【修改】图例标签更新为我们在论文里要写的专业术语
labels = {
    "Regular": "Q-Sum", 
    "Modified": "Q-Min"
}

for agent in agents:
    # 转换维度以便计算平均值和方差
    agent_scores = np.array(scores[agent])
    mean_scores = np.mean(agent_scores, axis=0)
    std_scores = np.std(agent_scores, axis=0)
    
    # 提取横坐标 (训练步数)
    steps = data[seeds[0]][agent][:, 0] /1000
    
    # 画平均值主线
    plt.plot(steps, mean_scores, label=labels[agent], color=colors[agent], linewidth=2.5)
    # 画方差阴影带（体现实验的严谨性和鲁棒性）
    plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, color=colors[agent], alpha=0.2)

# 4. 图表排版美化
# plt.title("Max-Min Fairness in UAV Emergency Network", fontsize=15, fontweight='bold')
plt.xlabel("Training Steps (× $10^3$)", fontsize=13)
plt.ylabel("Max-Min Rate (Mbps)", fontsize=13) 

# === 【新增：统一坐标轴范围】 ===
plt.xlim(0, 200)          # 锁定 X 轴为 0 到 20万步
plt.ylim(0.05, 0.95)      # 锁定 Y 轴，涵盖所有三个数据集的数据范围
# ==============================

plt.legend(loc="upper left", fontsize=12, framealpha=0.9)
# plt.grid(True, linestyle='--', alpha=0.6)
    
# 5. 保存并展示
plt.tight_layout()
plt.savefig("UAV_user_20.png", dpi=300) # 保存为 300dpi 的高清图，可直接贴入大论文
print("画图成功！已保存为高清图片 UAV_user_20.png")
plt.show()
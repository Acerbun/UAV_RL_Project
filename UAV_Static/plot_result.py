import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. 载入刚才跑出来的核心数据
data_path = "Models/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

# 2. 提取数据
seeds = list(data.keys())
agents = ["Regular", "Modified"]
scores = {agent: [] for agent in agents}

for seed in seeds:
    for agent in agents:
        # data[seed][agent] 是一个二维数组，索引 3 是 Score (瓶颈速率)
        scores[agent].append(data[seed][agent][:, 3])

# 3. 开始绘制高大上的 IEEE 风格折线图
plt.figure(figsize=(8, 6))
colors = {"Regular": "#E24A33", "Modified": "#348ABD"} # 使用学术期刊常用的色系
labels = {
    "Regular": "Baseline: Traditional Q-Sum", 
    "Modified": "Ours: Proposed Q-Min"
}

for agent in agents:
    # 转换维度以便计算平均值和方差
    agent_scores = np.array(scores[agent])
    mean_scores = np.mean(agent_scores, axis=0)
    std_scores = np.std(agent_scores, axis=0)
    
    # 提取横坐标 (训练步数)
    steps = data[seeds[0]][agent][:, 0] 
    
    # 画平均值主线
    plt.plot(steps, mean_scores, label=labels[agent], color=colors[agent], linewidth=2.5)
    # 画方差阴影带（体现实验的严谨性和鲁棒性）
    plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, color=colors[agent], alpha=0.2)

# 4. 图表排版美化
plt.title("Max-Min Fairness in UAV Emergency Network", fontsize=15, fontweight='bold')
plt.xlabel("Training Steps", fontsize=13)
plt.ylabel("Minimum User Rate (Bottleneck Rate)", fontsize=13)
plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)

# 5. 保存并展示
plt.tight_layout()
plt.savefig("UAV_Training_Curve.png", dpi=300) # 保存为 300dpi 的高清图，可直接贴入大论文
print("画图成功！已保存为高清图片 UAV_Training_Curve.png")
plt.show()
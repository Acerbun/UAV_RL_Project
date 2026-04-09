import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions, n_state_dims):
        super(DQN, self).__init__()
        # 针对无人机的一维状态向量（长度为 2K），我们使用三层全连接层 (MLP)
        self.fc1 = nn.Linear(n_state_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # 确保输入是 PyTorch 张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # 如果输入是 1D 向量 (单步预测)，自动增加 batch 维度变成 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
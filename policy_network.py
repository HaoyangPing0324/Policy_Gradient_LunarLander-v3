"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/23
@Desc    : 一个基于 PyTorch 的 Policy 网络
@License : MIT License (MIT)
@Version : 1.0

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    """
    基于 PyTorch 的 Policy 网络（灵活配置隐藏层结构）
    输入：LunarLander-v3 的 8 维状态向量
    输出：4 个离散动作的概率分布（对应 0=不动、1=主引擎、2=左引擎、3=右引擎）
    """

    def __init__(self, obs_dim=8, action_dim=4, hidden_layers=[64, 32], activation='relu', device='cuda'):
        """
        初始化网络结构（支持超参数配置，符合任务要求）
        :param obs_dim: 状态维度（LunarLander-v3 固定为 8）
        :param action_dim: 动作维度（LunarLander-v3 固定为 4）
        :param hidden_layers: 隐藏层配置列表（如 [128, 64] 表示2层隐藏层，分别64、32个神经元）
        :param activation: 激活函数（支持 'relu' 或 'sigmoid'，默认 'relu'）
        """
        super(PolicyNetwork, self).__init__()

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 校验隐藏层配置（至少1层隐藏层，符合任务要求）
        if not isinstance(hidden_layers, list) or len(hidden_layers) == 0:
            raise ValueError("hidden_layers 必须是非空列表（如 [64] 或 [128, 64]）")

        # 构建网络层（按 hidden_layers 动态生成）
        layers = []
        in_dim = obs_dim  # 初始输入维度为状态维度

        # 循环生成隐藏层（线性层 + 激活函数）
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))  # 线性变换
            # 激活函数（引入非线性，提升表达能力）
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("激活函数仅支持 'relu' 或 'sigmoid'")
            in_dim = out_dim  # 更新下一层输入维度

        # 输出层（线性变换 + Softmax 转为概率分布）
        layers.append(nn.Linear(in_dim, action_dim))

        # 封装为 Sequential 网络
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        """
        前向传播（核心计算逻辑）
        :param x: 输入的状态张量，形状为 (batch_size, obs_dim) 或 (obs_dim,)
        :return: 动作概率分布（输出概率和为 1）
        """
        # 前向传播计算
        logits = self.net(x)
        # 输出动作概率分布（Softmax 归一化，dim=-1 确保按最后一维归一化）
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def get_action(self, state):
        """
        训练时用：随机采样动作（探索环境）+ 返回动作对数概率
        :param state: 当前状态（8维向量，支持numpy数组或torch张量）
        :return: (action: 采样的动作索引, log_prob: 动作对应的对数概率)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.forward(state)
        # 基于概率分布采样动作（Categorical 分布适配离散动作）
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample()
        # 返回动作（转为Python整数）和对数概率（保持与动作概率同设备）
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def get_best_action(self, state):
        """
        测试/演示时用：选择概率最大的动作（无随机，纯利用）
        :param state: 当前状态（8维向量，支持numpy数组或torch张量）
        :return: 概率最大的动作索引
        """
        # 适配输入类型 + 设备
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.float().unsqueeze(0).to(self.device) if len(state.shape) == 1 else state.to(self.device)
        action_probs = self.forward(state)
        # 取概率最大的动作索引（argmax）
        best_action = torch.argmax(action_probs, dim=-1)
        return best_action.item()
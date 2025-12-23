"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/22
@Desc    : PG训练函数
@License : MIT License (MIT)
@Version : 1.0

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from policy_network import PolicyNetwork

def calculate_discounted_returns(rewards, gamma=0.99):
    discounted_returns = []
    cumulative_return = 0
    for reward in reversed(rewards):
        cumulative_return = reward + gamma * cumulative_return
        discounted_returns.insert(0, cumulative_return)
    return torch.tensor(discounted_returns, dtype=torch.float32)

def train(policy_net, device, save_path="policy_net.pth",
          lr=1e-3, gamma=0.99, max_episodes=1000):
    """
    训练函数：接收外部传入的 Policy 网络，按参数列表配置训练
    :param policy_net: 外部创建的 PolicyNetwork 实例（核心：模型由外部传入）
    :param device: 训练设备（cuda 或 cpu）
    :param lr: 学习率（可配置）
    :param gamma: 折扣因子（可配置）
    :param max_episodes: 最大训练回合数（可配置）
    :param batch_size: 批量更新的回合数（可配置）
    :return: episode_rewards（训练奖励列表，供绘图用）
    """
    # 初始化环境和优化器
    # 训练模式：不显示画面，运行速度快（适合批量训练）
    env = gym.make("LunarLander-v3")
    # 游戏在运行，但你看不到画面，只需要它默默计算得分和状态
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    episode_rewards = []

    # 核心训练循环
    for episode in range(max_episodes):
        trajectory = []
        current_state, _ = env.reset()
        done = False
        total_reward = 0

        # 蒙特卡洛采样整条轨迹
        while not done:
            action, log_prob = policy_net.get_action(current_state)# 从动作概率分布中随机采样  让智能体 “探索环境”（尝试不同动作，找到好的策略），避免 “只会走老路”
            next_state, reward, terminated, truncated, _ = env.step(action)
            '''
            truncated：布尔值，标记 “是否因超过最大步数（1000 步）而终止”
            terminated（中文译 “终止 / 完成”）标记任务是否因 “自然原因” 结束（比如成功着陆、坠毁）
            '''
            done = terminated or truncated
            trajectory.append((log_prob, reward))
            total_reward += reward
            current_state = next_state

        episode_rewards.append(total_reward)

        # 计算 Gt + Baseline 减少方差
        rewards = [t[1] for t in trajectory]
        log_probs = [t[0] for t in trajectory]
        # 2. 计算每一步的折扣累积回报Gt：从当前步到回合结束的所有奖励，按gamma打折求和
        # 作用：把"即时奖励"转换成"长期价值"，评价动作的长期好坏
        discounted_returns = calculate_discounted_returns(rewards, gamma).to(device)
        # 3. 计算基线（Baseline）：本轮所有Gt的平均值
        # 作用：作为"本轮平均水平基准"，避免单轮整体好坏导致的评价偏差
        baseline = discounted_returns.mean()
        # 4. 计算优势函数：用Gt减去基线，得到"相对回报"
        # 作用：减少训练方差（波动），让模型只关注"比平均好/差多少"，学得更稳
        advantages = discounted_returns - baseline

        policy_loss = []
        for log_prob, advantage_t in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage_t)
        total_loss = torch.cat(policy_loss).sum()
        # average_loss =torch.cat(policy_loss).mean() 用平均来反向传播会导致不收敛

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 打印训练进度
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else total_reward
            recent_avg_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else total_reward
            recent_avg_20 = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else total_reward
            recent_avg_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            print(f"Episode {episode+1:4d} |"
                  f" Reward: {total_reward:6.2f} |"
                  f" Recent Avg: {recent_avg:6.2f}")
            if recent_avg >= 200 and recent_avg_50 >= 200 and recent_avg_20 >= 200 and recent_avg_10 >= 200:
                print("最近100、50、20、10轮的平均分都高于200分，训练达标，提前终止")
                break

    # 训练结束后，保存模型参数（仅保存参数，不保存完整网络，占用空间小）
    torch.save(policy_net.state_dict(), save_path)
    print(f"\n模型参数已保存到：{save_path}")
    env.close()

    return episode_rewards

# --------------------------
# 执行入口：直接运行 train.py 时启动训练
# --------------------------
if __name__ == "__main__":
    # 1. 网络结构参数（增强模型表达能力）
    OBS_DIM = 8
    ACTION_DIM = 4
    hidden_layers = [256,128]
    activation = 'relu'  # 保留ReLU（对RL任务更高效）

    # 2. 训练超参数（降低方差+稳定梯度）
    LR = 0.001
    GAMMA = 0.99  # 略微降低折扣因子（让智能体更关注近期动作，减少长期回报的方差）
    MAX_EPISODES = 5000  # 增加训练轮数
    SAVE_PATH = "./lunar_lander_policy_gradient.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备：{device}（若显示 cuda:0 表示使用 GPU）")

    # 3. 创建 Policy 网络实例（搭建两层 MLP，符合任务要求）
    policy_net = PolicyNetwork(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_layers=hidden_layers,
        activation=activation,
        device=device
    )

    # 4. 启动训练（调用 train 函数，传入网络、设备和超参数）
    print("="*50)
    print("开始训练 LunarLander-v3 智能体...")
    print(f"超参数配置：学习率={LR}，隐藏层为={hidden_layers}，最大回合数={MAX_EPISODES}")
    print("="*50)
    episode_rewards = train(
        policy_net=policy_net,
        device=device,  # 传入训练设备
        save_path=SAVE_PATH,
        lr=LR,
        gamma=GAMMA,
        max_episodes=MAX_EPISODES
    )

    # 5. 训练结束后，绘制「训练回合得分曲线」
    plt.figure(figsize=(12, 6))
    # 配置 Matplotlib 支持中文（解决中文显示警告和乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认中文字体（Windows 用 SimHei/黑体）
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    # 绘制每轮得分曲线
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, color="#1f77b4", linewidth=1.5, alpha=0.7, label="单轮得分")
    # 绘制10轮滑动平均曲线（更清晰看趋势）
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode="valid")
        plt.plot(range(100, len(moving_avg) + 100), moving_avg, color="#ff7f0e", linewidth=2, label="100轮滑动平均")
    # 图表标注
    plt.xlabel("训练回合数", fontsize=12)
    plt.ylabel("回合得分", fontsize=12)
    plt.title("LunarLander-v3 训练回合得分变化曲线", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig("train_reward_curve.png", dpi=300, bbox_inches="tight")  # 保存图片
    plt.show()

    print("="*50)
    print(f"训练完成！模型已保存到：{SAVE_PATH}")
    print(f"训练回合数：{len(episode_rewards)}")
    print(f"最终100轮平均得分：{np.mean(episode_rewards[-100:]) if len(episode_rewards)>=100 else episode_rewards[-1]}")
    print(f"最终50轮平均得分：{np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_rewards[-1]}")
    print(f"最终20轮平均得分：{np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else episode_rewards[-1]}")
    print(f"最终10轮平均得分：{np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]}")
    print("="*50)
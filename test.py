"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/24
@Desc    : PG模型测试脚本（加载训练好的权重，验证LunarLander-v3性能）
@License : MIT License (MIT)
@Version : 1.0
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from policy_network import PolicyNetwork  # 导入你的策略网络类

def load_policy_model(model_path, obs_dim=8, action_dim=4, hidden_layers=[256,128], activation='relu'):
    """
    加载训练好的策略网络权重
    :param model_path: 模型权重文件路径（如 "./lunar_lander_policy_gradient.pth"）
    :param obs_dim: 状态维度（固定8）
    :param action_dim: 动作维度（固定4）
    :param hidden_layers: 隐藏层配置（需与训练时一致！）
    :param activation: 激活函数（需与训练时一致！）
    :return: 加载好权重的PolicyNetwork实例（eval模式）
    """
    # 1. 确定设备（与训练时一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备：{device}")

    # 2. 创建网络实例（结构需与训练完全匹配）
    policy_net = PolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        device=device
    )

    # 3. 加载权重（处理CPU/GPU兼容）
    try:
        # 兼容GPU训练、CPU测试的场景
        state_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(state_dict)
        print(f"✅ 模型权重加载成功：{model_path}")
    except Exception as e:
        raise ValueError(f"❌ 模型加载失败：{e}\n请确认模型路径正确，且网络结构与训练时一致！")

    # 4. 切换到评估模式（禁用Dropout/BN等训练层，不影响你的MLP，但规范）
    policy_net.eval()
    return policy_net

def test_policy(policy_net, num_test_episodes=10, render=True):
    """
    测试策略网络性能
    :param policy_net: 加载好的PolicyNetwork实例
    :param num_test_episodes: 测试回合数（推荐≥10，统计更稳定）
    :param render: 是否渲染画面（True=可视化测试，False=快速测试）
    :return: test_rewards（每回合得分列表）、avg_reward（平均得分）
    """
    # 初始化测试环境（render_mode控制是否显示画面）
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    test_rewards = []
    print(f"\n🚀 开始测试（共{num_test_episodes}回合）...")

    # 禁用梯度计算（测试时无需反向传播，提升速度）
    with torch.no_grad():
        for episode in range(num_test_episodes):
            current_state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # 测试时用「最优动作」（无随机探索），体现策略真实性能
                action = policy_net.get_best_action(current_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                current_state = next_state

                # 渲染时增加延迟，方便观察（可选）
                if render:
                    import time
                    time.sleep(0.01)

            test_rewards.append(total_reward)
            print(f"测试回合 {episode+1:2d} | 得分：{total_reward:6.2f}")

    env.close()

    # 计算统计指标
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)  # 得分标准差，反映稳定性
    print(f"\n📊 测试结果统计：")
    print(f"平均得分：{avg_reward:6.2f} | 得分标准差：{std_reward:5.2f}")
    print(f"最高得分：{np.max(test_rewards):6.2f} | 最低得分：{np.min(test_rewards):6.2f}")

    return test_rewards, avg_reward

def plot_test_results(test_rewards, avg_reward, save_path="test_reward_plot.png"):
    """
    绘制测试得分可视化图表
    :param test_rewards: 每回合得分列表
    :param avg_reward: 平均得分
    :param save_path: 图表保存路径
    """
    plt.figure(figsize=(10, 6))
    # 配置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制每回合得分
    x = range(1, len(test_rewards)+1)
    plt.plot(x, test_rewards, color="#1f77b4", linewidth=2, marker='o', label="单回合得分")
    # 绘制平均得分水平线
    plt.axhline(y=avg_reward, color="#ff7f0e", linewidth=2, linestyle='--', label=f"平均得分 ({avg_reward:.2f})")
    # 绘制达标线（200分）
    plt.axhline(y=200, color="#2ca02c", linewidth=1.5, linestyle=':', label="达标线（200分）")

    # 图表标注
    plt.xlabel("测试回合数", fontsize=12)
    plt.ylabel("回合得分", fontsize=12)
    plt.title("LunarLander-v3 策略测试得分分布", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(x)  # 强制显示所有测试回合

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📸 测试图表已保存：{save_path}")
    plt.show()

if __name__ == "__main__":
    # ===================== 配置参数（需与训练时一致） =====================
    MODEL_PATH = "./lunar_lander_policy_gradient.pth"  # 训练好的模型路径
    HIDDEN_LAYERS = [256, 128]  # 需与训练时的hidden_layers完全一致
    ACTIVATION = 'relu'         # 需与训练时的激活函数一致
    NUM_TEST_EPISODES = 10      # 测试回合数（建议10-20）
    RENDER_TEST = True          # 是否可视化测试（True=看画面，False=快速跑）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 执行测试流程 =====================
    # 1. 加载模型
    policy_net = load_policy_model(
        model_path=MODEL_PATH,
        hidden_layers=HIDDEN_LAYERS,
        activation=ACTIVATION
    ).to(device)

    # 2. 测试策略
    test_rewards, avg_reward = test_policy(
        policy_net=policy_net,
        num_test_episodes=NUM_TEST_EPISODES,
        render=RENDER_TEST
    )

    # 3. 绘制测试结果
    plot_test_results(test_rewards, avg_reward)
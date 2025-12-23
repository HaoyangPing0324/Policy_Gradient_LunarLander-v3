"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/23
@Desc    : LunarLander-v3 模型可视化演示脚本（实时渲染着陆过程）
@License : MIT License (MIT)
@Version : 1.0
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import torch
import gymnasium as gym
import time
import numpy as np
from policy_network import PolicyNetwork  # 导入你的策略网络

def load_trained_model(model_path="./lunar_lander_policy_gradient.pth"):
    """加载训练好的模型（与训练时的网络结构严格一致）"""
    # 固定参数（需与训练时完全匹配）
    OBS_DIM = 8
    ACTION_DIM = 4
    HIDDEN_LAYERS = [256, 128]
    ACTIVATION = 'relu'

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"演示设备：{device}")

    # 创建网络并加载权重
    policy_net = PolicyNetwork(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_layers=HIDDEN_LAYERS,
        activation=ACTIVATION,
        device=device
    )

    try:
        state_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(state_dict)
        policy_net.eval()  # 切换到评估模式
        print(f"✅ 模型加载成功：{model_path}")
        return policy_net
    except Exception as e:
        raise ValueError(f"❌ 模型加载失败：{e}\n请确认模型文件存在且网络结构匹配！")

def demo_lander(policy_net, num_demos=3, render_delay=0.01):
    """可视化演示飞船着陆"""
    # 创建带渲染的环境（human模式显示窗口）
    env = gym.make("LunarLander-v3", render_mode="human")
    print(f"\n🎮 开始可视化演示（共{num_demos}回合）...")
    print("提示：演示窗口可手动关闭，按Ctrl+C终止程序")

    # 禁用梯度计算（提升演示速度）
    with torch.no_grad():
        for demo_round in range(num_demos):
            print(f"\n===== 演示回合 {demo_round + 1} =====")
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                # 选择最优动作（无随机探索，体现模型真实能力）
                action = policy_net.get_best_action(state)
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # 累计奖励+计数
                total_reward += reward
                step += 1

                # 控制演示速度（避免画面过快）
                time.sleep(render_delay)

                # 更新状态
                state = next_state

                # 打印每步信息（可选）
                if step % 50 == 0:
                    print(f"  步数 {step} | 当前得分：{total_reward:.2f}")

            # 回合结束统计
            land_status = "成功着陆" if total_reward > 100 else "坠毁/未达标"
            print(f"演示回合 {demo_round + 1} 结束 | 总步数：{step} | 最终得分：{total_reward:.2f} | 状态：{land_status}")

    # 关闭环境
    env.close()
    print("\n🎉 所有演示回合结束！")

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "./lunar_lander_policy_gradient.pth"  # 训练好的模型路径
    NUM_DEMOS = 3  # 演示回合数（建议1-5）
    RENDER_DELAY = 0.01  # 画面延迟（越小越快，0.01为流畅速度）

    # 执行演示流程
    try:
        # 1. 加载模型
        policy_net = load_trained_model(MODEL_PATH)
        # 2. 可视化演示
        demo_lander(policy_net, num_demos=NUM_DEMOS, render_delay=RENDER_DELAY)
    except KeyboardInterrupt:
        print("\n⚠️ 演示被手动终止")
    except Exception as e:
        print(f"\n❌ 演示出错：{e}")
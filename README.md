# 项目名称：使用 Policy Gradient 方法训练 LunarLander-v3 环境智能体

## 项目来源
苏州大学未来科学与工程学院《人工智能》课程
任课老师：吴洪状

## 任务描述
LunarLander 是 OpenAI Gym 中的经典环境，模拟一个着陆器在月球表面软着陆的过程。目标是在着陆器不翻倒的情况下，平稳地降落在着陆点上。使用 PyTorch 实现基于 Policy Gradient 的强化学习算法，训练智能体在 LunarLander-v3 环境中获得高分。

## 任务要求
### 一、训练部分
1.搭建一个基于 PyTorch 的 Policy 网络（可以是简单的两层 MLP）。
2.使用 REINFORCE 算法训练智能体，具体包括：
o蒙特卡洛采样整条轨迹；
o计算每个动作对应的回报 $G_tG_tGt​$；
o使用 log-probability 和回报计算损失函数；
o执行梯度更新。

### 二、测试部分
1.加载训练好的策略网络。
2.运行若干个测试回合，计算平均得分。
3.绘图：回合得分随训练轮数的变化曲线。

### 三、演示部分
1.利用 gym.make('LunarLander-v3', render_mode='human')，展示训练好的策略在环境中的运行效果。

### 具体要求：
1、安装并使用gym库。pip install gym[box2d]
2、使用多文件结构：建立train.py/test.py/demo.py等文件，将训练的模型保存在本地，运行test和demo文件时加载模型
3、引入 baseline（例如平均回报）减少方差，即使用折扣累积奖励减去基线作为评价标准
4、模型结构和超参数支持配置（例如学习率、隐藏层大小）

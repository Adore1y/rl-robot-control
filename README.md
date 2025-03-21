# RL-Robot-Control

基于深度强化学习的工业机器人智能控制系统。该项目使用PPO等先进的强化学习算法，实现了机器人在工业环境中的智能控制。

## 项目结构

```
rl-robot-control/
│
├── agents/                  # 强化学习代理实现
│   ├── __init__.py
│   ├── base_agent.py        # 代理基类
│   └── ppo_agent.py         # PPO算法实现
│
├── configs/                 # 配置文件
│   └── ppo_config.yaml      # PPO算法配置
│
├── environments/            # 仿真环境
│   ├── __init__.py
│   ├── robot_env.py         # 机器人环境基类
│   └── tasks/               # 特定任务实现
│       ├── __init__.py
│       └── pick_place.py    # 抓取与放置任务
│
├── experiments/             # 实验脚本
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   └── compare_algorithms.py# 算法对比
│
├── logs/                    # 日志文件
│
├── models/                  # 保存的模型
│
├── notebooks/               # Jupyter笔记本
│
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   ├── memory.py            # 经验回放
│   └── visualizer.py        # 可视化工具
│
├── .gitignore               # Git忽略文件
├── LICENSE                  # 许可证
├── README.md                # 项目说明
├── requirements.txt         # 项目依赖
└── setup.py                 # 安装脚本
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/rl-robot-control.git
cd rl-robot-control

# 安装依赖
pip install -e .
```

## 使用方法

### 训练模型

```bash
python experiments/train.py --config configs/ppo_config.yaml
```

### 评估模型

```bash
python experiments/evaluate.py --model-path models/ppo_model.pth
```

### 算法对比

```bash
python experiments/compare_algorithms.py
```

## 项目特点

- 基于PyTorch实现的强化学习算法
- 使用PyBullet进行物理仿真
- 支持多种工业机器人任务
- 完整的训练、评估和可视化流程
- 模块化设计，易于扩展

## 依赖项

- Python 3.8+
- PyTorch 1.10+
- Gymnasium
- PyBullet
- NumPy
- Matplotlib
- PyYAML
- 更多依赖见requirements.txt

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT 
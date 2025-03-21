# 基于深度强化学习的工业机器人智能控制系统

本项目实现了一个基于深度强化学习的工业机器人控制系统，能够在复杂环境下实现自适应控制与优化决策，提高机器人操作精度与任务完成效率。

## 主要特点

- 基于PyTorch实现多种深度强化学习算法(PPO, SAC, TD3)
- 使用PyBullet构建高精度的工业机器人仿真环境
- 支持多种工业任务场景(抓取放置、轨迹跟踪等)
- 完整的实验分析与可视化工具
- 模块化设计，易于扩展和定制

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/rl-robot-control.git
cd rl-robot-control

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

## 快速开始

### 训练一个智能体

```bash
python experiments/train.py --config configs/ppo_config.yaml --task pick_place
```

### 评估训练好的模型

```bash
python experiments/evaluate.py --model models/ppo_pick_place_best.pt --episodes 10 --render
```

### 算法对比实验

```bash
python experiments/compare_algorithms.py --task tracking --algorithms ppo sac td3 --trials 5
```

## 项目结构

- `environments/`: 机器人环境定义
- `agents/`: 强化学习算法实现
- `utils/`: 工具函数
- `experiments/`: 实验脚本
- `configs/`: 配置文件
- `notebooks/`: 分析笔记本

## 引用

如果您在研究中使用了本项目，请引用:

```
@article{rl_robot_control2023,
  title={基于深度强化学习的工业机器人自适应控制系统研究与实现},
  author={Your Name},
  journal={},
  year={2023}
}
``` 
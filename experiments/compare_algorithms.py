import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# 导入自定义环境和智能体
from environments import PickPlaceEnv, TrackingEnv
from agents import PPOAgent, SACAgent, TD3Agent
from utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度强化学习算法对比实验')
    
    parser.add_argument('--task', type=str, required=True, choices=['pick_place', 'tracking'], help='任务类型')
    parser.add_argument('--algorithms', nargs='+', default=['ppo', 'sac', 'td3'], help='要对比的算法')
    parser.add_argument('--config_dir', type=str, default='configs', help='配置文件目录')
    parser.add_argument('--trials', type=int, default=5, help='每个算法的重复次数')
    parser.add_argument('--episodes', type=int, default=20, help='每次试验的回合数')
    parser.add_argument('--steps', type=int, default=5000, help='每个回合的最大步数')
    parser.add_argument('--seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--result_dir', type=str, default='results', help='结果保存目录')
    
    return parser.parse_args()

def load_configs(config_dir, algorithms):
    """加载算法配置"""
    configs = {}
    
    for algo in algorithms:
        config_path = os.path.join(config_dir, f"{algo}_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                configs[algo] = config.get(algo, {})
        else:
            print(f"警告: {config_path} 不存在，使用默认配置")
            configs[algo] = {}
    
    return configs

def create_env(task):
    """创建环境"""
    if task == 'pick_place':
        env = PickPlaceEnv(render_mode=None)
    elif task == 'tracking':
        env = TrackingEnv(render_mode=None)
    else:
        raise ValueError(f"未知任务类型: {task}")
    
    return env

def create_agent(algo, state_dim, action_dim, max_action, min_action, config):
    """创建智能体"""
    if algo == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action,
            hidden_dim=config.get('hidden_dim', 256),
            lr=config.get('lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_ratio=config.get('clip_ratio', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            update_epochs=config.get('update_epochs', 10)
        )
    elif algo == 'sac':
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action,
            hidden_dim=config.get('hidden_dim', 256),
            lr=config.get('lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            alpha=config.get('alpha', 0.2),
            auto_alpha=config.get('auto_alpha', True)
        )
    elif algo == 'td3':
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action,
            hidden_dim=config.get('hidden_dim', 256),
            lr=config.get('lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            policy_noise=config.get('policy_noise', 0.2),
            noise_clip=config.get('noise_clip', 0.5),
            policy_delay=config.get('policy_delay', 2)
        )
    else:
        raise ValueError(f"未知算法类型: {algo}")
    
    return agent

def run_trial(env, agent, episodes, max_steps):
    """运行单次试验"""
    episode_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
    
    return episode_rewards

def plot_comparison_results(results, result_dir):
    """绘制算法对比结果"""
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 8))
    
    # 创建数据框
    data = []
    for algo, trials in results.items():
        for trial_idx, rewards in enumerate(trials):
            for episode_idx, reward in enumerate(rewards):
                data.append({
                    'Algorithm': algo,
                    'Trial': trial_idx + 1,
                    'Episode': episode_idx + 1,
                    'Reward': reward
                })
    
    df = pd.DataFrame(data)
    
    # 按算法计算平均奖励
    algo_summary = df.groupby(['Algorithm', 'Episode'])['Reward'].agg(['mean', 'std']).reset_index()
    
    # 绘制平均奖励曲线
    for algo in results.keys():
        algo_data = algo_summary[algo_summary['Algorithm'] == algo]
        plt.plot(algo_data['Episode'], algo_data['mean'], label=algo.upper())
        plt.fill_between(
            algo_data['Episode'],
            algo_data['mean'] - algo_data['std'],
            algo_data['mean'] + algo_data['std'],
            alpha=0.2
        )
    
    plt.title('Algorithm Comparison: Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计数据
    summary = df.groupby('Algorithm')['Reward'].agg(['mean', 'std', 'max', 'min']).reset_index()
    summary.to_csv(os.path.join(result_dir, 'algorithm_summary.csv'), index=False)
    
    # 创建箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Algorithm', y='Reward', data=df)
    plt.title('Algorithm Comparison: Reward Distribution')
    plt.savefig(os.path.join(result_dir, 'algorithm_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"比较结果已保存到 {result_dir}")
    print("\n算法性能统计:")
    print(summary)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置基础随机种子
    base_seed = args.seed
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, f"{args.task}_compare_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(result_dir)
    logger.info(f"任务: {args.task}, 算法: {args.algorithms}, 试验次数: {args.trials}")
    
    # 加载算法配置
    configs = load_configs(args.config_dir, args.algorithms)
    logger.info(f"算法配置: {configs}")
    
    # 创建环境并获取环境信息
    env = create_env(args.task)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    logger.info(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 运行试验
    results = {algo: [] for algo in args.algorithms}
    
    for algo in args.algorithms:
        logger.info(f"开始评估算法: {algo}")
        
        for trial in range(args.trials):
            # 为每个试验设置不同的种子
            trial_seed = base_seed + trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            
            logger.info(f"  试验 {trial+1}/{args.trials}, 种子: {trial_seed}")
            
            # 创建智能体
            agent = create_agent(algo, state_dim, action_dim, max_action, min_action, configs[algo])
            
            # 运行试验
            trial_rewards = run_trial(env, agent, args.episodes, args.steps)
            results[algo].append(trial_rewards)
            
            # 记录试验结果
            logger.info(f"  平均奖励: {np.mean(trial_rewards):.2f}, 最大奖励: {np.max(trial_rewards):.2f}")
    
    # 绘制对比结果
    plot_comparison_results(results, result_dir)
    
    # 关闭环境
    env.close()
    
    logger.info(f"所有实验完成，结果已保存到 {result_dir}")

if __name__ == "__main__":
    main() 
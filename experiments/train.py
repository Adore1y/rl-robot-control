import argparse
import os
import yaml
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from collections import deque

# 导入自定义环境和智能体
from environments import PickPlaceEnv, TrackingEnv
from agents import PPOAgent, SACAgent, TD3Agent
from utils.logger import setup_logger
from utils.memory import RolloutBuffer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度强化学习机器人控制训练脚本')
    
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--task', type=str, required=True, choices=['pick_place', 'tracking'], help='任务类型')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac', 'td3'], help='算法类型')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--total_steps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--eval_freq', type=int, default=10000, help='评估频率')
    parser.add_argument('--save_freq', type=int, default=50000, help='保存模型频率')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_env(task, render=False):
    """创建环境"""
    render_mode = "human" if render else None
    
    if task == 'pick_place':
        env = PickPlaceEnv(render_mode=render_mode)
    elif task == 'tracking':
        env = TrackingEnv(render_mode=render_mode)
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

def evaluate(agent, env, n_episodes=10):
    """评估智能体性能"""
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    algo_config = config.get(args.algo, {})
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.task}_{args.algo}_{timestamp}")
    model_dir = os.path.join(args.model_dir, f"{args.task}_{args.algo}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(log_dir)
    logger.info(f"任务: {args.task}, 算法: {args.algo}, 种子: {args.seed}")
    logger.info(f"配置: {algo_config}")
    
    # 创建环境
    env = create_env(args.task, args.render)
    eval_env = create_env(args.task, False)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    logger.info(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    logger.info(f"动作范围: {min_action} ~ {max_action}")
    
    # 创建智能体
    agent = create_agent(args.algo, state_dim, action_dim, max_action, min_action, algo_config)
    agent.init_logging(log_dir)
    
    # 创建经验回放缓冲区
    if args.algo == 'ppo':
        memory = RolloutBuffer()
    else:
        # 这里需要根据SAC/TD3算法实现ReplayBuffer
        memory = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    
    # 训练参数
    steps = 0
    episodes = 0
    best_reward = -float('inf')
    episode_reward = 0
    episode_steps = 0
    
    # 获取初始状态
    obs, _ = env.reset(seed=args.seed)
    
    # 训练循环
    while steps < args.total_steps:
        # 选择动作
        if steps < config.get('random_steps', 10000) and args.algo != 'ppo':
            # 使用随机动作进行探索
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        
        # 存储经验
        if args.algo == 'ppo':
            memory.add(obs, action, reward, next_obs, float(done or truncated))
        else:
            memory.add(obs, action, reward, next_obs, float(done or truncated))
        
        # 更新状态
        obs = next_obs
        steps += 1
        agent.total_steps += 1
        
        # 如果回合结束，重置环境
        if done or truncated:
            logger.info(f"回合 {episodes+1} 结束，步数: {episode_steps}, 奖励: {episode_reward:.2f}")
            agent.log_episode_info({
                'reward': episode_reward,
                'steps': episode_steps
            })
            
            # 重置环境和状态
            obs, _ = env.reset()
            episodes += 1
            agent.episodes += 1
            episode_reward = 0
            episode_steps = 0
        
        # 更新智能体
        if args.algo == 'ppo':
            # PPO使用全部收集的经验进行批量更新
            if memory.size() >= config.get('batch_size', 2048):
                agent.train(memory, batch_size=config.get('minibatch_size', 64))
                memory.clear()
        else:
            # SAC和TD3使用经验回放进行更新
            if steps >= config.get('warmup_steps', 1000):
                agent.train(memory, batch_size=config.get('batch_size', 256))
        
        # 定期评估
        if steps % args.eval_freq == 0:
            eval_stats = evaluate(agent, eval_env)
            logger.info(f"评估: 步数 {steps}, 平均奖励: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            
            # 记录评估信息
            agent.log_training_info({
                'eval/mean_reward': eval_stats['mean_reward'],
                'eval/std_reward': eval_stats['std_reward'],
                'eval/min_reward': eval_stats['min_reward'],
                'eval/max_reward': eval_stats['max_reward']
            })
            
            # 保存最佳模型
            if eval_stats['mean_reward'] > best_reward:
                best_reward = eval_stats['mean_reward']
                best_model_path = os.path.join(model_dir, f"best_model.pt")
                agent.save(best_model_path)
                logger.info(f"新的最佳模型已保存到 {best_model_path}")
        
        # 定期保存模型
        if steps % args.save_freq == 0:
            model_path = os.path.join(model_dir, f"model_{steps}.pt")
            agent.save(model_path)
            logger.info(f"模型已保存到 {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model.pt")
    agent.save(final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 
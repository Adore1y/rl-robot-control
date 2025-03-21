#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
import logging
import time
from datetime import datetime

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import PickPlaceEnv, TrackingEnv
from agents import PPOAgent, SACAgent, TD3Agent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_env(task_name, render=False):
    """
    创建指定任务的环境
    
    Args:
        task_name: 任务名称 ['pick_place', 'tracking']
        render: 是否渲染环境
        
    Returns:
        gym环境
    """
    render_mode = "human" if render else None
    
    if task_name == "pick_place":
        env = PickPlaceEnv(render_mode=render_mode)
    elif task_name == "tracking":
        env = TrackingEnv(render_mode=render_mode)
    else:
        raise ValueError(f"不支持的任务: {task_name}")
    
    return env

def load_agent(model_path, algo, state_dim, action_dim, action_bounds=None):
    """
    加载指定算法的智能体
    
    Args:
        model_path: 模型文件路径
        algo: 算法名称 ['ppo', 'sac', 'td3']
        state_dim: 状态维度
        action_dim: 动作维度
        action_bounds: 动作范围 [min, max]
        
    Returns:
        强化学习智能体
    """
    if algo == "ppo":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=action_bounds[1],
            min_action=action_bounds[0],
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            update_epochs=10
        )
    elif algo == "sac":
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_tune=True
        )
    elif algo == "td3":
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bounds=action_bounds,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2
        )
    else:
        raise ValueError(f"不支持的算法: {algo}")
    
    # 加载模型
    agent.load(model_path)
    
    return agent

def evaluate(env, agent, num_episodes=10):
    """
    评估智能体在环境中的表现
    
    Args:
        env: 环境
        agent: 智能体
        num_episodes: 评估回合数
        
    Returns:
        平均奖励
    """
    rewards = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            obs = next_obs
            
            # 如果使用了GUI模式，添加一点延迟使可视化更流畅
            if env.render_mode == "human":
                time.sleep(0.01)
        
        rewards.append(ep_reward)
        logger.info(f"回合 {i+1}: 奖励 = {ep_reward:.2f}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    logger.info(f"评估完成: 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估强化学习模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--task", type=str, default="tracking", choices=["pick_place", "tracking"], help="任务名称")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"], help="算法名称")
    parser.add_argument("--num_episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    
    args = parser.parse_args()
    
    # 创建环境
    env = create_env(args.task, args.render)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取状态和动作信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = [
        env.action_space.low,
        env.action_space.high
    ]
    
    logger.info(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    logger.info(f"动作范围: {action_bounds[0]} ~ {action_bounds[1]}")
    
    # 加载智能体
    agent = load_agent(
        model_path=args.model_path,
        algo=args.algo,
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=action_bounds
    )
    
    # 评估智能体
    evaluate(env, agent, args.num_episodes)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main() 
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import time
from datetime import datetime

# 导入自定义环境和智能体
from environments import PickPlaceEnv, TrackingEnv
from agents import PPOAgent, SACAgent, TD3Agent
from utils.visualizer import plot_evaluation_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度强化学习机器人控制评估脚本')
    
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--task', type=str, required=True, choices=['pick_place', 'tracking'], help='任务类型')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac', 'td3'], help='算法类型')
    parser.add_argument('--episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--save_video', action='store_true', help='是否保存视频')
    parser.add_argument('--video_dir', type=str, default='videos', help='视频保存目录')
    
    return parser.parse_args()

def create_env(task, render=False, record=False, video_dir=None):
    """创建环境"""
    render_mode = "human" if render else "rgb_array" if record else None
    
    if task == 'pick_place':
        env = PickPlaceEnv(render_mode=render_mode)
    elif task == 'tracking':
        env = TrackingEnv(render_mode=render_mode)
    else:
        raise ValueError(f"未知任务类型: {task}")
    
    # 如果需要录制，使用环境包装器
    if record and video_dir:
        try:
            from gymnasium.wrappers import RecordVideo
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env, 
                video_folder=video_dir,
                episode_trigger=lambda x: True,  # 录制所有回合
                name_prefix=f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except ImportError:
            print("警告: 无法导入RecordVideo，视频录制已禁用")
    
    return env

def create_agent(algo, state_dim, action_dim, max_action, min_action):
    """创建智能体"""
    if algo == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action
        )
    elif algo == 'sac':
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action
        )
    elif algo == 'td3':
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action
        )
    else:
        raise ValueError(f"未知算法类型: {algo}")
    
    return agent

def evaluate(agent, env, episodes=10):
    """评估智能体性能"""
    episode_rewards = []
    episode_steps = []
    success_rate = 0.0
    
    for episode in range(episodes):
        start_time = time.time()
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            obs = next_obs
        
        # 记录结果
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # 检查是否成功完成任务
        if info.get('success', False) or episode_reward > 0:
            success_rate += 1.0
        
        # 打印回合信息
        elapsed_time = time.time() - start_time
        print(f"回合 {episode+1}/{episodes}, 奖励: {episode_reward:.2f}, 步数: {steps}, 时间: {elapsed_time:.2f}秒")
    
    # 计算统计信息
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_steps = np.mean(episode_steps)
    success_rate = (success_rate / episodes) * 100.0
    
    print(f"\n===== 评估结果 =====")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {mean_steps:.2f}")
    print(f"成功率: {success_rate:.2f}%")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_steps': mean_steps,
        'success_rate': success_rate,
        'rewards': episode_rewards,
        'steps': episode_steps
    }

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建目录
    if args.save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(args.video_dir, f"{args.task}_{args.algo}_{timestamp}")
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None
    
    # 创建环境
    env = create_env(args.task, args.render, args.save_video, video_dir)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"动作范围: {min_action} ~ {max_action}")
    
    # 创建智能体
    agent = create_agent(args.algo, state_dim, action_dim, max_action, min_action)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    agent.load(args.model)
    
    # 评估智能体
    print(f"开始评估 {args.episodes} 个回合...")
    eval_results = evaluate(agent, env, args.episodes)
    
    # 如果需要，绘制评估结果
    if args.save_video:
        plot_path = os.path.join(video_dir, "evaluation_results.png")
        plot_evaluation_results(eval_results, plot_path)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main() 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from agents.base_agent import BaseAgent

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic网络
    包含策略网络(Actor)和价值网络(Critic)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络 - 输出动作均值和标准差
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # 价值网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 当前状态
            
        Returns:
            action_mean: 动作分布的均值
            action_std: 动作分布的标准差
            value: 状态价值估计
        """
        features = self.feature_extractor(state)
        
        # 计算动作分布参数
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # 计算状态价值
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 所选动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        action_mean, action_std, value = self.forward(state)
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        
        # 采样动作或使用均值
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_action(self, state, action):
        """
        评估给定状态和动作的价值和概率
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            log_prob: 动作的对数概率
            value: 状态价值估计
            entropy: 策略的熵
        """
        action_mean, action_std, value = self.forward(state)
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # 计算策略熵
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class PPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization)智能体
    使用Actor-Critic架构和CLIP目标
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        min_action,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作上限
            min_action: 动作下限
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_ratio: PPO裁剪比率
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            update_epochs: 每批数据的更新次数
            device: 计算设备
        """
        super(PPOAgent, self).__init__(state_dim, action_dim, max_action, min_action, device)
        
        # PPO超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        # 创建Actor-Critic网络
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否处于评估模式
            
        Returns:
            所选动作
        """
        # 预处理状态
        state = self.preprocess_state(state)
        
        # 使用网络获取动作
        with torch.no_grad():
            action, _, _ = self.actor_critic.get_action(state, deterministic=evaluate)
        
        # 后处理动作
        return self.postprocess_action(action)
    
    def train(self, memory, batch_size=64):
        """
        使用收集的经验更新智能体
        
        Args:
            memory: 收集的经验
            batch_size: 批次大小
            
        Returns:
            训练信息字典
        """
        # 获取经验数据
        states = torch.FloatTensor(memory.states).to(self.device)
        actions = torch.FloatTensor(memory.actions).to(self.device)
        rewards = torch.FloatTensor(memory.rewards).to(self.device)
        next_states = torch.FloatTensor(memory.next_states).to(self.device)
        dones = torch.FloatTensor(memory.dones).to(self.device)
        
        # 计算旧的动作对数概率和状态价值
        with torch.no_grad():
            old_log_probs, _, _ = self.actor_critic.evaluate_action(states, actions)
            
            # 计算优势函数 (使用GAE)
            advantages = self._compute_gae(rewards, dones, states, next_states)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算价值目标
            returns = advantages + values
        
        # 多个epoch的训练
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # 随机采样批次
            indices = torch.randperm(states.size(0))
            
            for start in range(0, states.size(0), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略和价值
                new_log_probs, values, entropy = self.actor_critic.evaluate_action(
                    batch_states, batch_actions
                )
                
                # 计算策略比率和裁剪后的目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(values, batch_returns)
                
                # 计算熵损失
                entropy_loss = -entropy.mean()
                
                # 计算总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # 计算平均损失
        n_updates = self.update_epochs * ((states.size(0) + batch_size - 1) // batch_size)
        avg_loss = total_loss / n_updates
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        
        # 更新训练步数
        self.training_steps += 1
        
        # 返回训练信息
        info = {
            'loss/total': avg_loss,
            'loss/policy': avg_policy_loss,
            'loss/value': avg_value_loss,
            'loss/entropy': avg_entropy
        }
        
        # 记录训练信息
        self.log_training_info(info)
        
        return info
    
    def _compute_gae(self, rewards, dones, states, next_states):
        """
        计算GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励
            dones: 终止标志
            states: 状态
            next_states: 下一个状态
            
        Returns:
            advantages: 优势估计
        """
        with torch.no_grad():
            # 获取当前状态和下一个状态的价值
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)
            
            # 初始化
            advantages = torch.zeros_like(rewards)
            gae = 0
            
            # 从后向前计算GAE
            for t in reversed(range(len(rewards))):
                # 如果是终止状态，下一个状态的价值为0
                if t == len(rewards) - 1:
                    next_val = 0
                else:
                    next_val = next_values[t+1]
                
                # 计算TD误差
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
                
                # 计算GAE
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                
                # 存储优势
                advantages[t] = gae
        
        return advantages
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'total_steps': self.total_steps,
            'episodes': self.episodes
        }, path)
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.total_steps = checkpoint['total_steps']
        self.episodes = checkpoint['episodes'] 
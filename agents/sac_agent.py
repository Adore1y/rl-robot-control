import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from agents.base_agent import BaseAgent

class Actor(nn.Module):
    """
    SAC Actor网络
    输出动作均值和标准差
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.bias)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态输入
            
        Returns:
            mu: 动作均值
            std: 动作标准差
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        
        # 将log_std限制在一个合理的范围内
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mu, std
    
    def sample(self, state):
        """
        从策略分布中采样动作
        
        Args:
            state: 状态输入
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mu, std = self.forward(state)
        
        # 创建正态分布
        dist = Normal(mu, std)
        
        # 重参数化采样
        x = dist.rsample()
        
        # 应用tanh压缩到[-1, 1]
        y = torch.tanh(x)
        
        # 缩放到动作空间
        action = y * self.max_action
        
        # 计算对数概率 (考虑tanh变换的Jacobian)
        log_prob = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """
    SAC Critic网络
    评估状态-动作对的Q值
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 (双Q网络)
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        
        for m in [self.q1, self.q2]:
            nn.init.uniform_(m.weight, -3e-3, 3e-3)
            nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        """
        前向传播 (双Q网络)
        
        Args:
            state: 状态输入
            action: 动作输入
            
        Returns:
            q1_value: 第一个Q网络的输出
            q2_value: 第二个Q网络的输出
        """
        # 合并状态和动作
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1_value = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2_value = self.q2(q2)
        
        return q1_value, q2_value


class SACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) 智能体
    使用状态值函数和熵正则化的actor-critic算法
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
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作上限
            min_action: 动作下限
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            alpha: 熵系数
            auto_alpha: 是否自动调整熵系数
            device: 计算设备
        """
        super(SACAgent, self).__init__(state_dim, action_dim, max_action, min_action, device)
        
        # SAC超参数
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        
        # 动作缩放
        self.action_scale = torch.FloatTensor(
            (max_action - min_action) / 2.0
        ).to(device)
        self.action_bias = torch.FloatTensor(
            (max_action + min_action) / 2.0
        ).to(device)
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action=1.0).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 创建Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 创建目标Critic网络
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 自动调整熵系数
        if auto_alpha:
            # 目标熵 (根据动作空间维度自动设置)
            self.target_entropy = -action_dim
            
            # 可学习的log_alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否处于评估模式
            
        Returns:
            所选动作
        """
        state = self.preprocess_state(state)
        
        with torch.no_grad():
            if evaluate:
                # 在评估模式下，使用均值作为动作
                mu, _ = self.actor(state)
                action = torch.tanh(mu) * self.action_scale + self.action_bias
            else:
                # 在训练模式下，从分布中采样
                action, _ = self.actor.sample(state)
                # 缩放到实际动作空间
                action = action * self.action_scale + self.action_bias
        
        return self.postprocess_action(action)
    
    def train(self, replay_buffer, batch_size=256):
        """
        使用经验回放缓冲区的样本更新智能体
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            
        Returns:
            训练信息字典
        """
        # 从缓冲区中采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # 转换为张量
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # 计算当前alpha值
        if self.auto_alpha:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = self.alpha
        
        # 更新Critic
        with torch.no_grad():
            # 从目标策略中采样下一个动作和对数概率
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # 使用目标Critic评估下一个状态-动作对
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            
            # 计算目标Q值
            target_q = reward + (1 - done) * self.gamma * (next_q - alpha * next_log_prob)
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_action, new_log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        
        # 计算Actor损失
        actor_loss = (alpha * new_log_prob - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新Alpha (如果启用自动调整)
        if self.auto_alpha:
            alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            alpha = torch.exp(self.log_alpha)
        else:
            alpha_loss = torch.tensor(0.0)
        
        # 软更新目标网络
        self._update_target_networks()
        
        # 更新训练步数
        self.training_steps += 1
        
        # 训练信息
        info = {
            'loss/critic': critic_loss.item(),
            'loss/actor': actor_loss.item(),
            'loss/alpha': alpha_loss.item() if self.auto_alpha else 0.0,
            'alpha': alpha.item(),
            'q_value': q.mean().item(),
            'log_prob': new_log_prob.mean().item()
        }
        
        # 记录训练信息
        self.log_training_info(info)
        
        return info
    
    def _update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
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
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.training_steps = checkpoint['training_steps']
        self.total_steps = checkpoint['total_steps']
        self.episodes = checkpoint['episodes'] 
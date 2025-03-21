import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.base_agent import BaseAgent

class Actor(nn.Module):
    """
    TD3 Actor网络
    将状态映射到确定性动作
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态输入
            
        Returns:
            action: 确定性动作
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        
        return action


class Critic(nn.Module):
    """
    TD3 Critic网络
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
    
    def q1_forward(self, state, action):
        """
        只计算第一个Q网络的值
        
        Args:
            state: 状态输入
            action: 动作输入
            
        Returns:
            q1_value: 第一个Q网络的输出
        """
        x = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1_value = self.q1(q1)
        
        return q1_value


class TD3Agent(BaseAgent):
    """
    TD3 (Twin Delayed DDPG) 智能体
    使用双Q网络和延迟策略更新的DDPG变体
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
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化TD3智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作上限
            min_action: 动作下限
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            policy_noise: 添加到目标动作的噪声量
            noise_clip: 目标策略噪声的裁剪范围
            policy_delay: 策略更新的延迟步数
            device: 计算设备
        """
        super(TD3Agent, self).__init__(state_dim, action_dim, max_action, min_action, device)
        
        # TD3超参数
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action_tensor = torch.FloatTensor(max_action).to(device)
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action=1.0).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action=1.0).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 创建Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 动作缩放
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((max_action - min_action) / 2.0)
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((max_action + min_action) / 2.0)
        )
    
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
            # 获取确定性动作
            action = self.actor(state)
            
            # 在训练模式下添加探索噪声
            if not evaluate:
                noise = torch.randn_like(action) * 0.1
                action = (action + noise).clamp(-1.0, 1.0)
            
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
        
        # 更新Critic
        with torch.no_grad():
            # 获取目标动作
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # 计算下一个目标动作
            next_action = self.actor_target(next_state)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            next_action = next_action * self.action_scale + self.action_bias
            
            # 使用目标Critic评估下一个状态-动作对
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # 计算目标Q值
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        actor_loss = 0.0
        if self.training_steps % self.policy_delay == 0:
            # 计算Actor动作
            pi = self.actor(state) * self.action_scale + self.action_bias
            
            # 使用第一个Q网络计算Actor损失
            actor_loss = -self.critic.q1_forward(state, pi).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._update_target_networks()
        
        # 更新训练步数
        self.training_steps += 1
        
        # 训练信息
        info = {
            'loss/critic': critic_loss.item(),
            'loss/actor': actor_loss if isinstance(actor_loss, float) else actor_loss.item(),
            'q_value': current_q1.mean().item()
        }
        
        # 记录训练信息
        self.log_training_info(info)
        
        return info
    
    def _update_target_networks(self):
        """软更新目标网络"""
        # 更新目标Actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        # 更新目标Critic
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
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
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
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.total_steps = checkpoint['total_steps']
        self.episodes = checkpoint['episodes'] 
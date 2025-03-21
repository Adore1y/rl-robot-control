from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class BaseAgent(ABC):
    """
    基础智能体抽象类
    提供所有RL算法通用的方法和接口
    """
    
    def __init__(self, state_dim, action_dim, max_action, min_action, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化基础智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作的上限
            min_action: 动作的下限
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.device = device
        
        # 训练相关参数
        self.total_steps = 0
        self.training_steps = 0
        self.episodes = 0
        
        # Logger
        self.writer = None
    
    def init_logging(self, log_dir):
        """
        初始化TensorBoard日志
        
        Args:
            log_dir: 日志目录
        """
        self.writer = SummaryWriter(log_dir)
    
    @abstractmethod
    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否处于评估模式
            
        Returns:
            所选动作
        """
        pass
    
    @abstractmethod
    def train(self, replay_buffer, batch_size=256):
        """
        使用经验回放缓冲区的样本更新智能体
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            
        Returns:
            训练信息字典
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        pass
    
    def log_training_info(self, info, step=None):
        """
        记录训练信息到TensorBoard
        
        Args:
            info: 包含训练指标的字典
            step: 当前步数
        """
        if self.writer is None:
            return
        
        step = step or self.total_steps
        
        for key, value in info.items():
            self.writer.add_scalar(key, value, step)
    
    def log_episode_info(self, info, episode=None):
        """
        记录回合信息到TensorBoard
        
        Args:
            info: 包含回合指标的字典
            episode: 当前回合
        """
        if self.writer is None:
            return
            
        episode = episode or self.episodes
        
        for key, value in info.items():
            self.writer.add_scalar(f"episode/{key}", value, episode)
    
    def preprocess_state(self, state):
        """
        预处理状态以便NN使用
        
        Args:
            state: 原始状态
            
        Returns:
            处理后的状态张量
        """
        # 将numpy数组转换为张量并移动到正确的设备
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # 确保状态是正确的形状
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加批次维度
            
        return state
    
    def postprocess_action(self, action):
        """
        后处理动作以便环境使用
        
        Args:
            action: 网络输出的动作张量
            
        Returns:
            处理后的动作
        """
        # 将张量转换为numpy数组
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
            
        # 如果动作有批次维度但只有一个样本，则移除批次维度
        if len(action.shape) > 1 and action.shape[0] == 1:
            action = action.squeeze(0)
            
        # 确保动作在允许范围内
        action = np.clip(action, self.min_action, self.max_action)
            
        return action 
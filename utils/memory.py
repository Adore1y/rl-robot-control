import numpy as np

class RolloutBuffer:
    """
    用于存储PPO训练过程中收集的轨迹数据
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def size(self):
        """返回缓冲区大小"""
        return len(self.states)


class ReplayBuffer:
    """
    用于存储SAC/TD3训练过程中的经验回放数据
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """
        初始化经验回放缓冲区
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_size: 缓冲区最大容量
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            一批经验数据
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        ) 
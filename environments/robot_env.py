import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from abc import ABC, abstractmethod

class RobotEnv(gym.Env, ABC):
    """
    基础机器人环境类，实现了与PyBullet的接口和基本功能
    """
    
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps = 0
        self.physics_client_id = None
        self.robot_id = None
        
        # 初始化PyBullet
        self._setup_simulation()
        
        # 由子类实现的观察空间和动作空间
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
    
    def _setup_simulation(self):
        """设置PyBullet模拟环境"""
        if self.render_mode == "human":
            self.physics_client_id = p.connect(p.GUI)
        else:
            self.physics_client_id = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 加载地面
        p.loadURDF("plane.urdf")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.steps = 0
        
        # 重置模拟
        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # 加载机器人和环境对象
        self._load_robot()
        self._load_objects()
        
        # 获取初始观察
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """执行一步动作"""
        self.steps += 1
        
        # 将动作应用到机器人
        self._apply_action(action)
        
        # 模拟一步
        p.stepSimulation(physicsClientId=self.physics_client_id)
        
        # 获取观察、奖励和完成状态
        observation = self._get_observation()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境（如果使用GUI模式，PyBullet已经在渲染）"""
        if self.render_mode == "rgb_array":
            width, height = 320, 240
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=1.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(width)/height,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            rgb_array = np.array(px).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
    
    def close(self):
        """关闭环境"""
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
    
    @abstractmethod
    def _get_observation_space(self):
        """返回观察空间"""
        pass
    
    @abstractmethod
    def _get_action_space(self):
        """返回动作空间"""
        pass
    
    @abstractmethod
    def _load_robot(self):
        """加载机器人模型"""
        pass
    
    @abstractmethod
    def _load_objects(self):
        """加载环境中的对象"""
        pass
    
    @abstractmethod
    def _get_observation(self):
        """获取当前状态的观察"""
        pass
    
    @abstractmethod
    def _apply_action(self, action):
        """将动作应用到机器人"""
        pass
    
    @abstractmethod
    def _compute_reward(self, action):
        """计算奖励"""
        pass
    
    @abstractmethod
    def _is_terminated(self):
        """检查是否达到终止条件"""
        pass
    
    @abstractmethod
    def _get_info(self):
        """获取额外信息"""
        pass 
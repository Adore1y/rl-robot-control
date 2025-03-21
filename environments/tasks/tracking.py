import gymnasium as gym
import numpy as np
import pybullet as p
from gym import spaces
from environments.robot_env import RobotEnv

class TrackingEnv(RobotEnv):
    """
    机器人轨迹跟踪任务环境
    任务：控制机器人末端执行器跟踪指定的轨迹
    """
    
    def __init__(self, render_mode=None, max_steps=1000):
        self.target_trajectory = None
        self.current_trajectory_point = 0
        super().__init__(render_mode, max_steps)
    
    def _get_observation_space(self):
        """定义观察空间"""
        # 观察包含：
        # - 机器人末端执行器位置 (x, y, z) - 3维
        # - 机器人末端执行器速度 (vx, vy, vz) - 3维
        # - 当前目标点位置 (x, y, z) - 3维
        # - 下一个目标点位置 (x, y, z) - 3维
        return spaces.Box(
            low=np.array([-2, -2, 0, -1, -1, -1, -2, -2, 0, -2, -2, 0]),
            high=np.array([2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
            dtype=np.float32
        )
    
    def _get_action_space(self):
        """定义动作空间"""
        # 动作包含：
        # - 机器人末端执行器目标位置增量 (dx, dy, dz) - 3维
        return spaces.Box(
            low=np.array([-0.05, -0.05, -0.05]),
            high=np.array([0.05, 0.05, 0.05]),
            dtype=np.float32
        )
    
    def _load_robot(self):
        """加载UR5机器人模型"""
        self.robot_id = p.loadURDF(
            "ur5/ur5.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # 设置机器人初始关节角度
        initial_joint_positions = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        for i, pos in enumerate(initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)
    
    def _load_objects(self):
        """加载跟踪目标和轨迹可视化"""
        # 生成轨迹
        self._generate_trajectory()
        
        # 可视化当前目标点
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 0.7]
        )
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=self.target_trajectory[self.current_trajectory_point]
        )
        
        # 可视化轨迹
        for i in range(len(self.target_trajectory) - 1):
            p.addUserDebugLine(
                self.target_trajectory[i],
                self.target_trajectory[i + 1],
                lineColorRGB=[0, 1, 0],
                lineWidth=2
            )
    
    def _generate_trajectory(self):
        """生成要跟踪的轨迹"""
        # 创建一个圆形轨迹
        center = np.array([0.5, 0.0, 0.5])
        radius = 0.3
        num_points = 100
        
        trajectory = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            trajectory.append([x, y, z])
        
        self.target_trajectory = trajectory
        self.current_trajectory_point = 0
    
    def _get_observation(self):
        """获取当前环境状态的观察"""
        # 获取末端执行器位置和速度
        end_effector_pos, end_effector_orn = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_vel = p.getLinkState(self.robot_id, 5, computeLinkVelocity=True)[6]
        
        # 获取当前目标点
        current_target = self.target_trajectory[self.current_trajectory_point]
        
        # 获取下一个目标点
        next_point_idx = (self.current_trajectory_point + 1) % len(self.target_trajectory)
        next_target = self.target_trajectory[next_point_idx]
        
        # 构建观察向量
        observation = np.array(
            list(end_effector_pos) +
            list(end_effector_vel) +
            list(current_target) +
            list(next_target)
        )
        
        return observation
    
    def _apply_action(self, action):
        """应用动作到机器人"""
        # 解析动作
        dx, dy, dz = action
        
        # 获取当前末端执行器位置
        current_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        
        # 计算目标位置
        target_pos = [
            current_pos[0] + dx,
            current_pos[1] + dy,
            current_pos[2] + dz
        ]
        
        # 使用逆运动学计算关节角度
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            5,  # 末端执行器链接索引
            target_pos
        )
        
        # 控制机器人关节
        for i in range(6):  # UR5有6个关节
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                maxVelocity=1.0
            )
    
    def _compute_reward(self, action):
        """计算奖励"""
        # 获取末端执行器位置
        end_effector_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_pos = np.array(end_effector_pos)
        
        # 获取当前目标点
        current_target = np.array(self.target_trajectory[self.current_trajectory_point])
        
        # 计算与当前目标点的距离
        distance = np.linalg.norm(end_effector_pos - current_target)
        
        # 如果足够接近当前目标点，则移动到下一个目标点
        if distance < 0.05:
            self.current_trajectory_point = (self.current_trajectory_point + 1) % len(self.target_trajectory)
            
            # 更新可视化的目标点位置
            p.resetBasePositionAndOrientation(
                self.target_visual_id,
                self.target_trajectory[self.current_trajectory_point],
                [0, 0, 0, 1]
            )
            
            # 到达一个点给予额外奖励
            point_reward = 10.0
        else:
            point_reward = 0.0
        
        # 主要奖励是负距离（越近越好）
        distance_reward = -10.0 * distance
        
        # 能量消耗惩罚
        action_penalty = 0.1 * np.sum(np.square(action))
        
        # 总奖励
        reward = distance_reward + point_reward - action_penalty
        
        return reward
    
    def _is_terminated(self):
        """检查任务是否完成"""
        # 轨迹跟踪任务通常不会自动终止，而是达到最大步数
        return False
    
    def _get_info(self):
        """获取额外信息"""
        # 获取末端执行器位置
        end_effector_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_pos = np.array(end_effector_pos)
        
        # 获取当前目标点
        current_target = np.array(self.target_trajectory[self.current_trajectory_point])
        
        # 计算与当前目标点的距离
        distance = np.linalg.norm(end_effector_pos - current_target)
        
        return {
            'end_effector_position': end_effector_pos,
            'target_position': current_target,
            'distance': distance,
            'trajectory_progress': self.current_trajectory_point / len(self.target_trajectory)
        } 
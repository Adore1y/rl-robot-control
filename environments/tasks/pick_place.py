import gymnasium as gym
import numpy as np
import pybullet as p
from gym import spaces
from environments.robot_env import RobotEnv

class PickPlaceEnv(RobotEnv):
    """
    机器人抓取放置任务环境
    任务：控制机器人抓取目标物体并将其放置到指定位置
    """
    
    def __init__(self, render_mode=None, max_steps=1000):
        self.target_object_id = None
        self.target_position = None
        self.gripper_open = True
        super().__init__(render_mode, max_steps)
    
    def _get_observation_space(self):
        """定义观察空间"""
        # 观察包含：
        # - 机器人末端执行器位置 (x, y, z) - 3维
        # - 机器人末端执行器速度 (vx, vy, vz) - 3维
        # - 目标物体位置 (x, y, z) - 3维
        # - 目标位置 (x, y, z) - 3维
        # - 夹爪状态 (open/closed) - 1维
        return spaces.Box(
            low=np.array([-2, -2, 0, -1, -1, -1, -2, -2, 0, -2, -2, 0, 0]),
            high=np.array([2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]),
            dtype=np.float32
        )
    
    def _get_action_space(self):
        """定义动作空间"""
        # 动作包含：
        # - 机器人末端执行器目标位置增量 (dx, dy, dz) - 3维
        # - 夹爪控制 (open/close) - 1维
        return spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0]),
            high=np.array([0.05, 0.05, 0.05, 1]),
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
        
        # 加载夹爪
        self.gripper_id = p.loadURDF(
            "gripper/wsg50_one_motor_gripper.urdf",
            basePosition=[0.5, 0, 0.5],
            useFixedBase=False
        )
        
        # 将夹爪连接到机器人末端
        p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=5,
            childBodyUniqueId=self.gripper_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0.05]
        )
    
    def _load_objects(self):
        """加载目标物体和放置目标"""
        # 加载目标物体（方块）
        self.target_object_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.5, 0.0, 0.05],
            globalScaling=0.8
        )
        
        # 设置目标放置位置
        self.target_position = np.array([0.5, 0.5, 0.05])
        
        # 可视化目标位置（使用一个透明的虚拟方块）
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.02, 0.02, 0.02],
            rgbaColor=[0, 1, 0, 0.5]
        )
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=self.target_position
        )
    
    def _get_observation(self):
        """获取当前环境状态的观察"""
        # 获取末端执行器位置和速度
        end_effector_pos, end_effector_orn = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_vel = p.getLinkState(self.robot_id, 5, computeLinkVelocity=True)[6]
        
        # 获取目标物体位置
        target_object_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        
        # 构建观察向量
        observation = np.array(
            list(end_effector_pos) +
            list(end_effector_vel) +
            list(target_object_pos) +
            list(self.target_position) +
            [float(self.gripper_open)]
        )
        
        return observation
    
    def _apply_action(self, action):
        """应用动作到机器人"""
        # 解析动作
        dx, dy, dz, gripper_action = action
        
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
        
        # 控制夹爪
        gripper_open_new = gripper_action < 0.5
        if gripper_open_new != self.gripper_open:
            self.gripper_open = gripper_open_new
            
            # 设置夹爪位置
            if self.gripper_open:
                target_gripper_pos = 0.04  # 打开
            else:
                target_gripper_pos = 0.00  # 闭合
            
            p.setJointMotorControl2(
                bodyUniqueId=self.gripper_id,
                jointIndex=0,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_gripper_pos
            )
    
    def _compute_reward(self, action):
        """计算奖励"""
        # 获取当前物体位置
        object_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        object_pos = np.array(object_pos)
        
        # 获取末端执行器位置
        end_effector_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_pos = np.array(end_effector_pos)
        
        # 计算与目标物体的距离
        object_dist = np.linalg.norm(end_effector_pos - object_pos)
        
        # 计算物体与目标位置的距离
        target_dist = np.linalg.norm(object_pos - self.target_position)
        
        # 构建多阶段奖励
        reward = 0
        
        # 阶段1: 接近物体
        if not self._object_grasped():
            reward -= object_dist * 10  # 惩罚与物体的距离
            
            # 当接近物体且夹爪闭合时给予奖励
            if object_dist < 0.05 and not self.gripper_open:
                reward += 10
        
        # 阶段2: 物体已抓取，移向目标
        else:
            reward -= target_dist * 10  # 惩罚与目标的距离
            
            # 当物体接近目标位置时给予奖励
            if target_dist < 0.05:
                reward += 50  # 巨大的奖励用于鼓励将物体放到目标位置
        
        # 能量消耗惩罚
        action_penalty = 0.1 * np.sum(np.square(action[:3]))  # 只对移动部分进行惩罚
        reward -= action_penalty
        
        return reward
    
    def _is_terminated(self):
        """检查任务是否完成"""
        # 获取物体位置
        object_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        object_pos = np.array(object_pos)
        
        # 计算物体与目标位置的距离
        target_dist = np.linalg.norm(object_pos - self.target_position)
        
        # 如果物体足够接近目标位置且稳定，则任务完成
        return target_dist < 0.05 and self._object_stable()
    
    def _get_info(self):
        """获取额外信息"""
        # 获取物体位置
        object_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        object_pos = np.array(object_pos)
        
        # 获取末端执行器位置
        end_effector_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        end_effector_pos = np.array(end_effector_pos)
        
        # 计算相关距离
        object_dist = np.linalg.norm(end_effector_pos - object_pos)
        target_dist = np.linalg.norm(object_pos - self.target_position)
        
        return {
            'object_position': object_pos,
            'end_effector_position': end_effector_pos,
            'object_distance': object_dist,
            'target_distance': target_dist,
            'object_grasped': self._object_grasped(),
            'object_stable': self._object_stable()
        }
    
    def _object_grasped(self):
        """检查物体是否被抓取"""
        # 获取末端执行器位置
        end_effector_pos, _ = p.getLinkState(self.robot_id, 5)[0:2]
        
        # 获取物体位置
        object_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        
        # 计算距离
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(object_pos))
        
        # 如果夹爪闭合且物体与末端执行器足够接近，则认为物体被抓取
        return not self.gripper_open and distance < 0.05
    
    def _object_stable(self):
        """检查物体是否稳定"""
        # 获取物体的线速度和角速度
        linear_vel, angular_vel = p.getBaseVelocity(self.target_object_id)
        
        # 计算速度大小
        linear_speed = np.linalg.norm(linear_vel)
        angular_speed = np.linalg.norm(angular_vel)
        
        # 如果速度足够小，则认为物体稳定
        return linear_speed < 0.01 and angular_speed < 0.01 
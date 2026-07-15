"""
AGI统一框架 - 强化学习环境
实现各种强化学习环境接口和自定义环境
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod
import math


# ==================== 环境基类 ====================

class BaseEnvironment(ABC):
    """环境基类"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        pass
    
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Tuple[int, ...]:
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Tuple[int, ...]:
        pass


# ==================== 经典控制环境 ====================

class CartPoleEnv(BaseEnvironment):
    """CartPole环境"""
    
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        self.state = None
        self.steps_beyond_terminated = None
        
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.state is not None
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta + costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        terminated = bool(
            x < -self.x_threshold or
            x > self.x_threshold or
            theta < -self.theta_threshold_radians or
            theta > self.theta_threshold_radians
        )
        
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0
            
        return self.state.copy(), reward, terminated, {}
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (4,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (2,)


class MountainCarEnv(BaseEnvironment):
    """MountainCar环境"""
    
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.force = 0.001
        self.gravity = 0.0025
        
        self.state = None
        
    def reset(self) -> np.ndarray:
        self.state = np.array([-0.5, 0.0])
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        position, velocity = self.state
        
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        if position == self.min_position and velocity < 0:
            velocity = 0
            
        terminated = position >= self.goal_position
        reward = -1.0
        
        self.state = np.array([position, velocity])
        return self.state.copy(), reward, terminated, {}
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (2,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (3,)


class PendulumEnv(BaseEnvironment):
    """Pendulum环境"""
    
    def __init__(self):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        
        self.state = None
        
    def reset(self) -> np.ndarray:
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        th, thdot = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        costs = self._angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)
        
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * action) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}
    
    def _get_obs(self) -> np.ndarray:
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot])
    
    def _angle_normalize(self, x: float) -> float:
        return ((x + np.pi) % (2 * np.pi)) - np.pi
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (3,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (1,)


# ==================== 连续控制环境 ====================

class ReacherEnv(BaseEnvironment):
    """Reacher机械臂环境"""
    
    def __init__(self, arm_length: float = 1.0):
        self.arm_length = arm_length
        self.target_radius = 0.05
        self.dt = 0.02
        
        self.target = None
        self.joint_angles = None
        self.joint_velocities = None
        
    def reset(self) -> np.ndarray:
        self.target = np.random.uniform(-self.arm_length, self.arm_length, size=2)
        self.joint_angles = np.random.uniform(-np.pi, np.pi, size=2)
        self.joint_velocities = np.zeros(2)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, -1, 1)
        
        # 更新关节角度
        self.joint_velocities = action * 0.5
        self.joint_angles += self.joint_velocities * self.dt
        
        # 计算末端位置
        end_pos = self._get_end_position()
        
        # 计算奖励
        dist = np.linalg.norm(end_pos - self.target)
        reward = -dist - 0.01 * np.sum(action ** 2)
        
        terminated = dist < self.target_radius
        
        return self._get_obs(), reward, terminated, {}
    
    def _get_end_position(self) -> np.ndarray:
        x = self.arm_length * (np.cos(self.joint_angles[0]) + 
                               np.cos(self.joint_angles[0] + self.joint_angles[1]))
        y = self.arm_length * (np.sin(self.joint_angles[0]) + 
                               np.sin(self.joint_angles[0] + self.joint_angles[1]))
        return np.array([x, y])
    
    def _get_obs(self) -> np.ndarray:
        end_pos = self._get_end_position()
        return np.concatenate([
            self.target,
            end_pos,
            self.joint_angles,
            self.joint_velocities
        ])
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (8,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (2,)


class PointMassEnv(BaseEnvironment):
    """点质量环境"""
    
    def __init__(self, dim: int = 2, goal_radius: float = 0.1):
        self.dim = dim
        self.goal_radius = goal_radius
        self.max_speed = 1.0
        self.dt = 0.1
        
        self.position = None
        self.velocity = None
        self.goal = None
        
    def reset(self) -> np.ndarray:
        self.position = np.zeros(self.dim)
        self.velocity = np.zeros(self.dim)
        self.goal = np.random.uniform(-1, 1, size=self.dim)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, -1, 1)
        
        self.velocity = np.clip(self.velocity + action * self.dt, 
                               -self.max_speed, self.max_speed)
        self.position = np.clip(self.position + self.velocity * self.dt, -1, 1)
        
        dist = np.linalg.norm(self.position - self.goal)
        reward = -dist - 0.01 * np.sum(action ** 2)
        
        terminated = dist < self.goal_radius
        
        return self._get_obs(), reward, terminated, {}
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity, self.goal])
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (3 * self.dim,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (self.dim,)


# ==================== 多智能体环境 ====================

class MultiAgentEnv(BaseEnvironment):
    """多智能体环境基类"""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_states = [None] * num_agents
        
    @abstractmethod
    def reset(self) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        pass


class PursuitEvasionEnv(MultiAgentEnv):
    """追逃环境"""
    
    def __init__(self, num_pursuers: int = 2, num_evaders: int = 1,
                 world_size: float = 10.0, capture_radius: float = 0.5):
        super().__init__(num_pursuers + num_evaders)
        
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.world_size = world_size
        self.capture_radius = capture_radius
        self.dt = 0.1
        self.max_speed = 1.0
        
        self.pursuer_positions = None
        self.evader_positions = None
        
    def reset(self) -> List[np.ndarray]:
        self.pursuer_positions = np.random.uniform(
            0, self.world_size, size=(self.num_pursuers, 2)
        )
        self.evader_positions = np.random.uniform(
            0, self.world_size, size=(self.num_evaders, 2)
        )
        return self._get_obs()
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        actions = [np.clip(a, -1, 1) for a in actions]
        
        # 更新追捕者位置
        for i, action in enumerate(actions[:self.num_pursuers]):
            self.pursuer_positions[i] += action * self.max_speed * self.dt
            self.pursuer_positions[i] = np.clip(
                self.pursuer_positions[i], 0, self.world_size
            )
            
        # 更新逃避者位置
        for i, action in enumerate(actions[self.num_pursuers:]):
            self.evader_positions[i] += action * self.max_speed * self.dt
            self.evader_positions[i] = np.clip(
                self.evader_positions[i], 0, self.world_size
            )
            
        # 检查捕获
        captured = [False] * self.num_evaders
        for j, evader_pos in enumerate(self.evader_positions):
            for pursuer_pos in self.pursuer_positions:
                if np.linalg.norm(evader_pos - pursuer_pos) < self.capture_radius:
                    captured[j] = True
                    break
                    
        # 计算奖励
        rewards = []
        for i in range(self.num_pursuers):
            min_dist = min(
                np.linalg.norm(self.pursuer_positions[i] - evader_pos)
                for evader_pos in self.evader_positions
            )
            rewards.append(-min_dist / self.world_size)
            
        for j in range(self.num_evaders):
            if captured[j]:
                rewards.append(-10.0)
            else:
                min_dist = min(
                    np.linalg.norm(self.evader_positions[j] - pursuer_pos)
                    for pursuer_pos in self.pursuer_positions
                )
                rewards.append(min_dist / self.world_size)
                
        terminated = all(captured)
        terminated_list = [terminated] * self.num_agents
        
        return self._get_obs(), rewards, terminated_list, {}
    
    def _get_obs(self) -> List[np.ndarray]:
        obs = []
        
        for i in range(self.num_pursuers):
            obs.append(np.concatenate([
                self.pursuer_positions[i],
                self.evader_positions.flatten()
            ]))
            
        for j in range(self.num_evaders):
            obs.append(np.concatenate([
                self.evader_positions[j],
                self.pursuer_positions.flatten()
            ]))
            
        return obs
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (2 + 2 * self.num_evaders,)
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return (2,)


# ==================== 环境包装器 ====================

class EnvWrapper:
    """环境包装器基类"""
    
    def __init__(self, env: BaseEnvironment):
        self.env = env
        
    def reset(self) -> np.ndarray:
        return self.env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        return self.env.step(action)
    
    def render(self) -> Optional[np.ndarray]:
        return self.env.render()
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return self.env.observation_space
    
    @property
    def action_space(self) -> Tuple[int, ...]:
        return self.env.action_space


class NormalizeAction(EnvWrapper):
    """动作归一化包装器"""
    
    def __init__(self, env: BaseEnvironment, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # 假设原始动作空间是[0, 1]
        original_action = (action - self.low) / (self.high - self.low)
        return self.env.step(original_action)


class NormalizeObservation(EnvWrapper):
    """观测归一化包装器"""
    
    def __init__(self, env: BaseEnvironment, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.mean = np.zeros(env.observation_space[0])
        self.var = np.ones(env.observation_space[0])
        self.count = 0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        self._update(obs)
        return self._normalize(obs), reward, done, info
    
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._update(obs)
        return self._normalize(obs)
    
    def _update(self, obs: np.ndarray):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += delta * delta2
        
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)


class TimeLimit(EnvWrapper):
    """时间限制包装器"""
    
    def __init__(self, env: BaseEnvironment, max_steps: int = 1000):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True
            
        return obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.env.reset()


class FrameStack(EnvWrapper):
    """帧堆叠包装器"""
    
    def __init__(self, env: BaseEnvironment, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = []
        
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.frames = [obs] * self.num_frames
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate(self.frames)


class RewardScale(EnvWrapper):
    """奖励缩放包装器"""
    
    def __init__(self, env: BaseEnvironment, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward * self.scale, done, info


class ActionRepeat(EnvWrapper):
    """动作重复包装器"""
    
    def __init__(self, env: BaseEnvironment, repeat: int = 1):
        super().__init__(env)
        self.repeat = repeat
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        total_reward = 0.0
        done = False
        
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info

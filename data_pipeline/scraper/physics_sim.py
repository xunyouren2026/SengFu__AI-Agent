"""
物理仿真数据生成器 - Physics Simulation Data Generator

使用PyBullet生成物理仿真数据，用于训练SH-GNN和LeJEPA模型。

支持场景：
- 刚体碰撞
- 机械臂操作
- 球体滚动
- 积木堆叠
- 流体粒子（简化）
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from enum import Enum
import time


class SimulationType(Enum):
    """仿真类型"""
    BALL_ROLLING = "ball_rolling"
    BLOCK_STACKING = "block_stacking"
    ROBOT_ARM = "robot_arm"
    RIGID_COLLISION = "rigid_collision"
    PARTICLE_SYSTEM = "particle_system"


@dataclass
class PhysicsConfig:
    """物理仿真配置"""
    gravity: float = -9.81
    time_step: float = 1.0 / 240.0  # 240Hz
    num_substeps: int = 4
    real_time_factor: float = 1.0
    
    # 渲染
    render: bool = False
    width: int = 640
    height: int = 480
    
    # 数据采集
    record_frames: bool = True
    record_interval: int = 4  # 每4步记录一帧
    
    # 物理参数
    friction: float = 0.5
    restitution: float = 0.3
    linear_damping: float = 0.04
    angular_damping: float = 0.1


@dataclass
class SimulationState:
    """仿真状态"""
    step: int
    time: float
    objects: Dict[int, Dict[str, Any]]
    contacts: List[Dict[str, Any]]
    energy: Dict[str, float]


class PhysicsSimulator:
    """
    物理仿真器
    
    基于PyBullet的物理仿真，用于生成训练数据。
    """
    
    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PyBullet连接
        self.client_id = None
        self.plane_id = None
        self.objects: Dict[int, Dict[str, Any]] = {}
        
        # 数据记录
        self.states: List[SimulationState] = []
        self.frames: List[np.ndarray] = []
        
        # 统计
        self.stats = {
            'total_steps': 0,
            'total_collisions': 0,
            'total_energy_change': 0.0
        }
    
    def connect(self) -> None:
        """连接PyBullet"""
        if self.config.render:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # 设置物理参数
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            numSubSteps=self.config.num_substeps,
            physicsClientId=self.client_id
        )
        
        # 加载平面
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # 设置平面摩擦
        p.changeDynamics(
            self.plane_id, -1,
            lateralFriction=self.config.friction,
            restitution=self.config.restitution,
            physicsClientId=self.client_id
        )
        
        self.logger.info(f"PyBullet connected, client_id={self.client_id}")
    
    def disconnect(self) -> None:
        """断开连接"""
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None
            self.logger.info("PyBullet disconnected")
    
    def reset(self) -> None:
        """重置仿真"""
        self.objects.clear()
        self.states.clear()
        self.frames.clear()
        self.stats = {'total_steps': 0, 'total_collisions': 0, 'total_energy_change': 0.0}
        
        if self.client_id is not None:
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(0, 0, self.config.gravity, physicsClientId=self.client_id)
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
    
    def create_sphere(
        self,
        position: Tuple[float, float, float],
        radius: float = 0.1,
        mass: float = 1.0,
        color: Tuple[float, float, float, float] = (1, 0, 0, 1),
        velocity: Tuple[float, float, float] = (0, 0, 0)
    ) -> int:
        """创建球体"""
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client_id
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client_id
        )
        
        p.changeDynamics(
            body_id, -1,
            lateralFriction=self.config.friction,
            restitution=self.config.restitution,
            linearDamping=self.config.linear_damping,
            angularDamping=self.config.angular_damping,
            physicsClientId=self.client_id
        )
        
        if any(velocity):
            p.resetBaseVelocity(body_id, velocity, [0, 0, 0], physicsClientId=self.client_id)
        
        self.objects[body_id] = {
            'type': 'sphere',
            'radius': radius,
            'mass': mass,
            'color': color
        }
        
        return body_id
    
    def create_box(
        self,
        position: Tuple[float, float, float],
        half_extents: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        mass: float = 1.0,
        color: Tuple[float, float, float, float] = (0, 1, 0, 1),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)
    ) -> int:
        """创建立方体"""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client_id
        )
        
        p.changeDynamics(
            body_id, -1,
            lateralFriction=self.config.friction,
            restitution=self.config.restitution,
            linearDamping=self.config.linear_damping,
            angularDamping=self.config.angular_damping,
            physicsClientId=self.client_id
        )
        
        self.objects[body_id] = {
            'type': 'box',
            'half_extents': half_extents,
            'mass': mass,
            'color': color
        }
        
        return body_id
    
    def create_cylinder(
        self,
        position: Tuple[float, float, float],
        radius: float = 0.1,
        height: float = 0.2,
        mass: float = 1.0,
        color: Tuple[float, float, float, float] = (0, 0, 1, 1)
    ) -> int:
        """创建圆柱体"""
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.client_id
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self.client_id
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client_id
        )
        
        p.changeDynamics(
            body_id, -1,
            lateralFriction=self.config.friction,
            restitution=self.config.restitution,
            physicsClientId=self.client_id
        )
        
        self.objects[body_id] = {
            'type': 'cylinder',
            'radius': radius,
            'height': height,
            'mass': mass,
            'color': color
        }
        
        return body_id
    
    def get_object_state(self, body_id: int) -> Dict[str, Any]:
        """获取物体状态"""
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
        
        return {
            'position': np.array(pos),
            'orientation': np.array(orn),
            'linear_velocity': np.array(lin_vel),
            'angular_velocity': np.array(ang_vel)
        }
    
    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """获取所有物体状态"""
        return {body_id: self.get_object_state(body_id) for body_id in self.objects}
    
    def get_contacts(self) -> List[Dict[str, Any]]:
        """获取接触信息"""
        contacts = []
        all_contacts = p.getContactPoints(physicsClientId=self.client_id)
        
        for contact in all_contacts:
            contacts.append({
                'body_a': contact[1],
                'body_b': contact[2],
                'link_a': contact[3],
                'link_b': contact[4],
                'position': np.array(contact[5]),
                'normal': np.array(contact[7]),
                'force': contact[9],
                'friction_force': contact[10]
            })
        
        return contacts
    
    def compute_energy(self) -> Dict[str, float]:
        """计算系统总能量"""
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        for body_id, obj_info in self.objects.items():
            state = self.get_object_state(body_id)
            mass = obj_info['mass']
            
            # 动能 = 0.5 * m * v^2
            v = state['linear_velocity']
            kinetic_energy += 0.5 * mass * np.dot(v, v)
            
            # 势能 = m * g * h
            h = state['position'][2]
            potential_energy += mass * abs(self.config.gravity) * h
        
        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy
        }
    
    def step(self) -> SimulationState:
        """执行一步仿真"""
        p.stepSimulation(physicsClientId=self.client_id)
        
        self.stats['total_steps'] += 1
        current_time = self.stats['total_steps'] * self.config.time_step
        
        # 获取状态
        objects_state = self.get_all_states()
        contacts = self.get_contacts()
        energy = self.compute_energy()
        
        # 更新统计
        self.stats['total_collisions'] += len(contacts)
        if len(self.states) > 0:
            self.stats['total_energy_change'] += abs(
                energy['total'] - self.states[-1].energy['total']
            )
        
        # 创建状态记录
        state = SimulationState(
            step=self.stats['total_steps'],
            time=current_time,
            objects=objects_state,
            contacts=contacts,
            energy=energy
        )
        self.states.append(state)
        
        # 记录帧
        if self.config.record_frames and self.stats['total_steps'] % self.config.record_interval == 0:
            frame = self.get_camera_image()
            self.frames.append(frame)
        
        return state
    
    def get_camera_image(self) -> np.ndarray:
        """获取相机图像"""
        if not self.config.render and self.client_id is not None:
            # 使用DIRECT模式渲染
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[2, 2, 2],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.client_id
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=self.config.width / self.config.height,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.client_id
            )
            
            _, _, rgb, _, _ = p.getCameraImage(
                width=self.config.width,
                height=self.config.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client_id
            )
            
            return np.array(rgb[:, :, :3], dtype=np.uint8)
        
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
    
    def run_simulation(self, num_steps: int) -> List[SimulationState]:
        """运行仿真指定步数"""
        for _ in range(num_steps):
            self.step()
        return self.states
    
    def generate_ball_rolling_data(
        self,
        num_episodes: int = 100,
        steps_per_episode: int = 500,
        randomize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        生成球滚动数据集
        
        Returns:
            包含 positions, velocities, times 的字典
        """
        all_positions = []
        all_velocities = []
        all_times = []
        
        for episode in range(num_episodes):
            self.reset()
            
            # 随机初始位置和速度
            if randomize:
                start_x = np.random.uniform(-1, 1)
                start_y = np.random.uniform(-1, 1)
                start_z = np.random.uniform(0.5, 2.0)
                vel_x = np.random.uniform(-2, 2)
                vel_y = np.random.uniform(-2, 2)
                radius = np.random.uniform(0.05, 0.2)
                mass = np.random.uniform(0.5, 2.0)
            else:
                start_x, start_y, start_z = 0, 0, 1
                vel_x, vel_y = 1, 0
                radius, mass = 0.1, 1.0
            
            # 创建球
            ball_id = self.create_sphere(
                position=(start_x, start_y, start_z),
                radius=radius,
                mass=mass,
                velocity=(vel_x, vel_y, 0)
            )
            
            # 运行仿真
            states = self.run_simulation(steps_per_episode)
            
            # 提取轨迹
            positions = np.array([s.objects[ball_id]['position'] for s in states])
            velocities = np.array([s.objects[ball_id]['linear_velocity'] for s in states])
            times = np.array([s.time for s in states])
            
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_times.append(times)
            
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Generated {episode + 1}/{num_episodes} episodes")
        
        return {
            'positions': np.stack(all_positions),
            'velocities': np.stack(all_velocities),
            'times': np.stack(all_times)
        }
    
    def generate_block_stacking_data(
        self,
        num_episodes: int = 100,
        num_blocks: int = 5,
        steps_per_episode: int = 1000
    ) -> Dict[str, np.ndarray]:
        """生成积木堆叠数据集"""
        all_positions = []
        all_orientations = []
        all_stable = []
        
        for episode in range(num_episodes):
            self.reset()
            
            block_ids = []
            base_height = 0.1
            
            for i in range(num_blocks):
                # 随机位置偏移
                x = np.random.uniform(-0.1, 0.1)
                y = np.random.uniform(-0.1, 0.1)
                z = base_height + i * 0.25 + np.random.uniform(0.4, 0.6)
                
                # 随机尺寸
                half_extents = (
                    np.random.uniform(0.05, 0.15),
                    np.random.uniform(0.05, 0.15),
                    np.random.uniform(0.05, 0.1)
                )
                
                block_id = self.create_box(
                    position=(x, y, z),
                    half_extents=half_extents,
                    mass=np.random.uniform(0.5, 2.0),
                    color=(np.random.random(), np.random.random(), np.random.random(), 1)
                )
                block_ids.append(block_id)
            
            # 运行仿真
            states = self.run_simulation(steps_per_episode)
            
            # 检查稳定性（最后100步位置变化小）
            final_positions = []
            final_orientations = []
            for block_id in block_ids:
                pos_history = [s.objects[block_id]['position'] for s in states[-100:]]
                pos_std = np.std(pos_history, axis=0)
                is_stable = np.all(pos_std < 0.01)
                
                final_positions.append(states[-1].objects[block_id]['position'])
                final_orientations.append(states[-1].objects[block_id]['orientation'])
                all_stable.append(is_stable)
            
            all_positions.append(np.array(final_positions))
            all_orientations.append(np.array(final_orientations))
        
        return {
            'positions': np.stack(all_positions),
            'orientations': np.stack(all_orientations),
            'stable': np.array(all_stable)
        }
    
    def save_data(self, filepath: str) -> None:
        """保存仿真数据"""
        data = {
            'config': self.config.__dict__,
            'stats': self.stats,
            'states': [
                {
                    'step': s.step,
                    'time': s.time,
                    'objects': {k: {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv 
                                   for kk, vv in v.items()} 
                               for k, v in s.objects.items()},
                    'energy': s.energy
                }
                for s in self.states
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved simulation data to {filepath}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def generate_physics_dataset(
    output_dir: str,
    simulation_type: SimulationType = SimulationType.BALL_ROLLING,
    num_episodes: int = 1000,
    **kwargs
) -> Dict[str, str]:
    """
    生成物理仿真数据集
    
    Args:
        output_dir: 输出目录
        simulation_type: 仿真类型
        num_episodes: episode数量
        
    Returns:
        生成的文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = PhysicsConfig(**kwargs)
    
    with PhysicsSimulator(config) as simulator:
        if simulation_type == SimulationType.BALL_ROLLING:
            data = simulator.generate_ball_rolling_data(num_episodes)
        elif simulation_type == SimulationType.BLOCK_STACKING:
            data = simulator.generate_block_stacking_data(num_episodes)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        # 保存数据
        output_file = output_path / f"{simulation_type.value}.npz"
        np.savez(output_file, **data)
        
        # 保存元数据
        simulator.save_data(str(output_path / f"{simulation_type.value}_meta.json"))
        
        return {
            'data': str(output_file),
            'metadata': str(output_path / f"{simulation_type.value}_meta.json")
        }

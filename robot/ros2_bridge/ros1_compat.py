#!/usr/bin/env python3
"""
ROS1桥接器 - 完整实现（无外部依赖版本）
支持MoveIt运动规划、实时关节状态读取、数字/模拟IO、力控接口（需硬件支持）
注：移除了numpy依赖，使用纯Python math模块实现
"""

import threading
import time
import math
from typing import List, Tuple, Optional, Dict, Any
from collections import deque


class RobotControllerBase:
    """机器人控制器基类"""

    def __init__(self, robot_name: str = "robot"):
        self.robot_name = robot_name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """连接（基类默认实现：模拟连接）"""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """断开连接（基类默认实现）"""
        self._connected = False
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        """关节运动（基类默认实现）"""
        if not self._connected:
            return False
        return True

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        """笛卡尔空间运动（基类默认实现）"""
        if not self._connected:
            return False
        return True

    def get_joint_positions(self) -> List[float]:
        """获取关节位置（基类默认实现：返回零位）"""
        return [0.0] * 6

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取TCP位姿（基类默认实现：返回零位）"""
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def stop(self) -> bool:
        """停止运动（基类默认实现）"""
        return True

    def get_joint_velocities(self) -> List[float]:
        return [0.0] * 6

    def get_joint_torques(self) -> List[float]:
        return [0.0] * 6

    def get_joint_temperatures(self) -> List[float]:
        return [0.0] * 6


class ROS1RobotController(RobotControllerBase):
    """
    基于ROS1 MoveIt的完整机器人控制器
    
    注意：此类需要ROS1环境才能正常工作。
    如果ROS不可用，请使用MockRobotController替代。
    """

    def __init__(self, robot_name: str = "ur5", group_name: str = "manipulator"):
        super().__init__(robot_name)
        self.group_name = group_name
        self._move_group = None
        self._robot_commander = None
        self._scene = None
        self._joint_state_sub = None
        self._joint_positions = [0.0] * 6
        self._joint_velocities = [0.0] * 6
        self._joint_efforts = [0.0] * 6
        self._joint_names = []
        self._connected = False
        # IO发布器
        self._digital_out_pubs = {}
        self._analog_out_pubs = {}
        # 服务客户端（用于模拟IO服务）
        self._set_digital_srv = None
        self._set_analog_srv = None
        # ROS运行时引用（延迟导入）
        self._rospy = None
        self._moveit_commander = None

    def _import_ros(self):
        """延迟导入ROS模块"""
        if self._rospy is None:
            try:
                import rospy
                import moveit_commander
                self._rospy = rospy
                self._moveit_commander = moveit_commander
            except ImportError:
                raise RuntimeError("ROS1 not available. Please install ros-noetic-rospy and ros-noetic-moveit.")

    def connect(self) -> bool:
        self._import_ros()
        try:
            if not self._rospy.get_node_uri():
                self._rospy.init_node(f"{self.robot_name}_controller", anonymous=True)
            self._moveit_commander.roscpp_initialize([])
            self._robot_commander = self._moveit_commander.RobotCommander()
            self._scene = self._moveit_commander.PlanningSceneInterface()
            self._move_group = self._moveit_commander.MoveGroupCommander(self.group_name)
            # 获取关节名称
            self._joint_names = self._move_group.get_active_joints()
            self._joint_positions = [0.0] * len(self._joint_names)
            # 订阅关节状态
            import sensor_msgs.msg
            self._joint_state_sub = self._rospy.Subscriber("/joint_states", sensor_msgs.msg.JointState,
                                                           self._joint_state_callback)
            # 等待第一个消息
            self._rospy.sleep(0.5)
            self._connected = True
            print(f"Connected to ROS1 MoveIt for {self.robot_name}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def _joint_state_callback(self, msg):
        """更新关节状态缓存"""
        for i, name in enumerate(self._joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self._joint_positions[i] = msg.position[idx]
                self._joint_velocities[i] = msg.velocity[idx] if len(msg.velocity) > idx else 0.0
                self._joint_efforts[i] = msg.effort[idx] if len(msg.effort) > idx else 0.0

    def disconnect(self) -> bool:
        if self._joint_state_sub:
            self._joint_state_sub.unregister()
        if self._moveit_commander:
            self._moveit_commander.roscpp_shutdown()
        self._connected = False
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        if not self._connected or len(joint_positions) != len(self._joint_names):
            return False
        self._move_group.set_max_velocity_scaling_factor(velocity)
        self._move_group.set_max_acceleration_scaling_factor(acceleration)
        plan = self._move_group.go(joint_positions, wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()
        return plan

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        if not self._connected:
            return False
        import geometry_msgs.msg
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x, target_pose.position.y, target_pose.position.z = pose[0], pose[1], pose[2]
        q = self._euler_to_quaternion(pose[3], pose[4], pose[5])
        target_pose.orientation.x, target_pose.orientation.y = q[0], q[1]
        target_pose.orientation.z, target_pose.orientation.w = q[2], q[3]
        self._move_group.set_pose_target(target_pose)
        self._move_group.set_max_velocity_scaling_factor(velocity)
        self._move_group.set_max_acceleration_scaling_factor(acceleration)
        plan = self._move_group.go(wait=True)
        self._move_group.stop()
        self._move_group.clear_pose_targets()
        return plan

    def get_joint_positions(self) -> List[float]:
        return self._joint_positions.copy()

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        if not self._connected:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pose = self._move_group.get_current_pose().pose
        rx, ry, rz = self._quaternion_to_euler(pose.orientation.x, pose.orientation.y,
                                               pose.orientation.z, pose.orientation.w)
        return (pose.position.x, pose.position.y, pose.position.z, rx, ry, rz)

    def stop(self) -> bool:
        if self._connected:
            self._move_group.stop()
        return True

    # ========== 力控（需要硬件支持，提供ROS接口） ==========
    def set_force_torque(self, force: Tuple[float, float, float], torque: Tuple[float, float, float]) -> bool:
        self._import_ros()
        # 发布到力控控制器（假设存在 /force_control/set_wrench 话题）
        import geometry_msgs.msg
        pub = self._rospy.Publisher(f"/{self.robot_name}/force_control/wrench", geometry_msgs.msg.Wrench, queue_size=1)
        wrench = geometry_msgs.msg.Wrench()
        wrench.force.x, wrench.force.y, wrench.force.z = force
        wrench.torque.x, wrench.torque.y, wrench.torque.z = torque
        pub.publish(wrench)
        return True

    def force_control_enable(self, enable: bool) -> bool:
        self._import_ros()
        # 调用服务启用/禁用力控
        try:
            from std_srvs.srv import SetBool, SetBoolResponse
            self._rospy.wait_for_service(f"/{self.robot_name}/force_control/enable", timeout=1.0)
            srv = self._rospy.ServiceProxy(f"/{self.robot_name}/force_control/enable", SetBool)
            resp = srv(enable)
            return resp.success
        except:
            return False

    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        self._import_ros()
        # 从 /force_torque_sensor 话题读取
        try:
            import geometry_msgs.msg
            msg = self._rospy.wait_for_message(f"/{self.robot_name}/force_torque_sensor", 
                                                geometry_msgs.msg.WrenchStamped, timeout=0.1)
            f = (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)
            t = (msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z)
            return (f, t)
        except:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # ========== IO（通过ROS topic/service） ==========
    def set_digital_out(self, pin: int, value: bool) -> bool:
        self._import_ros()
        import std_msgs.msg
        topic = f"/{self.robot_name}/digital_out/{pin}"
        if topic not in self._digital_out_pubs:
            self._digital_out_pubs[topic] = self._rospy.Publisher(topic, std_msgs.msg.Bool, queue_size=1)
        self._digital_out_pubs[topic].publish(std_msgs.msg.Bool(data=value))
        return True

    def get_digital_in(self, pin: int) -> bool:
        self._import_ros()
        import std_msgs.msg
        topic = f"/{self.robot_name}/digital_in/{pin}"
        try:
            msg = self._rospy.wait_for_message(topic, std_msgs.msg.Bool, timeout=0.1)
            return msg.data
        except:
            return False

    def set_analog_out(self, pin: int, value: float) -> bool:
        self._import_ros()
        import std_msgs.msg
        topic = f"/{self.robot_name}/analog_out/{pin}"
        if topic not in self._analog_out_pubs:
            self._analog_out_pubs[topic] = self._rospy.Publisher(topic, std_msgs.msg.Float64, queue_size=1)
        self._analog_out_pubs[topic].publish(std_msgs.msg.Float64(data=value))
        return True

    def get_analog_in(self, pin: int) -> float:
        self._import_ros()
        import std_msgs.msg
        topic = f"/{self.robot_name}/analog_in/{pin}"
        try:
            msg = self._rospy.wait_for_message(topic, std_msgs.msg.Float64, timeout=0.1)
            return msg.data
        except:
            return 0.0

    # ========== 辅助（使用纯Python math替代numpy） ==========
    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        """欧拉角转四元数（纯Python实现）"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return (qx, qy, qz, qw)

    @staticmethod
    def _quaternion_to_euler(x, y, z, w):
        """四元数转欧拉角（纯Python实现）"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw


class MockRobotController(RobotControllerBase):
    """
    模拟机器人控制器（用于测试和无ROS环境）
    """

    def __init__(self, robot_name: str = "mock_robot", num_joints: int = 6):
        super().__init__(robot_name)
        self.num_joints = num_joints
        self._joint_positions = [0.0] * num_joints
        self._joint_velocities = [0.0] * num_joints
        self._joint_torques = [0.0] * num_joints
        self._tcp_pose = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        self._velocity_scale = 1.0

    def connect(self) -> bool:
        self._connected = True
        print(f"Mock robot {self.robot_name} connected")
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5) -> bool:
        if len(joint_positions) == self.num_joints:
            self._joint_positions = list(joint_positions)
            print(f"Mock: moved to joints {joint_positions}")
            return True
        return False

    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        self._tcp_pose = list(pose)
        print(f"Mock: moved to pose {pose}")
        return True

    def get_joint_positions(self) -> List[float]:
        return self._joint_positions.copy()

    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(self._tcp_pose)

    def stop(self) -> bool:
        print("Mock: stopped")
        return True

    def set_velocity_scale(self, scale: float):
        self._velocity_scale = max(0.0, min(1.0, scale))

    def get_velocity_scale(self) -> float:
        return self._velocity_scale

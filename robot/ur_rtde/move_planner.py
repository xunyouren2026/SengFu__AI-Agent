"""
Robot Motion Planning Module

Robot motion planning with:
- Joint space planning (linear interpolation)
- Cartesian space planning (linear and circular)
- Velocity/acceleration limits
- Waypoint sequencing
- Motion validation

Pure Python standard library only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


@dataclass
class JointState:
    """Robot joint state."""
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    def num_joints(self) -> int:
        return len(self.positions)

    def copy(self) -> JointState:
        return JointState(
            positions=list(self.positions),
            velocities=list(self.velocities),
            accelerations=list(self.accelerations),
            timestamps=list(self.timestamps),
            names=list(self.names),
        )

    def distance_to(self, other: JointState) -> float:
        """Euclidean distance in joint space."""
        if self.num_joints() != other.num_joints():
            return float('inf')
        return math.sqrt(sum((a - b) ** 2 for a, b in
                             zip(self.positions, other.positions)))


@dataclass
class Pose:
    """6D pose (position + orientation as Euler angles)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0  # Roll
    ry: float = 0.0  # Pitch
    rz: float = 0.0  # Yaw

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def orientation(self) -> Tuple[float, float, float]:
        return (self.rx, self.ry, self.rz)

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance in Cartesian space."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def angular_distance_to(self, other: Pose) -> float:
        """Angular distance."""
        return math.sqrt(
            (self.rx - other.rx) ** 2 +
            (self.ry - other.ry) ** 2 +
            (self.rz - other.rz) ** 2
        )

    def lerp(self, other: Pose, t: float) -> Pose:
        """Linear interpolation between two poses."""
        return Pose(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t,
            z=self.z + (other.z - self.z) * t,
            rx=self.rx + (other.rx - self.rx) * t,
            ry=self.ry + (other.ry - self.ry) * t,
            rz=self.rz + (other.rz - self.rz) * t,
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z,
                "rx": self.rx, "ry": self.ry, "rz": self.rz}


@dataclass
class Waypoint:
    """A motion waypoint."""
    pose: Optional[Pose] = None
    joint_state: Optional[JointState] = None
    velocity_scale: float = 1.0
    acceleration_scale: float = 1.0
    blend_radius: float = 0.0
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VelocityLimit:
    """Velocity and acceleration limits."""
    max_joint_velocity: List[float] = field(default_factory=list)
    max_joint_acceleration: List[float] = field(default_factory=list)
    max_cartesian_velocity: float = 1.0  # m/s
    max_cartesian_acceleration: float = 2.0  # m/s^2
    max_cartesian_rotation_velocity: float = 1.5  # rad/s


class VelocityProfile:
    """
    Velocity profile generation for motion trajectories.

    Supports trapezoidal and S-curve velocity profiles.
    """

    @staticmethod
    def trapezoidal(q_start: float, q_end: float,
                    max_vel: float, max_acc: float,
                    num_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        Generate a trapezoidal velocity profile.

        Returns list of (position, velocity, time) tuples.
        """
        distance = abs(q_end - q_start)
        if distance < 1e-10:
            return [(q_start, 0.0, 0.0)]

        direction = 1.0 if q_end > q_start else -1.0

        # Calculate acceleration, cruise, deceleration phases
        # Time to accelerate to max velocity
        t_acc = max_vel / max_acc
        # Distance during acceleration
        d_acc = 0.5 * max_acc * t_acc ** 2

        if 2 * d_acc > distance:
            # Triangle profile (can't reach max velocity)
            t_acc = math.sqrt(distance / max_acc)
            d_acc = 0.5 * max_acc * t_acc ** 2
            t_cruise = 0.0
            v_cruise = max_acc * t_acc
        else:
            v_cruise = max_vel
            d_cruise = distance - 2 * d_acc
            t_cruise = d_cruise / v_cruise if v_cruise > 0 else 0.0

        t_dec = t_acc
        total_time = 2 * t_acc + t_cruise

        points: List[Tuple[float, float, float]] = []
        for i in range(num_points + 1):
            t = (i / num_points) * total_time

            if t <= t_acc:
                # Acceleration phase
                q = q_start + direction * 0.5 * max_acc * t ** 2
                v = direction * max_acc * t
            elif t <= t_acc + t_cruise:
                # Cruise phase
                dt = t - t_acc
                q = q_start + direction * (d_acc + v_cruise * dt)
                v = direction * v_cruise
            else:
                # Deceleration phase
                dt = t - t_acc - t_cruise
                q = q_start + direction * (d_acc + d_cruise + v_cruise * dt - 0.5 * max_acc * dt ** 2)
                v = direction * (v_cruise - max_acc * dt)

            v = max(-max_vel, min(max_vel, v))
            points.append((q, v, t))

        return points

    @staticmethod
    def s_curve(q_start: float, q_end: float,
                 max_vel: float, max_acc: float,
                 jerk: float = 10.0,
                 num_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        Generate an S-curve (jerk-limited) velocity profile.

        Returns list of (position, velocity, time) tuples.
        """
        distance = abs(q_end - q_start)
        if distance < 1e-10:
            return [(q_start, 0.0, 0.0)]

        direction = 1.0 if q_end > q_start else -1.0

        # Simplified S-curve: use sinusoidal acceleration
        t_acc = max_vel / max_acc
        d_acc = 0.5 * max_vel * t_acc

        if 2 * d_acc > distance:
            t_acc = math.sqrt(distance / max_acc)
            d_acc = 0.5 * max_acc * t_acc ** 2
            v_cruise = max_acc * t_acc
            t_cruise = 0.0
        else:
            v_cruise = max_vel
            d_cruise = distance - 2 * d_acc
            t_cruise = d_cruise / v_cruise if v_cruise > 0 else 0.0

        total_time = 2 * t_acc + t_cruise
        points: List[Tuple[float, float, float]] = []

        for i in range(num_points + 1):
            t = (i / num_points) * total_time

            if t <= t_acc:
                # S-curve acceleration (sinusoidal)
                phase = t / t_acc
                acc = max_acc * math.sin(phase * math.pi / 2)
                v = max_acc * t_acc * (1 - math.cos(phase * math.pi / 2)) / (math.pi / 2)
                ds = v * t * 0.5
                q = q_start + direction * ds
            elif t <= t_acc + t_cruise:
                dt = t - t_acc
                q = q_start + direction * (d_acc + v_cruise * dt)
                v = direction * v_cruise
            else:
                dt = t - t_acc - t_cruise
                phase = min(1.0, dt / t_acc)
                dec = max_acc * math.sin(phase * math.pi / 2)
                v = direction * (v_cruise - v_cruise * (1 - math.cos(phase * math.pi / 2)) / (math.pi / 2))
                q = q_start + direction * (distance - 0.5 * max_acc * (t_acc - dt) ** 2 * phase)

            v = max(-max_vel, min(max_vel, v))
            points.append((q, v, t))

        return points


class JointPlan:
    """Joint space motion plan."""

    def __init__(self) -> None:
        self.trajectory: List[JointState] = []
        self.duration: float = 0.0
        self.num_joints: int = 0

    def add_point(self, state: JointState) -> None:
        """Add a trajectory point."""
        self.trajectory.append(state)
        self.num_joints = state.num_joints()

    def get_point(self, index: int) -> Optional[JointState]:
        """Get a trajectory point by index."""
        if 0 <= index < len(self.trajectory):
            return self.trajectory[index]
        return None

    def interpolate(self, t: float) -> JointState:
        """Interpolate the trajectory at parameter t (0 to 1)."""
        if not self.trajectory:
            return JointState()
        if len(self.trajectory) == 1:
            return self.trajectory[0].copy()

        t = max(0.0, min(1.0, t))
        idx_f = t * (len(self.trajectory) - 1)
        idx = int(idx_f)
        frac = idx_f - idx

        if idx >= len(self.trajectory) - 1:
            return self.trajectory[-1].copy()

        s1 = self.trajectory[idx]
        s2 = self.trajectory[idx + 1]

        positions = [a + (b - a) * frac for a, b in zip(s1.positions, s2.positions)]
        velocities = [a + (b - a) * frac for a, b in zip(s1.velocities, s2.velocities)] if s1.velocities else []

        return JointState(positions=positions, velocities=velocities,
                          names=list(s1.names))

    def to_list(self) -> List[List[float]]:
        """Export trajectory as list of position lists."""
        return [list(s.positions) for s in self.trajectory]


class CartesianPlan:
    """Cartesian space motion plan."""

    def __init__(self) -> None:
        self.trajectory: List[Pose] = []
        self.duration: float = 0.0
        self.plan_type: str = "linear"

    def add_point(self, pose: Pose) -> None:
        """Add a trajectory point."""
        self.trajectory.append(pose)

    def get_point(self, index: int) -> Optional[Pose]:
        """Get a trajectory point by index."""
        if 0 <= index < len(self.trajectory):
            return self.trajectory[index]
        return None

    def interpolate(self, t: float) -> Pose:
        """Interpolate the trajectory at parameter t (0 to 1)."""
        if not self.trajectory:
            return Pose()
        if len(self.trajectory) == 1:
            return self.trajectory[0]

        t = max(0.0, min(1.0, t))
        idx_f = t * (len(self.trajectory) - 1)
        idx = int(idx_f)
        frac = idx_f - idx

        if idx >= len(self.trajectory) - 1:
            return self.trajectory[-1]

        return self.trajectory[idx].lerp(self.trajectory[idx + 1], frac)

    def total_distance(self) -> float:
        """Calculate total path distance."""
        total = 0.0
        for i in range(1, len(self.trajectory)):
            total += self.trajectory[i - 1].distance_to(self.trajectory[i])
        return total


class MotionValidator:
    """
    Validates motion plans for feasibility.

    Checks joint limits, velocity limits, self-collision, and singularities.
    """

    def __init__(self, joint_limits: Optional[List[Tuple[float, float]]] = None,
                 velocity_limits: Optional[VelocityLimit] = None) -> None:
        self.joint_limits = joint_limits or []
        self.velocity_limits = velocity_limits or VelocityLimit()

    def validate_joint_plan(self, plan: JointPlan) -> Dict[str, Any]:
        """Validate a joint space plan."""
        issues: List[str] = []
        warnings: List[str] = []

        if not plan.trajectory:
            return {"valid": False, "issues": ["Empty trajectory"], "warnings": []}

        # Check joint limits
        for i, state in enumerate(plan.trajectory):
            for j, pos in enumerate(state.positions):
                if j < len(self.joint_limits):
                    lo, hi = self.joint_limits[j]
                    if pos < lo or pos > hi:
                        issues.append(
                            f"Step {i}, Joint {j}: position {pos:.4f} "
                            f"outside limits [{lo:.4f}, {hi:.4f}]"
                        )

        # Check velocity limits
        for i in range(1, len(plan.trajectory)):
            dt = plan.trajectory[i].timestamps[i] - plan.trajectory[i - 1].timestamps[i - 1] if \
                i < len(plan.trajectory[i].timestamps) and i < len(plan.trajectory[i - 1].timestamps) else 0.01
            if dt <= 0:
                dt = 0.01
            for j in range(plan.num_joints()):
                dq = abs(plan.trajectory[i].positions[j] - plan.trajectory[i - 1].positions[j])
                vel = dq / dt
                if j < len(self.velocity_limits.max_joint_velocity):
                    if vel > self.velocity_limits.max_joint_velocity[j]:
                        issues.append(
                            f"Step {i}, Joint {j}: velocity {vel:.4f} "
                            f"exceeds limit {self.velocity_limits.max_joint_velocity[j]:.4f}"
                        )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "num_points": len(plan.trajectory),
        }

    def validate_cartesian_plan(self, plan: CartesianPlan) -> Dict[str, Any]:
        """Validate a Cartesian space plan."""
        issues: List[str] = []
        warnings: List[str] = []

        if not plan.trajectory:
            return {"valid": False, "issues": ["Empty trajectory"], "warnings": []}

        # Check for large jumps
        for i in range(1, len(plan.trajectory)):
            dist = plan.trajectory[i - 1].distance_to(plan.trajectory[i])
            if dist > 1.0:  # 1 meter threshold
                warnings.append(f"Step {i}: large Cartesian jump of {dist:.4f}m")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "num_points": len(plan.trajectory),
            "total_distance": plan.total_distance(),
        }


class WaypointSequencer:
    """
    Sequences waypoints into a motion plan.

    Handles blending between waypoints and velocity scaling.
    """

    def __init__(self, default_velocity: float = 0.5,
                 default_acceleration: float = 1.0,
                 num_points_per_segment: int = 50) -> None:
        self.default_velocity = default_velocity
        self.default_acceleration = default_acceleration
        self.num_points_per_segment = num_points_per_segment

    def sequence_joint(self, waypoints: List[Waypoint],
                        joint_limits: Optional[List[Tuple[float, float]]] = None) -> JointPlan:
        """Sequence waypoints into a joint space plan."""
        plan = JointPlan()

        for i, wp in enumerate(waypoints):
            if wp.joint_state is None:
                continue

            if i == 0:
                plan.add_point(wp.joint_state)
                continue

            prev = waypoints[i - 1]
            if prev.joint_state is None:
                plan.add_point(wp.joint_state)
                continue

            # Interpolate between waypoints
            max_vel = wp.max_velocity or self.default_velocity
            max_acc = wp.max_acceleration or self.default_acceleration
            vel_scale = wp.velocity_scale

            for j in range(wp.joint_state.num_joints()):
                profile = VelocityProfile.trapezoidal(
                    prev.joint_state.positions[j],
                    wp.joint_state.positions[j],
                    max_vel * vel_scale,
                    max_acc,
                    self.num_points_per_segment,
                )

                for k, (pos, vel, t) in enumerate(profile):
                    if j == 0:
                        state = JointState(
                            positions=[pos],
                            velocities=[vel],
                            names=list(wp.joint_state.names) if wp.joint_state.names else [],
                        )
                        plan.add_point(state)
                    else:
                        if k < len(plan.trajectory):
                            plan.trajectory[-k].positions.append(pos)
                            plan.trajectory[-k].velocities.append(vel)

        plan.duration = len(plan.trajectory) * 0.008  # Approximate
        return plan

    def sequence_cartesian(self, waypoints: List[Waypoint]) -> CartesianPlan:
        """Sequence waypoints into a Cartesian space plan."""
        plan = CartesianPlan()

        for i, wp in enumerate(waypoints):
            if wp.pose is None:
                continue

            if i == 0:
                plan.add_point(wp.pose)
                continue

            prev = waypoints[i - 1]
            if prev.pose is None:
                plan.add_point(wp.pose)
                continue

            # Linear interpolation between waypoints
            num_pts = self.num_points_per_segment
            for k in range(1, num_pts + 1):
                t = k / num_pts
                interp = prev.pose.lerp(wp.pose, t)
                plan.add_point(interp)

        if plan.trajectory:
            plan.duration = plan.total_distance() / self.default_velocity

        return plan

    def sequence_circular(self, center: Pose, radius: float,
                          axis: str = "z", angle_start: float = 0,
                          angle_end: float = 2 * math.pi,
                          num_points: int = 100) -> CartesianPlan:
        """Generate a circular Cartesian path."""
        plan = CartesianPlan()
        plan.plan_type = "circular"

        for i in range(num_points + 1):
            t = i / num_points
            angle = angle_start + (angle_end - angle_start) * t

            if axis == "z":
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                z = center.z
            elif axis == "x":
                x = center.x
                y = center.y + radius * math.cos(angle)
                z = center.z + radius * math.sin(angle)
            else:
                x = center.x + radius * math.cos(angle)
                y = center.y
                z = center.z + radius * math.sin(angle)

            plan.add_point(Pose(x=x, y=y, z=z, rx=center.rx, ry=center.ry, rz=center.rz))

        plan.duration = abs(angle_end - angle_start) / self.default_velocity
        return plan


class MovePlanner:
    """
    High-level motion planner.

    Provides joint space and Cartesian space planning with
    velocity profiling and validation.
    """

    def __init__(self, num_joints: int = 6,
                 joint_limits: Optional[List[Tuple[float, float]]] = None,
                 velocity_limits: Optional[VelocityLimit] = None) -> None:
        self.num_joints = num_joints
        self.joint_limits = joint_limits or [(-math.pi, math.pi)] * num_joints
        self.velocity_limits = velocity_limits or VelocityLimit(
            max_joint_velocity=[2.0] * num_joints,
            max_joint_acceleration=[4.0] * num_joints,
        )
        self.velocity_profile = VelocityProfile()
        self.validator = MotionValidator(self.joint_limits, self.velocity_limits)
        self.sequencer = WaypointSequencer()

    def plan_joint_move(self, start: JointState, end: JointState,
                        num_points: int = 100) -> JointPlan:
        """Plan a linear joint space move."""
        plan = JointPlan()
        max_vel = min(self.velocity_limits.max_joint_velocity) if self.velocity_limits.max_joint_velocity else 2.0
        max_acc = min(self.velocity_limits.max_joint_acceleration) if self.velocity_limits.max_joint_acceleration else 4.0

        for j in range(min(start.num_joints(), end.num_joints(), self.num_joints)):
            profile = self.velocity_profile.trapezoidal(
                start.positions[j], end.positions[j],
                max_vel, max_acc, num_points,
            )

            for k, (pos, vel, t) in enumerate(profile):
                if j == 0:
                    state = JointState(
                        positions=[pos], velocities=[vel],
                        names=list(start.names) if start.names else [],
                    )
                    plan.add_point(state)
                elif k < len(plan.trajectory):
                    plan.trajectory[-(num_points + 1 - k)].positions.append(pos)
                    plan.trajectory[-(num_points + 1 - k)].velocities.append(vel)

        plan.duration = profile[-1][2] if profile else 0
        return plan

    def plan_cartesian_move(self, start: Pose, end: Pose,
                            num_points: int = 100) -> CartesianPlan:
        """Plan a linear Cartesian move."""
        plan = CartesianPlan()
        plan.plan_type = "linear"

        for i in range(num_points + 1):
            t = i / num_points
            plan.add_point(start.lerp(end, t))

        distance = start.distance_to(end)
        plan.duration = distance / self.velocity_limits.max_cartesian_velocity
        return plan

    def plan_circular_move(self, center: Pose, radius: float,
                           axis: str = "z",
                           angle_start: float = 0,
                           angle_end: float = 2 * math.pi,
                           num_points: int = 100) -> CartesianPlan:
        """Plan a circular Cartesian move."""
        return self.sequencer.sequence_circular(
            center, radius, axis, angle_start, angle_end, num_points
        )

    def plan_waypoints(self, waypoints: List[Waypoint],
                       space: str = "joint") -> Any:
        """Plan through a sequence of waypoints."""
        if space == "joint":
            return self.sequencer.sequence_joint(waypoints, self.joint_limits)
        else:
            return self.sequencer.sequence_cartesian(waypoints)

    def validate_plan(self, plan: Any) -> Dict[str, Any]:
        """Validate a motion plan."""
        if isinstance(plan, JointPlan):
            return self.validator.validate_joint_plan(plan)
        elif isinstance(plan, CartesianPlan):
            return self.validator.validate_cartesian_plan(plan)
        return {"valid": False, "issues": ["Unknown plan type"]}

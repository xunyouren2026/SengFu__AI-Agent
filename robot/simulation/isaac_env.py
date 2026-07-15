"""
Isaac Sim Environment Wrapper Module

Simulates the Isaac Sim environment for robot simulation:
- Scene management
- Physics stepping
- Sensor simulation (camera, lidar, force/torque)
- Domain randomization
- Asset management
- Simulation clock

Pure Python standard library only.
"""

from __future__ import annotations

import math
import time
import random
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


class PhysicsEngine(Enum):
    """Physics engine types."""
    PHYSX = "physx"
    ODE = "ode"
    BULLET = "bullet"


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> Vector3:
        m = self.magnitude()
        return Vector3(self.x / m, self.y / m, self.z / m) if m > 0 else Vector3()

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: List[float]) -> Vector3:
        if len(values) >= 3:
            return cls(values[0], values[1], values[2])
        return cls()


@dataclass
class Quaternion:
    """Quaternion rotation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalize(self) -> Quaternion:
        m = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if m > 0:
            return Quaternion(self.w/m, self.x/m, self.y/m, self.z/m)
        return Quaternion()

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion."""
        qv = Quaternion(0, v.x, v.y, v.z)
        q_conj = Quaternion(self.w, -self.x, -self.y, -self.z)
        result = self._multiply(qv)._multiply(q_conj)
        return Vector3(result.x, result.y, result.z)

    def _multiply(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
        )

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        return cls(cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                   cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


@dataclass
class Transform:
    """3D transform (position + rotation)."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    def to_matrix(self) -> List[List[float]]:
        """Convert to 4x4 transformation matrix."""
        q = self.rotation.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        return [
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), self.position.x],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), self.position.y],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), self.position.z],
            [0, 0, 0, 1],
        ]


@dataclass
class RigidBody:
    """A rigid body in the physics scene."""
    name: str
    transform: Transform = field(default_factory=Transform)
    velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3
    is_static: bool = False
    collision_enabled: bool = True
    asset_path: str = ""


@dataclass
class SensorData:
    """Data from a simulated sensor."""
    sensor_name: str
    data_type: str
    timestamp: float
    data: Any = None
    shape: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationClock:
    """Manages simulation time stepping."""

    def __init__(self, time_step: float = 1.0 / 60.0) -> None:
        self.time_step = time_step
        self.current_time: float = 0.0
        self.step_count: int = 0
        self._start_wall_time: float = time.monotonic()
        self._paused: bool = False
        self._time_scale: float = 1.0

    def step(self) -> float:
        """Advance the simulation by one time step."""
        if self._paused:
            return self.current_time
        self.current_time += self.time_step * self._time_scale
        self.step_count += 1
        return self.current_time

    def reset(self) -> None:
        """Reset the clock."""
        self.current_time = 0.0
        self.step_count = 0
        self._start_wall_time = time.monotonic()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def wall_time(self) -> float:
        return time.monotonic() - self._start_wall_time

    @property
    def fps(self) -> float:
        if self.wall_time > 0:
            return self.step_count / self.wall_time
        return 0.0

    def set_time_scale(self, scale: float) -> None:
        self._time_scale = max(0.0, min(10.0, scale))


class PhysicsScene:
    """Simulated physics scene."""

    def __init__(self, gravity: Vector3 = None, engine: PhysicsEngine = PhysicsEngine.PHYSX) -> None:
        self.gravity = gravity or Vector3(0, -9.81, 0)
        self.engine = engine
        self.bodies: Dict[str, RigidBody] = {}
        self._contacts: List[Dict[str, Any]] = []

    def add_body(self, body: RigidBody) -> None:
        self.bodies[body.name] = body

    def remove_body(self, name: str) -> bool:
        return self.bodies.pop(name, None) is not None

    def get_body(self, name: str) -> Optional[RigidBody]:
        return self.bodies.get(name)

    def step(self, dt: float) -> List[Dict[str, Any]]:
        """Step the physics simulation."""
        contacts: List[Dict[str, Any]] = []
        for body in self.bodies.values():
            if body.is_static:
                continue
            # Apply gravity
            body.velocity = body.velocity + self.gravity * dt
            # Update position
            body.transform.position = body.transform.position + body.velocity * dt
            # Simple ground collision
            if body.transform.position.y < 0:
                body.transform.position.y = 0
                body.velocity = Vector3(body.velocity.x, -body.velocity.y * body.restitution, body.velocity.z)
                body.velocity = body.velocity * (1 - body.friction * dt)
                contacts.append({"body_a": body.name, "body_b": "ground", "point": [body.transform.position.x, 0, body.transform.position.z]})
        self._contacts = contacts
        return contacts

    def get_contacts(self) -> List[Dict[str, Any]]:
        return list(self._contacts)

    def raycast(self, origin: Vector3, direction: Vector3,
                 max_distance: float = 100.0) -> Optional[Dict[str, Any]]:
        """Simple raycast simulation."""
        direction = direction.normalized()
        closest: Optional[Dict[str, Any]] = None
        closest_dist = max_distance
        for body in self.bodies.values():
            if not body.collision_enabled:
                continue
            to_body = body.transform.position - origin
            proj = to_body.dot(direction)
            if proj < 0 or proj > closest_dist:
                continue
            closest_point = origin + direction * proj
            dist = closest_point.distance_to(body.transform.position)
            if dist < 0.5 and proj < closest_dist:
                closest_dist = proj
                closest = {"body": body.name, "distance": proj, "point": closest_point.to_list()}
        return closest


class SensorSuite:
    """Simulated sensor suite."""

    def __init__(self) -> None:
        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._data_buffer: Dict[str, List[SensorData]] = {}
        self._callbacks: Dict[str, List[Callable[[SensorData], None]]] = {}

    def add_sensor(self, name: str, sensor_type: str,
                   transform: Transform = None, config: Dict[str, Any] = None) -> None:
        self._sensors[name] = {
            "type": sensor_type,
            "transform": transform or Transform(),
            "config": config or {},
            "active": True,
        }
        self._data_buffer[name] = []

    def remove_sensor(self, name: str) -> bool:
        self._sensors.pop(name, None)
        self._data_buffer.pop(name, None)
        return True

    def register_callback(self, sensor_name: str,
                          callback: Callable[[SensorData], None]) -> None:
        if sensor_name not in self._callbacks:
            self._callbacks[sensor_name] = []
        self._callbacks[sensor_name].append(callback)

    def update(self, scene: PhysicsScene, sim_time: float) -> Dict[str, SensorData]:
        """Update all sensors and return latest data."""
        results: Dict[str, SensorData] = {}
        for name, sensor in self._sensors.items():
            if not sensor["active"]:
                continue
            data = self._simulate_sensor(name, sensor, scene, sim_time)
            self._data_buffer[name].append(data)
            if len(self._data_buffer[name]) > 100:
                self._data_buffer[name] = self._data_buffer[name][-100:]
            results[name] = data
            for cb in self._callbacks.get(name, []):
                try:
                    cb(data)
                except Exception:
                    pass
        return results

    def _simulate_sensor(self, name: str, sensor: Dict[str, Any],
                         scene: PhysicsScene, sim_time: float) -> SensorData:
        """Simulate sensor data generation."""
        stype = sensor["type"]
        if stype == "camera":
            return self._simulate_camera(name, sensor, sim_time)
        elif stype == "lidar":
            return self._simulate_lidar(name, sensor, scene, sim_time)
        elif stype == "force_torque":
            return self._simulate_ft(name, sensor, sim_time)
        elif stype == "imu":
            return self._simulate_imu(name, sensor, sim_time)
        return SensorData(name, stype, sim_time)

    def _simulate_camera(self, name: str, sensor: Dict[str, Any],
                         sim_time: float) -> SensorData:
        w = sensor["config"].get("width", 640)
        h = sensor["config"].get("height", 480)
        rng = random.Random(hash(name) + int(sim_time * 10))
        # Simulate RGB image data as flattened array
        data = [rng.randint(0, 255) for _ in range(w * h * 3)]
        return SensorData(name, "camera_rgb", sim_time, data, (h, w, 3),
                          {"width": w, "height": h, "channels": 3})

    def _simulate_lidar(self, name: str, sensor: Dict[str, Any],
                        scene: PhysicsScene, sim_time: float) -> SensorData:
        num_points = sensor["config"].get("num_points", 360)
        max_range = sensor["config"].get("max_range", 10.0)
        origin = sensor["transform"].position
        rng = random.Random(hash(name) + int(sim_time * 10))
        points: List[List[float]] = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = rng.uniform(0.5, max_range)
            x = origin.x + dist * math.cos(angle)
            z = origin.z + dist * math.sin(angle)
            y = origin.y + rng.gauss(0, 0.05)
            points.append([x, y, z])
        return SensorData(name, "lidar", sim_time, points, (num_points, 3),
                          {"num_points": num_points, "max_range": max_range})

    def _simulate_ft(self, name: str, sensor: Dict[str, Any],
                     sim_time: float) -> SensorData:
        rng = random.Random(hash(name) + int(sim_time * 100))
        force = [rng.gauss(0, 0.1) for _ in range(3)]
        torque = [rng.gauss(0, 0.01) for _ in range(3)]
        return SensorData(name, "force_torque", sim_time,
                          {"force": force, "torque": torque}, (6,))

    def _simulate_imu(self, name: str, sensor: Dict[str, Any],
                      sim_time: float) -> SensorData:
        rng = random.Random(hash(name) + int(sim_time * 100))
        accel = [rng.gauss(0, 0.01), rng.gauss(-9.81, 0.01), rng.gauss(0, 0.01)]
        gyro = [rng.gauss(0, 0.001) for _ in range(3)]
        return SensorData(name, "imu", sim_time,
                          {"acceleration": accel, "gyroscope": gyro}, (6,))

    def get_data(self, sensor_name: str) -> List[SensorData]:
        return list(self._data_buffer.get(sensor_name, []))

    def get_latest(self, sensor_name: str) -> Optional[SensorData]:
        buf = self._data_buffer.get(sensor_name, [])
        return buf[-1] if buf else None


class DomainRandomizer:
    """Domain randomization for sim2real transfer."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self._active: bool = True

    def randomize_lighting(self) -> Dict[str, float]:
        return {
            "ambient_intensity": self.rng.uniform(0.3, 1.0),
            "directional_intensity": self.rng.uniform(0.5, 2.0),
            "light_x": self.rng.uniform(-5, 5),
            "light_y": self.rng.uniform(5, 15),
            "light_z": self.rng.uniform(-5, 5),
        }

    def randomize_texture(self) -> Dict[str, Any]:
        textures = ["checkerboard", "noise", "solid", "grid", "wood", "metal"]
        return {
            "texture_type": self.rng.choice(textures),
            "texture_scale": self.rng.uniform(0.5, 5.0),
            "roughness": self.rng.uniform(0.1, 1.0),
            "metalness": self.rng.uniform(0.0, 1.0),
        }

    def randomize_camera(self) -> Dict[str, Any]:
        return {
            "focal_length": self.rng.uniform(300, 800),
            "sensor_width": self.rng.choice([640, 800, 1024, 1280]),
            "sensor_height": self.rng.choice([480, 600, 768, 960]),
            "noise_std": self.rng.uniform(0, 0.05),
        }

    def randomize_physics(self) -> Dict[str, float]:
        return {
            "gravity_scale": self.rng.uniform(0.9, 1.1),
            "friction_range": (self.rng.uniform(0.3, 0.7), self.rng.uniform(0.3, 0.7)),
            "restitution_range": (self.rng.uniform(0.1, 0.5), self.rng.uniform(0.1, 0.5)),
        }

    def randomize_object_pose(self, x_range: Tuple[float, float] = (-0.5, 0.5),
                                y_range: Tuple[float, float] = (0, 0.5),
                                z_range: Tuple[float, float] = (-0.5, 0.5)) -> Transform:
        return Transform(
            position=Vector3(
                self.rng.uniform(*x_range),
                self.rng.uniform(*y_range),
                self.rng.uniform(*z_range),
            ),
            rotation=Quaternion.from_euler(
                self.rng.uniform(0, 2*math.pi),
                self.rng.uniform(0, 2*math.pi),
                self.rng.uniform(0, 2*math.pi),
            ),
        )

    def randomize_all(self) -> Dict[str, Any]:
        return {
            "lighting": self.randomize_lighting(),
            "texture": self.randomize_texture(),
            "camera": self.randomize_camera(),
            "physics": self.randomize_physics(),
        }


class AssetManager:
    """Manages simulation assets."""

    def __init__(self) -> None:
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._loaded: Dict[str, bool] = {}

    def register_asset(self, name: str, asset_path: str,
                       asset_type: str = "usd") -> None:
        self._assets[name] = {"path": asset_path, "type": asset_type, "loaded": False}

    def load_asset(self, name: str) -> bool:
        if name in self._assets:
            self._assets[name]["loaded"] = True
            self._loaded[name] = True
            return True
        return False

    def unload_asset(self, name: str) -> bool:
        if name in self._assets:
            self._assets[name]["loaded"] = False
            self._loaded.pop(name, None)
            return True
        return False

    def is_loaded(self, name: str) -> bool:
        return self._loaded.get(name, False)

    def list_assets(self) -> List[str]:
        return list(self._assets.keys())

    def get_asset_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self._assets.get(name)


class IsaacEnvironment:
    """High-level Isaac Sim environment wrapper."""

    def __init__(self, physics_engine: PhysicsEngine = PhysicsEngine.PHYSX,
                 time_step: float = 1.0/60.0) -> None:
        self.clock = SimulationClock(time_step)
        self.physics = PhysicsScene(engine=physics_engine)
        self.sensors = SensorSuite()
        self.domain_randomizer = DomainRandomizer()
        self.asset_manager = AssetManager()
        self._running = False
        self._step_callbacks: List[Callable[[float], None]] = []

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        gravity = config.get("gravity", [0, -9.81, 0])
        self.physics.gravity = Vector3(*gravity)
        ts = config.get("time_step", 1.0/60.0)
        self.clock = SimulationClock(ts)

    def step(self, num_steps: int = 1) -> Dict[str, Any]:
        results: Dict[str, Any] = {"steps": num_steps, "time": self.clock.current_time}
        for _ in range(num_steps):
            self.physics.step(self.clock.time_step)
            self.clock.step()
            sensor_data = self.sensors.update(self.physics, self.clock.current_time)
            results["sensor_data"] = sensor_data
            for cb in self._step_callbacks:
                try:
                    cb(self.clock.current_time)
                except Exception:
                    pass
        return results

    def reset(self) -> None:
        self.clock.reset()
        for body in self.physics.bodies.values():
            body.velocity = Vector3()
            body.angular_velocity = Vector3()

    def register_step_callback(self, callback: Callable[[float], None]) -> None:
        self._step_callbacks.append(callback)

    def get_state(self) -> Dict[str, Any]:
        bodies = {}
        for name, body in self.physics.bodies.items():
            bodies[name] = {
                "position": body.transform.position.to_list(),
                "velocity": body.velocity.to_list(),
            }
        return {
            "time": self.clock.current_time,
            "step_count": self.clock.step_count,
            "bodies": bodies,
            "fps": self.clock.fps,
        }

    def save_state(self) -> Dict[str, Any]:
        return {"time": self.clock.current_time, "bodies": self.get_state()["bodies"]}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.clock.current_time = state.get("time", 0)
        for name, body_state in state.get("bodies", {}).items():
            body = self.physics.bodies.get(name)
            if body:
                pos = body_state.get("position", [0, 0, 0])
                body.transform.position = Vector3.from_list(pos)

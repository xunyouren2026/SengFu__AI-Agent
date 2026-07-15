"""
Collision Avoidance Module

Robot collision avoidance algorithms:
- AABB collision detection
- GJK (Gilbert-Johnson-Keerthi) algorithm
- Separating Axis Theorem (SAT)
- Potential field method
- RRT (Rapidly-exploring Random Tree) path planning
- Configuration space obstacle building

Pure Python standard library only.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable


@dataclass
class Vec3:
    """3D vector for collision math."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: Vec3) -> Vec3:
        return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)

    def __sub__(self, o: Vec3) -> Vec3:
        return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x*s, self.y*s, self.z*s)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o: Vec3) -> float:
        return self.x*o.x + self.y*o.y + self.z*o.z

    def cross(self, o: Vec3) -> Vec3:
        return Vec3(self.y*o.z-self.z*o.y, self.z*o.x-self.x*o.z, self.x*o.y-self.y*o.x)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def length_sq(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    def normalized(self) -> Vec3:
        l = self.length()
        return Vec3(self.x/l, self.y/l, self.z/l) if l > 1e-10 else Vec3()

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""
    min_corner: Vec3
    max_corner: Vec3

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.min_corner.x + self.max_corner.x) / 2,
            (self.min_corner.y + self.max_corner.y) / 2,
            (self.min_corner.z + self.max_corner.z) / 2,
        )

    @property
    def half_extents(self) -> Vec3:
        c = self.center
        return Vec3(self.max_corner.x - c.x, self.max_corner.y - c.y, self.max_corner.z - c.z)

    def intersects(self, other: AABB) -> bool:
        return (self.min_corner.x <= other.max_corner.x and
                self.max_corner.x >= other.min_corner.x and
                self.min_corner.y <= other.max_corner.y and
                self.max_corner.y >= other.min_corner.y and
                self.min_corner.z <= other.max_corner.z and
                self.max_corner.z >= other.min_corner.z)

    def contains_point(self, p: Vec3) -> bool:
        return (self.min_corner.x <= p.x <= self.max_corner.x and
                self.min_corner.y <= p.y <= self.max_corner.y and
                self.min_corner.z <= p.z <= self.max_corner.z)

    def merge(self, other: AABB) -> AABB:
        return AABB(
            Vec3(min(self.min_corner.x, other.min_corner.x),
                 min(self.min_corner.y, other.min_corner.y),
                 min(self.min_corner.z, other.min_corner.z)),
            Vec3(max(self.max_corner.x, other.max_corner.x),
                 max(self.max_corner.y, other.max_corner.y),
                 max(self.max_corner.z, other.max_corner.z)),
        )

    def expand(self, margin: float) -> AABB:
        m = Vec3(margin, margin, margin)
        return AABB(self.min_corner - m, self.max_corner + m)

    def surface_area(self) -> float:
        d = self.max_corner - self.min_corner
        return 2 * (d.x*d.y + d.y*d.z + d.x*d.z)

    def volume(self) -> float:
        d = self.max_corner - self.min_corner
        return d.x * d.y * d.z


class AABBTree:
    """Bounding Volume Hierarchy using AABBs."""

    def __init__(self) -> None:
        self.root: Optional[_AABBNode] = None
        self._objects: List[Tuple[AABB, Any]] = []

    def build(self, objects: List[Tuple[AABB, Any]]) -> None:
        self._objects = objects
        if not objects:
            self.root = None
            return
        self.root = self._build_recursive(list(range(len(objects))))

    def _build_recursive(self, indices: List[int]) -> _AABBNode:
        if len(indices) == 1:
            aabb, obj = self._objects[indices[0]]
            return _AABBNode(aabb, obj=obj)
        # Compute merged AABB
        merged = self._objects[indices[0]][0]
        for i in indices[1:]:
            merged = merged.merge(self._objects[i][0])
        # Split along longest axis
        d = merged.max_corner - merged.min_corner
        axis = 0
        if d.y > d.x:
            axis = 1
        if d.z > d.y and d.z > d.x:
            axis = 2
        indices.sort(key=lambda i: (self._objects[i][0].min_corner.x,
                                     self._objects[i][0].min_corner.y,
                                     self._objects[i][0].min_corner.z)[axis])
        mid = len(indices) // 2
        left = self._build_recursive(indices[:mid])
        right = self._build_recursive(indices[mid:])
        return _AABBNode(merged, left=left, right=right)

    def query(self, aabb: AABB) -> List[Any]:
        results: List[Any] = []
        if self.root:
            self._query_recursive(self.root, aabb, results)
        return results

    def _query_recursive(self, node: _AABBNode, aabb: AABB, results: List[Any]) -> None:
        if not node.aabb.intersects(aabb):
            return
        if node.obj is not None:
            results.append(node.obj)
        if node.left:
            self._query_recursive(node.left, aabb, results)
        if node.right:
            self._query_recursive(node.right, aabb, results)

    def query_point(self, point: Vec3) -> List[Any]:
        results: List[Any] = []
        if self.root:
            self._query_point_recursive(self.root, point, results)
        return results

    def _query_point_recursive(self, node: _AABBNode, point: Vec3, results: List[Any]) -> None:
        if not node.aabb.contains_point(point):
            return
        if node.obj is not None:
            results.append(node.obj)
        if node.left:
            self._query_point_recursive(node.left, point, results)
        if node.right:
            self._query_point_recursive(node.right, point, results)


class _AABBNode:
    def __init__(self, aabb: AABB, obj: Any = None,
                 left: Optional[_AABBNode] = None,
                 right: Optional[_AABBNode] = None) -> None:
        self.aabb = aabb
        self.obj = obj
        self.left = left
        self.right = right


class CollisionChecker:
    """High-level collision checking interface."""

    def __init__(self) -> None:
        self._static_tree = AABBTree()
        self._dynamic_objects: List[Tuple[AABB, Any]] = []

    def set_static_obstacles(self, obstacles: List[Tuple[AABB, Any]]) -> None:
        self._static_tree.build(obstacles)

    def add_dynamic(self, aabb: AABB, obj: Any) -> None:
        self._dynamic_objects.append((aabb, obj))

    def clear_dynamic(self) -> None:
        self._dynamic_objects.clear()

    def check_static(self, aabb: AABB) -> List[Any]:
        return self._static_tree.query(aabb)

    def check_dynamic(self, aabb: AABB) -> List[Any]:
        results: List[Any] = []
        for other_aabb, obj in self._dynamic_objects:
            if aabb.intersects(other_aabb):
                results.append(obj)
        return results

    def check_all(self, aabb: AABB) -> List[Any]:
        return self.check_static(aabb) + self.check_dynamic(aabb)

    def check_point(self, point: Vec3) -> List[Any]:
        return self._static_tree.query_point(point)


class GJKSolver:
    """
    Gilbert-Johnson-Keerthi (GJK) algorithm for convex collision detection.

    Determines if two convex shapes are intersecting using the Minkowski
    difference and support functions.
    """

    @staticmethod
    def support(vertices: List[Vec3], direction: Vec3) -> Vec3:
        """Get the support point of a convex shape in a direction."""
        best = vertices[0]
        best_dot = best.dot(direction)
        for v in vertices[1:]:
            d = v.dot(direction)
            if d > best_dot:
                best_dot = d
                best = v
        return best

    @staticmethod
    def minkowski_support(verts_a: List[Vec3], verts_b: List[Vec3],
                          direction: Vec3) -> Vec3:
        """Support function for the Minkowski difference A - B."""
        sa = GJKSolver.support(verts_a, direction)
        sb = GJKSolver.support(verts_b, -direction)
        return sa - sb

    @staticmethod
    def triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
        """Triple product: (A x B) x C."""
        return b * a.dot(c) - a * b.dot(c)

    def intersects(self, verts_a: List[Vec3], verts_b: List[Vec3],
                   max_iterations: int = 64) -> bool:
        """Check if two convex shapes intersect using GJK."""
        if len(verts_a) < 4 or len(verts_b) < 4:
            return self._fallback_check(verts_a, verts_b)

        direction = Vec3(1, 0, 0)
        simplex: List[Vec3] = [self.minkowski_support(verts_a, verts_b, direction)]
        direction = -simplex[0]

        for _ in range(max_iterations):
            if direction.length() < 1e-10:
                return True

            new_point = self.minkowski_support(verts_a, verts_b, direction)
            if new_point.dot(direction) < 0:
                return False

            simplex.append(new_point)
            result = self._process_simplex(simplex, direction)
            if result:
                return True

        return False

    def _process_simplex(self, simplex: List[Vec3], direction: Vec3) -> bool:
        """Process the simplex and update direction."""
        n = len(simplex)
        if n == 2:
            return self._line_case(simplex, direction)
        elif n == 3:
            return self._triangle_case(simplex, direction)
        elif n == 4:
            return self._tetrahedron_case(simplex, direction)
        return False

    def _line_case(self, simplex: List[Vec3], direction: Vec3) -> bool:
        a, b = simplex[1], simplex[0]
        ab = b - a
        ao = -a
        if ab.dot(ao) > 0:
            direction.x = ab.y * ao.z - ab.z * ao.y
            direction.y = ab.z * ao.x - ab.x * ao.z
            direction.z = ab.x * ao.y - ab.y * ao.x
        else:
            simplex.clear()
            simplex.append(a)
            direction = ao
        return False

    def _triangle_case(self, simplex: List[Vec3], direction: Vec3) -> bool:
        a, b, c = simplex[2], simplex[1], simplex[0]
        ab = b - a
        ac = c - a
        ao = -a
        abc = ab.cross(ac)

        if abc.cross(ac).dot(ao) > 0:
            if ac.dot(ao) > 0:
                simplex.pop(1)
                return self._line_case([a, c], direction)
            else:
                simplex.pop(0)
                return self._line_case([a, b], direction)
        else:
            if ab.cross(abc).dot(ao) > 0:
                simplex.pop(0)
                return self._line_case([a, b], direction)
            else:
                if abc.dot(ao) > 0:
                    direction = abc
                else:
                    simplex.reverse()
                    direction = -abc
                return False
        return False

    def _tetrahedron_case(self, simplex: List[Vec3], direction: Vec3) -> bool:
        a, b, c, d = simplex[3], simplex[2], simplex[1], simplex[0]
        ab = b - a
        ac = c - a
        ad = d - a
        ao = -a

        abc = ab.cross(ac)
        acd = ac.cross(ad)
        adb = ad.cross(ab)

        if abc.dot(ao) > 0:
            if adb.dot(ao) > 0:
                simplex.pop(1)
                simplex.pop(1)
                return self._triangle_case([a, d, b], direction)
            elif acd.dot(ao) > 0:
                simplex.pop(2)
                return self._triangle_case([a, c, d], direction)
            else:
                simplex.pop(3)
                return self._triangle_case([a, b, c], direction)
        elif acd.dot(ao) > 0:
            if adb.dot(ao) > 0:
                simplex.pop(1)
                simplex.pop(1)
                return self._triangle_case([a, d, b], direction)
            else:
                simplex.pop(3)
                simplex.pop(2)
                return self._triangle_case([a, c, d], direction)
        elif adb.dot(ao) > 0:
            simplex.pop(2)
            return self._triangle_case([a, d, b], direction)
        else:
            return True  # Origin is inside tetrahedron

    def _fallback_check(self, verts_a: List[Vec3], verts_b: List[Vec3]) -> bool:
        aabb_a = self._compute_aabb(verts_a)
        aabb_b = self._compute_aabb(verts_b)
        return aabb_a.intersects(aabb_b)

    @staticmethod
    def _compute_aabb(vertices: List[Vec3]) -> AABB:
        if not vertices:
            return AABB(Vec3(), Vec3())
        mn = Vec3(float('inf'), float('inf'), float('inf'))
        mx = Vec3(float('-inf'), float('-inf'), float('-inf'))
        for v in vertices:
            mn.x = min(mn.x, v.x); mn.y = min(mn.y, v.y); mn.z = min(mn.z, v.z)
            mx.x = max(mx.x, v.x); mx.y = max(mx.y, v.y); mx.z = max(mx.z, v.z)
        return AABB(mn, mx)


class PotentialField:
    """Potential field method for collision avoidance."""

    def __init__(self, attractive_gain: float = 1.0,
                 repulsive_gain: float = 5.0,
                 influence_distance: float = 2.0,
                 obstacle_threshold: float = 0.5) -> None:
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance
        self.obstacle_threshold = obstacle_threshold

    def compute_force(self, position: Vec3, goal: Vec3,
                      obstacles: List[Vec3]) -> Vec3:
        """Compute the resultant potential field force."""
        # Attractive force toward goal
        to_goal = goal - position
        dist_goal = to_goal.length()
        if dist_goal > 1e-10:
            f_attract = to_goal.normalized() * self.attractive_gain * dist_goal
        else:
            f_attract = Vec3()

        # Repulsive forces from obstacles
        f_repulse = Vec3()
        for obs in obstacles:
            to_robot = position - obs
            dist = to_robot.length()
            if dist < self.influence_distance and dist > 1e-10:
                magnitude = self.repulsive_gain * (1.0/dist - 1.0/self.influence_distance) / (dist**2)
                f_repulse = f_repulse + to_robot.normalized() * magnitude

        return f_attract + f_repulse

    def compute_path(self, start: Vec3, goal: Vec3,
                     obstacles: List[Vec3],
                     step_size: float = 0.1,
                     max_steps: int = 1000) -> List[Vec3]:
        """Follow the potential field to generate a path."""
        path: List[Vec3] = [start]
        current = start
        for _ in range(max_steps):
            force = self.compute_force(current, goal, obstacles)
            if force.length() < 1e-6:
                break
            direction = force.normalized()
            current = current + direction * step_size
            path.append(current)
            if (current - goal).length() < step_size:
                path.append(goal)
                break
        return path


class RRTPlanner:
    """Rapidly-exploring Random Tree path planner."""

    def __init__(self, step_size: float = 0.5,
                 max_iterations: int = 5000,
                 goal_bias: float = 0.1,
                 goal_threshold: float = 0.5,
                 seed: Optional[int] = None) -> None:
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.rng = random.Random(seed)

    def plan(self, start: Vec3, goal: Vec3,
             collision_fn: Callable[[Vec3, Vec3], bool],
             bounds: Optional[Tuple[Vec3, Vec3]] = None) -> List[Vec3]:
        """Plan a path from start to goal avoiding obstacles."""
        nodes: List[Vec3] = [start]
        parents: Dict[int, int] = {}

        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if self.rng.random() < self.goal_bias:
                q_rand = goal
            else:
                if bounds:
                    mn, mx = bounds
                    q_rand = Vec3(
                        self.rng.uniform(mn.x, mx.x),
                        self.rng.uniform(mn.y, mx.y),
                        self.rng.uniform(mn.z, mx.z),
                    )
                else:
                    q_rand = Vec3(
                        self.rng.uniform(start.x - 5, goal.x + 5),
                        self.rng.uniform(start.y - 5, goal.y + 5),
                        self.rng.uniform(start.z - 5, goal.z + 5),
                    )

            # Find nearest node
            nearest_idx = 0
            nearest_dist = (nodes[0] - q_rand).length_sq()
            for i in range(1, len(nodes)):
                d = (nodes[i] - q_rand).length_sq()
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i

            # Steer toward random point
            direction = (q_rand - nodes[nearest_idx]).normalized()
            q_new = nodes[nearest_idx] + direction * self.step_size

            # Collision check
            if collision_fn(nodes[nearest_idx], q_new):
                continue

            new_idx = len(nodes)
            nodes.append(q_new)
            parents[new_idx] = nearest_idx

            # Check if we reached the goal
            if (q_new - goal).length() < self.goal_threshold:
                nodes.append(goal)
                parents[len(nodes) - 1] = new_idx
                return self._extract_path(nodes, parents, len(nodes) - 1)

        return []  # No path found

    def _extract_path(self, nodes: List[Vec3], parents: Dict[int, int],
                      goal_idx: int) -> List[Vec3]:
        """Extract path from tree by backtracking from goal."""
        path: List[Vec3] = []
        idx = goal_idx
        while idx != 0:
            path.append(nodes[idx])
            idx = parents[idx]
        path.append(nodes[0])
        path.reverse()
        return self._smooth_path(path)

    def _smooth_path(self, path: List[Vec3], iterations: int = 50) -> List[Vec3]:
        """Smooth the path by shortcutting."""
        if len(path) <= 2:
            return path
        smoothed = list(path)
        for _ in range(iterations):
            if len(smoothed) <= 2:
                break
            i = self.rng.randint(0, len(smoothed) - 2)
            j = self.rng.randint(i + 1, min(i + 10, len(smoothed) - 1))
            shortcut = smoothed[i:j+1]
            if len(shortcut) >= 2:
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed


class CSpaceBuilder:
    """Builds configuration space obstacle representations."""

    def __init__(self, resolution: float = 0.05) -> None:
        self.resolution = resolution
        self._grid: Dict[Tuple[int, int, int], bool] = {}

    def build_from_obstacles(self, obstacles: List[Tuple[AABB, Any]],
                              robot_radius: float = 0.3,
                              bounds: Optional[Tuple[Vec3, Vec3]] = None) -> None:
        """Build C-space grid from obstacle AABBs."""
        self._grid.clear()
        if bounds is None:
            return
        mn, mx = bounds
        nx = int((mx.x - mn.x) / self.resolution)
        ny = int((mx.y - mn.y) / self.resolution)
        nz = int((mx.z - mn.z) / self.resolution)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x = mn.x + ix * self.resolution
                    y = mn.y + iy * self.resolution
                    z = mn.z + iz * self.resolution
                    point = Vec3(x, y, z)
                    for aabb, _ in obstacles:
                        expanded = aabb.expand(robot_radius)
                        if expanded.contains_point(point):
                            self._grid[(ix, iy, iz)] = True
                            break

    def is_occupied(self, point: Vec3, bounds: Optional[Tuple[Vec3, Vec3]] = None) -> bool:
        if bounds:
            mn = bounds[0]
            ix = int((point.x - mn.x) / self.resolution)
            iy = int((point.y - mn.y) / self.resolution)
            iz = int((point.z - mn.z) / self.resolution)
            return self._grid.get((ix, iy, iz), False)
        return False

    def get_occupied_count(self) -> int:
        return sum(1 for v in self._grid.values() if v)

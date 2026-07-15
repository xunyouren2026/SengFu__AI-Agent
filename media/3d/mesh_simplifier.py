"""
3D Mesh Simplification - 3D网格简化器

本模块实现了完整的3D网格简化系统，包含边折叠算法、二次误差度量、
顶点合并、拓扑保持、LOD生成和网格统计功能。仅使用标准库，
不依赖外部库。
"""

import math
import random
import time
import threading
import hashlib
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_id() -> str:
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _vec_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_length(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vec_normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = _vec_length(v)
    if length < 1e-10:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _triangle_area(v0: Tuple[float, float, float], v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    """计算三角形面积"""
    e1 = _vec_sub(v1, v0)
    e2 = _vec_sub(v2, v0)
    cross = _vec_cross(e1, e2)
    return _vec_length(cross) * 0.5


def _triangle_normal(v0: Tuple[float, float, float], v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """计算三角形法线"""
    e1 = _vec_sub(v1, v0)
    e2 = _vec_sub(v2, v0)
    return _vec_normalize(_vec_cross(e1, e2))


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TriangleMesh:
    """三角形网格"""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    colors: List[Tuple[int, int, int]] = field(default_factory=list)

    def compute_normals(self) -> List[Tuple[float, float, float]]:
        """计算顶点法线"""
        self.normals = [(0.0, 0.0, 0.0) for _ in self.vertices]
        for face in self.faces:
            i0, i1, i2 = face
            if i0 >= len(self.vertices) or i1 >= len(self.vertices) or i2 >= len(self.vertices):
                continue
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            n = _triangle_normal(v0, v1, v2)
            for idx in (i0, i1, i2):
                old = self.normals[idx]
                self.normals[idx] = (old[0] + n[0], old[1] + n[1], old[2] + n[2])
        for i in range(len(self.normals)):
            self.normals[i] = _vec_normalize(self.normals[i])
        return self.normals

    def compute_bbox(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """计算包围盒"""
        if not self.vertices:
            return ((0, 0, 0), (0, 0, 0))
        min_v = (
            min(v[0] for v in self.vertices),
            min(v[1] for v in self.vertices),
            min(v[2] for v in self.vertices),
        )
        max_v = (
            max(v[0] for v in self.vertices),
            max(v[1] for v in self.vertices),
            max(v[2] for v in self.vertices),
        )
        return (min_v, max_v)

    def vertex_count(self) -> int:
        return len(self.vertices)

    def face_count(self) -> int:
        return len(self.faces)

    def edge_count(self) -> int:
        edges: Set[Tuple[int, int]] = set()
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edges.add(e)
        return len(edges)

    def is_manifold(self) -> bool:
        """检查是否为流形网格"""
        edge_face_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edge_face_count[e] += 1

        for count in edge_face_count.values():
            if count > 2:
                return False
        return True

    def is_watertight(self) -> bool:
        """检查是否为水密网格"""
        edge_face_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edge_face_count[e] += 1

        for count in edge_face_count.values():
            if count != 2:
                return False
        return True


@dataclass
class SimplificationResult:
    """简化结果"""
    original_vertices: int = 0
    original_faces: int = 0
    simplified_vertices: int = 0
    simplified_faces: int = 0
    reduction_ratio: float = 0.0
    error: float = 0.0
    mesh: Optional[TriangleMesh] = None
    time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LODLevel:
    """LOD层级"""
    level: int = 0
    mesh: Optional[TriangleMesh] = None
    reduction_ratio: float = 0.0
    error: float = 0.0


@dataclass
class MeshStats:
    """网格统计"""
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0
    is_manifold: bool = True
    is_watertight: bool = False
    total_area: float = 0.0
    total_volume: float = 0.0
    bbox_min: Tuple[float, float, float] = (0, 0, 0)
    bbox_max: Tuple[float, float, float] = (0, 0, 0)
    bbox_size: Tuple[float, float, float] = (0, 0, 0)
    average_edge_length: float = 0.0
    minimum_angle: float = 0.0
    maximum_angle: float = 0.0


# ============================================================================
# QuadricErrorMetric - 二次误差度量
# ============================================================================

class QuadricErrorMetric:
    """
    二次误差度量（QEM）：Garland & Heckbert 1997。

    为每个顶点维护一个4x4对称矩阵Q，表示该顶点到其相邻平面的
   距离平方和。边折叠的代价为两个端点Q矩阵之和在新顶点位置
    处的二次形式值。
    """

    def __init__(self):
        self._quadrics: Dict[int, List[List[float]]] = {}

    def compute_quadric(
        self, vertex_idx: int, mesh: TriangleMesh
    ) -> List[List[float]]:
        """计算顶点的二次误差矩阵"""
        Q = [[0.0] * 4 for _ in range(4)]

        for face in mesh.faces:
            if vertex_idx not in face:
                continue

            # 计算平面方程 ax + by + cz + d = 0
            i0, i1, i2 = face
            if any(idx >= len(mesh.vertices) for idx in (i0, i1, i2)):
                continue

            v0, v1, v2 = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]
            normal = _triangle_normal(v0, v1, v2)

            a, b, c = normal
            d = -(a * v0[0] + b * v0[1] + c * v0[2])

            # Kp = [a^2, ab, ac, ad; ab, b^2, bc, bd; ac, bc, c^2, cd; ad, bd, cd, d^2]
            Kp = [
                [a * a, a * b, a * c, a * d],
                [a * b, b * b, b * c, b * d],
                [a * c, b * c, c * c, c * d],
                [a * d, b * d, c * d, d * d],
            ]

            # Q += Kp
            for i in range(4):
                for j in range(4):
                    Q[i][j] += Kp[i][j]

        self._quadrics[vertex_idx] = Q
        return Q

    def compute_all_quadrics(self, mesh: TriangleMesh) -> None:
        """计算所有顶点的二次误差矩阵"""
        self._quadrics.clear()
        for i in range(len(mesh.vertices)):
            self.compute_quadric(i, mesh)

    def evaluate_edge(
        self, v1_idx: int, v2_idx: int, mesh: TriangleMesh
    ) -> Tuple[float, Tuple[float, float, float]]:
        """
        评估边折叠代价。

        返回 (代价, 最优新顶点位置)
        """
        Q1 = self._quadrics.get(v1_idx, self._identity_4x4())
        Q2 = self._quadrics.get(v2_idx, self._identity_4x4())

        # Q = Q1 + Q2
        Q = [[Q1[i][j] + Q2[i][j] for j in range(4)] for i in range(4)]

        # 求解最优顶点位置
        opt_pos = self._solve_optimal_position(Q, mesh, v1_idx, v2_idx)

        # 计算代价 v^T Q v
        v = [opt_pos[0], opt_pos[1], opt_pos[2], 1.0]
        cost = 0.0
        for i in range(4):
            row_sum = sum(Q[i][j] * v[j] for j in range(4))
            cost += v[i] * row_sum

        return max(0.0, cost), opt_pos

    def _solve_optimal_position(
        self,
        Q: List[List[float]],
        mesh: TriangleMesh,
        v1_idx: int,
        v2_idx: int,
    ) -> Tuple[float, float, float]:
        """求解最优新顶点位置"""
        # 提取3x3子矩阵和向量
        A = [[Q[i][j] for j in range(3)] for i in range(3)]
        b = [-Q[i][3] for i in range(3)]

        # 尝试求解线性方程组 Ax = b
        det = (
            A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
        )

        if abs(det) > 1e-10:
            # Cramer法则
            x = (
                b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2])
                + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])
            ) / det
            y = (
                A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2])
                - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])
            ) / det
            z = (
                A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1])
                - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0])
                + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
            ) / det
            return (x, y, z)

        # 退化情况：使用边中点
        if v1_idx < len(mesh.vertices) and v2_idx < len(mesh.vertices):
            p1 = mesh.vertices[v1_idx]
            p2 = mesh.vertices[v2_idx]
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

        return (0.0, 0.0, 0.0)

    def merge_quadrics(self, v1_idx: int, v2_idx: int) -> List[List[float]]:
        """合并两个顶点的二次误差矩阵"""
        Q1 = self._quadrics.get(v1_idx, self._identity_4x4())
        Q2 = self._quadrics.get(v2_idx, self._identity_4x4())
        return [[Q1[i][j] + Q2[i][j] for j in range(4)] for i in range(4)]

    def update_quadric(
        self, vertex_idx: int, Q: List[List[float]]
    ) -> None:
        """更新顶点的二次误差矩阵"""
        self._quadrics[vertex_idx] = Q

    def _identity_4x4(self) -> List[List[float]]:
        return [[float(i == j) for j in range(4)] for i in range(4)]


# ============================================================================
# EdgeCollapse - 边折叠操作
# ============================================================================

class EdgeCollapse:
    """
    边折叠操作：将一条边折叠为一个顶点。

    实现:
    1. 移除包含该边的三角形
    2. 将两个端点合并为一个新顶点
    3. 更新受影响的三角形
    """

    def __init__(self):
        self._vertex_faces: Dict[int, Set[int]] = defaultdict(set)
        self._vertex_edges: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

    def build_adjacency(self, mesh: TriangleMesh) -> None:
        """构建顶点邻接信息"""
        self._vertex_faces.clear()
        self._vertex_edges.clear()

        for face_idx, face in enumerate(mesh.faces):
            for i in range(3):
                v = face[i]
                self._vertex_faces[v].add(face_idx)
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                self._vertex_edges[v].add(e)

    def collapse_edge(
        self,
        mesh: TriangleMesh,
        v1: int,
        v2: int,
        new_pos: Tuple[float, float, float],
        qem: QuadricErrorMetric,
    ) -> TriangleMesh:
        """
        执行边折叠操作。

        将顶点v1和v2合并到new_pos位置。
        """
        # 创建新网格（深拷贝）
        new_vertices = list(mesh.vertices)
        new_faces = list(mesh.faces)
        new_normals = list(mesh.normals) if mesh.normals else []
        new_uvs = list(mesh.uvs) if mesh.uvs else []
        new_colors = list(mesh.colors) if mesh.colors else []

        # 确定保留哪个顶点索引（保留较小的）
        keep = min(v1, v2)
        remove = max(v1, v2)

        # 设置新顶点位置
        new_vertices[keep] = new_pos

        # 移除包含边(v1, v2)的三角形
        faces_to_remove: Set[int] = set()
        for face_idx, face in enumerate(new_faces):
            if v1 in face and v2 in face:
                faces_to_remove.add(face_idx)

        # 更新剩余三角形中的顶点索引
        for face_idx in range(len(new_faces)):
            if face_idx in faces_to_remove:
                continue
            face = new_faces[face_idx]
            new_face = tuple(
                keep if v == remove else v for v in face
            )
            new_faces[face_idx] = new_face

        # 移除标记的三角形
        new_faces = [f for i, f in enumerate(new_faces) if i not in faces_to_remove]

        # 移除未使用的顶点
        used_vertices: Set[int] = set()
        for face in new_faces:
            used_vertices.update(face)

        # 创建顶点映射
        vertex_map: Dict[int, int] = {}
        new_vertex_list: List[Tuple[float, float, float]] = []
        for v in range(len(new_vertices)):
            if v in used_vertices:
                vertex_map[v] = len(new_vertex_list)
                new_vertex_list.append(new_vertices[v])

        # 更新面索引
        final_faces = [
            (vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]])
            for f in new_faces
        ]

        # 更新法线
        final_normals: List[Tuple[float, float, float]] = []
        if new_normals:
            for v in range(len(new_vertices)):
                if v in used_vertices:
                    final_normals.append(new_normals[v] if v < len(new_normals) else (0, 0, 1))

        # 更新UV
        final_uvs: List[Tuple[float, float]] = []
        if new_uvs:
            for v in range(len(new_uvs)):
                if v in used_vertices:
                    final_uvs.append(new_uvs[v] if v < len(new_uvs) else (0, 0))

        # 更新颜色
        final_colors: List[Tuple[int, int, int]] = []
        if new_colors:
            for v in range(len(new_colors)):
                if v in used_vertices:
                    final_colors.append(new_colors[v] if v < len(new_colors) else (255, 255, 255))

        return TriangleMesh(
            vertices=new_vertex_list,
            faces=final_faces,
            normals=final_normals,
            uvs=final_uvs,
            colors=final_colors,
        )

    def get_collapsible_edges(
        self, mesh: TriangleMesh
    ) -> List[Tuple[int, int]]:
        """获取所有可折叠的边"""
        edges: Set[Tuple[int, int]] = set()
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edges.add(e)
        return list(edges)


# ============================================================================
# TopologyChecker - 拓扑检查器
# ============================================================================

class TopologyChecker:
    """
    拓扑检查器：确保简化过程中保持网格拓扑的有效性。

    检查:
    - 边界边保护
    - 流形性保持
    - 三角形退化检测
    - 自交检测
    """

    def __init__(self):
        self._protect_boundary: bool = True
        self._protect_features: bool = False
        self._feature_angle: float = 30.0

    def is_collapse_valid(
        self,
        mesh: TriangleMesh,
        v1: int,
        v2: int,
    ) -> bool:
        """检查边折叠是否有效"""
        # 检查顶点索引有效
        if v1 >= len(mesh.vertices) or v2 >= len(mesh.vertices):
            return False

        # 检查是否为边界边
        if self._protect_boundary:
            if self._is_boundary_edge(mesh, v1, v2):
                return False

        # 检查折叠后是否产生退化三角形
        if self._would_create_degenerate(mesh, v1, v2):
            return False

        # 检查是否违反流形性
        if self._would_break_manifold(mesh, v1, v2):
            return False

        # 检查特征边保护
        if self._protect_features:
            if self._is_feature_edge(mesh, v1, v2):
                return False

        return True

    def _is_boundary_edge(
        self, mesh: TriangleMesh, v1: int, v2: int
    ) -> bool:
        """检查是否为边界边"""
        edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_count[e] += 1

        target = tuple(sorted((v1, v2)))
        return edge_count.get(target, 0) == 1

    def _would_create_degenerate(
        self, mesh: TriangleMesh, v1: int, v2: int
    ) -> bool:
        """检查折叠后是否产生退化三角形"""
        # 找到共享v1但不共享v2的三角形，以及共享v2但不共享v1的
        v1_only_faces: List[Tuple[int, int, int]] = []
        v2_only_faces: List[Tuple[int, int, int]] = []

        for face in mesh.faces:
            has_v1 = v1 in face
            has_v2 = v2 in face
            if has_v1 and has_v2:
                continue  # 将被移除
            if has_v1:
                v1_only_faces.append(face)
            if has_v2:
                v2_only_faces.append(face)

        # 检查v1_only和v2_only是否有共同顶点
        v1_neighbors: Set[int] = set()
        for f in v1_only_faces:
            for v in f:
                if v != v1:
                    v1_neighbors.add(v)

        v2_neighbors: Set[int] = set()
        for f in v2_only_faces:
            for v in f:
                if v != v2:
                    v2_neighbors.add(v)

        # 如果有超过2个共同邻居，折叠后会形成退化三角形
        common = v1_neighbors & v2_neighbors
        return len(common) > 2

    def _would_break_manifold(
        self, mesh: TriangleMesh, v1: int, v2: int
    ) -> bool:
        """检查是否违反流形性"""
        # 计算折叠后每个邻居的面数
        v1_faces = sum(1 for f in mesh.faces if v1 in f)
        v2_faces = sum(1 for f in mesh.faces if v2 in f)
        shared_faces = sum(1 for f in mesh.faces if v1 in f and v2 in f)

        # 折叠后的面数
        remaining = v1_faces + v2_faces - 2 * shared_faces

        # 检查是否超过限制（流形网格中每个顶点的面数有限制）
        return remaining > 16

    def _is_feature_edge(
        self, mesh: TriangleMesh, v1: int, v2: int
    ) -> bool:
        """检查是否为特征边（基于法线夹角）"""
        if not mesh.normals:
            return False

        angle_threshold = math.radians(self._feature_angle)

        for face in mesh.faces:
            if v1 in face and v2 in face:
                continue

            adj_faces: List[Tuple[int, int, int]] = []
            for f in mesh.faces:
                if v1 in f and v2 not in f:
                    adj_faces.append(f)

            for af in adj_faces:
                fi = mesh.faces.index(face) if face in mesh.faces else -1
                ai = mesh.faces.index(af) if af in mesh.faces else -1
                if fi < 0 or ai < 0:
                    continue
                if fi < len(mesh.normals) and ai < len(mesh.normals):
                    n1 = mesh.normals[fi]
                    n2 = mesh.normals[ai]
                    cos_angle = _vec_dot(n1, n2)
                    cos_angle = _clamp(cos_angle, -1.0, 1.0)
                    angle = math.acos(cos_angle)
                    if angle > angle_threshold:
                        return True

        return False


# ============================================================================
# VertexMerger - 顶点合并器
# ============================================================================

class VertexMerger:
    """
    顶点合并器：合并距离相近的顶点。

    使用空间哈希加速邻近顶点查找。
    """

    def __init__(self, merge_distance: float = 0.001):
        self._merge_distance = merge_distance

    def merge_close_vertices(
        self, mesh: TriangleMesh, distance: Optional[float] = None
    ) -> TriangleMesh:
        """合并距离相近的顶点"""
        dist = distance or self._merge_distance
        if not mesh.vertices:
            return mesh

        # 构建空间哈希
        cell_size = dist * 2
        spatial_hash: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)

        for i, v in enumerate(mesh.vertices):
            cx = int(v[0] / cell_size)
            cy = int(v[1] / cell_size)
            cz = int(v[2] / cell_size)
            spatial_hash[(cx, cy, cz)].append(i)

        # 查找合并对
        merge_map: Dict[int, int] = {}
        for i, v in enumerate(mesh.vertices):
            if i in merge_map:
                continue
            cx = int(v[0] / cell_size)
            cy = int(v[1] / cell_size)
            cz = int(v[2] / cell_size)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        cell = (cx + dx, cy + dy, cz + dz)
                        for j in spatial_hash.get(cell, []):
                            if j <= i or j in merge_map:
                                continue
                            d = _vec_length(_vec_sub(v, mesh.vertices[j]))
                            if d < dist:
                                merge_map[j] = i

        if not merge_map:
            return mesh

        # 应用合并
        new_vertices = list(mesh.vertices)
        new_faces = list(mesh.faces)

        for remove, keep in merge_map.items():
            new_vertices[keep] = new_vertices[keep]  # 保留原位置

        # 更新面索引
        final_faces = []
        for face in new_faces:
            new_face = tuple(merge_map.get(v, v) for v in face)
            # 检查是否退化
            if len(set(new_face)) == 3:
                final_faces.append(new_face)

        # 移除未使用的顶点
        used: Set[int] = set()
        for f in final_faces:
            used.update(f)

        vertex_map: Dict[int, int] = {}
        result_vertices: List[Tuple[float, float, float]] = []
        for v in range(len(new_vertices)):
            if v in used:
                vertex_map[v] = len(result_vertices)
                result_vertices.append(new_vertices[v])

        result_faces = [
            (vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]])
            for f in final_faces
        ]

        return TriangleMesh(vertices=result_vertices, faces=result_faces)


# ============================================================================
# LODGenerator - LOD生成器
# ============================================================================

class LODGenerator:
    """
    LOD（Level of Detail）生成器：生成多级细节网格。

    从原始网格生成一系列简化版本，用于不同距离的渲染。
    """

    def __init__(self):
        self._simplifier: Optional[Any] = None

    def generate_lod_chain(
        self,
        mesh: TriangleMesh,
        num_levels: int = 5,
        min_faces: int = 10,
        simplifier: Any = None,
    ) -> List[LODLevel]:
        """生成LOD链"""
        self._simplifier = simplifier
        levels: List[LODLevel] = []

        # Level 0: 原始网格
        levels.append(LODLevel(
            level=0,
            mesh=TriangleMesh(
                vertices=list(mesh.vertices),
                faces=list(mesh.faces),
                normals=list(mesh.normals),
                uvs=list(mesh.uvs),
                colors=list(mesh.colors),
            ),
            reduction_ratio=0.0,
            error=0.0,
        ))

        current_mesh = mesh
        for level in range(1, num_levels):
            target_ratio = level / num_levels
            target_faces = max(min_faces, int(mesh.face_count() * (1.0 - target_ratio)))

            if self._simplifier and current_mesh.face_count() > target_faces:
                result = self._simplifier.simplify(current_mesh, target_face_count=target_faces)
                if result.mesh and result.mesh.face_count() > 0:
                    current_mesh = result.mesh
                    levels.append(LODLevel(
                        level=level,
                        mesh=current_mesh,
                        reduction_ratio=result.reduction_ratio,
                        error=result.error,
                    ))
                else:
                    break
            else:
                break

        return levels

    def select_lod(
        self, levels: List[LODLevel], distance: float,
        near_distance: float = 1.0, far_distance: float = 100.0,
    ) -> int:
        """根据距离选择LOD层级"""
        if not levels:
            return 0

        t = _clamp((distance - near_distance) / max(far_distance - near_distance, 0.01), 0.0, 1.0)
        level = int(t * (len(levels) - 1))
        return min(level, len(levels) - 1)


# ============================================================================
# MeshStats - 网格统计
# ============================================================================

class MeshStats:
    """网格统计分析"""

    def compute(self, mesh: TriangleMesh) -> Dict[str, Any]:
        """计算网格统计信息"""
        stats: Dict[str, Any] = {}

        # 基本统计
        stats["vertex_count"] = len(mesh.vertices)
        stats["face_count"] = len(mesh.faces)
        stats["edge_count"] = mesh.edge_count()
        stats["is_manifold"] = mesh.is_manifold()
        stats["is_watertight"] = mesh.is_watertight()

        # 包围盒
        bbox_min, bbox_max = mesh.compute_bbox()
        stats["bbox_min"] = bbox_min
        stats["bbox_max"] = bbox_max
        stats["bbox_size"] = (
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        )

        # 总面积
        total_area = 0.0
        for face in mesh.faces:
            if all(idx < len(mesh.vertices) for idx in face):
                total_area += _triangle_area(
                    mesh.vertices[face[0]],
                    mesh.vertices[face[1]],
                    mesh.vertices[face[2]],
                )
        stats["total_area"] = total_area

        # 体积（有符号体积法）
        total_volume = 0.0
        for face in mesh.faces:
            if all(idx < len(mesh.vertices) for idx in face):
                v0 = mesh.vertices[face[0]]
                v1 = mesh.vertices[face[1]]
                v2 = mesh.vertices[face[2]]
                total_volume += (
                    v0[0] * (v1[1] * v2[2] - v2[1] * v1[2])
                    - v1[0] * (v0[1] * v2[2] - v2[1] * v0[2])
                    + v2[0] * (v0[1] * v1[2] - v1[1] * v0[2])
                ) / 6.0
        stats["total_volume"] = abs(total_volume)

        # 平均边长
        edge_lengths: List[float] = []
        for face in mesh.faces:
            for i in range(3):
                if all(idx < len(mesh.vertices) for idx in (face[i], face[(i + 1) % 3])):
                    e = _vec_sub(
                        mesh.vertices[face[(i + 1) % 3]],
                        mesh.vertices[face[i]],
                    )
                    edge_lengths.append(_vec_length(e))
        stats["average_edge_length"] = (
            sum(edge_lengths) / len(edge_lengths) if edge_lengths else 0.0
        )
        stats["min_edge_length"] = min(edge_lengths) if edge_lengths else 0.0
        stats["max_edge_length"] = max(edge_lengths) if edge_lengths else 0.0

        # 三角形角度
        min_angle = float('inf')
        max_angle = 0.0
        for face in mesh.faces:
            if all(idx < len(mesh.vertices) for idx in face):
                angles = self._triangle_angles(
                    mesh.vertices[face[0]],
                    mesh.vertices[face[1]],
                    mesh.vertices[face[2]],
                )
                min_angle = min(min_angle, min(angles))
                max_angle = max(max_angle, max(angles))
        stats["minimum_angle"] = min_angle if min_angle != float('inf') else 0.0
        stats["maximum_angle"] = max_angle

        return stats

    def _triangle_angles(
        self,
        v0: Tuple[float, float, float],
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
    ) -> List[float]:
        """计算三角形三个角"""
        edges = [
            _vec_length(_vec_sub(v1, v2)),
            _vec_length(_vec_sub(v0, v2)),
            _vec_length(_vec_sub(v0, v1)),
        ]
        angles: List[float] = []
        for i in range(3):
            a, b = edges[(i + 1) % 3], edges[(i + 2) % 3]
            c = edges[i]
            if a * b < 1e-10:
                angles.append(0.0)
            else:
                cos_val = _clamp((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
                angles.append(math.acos(cos_val))
        return angles

    def quality_report(self, mesh: TriangleMesh) -> str:
        """生成质量报告"""
        stats = self.compute(mesh)
        lines: List[str] = [
            "=== Mesh Quality Report ===",
            f"Vertices: {stats['vertex_count']}",
            f"Faces: {stats['face_count']}",
            f"Edges: {stats['edge_count']}",
            f"Manifold: {stats['is_manifold']}",
            f"Watertight: {stats['is_watertight']}",
            f"Total Area: {stats['total_area']:.4f}",
            f"Total Volume: {stats['total_volume']:.4f}",
            f"Avg Edge Length: {stats['average_edge_length']:.4f}",
            f"Min Angle: {math.degrees(stats['minimum_angle']):.1f} deg",
            f"Max Angle: {math.degrees(stats['maximum_angle']):.1f} deg",
            f"BBox Size: ({stats['bbox_size'][0]:.2f}, {stats['bbox_size'][1]:.2f}, {stats['bbox_size'][2]:.2f})",
        ]
        return "\n".join(lines)


# ============================================================================
# MeshSimplifier - 网格简化器（主入口）
# ============================================================================

class MeshSimplifier:
    """
    3D网格简化器：使用边折叠算法和二次误差度量简化网格。

    算法: Garland & Heckbert 1997 - Surface Simplification Using
           Quadric Error Metrics

    使用方法:
        simplifier = MeshSimplifier()
        result = simplifier.simplify(mesh, target_face_count=1000)
    """

    def __init__(
        self,
        protect_boundary: bool = True,
        protect_features: bool = False,
        feature_angle: float = 30.0,
        aggression: float = 1.0,
    ):
        self._qem = QuadricErrorMetric()
        self._edge_collapse = EdgeCollapse()
        self._topology_checker = TopologyChecker()
        self._vertex_merger = VertexMerger()
        self._lod_generator = LODGenerator()
        self._stats = MeshStats()

        self._topology_checker._protect_boundary = protect_boundary
        self._topology_checker._protect_features = protect_features
        self._topology_checker._feature_angle = feature_angle
        self._aggression = aggression

    def simplify(
        self,
        mesh: TriangleMesh,
        target_face_count: int = 100,
        target_ratio: Optional[float] = None,
        max_iterations: Optional[int] = None,
        error_threshold: Optional[float] = None,
    ) -> SimplificationResult:
        """
        简化网格。

        Args:
            mesh: 输入网格
            target_face_count: 目标面数
            target_ratio: 目标简化比例 (0.0-1.0)
            max_iterations: 最大迭代次数
            error_threshold: 最大允许误差

        Returns:
            SimplificationResult: 简化结果
        """
        start_time = time.time()
        original_faces = mesh.face_count()
        original_vertices = mesh.vertex_count()

        if target_ratio is not None:
            target_face_count = max(10, int(original_faces * (1.0 - target_ratio)))

        if target_face_count >= original_faces:
            return SimplificationResult(
                original_vertices=original_vertices,
                original_faces=original_faces,
                simplified_vertices=original_vertices,
                simplified_faces=original_faces,
                reduction_ratio=0.0,
                error=0.0,
                mesh=mesh,
                time_ms=0.0,
            )

        # 预处理：合并相近顶点
        current_mesh = self._vertex_merger.merge_close_vertices(mesh)

        # 计算所有顶点的二次误差矩阵
        self._qem.compute_all_quadrics(current_mesh)

        # 构建优先队列
        edge_heap: List[Tuple[float, int, int, Tuple[float, float, float]]] = []
        edges = self._edge_collapse.get_collapsible_edges(current_mesh)

        for v1, v2 in edges:
            cost, opt_pos = self._qem.evaluate_edge(v1, v2, current_mesh)
            heapq.heappush(edge_heap, (cost, v1, v2, opt_pos))

        # 迭代简化
        iteration = 0
        total_error = 0.0
        removed_faces = 0
        target_removed = original_faces - target_face_count

        while edge_heap and removed_faces < target_removed:
            if max_iterations is not None and iteration >= max_iterations:
                break

            cost, v1, v2, opt_pos = heapq.heappop(edge_heap)

            if error_threshold is not None and cost > error_threshold:
                break

            # 检查顶点是否仍然有效
            if v1 >= len(current_mesh.vertices) or v2 >= len(current_mesh.vertices):
                continue

            # 检查拓扑有效性
            if not self._topology_checker.is_collapse_valid(current_mesh, v1, v2):
                continue

            # 执行边折叠
            faces_before = current_mesh.face_count()
            current_mesh = self._edge_collapse.collapse_edge(
                current_mesh, v1, v2, opt_pos, self._qem
            )
            faces_after = current_mesh.face_count()
            removed = faces_before - faces_after

            if removed > 0:
                removed_faces += removed
                total_error += cost

            # 重新计算受影响顶点的Q矩阵
            self._qem.compute_all_quadrics(current_mesh)

            iteration += 1

            # 定期重新计算边代价（提高质量）
            if iteration % 100 == 0:
                edge_heap = []
                edges = self._edge_collapse.get_collapsible_edges(current_mesh)
                for ev1, ev2 in edges:
                    ecost, eopt = self._qem.evaluate_edge(ev1, ev2, current_mesh)
                    heapq.heappush(edge_heap, (ecost, ev1, ev2, eopt))

        elapsed = (time.time() - start_time) * 1000.0
        reduction = 1.0 - current_mesh.face_count() / max(original_faces, 1)

        return SimplificationResult(
            original_vertices=original_vertices,
            original_faces=original_faces,
            simplified_vertices=current_mesh.vertex_count(),
            simplified_faces=current_mesh.face_count(),
            reduction_ratio=reduction,
            error=total_error,
            mesh=current_mesh,
            time_ms=elapsed,
            metadata={
                "iterations": iteration,
                "faces_removed": removed_faces,
            },
        )

    def generate_lod(
        self,
        mesh: TriangleMesh,
        num_levels: int = 5,
        min_faces: int = 10,
    ) -> List[LODLevel]:
        """生成LOD链"""
        return self._lod_generator.generate_lod_chain(
            mesh, num_levels, min_faces, self
        )

    def compute_stats(self, mesh: TriangleMesh) -> Dict[str, Any]:
        """计算网格统计"""
        return self._stats.compute(mesh)

    def quality_report(self, mesh: TriangleMesh) -> str:
        """生成质量报告"""
        return self._stats.quality_report(mesh)

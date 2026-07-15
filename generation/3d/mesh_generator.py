"""
3D网格生成器 - 纯Python实现
包含3D网格数据结构、多视角生成、单图到3D、程序化生成和纹理生成功能。
仅使用标准库，不依赖外部库。
"""

import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# 公共数据结构
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """统一生成结果"""
    data: Any = None
    format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Mesh3D - 3D网格数据结构
# ---------------------------------------------------------------------------

class Mesh3D:
    """3D网格数据结构"""

    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.uvs: List[Tuple[float, float]] = []
        self.colors: List[Tuple[int, int, int]] = []

    def compute_normals(self) -> List[Tuple[float, float, float]]:
        """计算法线"""
        self.normals = [(0.0, 0.0, 0.0) for _ in self.vertices]
        for face in self.faces:
            i0, i1, i2 = face
            if i0 >= len(self.vertices) or i1 >= len(self.vertices) or i2 >= len(self.vertices):
                continue
            v0 = self.vertices[i0]
            v1 = self.vertices[i1]
            v2 = self.vertices[i2]
            e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx = e1[1] * e2[2] - e1[2] * e2[1]
            ny = e1[2] * e2[0] - e1[0] * e2[2]
            nz = e1[0] * e2[1] - e1[1] * e2[0]
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 1e-10:
                nx /= length
                ny /= length
                nz /= length
            for idx in (i0, i1, i2):
                old = self.normals[idx]
                self.normals[idx] = (old[0] + nx, old[1] + ny, old[2] + nz)
        for i in range(len(self.normals)):
            n = self.normals[i]
            length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            if length > 1e-10:
                self.normals[i] = (n[0] / length, n[1] / length, n[2] / length)
        return self.normals

    def compute_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """计算包围盒"""
        if not self.vertices:
            return ((0, 0, 0), (0, 0, 0))
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        min_z = min(v[2] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        max_z = max(v[2] for v in self.vertices)
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

    def simplify(self, target_faces: int) -> 'Mesh3D':
        """简化网格 - 边折叠算法(简化版)"""
        if len(self.faces) <= target_faces:
            return self._copy()
        mesh = self._copy()
        mesh.compute_normals()
        edge_costs: Dict[Tuple[int, int], float] = {}
        for face in mesh.faces:
            edges = [
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[0], face[2]), max(face[0], face[2])),
            ]
            for edge in edges:
                if edge not in edge_costs:
                    v0 = mesh.vertices[edge[0]]
                    v1 = mesh.vertices[edge[1]]
                    dx = v0[0] - v1[0]
                    dy = v0[1] - v1[1]
                    dz = v0[2] - v1[2]
                    cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if edge[0] < len(mesh.normals) and edge[1] < len(mesh.normals):
                        n0 = mesh.normals[edge[0]]
                        n1 = mesh.normals[edge[1]]
                        dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
                        cost *= (2.0 - dot)
                    edge_costs[edge] = cost
        while len(mesh.faces) > target_faces and edge_costs:
            best_edge = min(edge_costs, key=edge_costs.get)
            del edge_costs[best_edge]
            i0, i1 = best_edge
            v0 = mesh.vertices[i0]
            v1 = mesh.vertices[i1]
            mid = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2, (v0[2] + v1[2]) / 2)
            mesh.vertices[i0] = mid
            new_faces: List[Tuple[int, int, int]] = []
            for face in mesh.faces:
                if i1 in face:
                    new_face = tuple(i0 if v == i1 else v for v in face)
                    if len(set(new_face)) == 3:
                        new_faces.append(new_face)
                else:
                    new_faces.append(face)
            mesh.faces = new_faces
        return mesh

    def subdivide(self, level: int = 1) -> 'Mesh3D':
        """细分网格 - Loop细分(简化版)"""
        mesh = self._copy()
        for _ in range(level):
            mesh = mesh._subdivide_once()
        return mesh

    def _subdivide_once(self) -> 'Mesh3D':
        """单次细分"""
        new_mesh = Mesh3D()
        edge_midpoints: Dict[Tuple[int, int], int] = {}
        new_vertices = list(self.vertices)
        for face in self.faces:
            edges = [
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[0], face[2]), max(face[0], face[2])),
            ]
            mid_indices: List[int] = []
            for edge in edges:
                if edge not in edge_midpoints:
                    v0 = self.vertices[edge[0]]
                    v1 = self.vertices[edge[1]]
                    mid = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2, (v0[2] + v1[2]) / 2)
                    new_vertices.append(mid)
                    edge_midpoints[edge] = len(new_vertices) - 1
                mid_indices.append(edge_midpoints[edge])
            a, b, c = face
            ab, bc, ca = mid_indices
            new_mesh.faces.append((a, ab, ca))
            new_mesh.faces.append((b, bc, ab))
            new_mesh.faces.append((c, ca, bc))
            new_mesh.faces.append((ab, bc, ca))
        new_mesh.vertices = new_vertices
        new_mesh.compute_normals()
        return new_mesh

    def merge(self, other: 'Mesh3D') -> 'Mesh3D':
        """合并网格"""
        result = self._copy()
        offset = len(result.vertices)
        result.vertices.extend(other.vertices)
        result.normals.extend(other.normals)
        result.uvs.extend(other.uvs)
        result.colors.extend(other.colors)
        for face in other.faces:
            result.faces.append((face[0] + offset, face[1] + offset, face[2] + offset))
        return result

    def to_obj(self) -> str:
        """导出OBJ格式"""
        lines: List[str] = ["# Generated by AGI Unified Framework"]
        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for n in self.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        for uv in self.uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        for face in self.faces:
            if self.normals and self.uvs:
                i0, i1, i2 = face
                lines.append(f"f {i0+1}/{i0+1}/{i0+1} {i1+1}/{i1+1}/{i1+1} {i2+1}/{i2+1}/{i2+1}")
            else:
                lines.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "vertices": [list(v) for v in self.vertices],
            "faces": [list(f) for f in self.faces],
            "normals": [list(n) for n in self.normals],
            "uvs": [list(u) for u in self.uvs],
            "colors": [list(c) for c in self.colors],
        }

    def _copy(self) -> 'Mesh3D':
        """深拷贝"""
        m = Mesh3D()
        m.vertices = list(self.vertices)
        m.faces = list(self.faces)
        m.normals = list(self.normals)
        m.uvs = list(self.uvs)
        m.colors = list(self.colors)
        return m


# ---------------------------------------------------------------------------
# 2. MultiViewGenerator - 多视角生成
# ---------------------------------------------------------------------------

class MultiViewGenerator:
    """多视角生成器(Zero123风格)"""

    def __init__(self):
        self._num_views: int = 6
        self._elevation_angles: List[float] = [0.0, 0.0, 30.0, -30.0, 0.0, 0.0]
        self._azimuth_angles: List[float] = [0.0, 90.0, 45.0, 135.0, 180.0, 270.0]
        self._image_size: int = 256

    def generate_multiview(self, image_prompt: str) -> List[GenerationResult]:
        """多视角生成"""
        results: List[GenerationResult] = []
        for i in range(self._num_views):
            elev = self._elevation_angles[i % len(self._elevation_angles)]
            azim = self._azimuth_angles[i % len(self._azimuth_angles)]
            camera = self._compute_camera_matrix(elev, azim, 2.0)
            depth_map = self._estimate_depth_from_prompt(image_prompt, i)
            mesh = self._depth_to_3d(depth_map, camera)
            results.append(GenerationResult(
                data=mesh,
                format="mesh",
                metadata={
                    "view_index": i,
                    "elevation": elev,
                    "azimuth": azim,
                    "prompt": image_prompt,
                },
            ))
        return results

    def _estimate_depth_from_prompt(self, prompt: str, view_index: int) -> List[List[float]]:
        """根据提示词估计深度图"""
        random.seed(hash(prompt) + view_index)
        size = self._image_size
        depth_map: List[List[float]] = []
        for y in range(size):
            row: List[float] = []
            for x in range(size):
                cx, cy = size / 2, size / 2
                dx = (x - cx) / cx
                dy = (y - cy) / cy
                dist = math.sqrt(dx * dx + dy * dy)
                depth = 1.0 / (1.0 + dist * 2.0)
                depth += random.gauss(0, 0.02)
                depth = max(0.1, min(1.0, depth))
                row.append(depth)
            depth_map.append(row)
        return depth_map

    def _compute_camera_matrix(self, elevation: float, azimuth: float,
                                distance: float) -> List[List[float]]:
        """计算相机矩阵"""
        el_rad = math.radians(elevation)
        az_rad = math.radians(azimuth)
        eye_x = distance * math.cos(el_rad) * math.sin(az_rad)
        eye_y = distance * math.sin(el_rad)
        eye_z = distance * math.cos(el_rad) * math.cos(az_rad)
        forward = self._normalize((-eye_x, -eye_y, -eye_z))
        world_up = (0.0, 1.0, 0.0)
        right = self._cross(forward, world_up)
        right = self._normalize(right)
        up = self._cross(right, forward)
        up = self._normalize(up)
        matrix = [
            [right[0], right[1], right[2], -self._dot(right, (eye_x, eye_y, eye_z))],
            [up[0], up[1], up[2], -self._dot(up, (eye_x, eye_y, eye_z))],
            [-forward[0], -forward[1], -forward[2], self._dot(forward, (eye_x, eye_y, eye_z))],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return matrix

    def _project_3d_to_2d(self, point_3d: Tuple[float, float, float],
                           camera_matrix: List[List[float]]) -> Tuple[float, float]:
        """3D到2D投影"""
        x, y, z = point_3d
        m = camera_matrix
        px = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
        py = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
        pz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
        if abs(pz) < 1e-10:
            return (0.0, 0.0)
        fov = 60.0
        f = 1.0 / math.tan(math.radians(fov / 2))
        sx = px * f / pz
        sy = py * f / pz
        return (sx, sy)

    def _estimate_depth(self, image: Any) -> List[List[float]]:
        """深度估计"""
        size = self._image_size
        depth_map: List[List[float]] = []
        random.seed(42)
        for y in range(size):
            row: List[float] = []
            for x in range(size):
                cx, cy = size / 2, size / 2
                dx = (x - cx) / cx
                dy = (y - cy) / cy
                depth = 1.0 - math.sqrt(dx * dx + dy * dy) * 0.5
                depth += random.gauss(0, 0.03)
                row.append(max(0.1, min(1.0, depth)))
            depth_map.append(row)
        return depth_map

    def _depth_to_3d(self, depth_map: List[List[float]],
                      camera_matrix: List[List[float]]) -> Mesh3D:
        """深度图转3D"""
        mesh = Mesh3D()
        h = len(depth_map)
        w = len(depth_map[0]) if h > 0 else 0
        step = max(1, min(w, h) // 64)
        for y in range(0, h, step):
            for x in range(0, w, step):
                depth = depth_map[y][x]
                px = (x / w - 0.5) * 2.0
                py = (y / h - 0.5) * 2.0
                pz = depth * 2.0
                inv_m = self._invert_matrix_3x3(camera_matrix)
                wx = inv_m[0][0] * px + inv_m[0][1] * py + inv_m[0][2] * pz
                wy = inv_m[1][0] * px + inv_m[1][1] * py + inv_m[1][2] * pz
                wz = inv_m[2][0] * px + inv_m[2][1] * py + inv_m[2][2] * pz
                mesh.vertices.append((wx, wy, wz))
        cols = max(1, w // step)
        for y in range(max(0, h // step - 1)):
            for x in range(cols - 1):
                i0 = y * cols + x
                i1 = i0 + 1
                i2 = i0 + cols
                i3 = i2 + 1
                if i3 < len(mesh.vertices):
                    mesh.faces.append((i0, i1, i2))
                    mesh.faces.append((i1, i3, i2))
        mesh.compute_normals()
        return mesh

    @staticmethod
    def _invert_matrix_3x3(m: List[List[float]]) -> List[List[float]]:
        """3x3矩阵求逆"""
        a = m[0][0]; b = m[0][1]; c = m[0][2]
        d = m[1][0]; e = m[1][1]; f = m[1][2]
        g = m[2][0]; h = m[2][1]; i = m[2][2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(det) < 1e-10:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        inv_det = 1.0 / det
        return [
            [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
            [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
            [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
        ]

    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if length < 1e-10:
            return (0.0, 0.0, 0.0)
        return (v[0] / length, v[1] / length, v[2] / length)

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ---------------------------------------------------------------------------
# 3. TripoSRStyle - 快速单图到3D
# ---------------------------------------------------------------------------

class TripoSRStyle:
    """快速单图到3D生成(TripoSR风格)"""

    def __init__(self):
        self._encoder: Any = None
        self._decoder: Any = None
        self._grid_resolution: int = 32

    def generate(self, image_description: str) -> GenerationResult:
        """从图像描述生成3D模型"""
        features = self._encode_image_features(image_description)
        mesh = self._decode_to_mesh(features)
        return GenerationResult(
            data=mesh,
            format="mesh",
            metadata={"description": image_description, "method": "tripo_sr_style"},
        )

    def _encode_image_features(self, description: str) -> List[float]:
        """编码图像特征"""
        h = hashlib.sha256(description.encode()).hexdigest()
        features: List[float] = []
        for i in range(0, len(h), 2):
            val = int(h[i:i + 2], 16) / 255.0
            features.append(val)
        while len(features) < 128:
            features.append(0.0)
        return features[:128]

    def _decode_to_mesh(self, features: List[float]) -> Mesh3D:
        """解码到网格"""
        res = self._grid_resolution
        sdf_grid: List[List[List[float]]] = []
        random.seed(int(features[0] * 1000) if features else 42)
        for x in range(res):
            slice_2d: List[List[float]] = []
            for y in range(res):
                row: List[float] = []
                for z in range(res):
                    nx = (x / (res - 1)) * 2 - 1
                    ny = (y / (res - 1)) * 2 - 1
                    nz = (z / (res - 1)) * 2 - 1
                    dist = math.sqrt(nx * nx + ny * ny + nz * nz) - 0.8
                    feat_val = 0.0
                    for k in range(min(16, len(features))):
                        feat_val += features[k] * math.sin(nx * (k + 1) * 0.5) * math.cos(nz * (k + 1) * 0.5)
                    dist += feat_val * 0.3
                    row.append(dist)
                slice_2d.append(row)
            sdf_grid.append(slice_2d)
        mesh = self._marching_cubes(sdf_grid, threshold=0.0)
        return mesh

    def _marching_cubes(self, sdf_grid: List[List[List[float]]], threshold: float) -> Mesh3D:
        """Marching Cubes算法(纯Python)"""
        mesh = Mesh3D()
        res_x = len(sdf_grid)
        res_y = len(sdf_grid[0]) if res_x > 0 else 0
        res_z = len(sdf_grid[0][0]) if res_y > 0 else 0
        vertex_map: Dict[Tuple[int, int, int], int] = {}
        edge_table = self._build_edge_table()
        tri_table = self._build_tri_table()
        for x in range(res_x - 1):
            for y in range(res_y - 1):
                for z in range(res_z - 1):
                    cube_idx = 0
                    corners = [
                        (x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z),
                        (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1),
                    ]
                    for i, (cx, cy, cz) in enumerate(corners):
                        if sdf_grid[cx][cy][cz] < threshold:
                            cube_idx |= (1 << i)
                    if cube_idx == 0 or cube_idx == 255:
                        continue
                    edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]
                    edge_verts: Dict[int, int] = {}
                    for ei, (a, b) in enumerate(edges):
                        va = sdf_grid[corners[a][0]][corners[a][1]][corners[a][2]]
                        vb = sdf_grid[corners[b][0]][corners[b][1]][corners[b][2]]
                        if (va < threshold) != (vb < threshold):
                            t = (threshold - va) / max(1e-10, vb - va)
                            ca, cb = corners[a], corners[b]
                            ix = ca[0] + t * (cb[0] - ca[0])
                            iy = ca[1] + t * (cb[1] - ca[1])
                            iz = ca[2] + t * (cb[2] - ca[2])
                            key = (round(ix, 4), round(iy, 4), round(iz, 4))
                            if key not in vertex_map:
                                vertex_map[key] = len(mesh.vertices)
                                mesh.vertices.append((ix, iy, iz))
                            edge_verts[ei] = vertex_map[key]
                    tris = tri_table.get(cube_idx, [])
                    for i in range(0, len(tris) - 2, 3):
                        e0, e1, e2 = tris[i], tris[i + 1], tris[i + 2]
                        if e0 in edge_verts and e1 in edge_verts and e2 in edge_verts:
                            mesh.faces.append((edge_verts[e0], edge_verts[e1], edge_verts[e2]))
        mesh.compute_normals()
        return mesh

    def _trilinear_interpolation(self, grid: List[List[List[float]]],
                                  x: float, y: float, z: float) -> float:
        """三线性插值"""
        res_x = len(grid)
        res_y = len(grid[0]) if res_x > 0 else 0
        res_z = len(grid[0][0]) if res_y > 0 else 0
        x = max(0.0, min(res_x - 1.001, x))
        y = max(0.0, min(res_y - 1.001, y))
        z = max(0.0, min(res_z - 1.001, z))
        x0, y0, z0 = int(x), int(y), int(z)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = x - x0, y - y0, z - z0
        c000 = grid[x0][y0][z0]
        c100 = grid[x1][y0][z0]
        c010 = grid[x0][y1][z0]
        c110 = grid[x1][y1][z0]
        c001 = grid[x0][y0][z1]
        c101 = grid[x1][y0][z1]
        c011 = grid[x0][y1][z1]
        c111 = grid[x1][y1][z1]
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        return c0 * (1 - zd) + c1 * zd

    def _compute_sdf(self, point: Tuple[float, float, float], features: List[float]) -> float:
        """计算符号距离"""
        x, y, z = point
        dist = math.sqrt(x * x + y * y + z * z) - 0.8
        for k in range(min(16, len(features))):
            dist += features[k] * 0.1 * math.sin(x * (k + 1)) * math.cos(z * (k + 1))
        return dist

    @staticmethod
    def _build_edge_table() -> List[int]:
        """构建Marching Cubes边表"""
        return [0] * 256

    @staticmethod
    def _build_tri_table() -> Dict[int, List[int]]:
        """构建Marching Cubes三角表(简化版)"""
        table: Dict[int, List[int]] = {}
        table[1] = [0, 8, 3]
        table[2] = [0, 1, 9]
        table[3] = [1, 8, 3, 9, 8, 1]
        table[4] = [1, 2, 10]
        table[5] = [0, 8, 3, 1, 2, 10]
        table[6] = [9, 2, 10, 0, 2, 9]
        table[7] = [2, 8, 3, 2, 10, 8, 10, 9, 8]
        table[8] = [3, 11, 2]
        table[9] = [0, 11, 2, 8, 11, 0]
        table[10] = [1, 9, 0, 2, 3, 11]
        table[11] = [1, 11, 2, 1, 9, 11, 9, 8, 11]
        table[12] = [3, 10, 1, 11, 10, 3]
        table[13] = [0, 10, 1, 0, 8, 10, 8, 11, 10]
        table[14] = [3, 9, 0, 3, 11, 9, 11, 10, 9]
        table[15] = [9, 8, 10, 10, 8, 11]
        table[254] = [0, 3, 8]
        table[253] = [0, 1, 9, 8, 3, 1]
        table[252] = [1, 2, 10, 8, 3, 0]
        table[251] = [3, 8, 2, 8, 10, 2, 8, 9, 10]
        return table


# ---------------------------------------------------------------------------
# 4. ProceduralGenerator - 程序化3D生成
# ---------------------------------------------------------------------------

class ProceduralGenerator:
    """程序化3D生成器"""

    def generate_primitive(self, shape: str, params: Optional[dict] = None) -> Mesh3D:
        """生成基本体"""
        if params is None:
            params = {}
        shape = shape.lower()
        if shape == "cube":
            return self._generate_cube(params.get("size", 1.0))
        elif shape == "sphere":
            return self._generate_sphere(params.get("radius", 0.5), params.get("segments", 16))
        elif shape == "cylinder":
            return self._generate_cylinder(
                params.get("radius", 0.5), params.get("height", 1.0), params.get("segments", 16))
        elif shape == "torus":
            return self._generate_torus(
                params.get("major_r", 0.5), params.get("minor_r", 0.2), params.get("segments", 16))
        elif shape == "plane":
            return self._generate_plane(
                params.get("width", 1.0), params.get("height", 1.0), params.get("subdivisions", 4))
        else:
            return self._generate_cube(1.0)

    def _generate_cube(self, size: float) -> Mesh3D:
        """生成立方体"""
        mesh = Mesh3D()
        h = size / 2
        mesh.vertices = [
            (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
            (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
        ]
        mesh.faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (0, 3, 7), (0, 7, 4),
            (1, 5, 6), (1, 6, 2),
        ]
        mesh.compute_normals()
        return mesh

    def _generate_sphere(self, radius: float, segments: int) -> Mesh3D:
        """生成球体"""
        mesh = Mesh3D()
        rings = segments
        sectors = segments
        for r in range(rings + 1):
            phi = math.pi * r / rings
            for s in range(sectors + 1):
                theta = 2.0 * math.pi * s / sectors
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                mesh.vertices.append((x, y, z))
        for r in range(rings):
            for s in range(sectors):
                a = r * (sectors + 1) + s
                b = a + sectors + 1
                mesh.faces.append((a, b, a + 1))
                mesh.faces.append((b, b + 1, a + 1))
        mesh.compute_normals()
        return mesh

    def _generate_cylinder(self, radius: float, height: float, segments: int) -> Mesh3D:
        """生成圆柱体"""
        mesh = Mesh3D()
        h = height / 2
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            mesh.vertices.append((x, -h, z))
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            mesh.vertices.append((x, h, z))
        for i in range(segments):
            next_i = (i + 1) % segments
            mesh.faces.append((i, next_i, i + segments))
            mesh.faces.append((next_i, next_i + segments, i + segments))
        center_bottom = len(mesh.vertices)
        mesh.vertices.append((0, -h, 0))
        center_top = center_bottom + 1
        mesh.vertices.append((0, h, 0))
        for i in range(segments):
            next_i = (i + 1) % segments
            mesh.faces.append((center_bottom, i, next_i))
            mesh.faces.append((center_top, i + segments, next_i + segments))
        mesh.compute_normals()
        return mesh

    def _generate_torus(self, major_r: float, minor_r: float, segments: int) -> Mesh3D:
        """生成环面"""
        mesh = Mesh3D()
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            for j in range(segments):
                phi = 2.0 * math.pi * j / segments
                x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
                y = minor_r * math.sin(phi)
                z = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
                mesh.vertices.append((x, y, z))
        for i in range(segments):
            for j in range(segments):
                a = i * segments + j
                b = i * segments + (j + 1) % segments
                c = ((i + 1) % segments) * segments + j
                d = ((i + 1) % segments) * segments + (j + 1) % segments
                mesh.faces.append((a, b, d))
                mesh.faces.append((a, d, c))
        mesh.compute_normals()
        return mesh

    def _generate_plane(self, width: float, height: float, subdivisions: int) -> Mesh3D:
        """生成平面"""
        mesh = Mesh3D()
        for y in range(subdivisions + 1):
            for x in range(subdivisions + 1):
                px = (x / subdivisions - 0.5) * width
                py = 0.0
                pz = (y / subdivisions - 0.5) * height
                mesh.vertices.append((px, py, pz))
                mesh.uvs.append((x / subdivisions, y / subdivisions))
        for y in range(subdivisions):
            for x in range(subdivisions):
                a = y * (subdivisions + 1) + x
                b = a + 1
                c = a + subdivisions + 1
                d = c + 1
                mesh.faces.append((a, c, b))
                mesh.faces.append((b, c, d))
        mesh.compute_normals()
        return mesh

    def apply_transform(self, mesh: Mesh3D, transform: Dict[str, Any]) -> Mesh3D:
        """应用变换"""
        result = mesh._copy()
        if "rotation" in transform:
            rot = transform["rotation"]
            axis = rot.get("axis", "y")
            angle = rot.get("angle", 0.0)
            result = self._rotate(result, axis, angle)
        if "scale" in transform:
            factors = transform["scale"]
            if isinstance(factors, (int, float)):
                factors = (factors, factors, factors)
            result = self._scale(result, factors)
        if "translation" in transform:
            offset = transform["translation"]
            result = self._translate(result, offset)
        return result

    def _rotate(self, mesh: Mesh3D, axis: str, angle: float) -> Mesh3D:
        """旋转"""
        result = mesh._copy()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_vertices: List[Tuple[float, float, float]] = []
        for v in result.vertices:
            x, y, z = v
            if axis == "x":
                ny = y * cos_a - z * sin_a
                nz = y * sin_a + z * cos_a
                new_vertices.append((x, ny, nz))
            elif axis == "y":
                nx = x * cos_a + z * sin_a
                nz = -x * sin_a + z * cos_a
                new_vertices.append((nx, y, nz))
            elif axis == "z":
                nx = x * cos_a - y * sin_a
                ny = x * sin_a + y * cos_a
                new_vertices.append((nx, ny, z))
            else:
                new_vertices.append(v)
        result.vertices = new_vertices
        if result.normals:
            result.compute_normals()
        return result

    def _scale(self, mesh: Mesh3D, factors: Tuple[float, float, float]) -> Mesh3D:
        """缩放"""
        result = mesh._copy()
        sx, sy, sz = factors
        result.vertices = [(v[0] * sx, v[1] * sy, v[2] * sz) for v in result.vertices]
        if result.normals:
            result.compute_normals()
        return result

    def _translate(self, mesh: Mesh3D, offset: Tuple[float, float, float]) -> Mesh3D:
        """平移"""
        result = mesh._copy()
        dx, dy, dz = offset
        result.vertices = [(v[0] + dx, v[1] + dy, v[2] + dz) for v in result.vertices]
        return result

    def boolean_union(self, mesh_a: Mesh3D, mesh_b: Mesh3D) -> Mesh3D:
        """布尔并集(简化实现 - 合并网格并移除内部面)"""
        result = mesh_a.merge(mesh_b)
        bbox_a = mesh_a.compute_bounding_box()
        bbox_b = mesh_b.compute_bounding_box()
        overlap = (
            bbox_a[0][0] < bbox_b[1][0] and bbox_a[1][0] > bbox_b[0][0] and
            bbox_a[0][1] < bbox_b[1][1] and bbox_a[1][1] > bbox_b[0][1] and
            bbox_a[0][2] < bbox_b[1][2] and bbox_a[1][2] > bbox_b[0][2],
        )
        if overlap[0]:
            center_a = (
                (bbox_a[0][0] + bbox_a[1][0]) / 2,
                (bbox_a[0][1] + bbox_a[1][1]) / 2,
                (bbox_a[0][2] + bbox_a[1][2]) / 2,
            )
            center_b = (
                (bbox_b[0][0] + bbox_b[1][0]) / 2,
                (bbox_b[0][1] + bbox_b[1][1]) / 2,
                (bbox_b[0][2] + bbox_b[1][2]) / 2,
            )
            sep_plane_normal = (
                center_b[0] - center_a[0],
                center_b[1] - center_a[1],
                center_b[2] - center_a[2],
            )
            length = math.sqrt(sum(c * c for c in sep_plane_normal))
            if length > 1e-10:
                sep_plane_normal = tuple(c / length for c in sep_plane_normal)
            sep_point = tuple((a + b) / 2 for a, b in zip(center_a, center_b))
            kept_faces: List[Tuple[int, int, int]] = []
            n_a = len(mesh_a.vertices)
            for face in result.faces:
                face_center = (
                    sum(result.vertices[v][0] for v in face) / 3,
                    sum(result.vertices[v][1] for v in face) / 3,
                    sum(result.vertices[v][2] for v in face) / 3,
                )
                dot = sum(fc * sn for fc, sn in zip(face_center, sep_plane_normal))
                sp_dot = sum(sp * sn for sp, sn in zip(sep_point, sep_plane_normal))
                is_a = all(v < n_a for v in face)
                is_b = all(v >= n_a for v in face)
                if is_a and dot < sp_dot:
                    kept_faces.append(face)
                elif is_b and dot >= sp_dot:
                    kept_faces.append(face)
                elif not is_a and not is_b:
                    kept_faces.append(face)
            result.faces = kept_faces
        result.compute_normals()
        return result

    def boolean_difference(self, mesh_a: Mesh3D, mesh_b: Mesh3D) -> Mesh3D:
        """布尔差集(简化实现)"""
        result = mesh_a._copy()
        bbox_b = mesh_b.compute_bounding_box()
        center_b = (
            (bbox_b[0][0] + bbox_b[1][0]) / 2,
            (bbox_b[0][1] + bbox_b[1][1]) / 2,
            (bbox_b[0][2] + bbox_b[1][2]) / 2,
        )
        max_r = max(
            bbox_b[1][0] - bbox_b[0][0],
            bbox_b[1][1] - bbox_b[0][1],
            bbox_b[1][2] - bbox_b[0][2],
        ) / 2
        kept_faces: List[Tuple[int, int, int]] = []
        for face in result.faces:
            face_center = (
                sum(result.vertices[v][0] for v in face) / 3,
                sum(result.vertices[v][1] for v in face) / 3,
                sum(result.vertices[v][2] for v in face) / 3,
            )
            dist = math.sqrt(sum((fc - cc) ** 2 for fc, cc in zip(face_center, center_b)))
            if dist > max_r * 0.8:
                kept_faces.append(face)
        result.faces = kept_faces
        result.compute_normals()
        return result


# ---------------------------------------------------------------------------
# 5. TextureGenerator - 纹理生成
# ---------------------------------------------------------------------------

class TextureGenerator:
    """纹理生成器"""

    def generate_texture(self, prompt: str, resolution: int = 512) -> List[List[Tuple[int, int, int]]]:
        """生成纹理"""
        prompt_lower = prompt.lower()
        if "wood" in prompt_lower or "wooden" in prompt_lower:
            return self._procedural_texture("wood", {"resolution": resolution})
        elif "marble" in prompt_lower:
            return self._procedural_texture("marble", {"resolution": resolution})
        elif "brick" in prompt_lower:
            return self._procedural_texture("brick", {"resolution": resolution})
        elif "noise" in prompt_lower or "static" in prompt_lower:
            noise = self._perlin_noise_2d(resolution, resolution, 50.0)
            return self._float_to_color(noise, 0.0, 1.0)
        elif "voronoi" in prompt_lower or "cellular" in prompt_lower:
            random.seed(42)
            points = [(random.randint(0, resolution), random.randint(0, resolution)) for _ in range(30)]
            return self._voronoi_texture(resolution, resolution, points)
        elif "checker" in prompt_lower or "checkered" in prompt_lower:
            return self._checker_texture(resolution, resolution, resolution // 8)
        elif "gradient" in prompt_lower:
            colors = [(255, 0, 0), (0, 0, 255)]
            return self._gradient_texture(resolution, resolution, colors)
        elif "metal" in prompt_lower or "metallic" in prompt_lower:
            return self._procedural_texture("metal", {"resolution": resolution})
        elif "fabric" in prompt_lower or "cloth" in prompt_lower:
            return self._procedural_texture("fabric", {"resolution": resolution})
        else:
            noise = self._perlin_noise_2d(resolution, resolution, 30.0)
            return self._float_to_color(noise, 0.0, 1.0)

    def _procedural_texture(self, type_: str, params: dict) -> List[List[Tuple[int, int, int]]]:
        """程序化纹理"""
        res = params.get("resolution", 256)
        texture: List[List[Tuple[int, int, int]]] = []
        random.seed(42)
        if type_ == "wood":
            noise = self._perlin_noise_2d(res, res, 200.0)
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                for x in range(res):
                    ring = math.sin((x + noise[y][x] * 50) * 0.05) * 0.5 + 0.5
                    r = int(139 + ring * 60)
                    g = int(90 + ring * 40)
                    b = int(43 + ring * 20)
                    row.append((min(255, r), min(255, g), min(255, b)))
                texture.append(row)
        elif type_ == "marble":
            noise = self._perlin_noise_2d(res, res, 80.0)
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                for x in range(res):
                    val = math.sin((x + y) * 0.02 + noise[y][x] * 10) * 0.5 + 0.5
                    r = int(200 + val * 55)
                    g = int(200 + val * 55)
                    b = int(210 + val * 45)
                    row.append((min(255, r), min(255, g), min(255, b)))
                texture.append(row)
        elif type_ == "brick":
            brick_h = res // 8
            brick_w = res // 4
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                offset = (brick_w // 2) if (y // brick_h) % 2 else 0
                for x in range(res):
                    bx = (x + offset) % brick_w
                    by = y % brick_h
                    is_mortar = bx < 2 or by < 2
                    if is_mortar:
                        row.append((180, 180, 170))
                    else:
                        variation = random.randint(-15, 15)
                        row.append((180 + variation, 80 + variation, 60 + variation))
                texture.append(row)
        elif type_ == "metal":
            noise = self._perlin_noise_2d(res, res, 100.0)
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                for x in range(res):
                    base = 160 + int(noise[y][x] * 60)
                    spec = int(abs(math.sin(x * 0.1 + noise[y][x] * 5)) * 40)
                    row.append((min(255, base + spec), min(255, base + spec), min(255, base + spec + 10)))
                texture.append(row)
        elif type_ == "fabric":
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                for x in range(res):
                    warp = math.sin(x * 0.5) * 0.5 + 0.5
                    weft = math.sin(y * 0.5) * 0.5 + 0.5
                    val = (warp + weft) / 2
                    row.append((int(val * 100 + 50), int(val * 50 + 30), int(val * 120 + 80)))
                texture.append(row)
        else:
            for y in range(res):
                row: List[Tuple[int, int, int]] = []
                for x in range(res):
                    v = random.randint(100, 200)
                    row.append((v, v, v))
                texture.append(row)
        return texture

    def _perlin_noise_2d(self, width: int, height: int, scale: float) -> List[List[float]]:
        """Perlin噪声(简化版)"""
        random.seed(42)
        grid_size = max(2, int(max(width, height) / scale) + 2)
        gradients: List[List[Tuple[float, float]]] = []
        for i in range(grid_size + 1):
            row: List[Tuple[float, float]] = []
            for j in range(grid_size + 1):
                angle = random.uniform(0, 2 * math.pi)
                row.append((math.cos(angle), math.sin(angle)))
            gradients.append(row)

        def _fade(t: float) -> float:
            return t * t * t * (t * (t * 6 - 15) + 10)

        def _lerp(a: float, b: float, t: float) -> float:
            return a + t * (b - a)

        noise: List[List[float]] = []
        for y in range(height):
            row: List[float] = []
            for x in range(width):
                gx = x / scale
                gy = y / scale
                x0 = int(gx)
                y0 = int(gy)
                x1 = x0 + 1
                y1 = y0 + 1
                xf = gx - x0
                yf = gy - y0
                u = _fade(xf)
                v = _fade(yf)
                x0 = min(x0, grid_size)
                x1 = min(x1, grid_size)
                y0 = min(y0, grid_size)
                y1 = min(y1, grid_size)
                g00 = gradients[y0][x0]
                g10 = gradients[y0][x1]
                g01 = gradients[y1][x0]
                g11 = gradients[y1][x1]
                d00 = g00[0] * xf + g00[1] * yf
                d10 = g10[0] * (xf - 1) + g10[1] * yf
                d01 = g01[0] * xf + g01[1] * (yf - 1)
                d11 = g11[0] * (xf - 1) + g11[1] * (yf - 1)
                val = _lerp(_lerp(d00, d10, u), _lerp(d01, d11, u), v)
                val = val * 0.5 + 0.5
                row.append(max(0.0, min(1.0, val)))
            noise.append(row)
        return noise

    def _voronoi_texture(self, width: int, height: int,
                          points: List[Tuple[int, int]]) -> List[List[Tuple[int, int, int]]]:
        """Voronoi纹理"""
        texture: List[List[Tuple[int, int, int]]] = []
        random.seed(42)
        point_colors = [(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                        for _ in range(len(points))]
        for y in range(height):
            row: List[Tuple[int, int, int]] = []
            for x in range(width):
                min_dist = float('inf')
                min_idx = 0
                for idx, (px, py) in enumerate(points):
                    dx = x - px
                    dy = y - py
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
                edge_factor = min(1.0, min_dist / 20.0)
                base = point_colors[min_idx]
                r = int(base[0] * edge_factor)
                g = int(base[1] * edge_factor)
                b = int(base[2] * edge_factor)
                row.append((min(255, r), min(255, g), min(255, b)))
            texture.append(row)
        return texture

    def _checker_texture(self, width: int, height: int, size: int) -> List[List[Tuple[int, int, int]]]:
        """棋盘格纹理"""
        texture: List[List[Tuple[int, int, int]]] = []
        for y in range(height):
            row: List[Tuple[int, int, int]] = []
            for x in range(width):
                if ((x // size) + (y // size)) % 2 == 0:
                    row.append((255, 255, 255))
                else:
                    row.append((0, 0, 0))
            texture.append(row)
        return texture

    def _gradient_texture(self, width: int, height: int,
                           colors: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """渐变纹理"""
        texture: List[List[Tuple[int, int, int]]] = []
        n_colors = len(colors)
        for y in range(height):
            row: List[Tuple[int, int, int]] = []
            t = y / max(1, height - 1)
            seg = t * (n_colors - 1)
            idx = min(int(seg), n_colors - 2)
            frac = seg - idx
            c0 = colors[idx]
            c1 = colors[idx + 1]
            r = int(c0[0] + (c1[0] - c0[0]) * frac)
            g = int(c0[1] + (c1[1] - c0[1]) * frac)
            b = int(c0[2] + (c1[2] - c0[2]) * frac)
            for x in range(width):
                row.append((r, g, b))
            texture.append(row)
        return texture

    def _apply_normal_map(self, heightmap: List[List[float]]) -> List[List[Tuple[int, int, int]]]:
        """法线贴图生成"""
        h = len(heightmap)
        w = len(heightmap[0]) if h > 0 else 0
        normal_map: List[List[Tuple[int, int, int]]] = []
        strength = 2.0
        for y in range(h):
            row: List[Tuple[int, int, int]] = []
            for x in range(w):
                left = heightmap[y][max(0, x - 1)]
                right = heightmap[y][min(w - 1, x + 1)]
                up = heightmap[max(0, y - 1)][x]
                down = heightmap[min(h - 1, y + 1)][x]
                dx = (left - right) * strength
                dy = (up - down) * strength
                dz = 1.0
                length = math.sqrt(dx * dx + dy * dy + dz * dz)
                if length > 1e-10:
                    dx /= length
                    dy /= length
                    dz /= length
                r = int((dx * 0.5 + 0.5) * 255)
                g = int((dy * 0.5 + 0.5) * 255)
                b = int((dz * 0.5 + 0.5) * 255)
                row.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            normal_map.append(row)
        return normal_map

    @staticmethod
    def _float_to_color(data: List[List[float]], vmin: float, vmax: float) -> List[List[Tuple[int, int, int]]]:
        """浮点数据转颜色"""
        result: List[List[Tuple[int, int, int]]] = []
        for row in data:
            color_row: List[Tuple[int, int, int]] = []
            for val in row:
                if vmax > vmin:
                    t = (val - vmin) / (vmax - vmin)
                else:
                    t = 0.0
                t = max(0.0, min(1.0, t))
                c = int(t * 255)
                color_row.append((c, c, c))
            result.append(color_row)
        return result

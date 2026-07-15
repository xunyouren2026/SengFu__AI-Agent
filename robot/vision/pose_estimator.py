"""
6D Pose Estimation Module

Provides 6-DoF pose estimation from 2D/3D correspondences:
- EPnP (Efficient Perspective-n-Point) solver
- ICP (Iterative Closest Point) refinement
- Pose scoring and hypothesis generation
- Covariance/uncertainty estimation
"""

from __future__ import annotations

import math
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

try:
    import json
except ImportError:
    json = None  # type: ignore


class PoseComponent(Enum):
    """Components of a 6D pose."""
    TRANSLATION = "translation"
    ROTATION = "rotation"
    FULL = "full"


@dataclass
class Point3D:
    """A 3D point with x, y, z coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Point3D:
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: Point3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Point3D) -> Point3D:
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> Point3D:
        n = self.norm()
        if n < 1e-12:
            return Point3D(0.0, 0.0, 0.0)
        return Point3D(self.x / n, self.y / n, self.z / n)

    def distance_to(self, other: Point3D) -> float:
        return (self - other).norm()

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class Point2D:
    """A 2D point with u, v coordinates."""
    u: float = 0.0
    v: float = 0.0

    def distance_to(self, other: Point2D) -> float:
        du = self.u - other.u
        dv = self.v - other.v
        return math.sqrt(du * du + dv * dv)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.u, self.v)


@dataclass
class RotationMatrix:
    """3x3 rotation matrix stored as row-major list of lists."""
    data: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    @staticmethod
    def identity() -> RotationMatrix:
        return RotationMatrix()

    def multiply(self, other: RotationMatrix) -> RotationMatrix:
        result = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += self.data[i][k] * other.data[k][j]
                result[i][j] = s
        return RotationMatrix(result)

    def transform_point(self, p: Point3D) -> Point3D:
        r = self.data
        return Point3D(
            r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
            r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
            r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z,
        )

    def transpose(self) -> RotationMatrix:
        result = [[self.data[j][i] for j in range(3)] for i in range(3)]
        return RotationMatrix(result)

    def determinant(self) -> float:
        d = self.data
        return (
            d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
        )

    def is_valid_rotation(self, tol: float = 1e-6) -> bool:
        det = abs(self.determinant())
        if abs(det - 1.0) > tol:
            return False
        rt = self.multiply(self.transpose())
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                if abs(rt.data[i][j] - expected) > tol:
                    return False
        return True


@dataclass
class Pose6D:
    """6-DoF pose: rotation + translation."""
    rotation: RotationMatrix = field(default_factory=RotationMatrix.identity)
    translation: Point3D = field(default_factory=Point3D)

    def transform_point(self, p: Point3D) -> Point3D:
        rotated = self.rotation.transform_point(p)
        return rotated + self.translation

    def inverse(self) -> Pose6D:
        rt = self.rotation.transpose()
        t_inv = rt.transform_point(Point3D(-self.translation.x, -self.translation.y, -self.translation.z))
        return Pose6D(rotation=rt, translation=t_inv)

    def to_matrix_4x4(self) -> List[List[float]]:
        m = [[0.0] * 4 for _ in range(4)]
        for i in range(3):
            for j in range(3):
                m[i][j] = self.rotation.data[i][j]
        m[0][3] = self.translation.x
        m[1][3] = self.translation.y
        m[2][3] = self.translation.z
        m[3][3] = 1.0
        return m

    @staticmethod
    def from_matrix_4x4(m: List[List[float]]) -> Pose6D:
        rot = RotationMatrix([[m[i][j] for j in range(3)] for i in range(3)])
        trans = Point3D(m[0][3], m[1][3], m[2][3])
        return Pose6D(rotation=rot, translation=trans)

    def copy(self) -> Pose6D:
        return Pose6D(
            rotation=RotationMatrix([row[:] for row in self.rotation.data]),
            translation=Point3D(self.translation.x, self.translation.y, self.translation.z),
        )


@dataclass
class PoseHypothesis:
    """A pose hypothesis with an associated score."""
    pose: Pose6D
    score: float = 0.0
    inlier_count: int = 0
    reprojection_error: float = 0.0
    source: str = ""


class PNPSolver:
    """Efficient Perspective-n-Point (EPnP) solver.

    Solves the camera pose given 3D object points and their 2D projections.
    Uses the EPnP algorithm with 4 control points.
    """

    def __init__(self, camera_matrix: Optional[List[List[float]]] = None) -> None:
        self.camera_matrix = camera_matrix or [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
        self.dist_coeffs: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def _select_control_points(
        self, object_points: List[Point3D]
    ) -> List[Point3D]:
        """Select 4 control points from the object points."""
        n = len(object_points)
        cx = sum(p.x for p in object_points) / n
        cy = sum(p.y for p in object_points) / n
        cz = sum(p.z for p in object_points) / n
        centroid = Point3D(cx, cy, cz)

        dx = [p.x - cx for p in object_points]
        dy = [p.y - cy for p in object_points]
        dz = [p.z - cz for p in object_points]

        cov_xx = sum(d * d for d in dx) / n
        cov_yy = sum(d * d for d in dy) / n
        cov_zz = sum(d * d for d in dz) / n
        cov_xy = sum(dx[i] * dy[i] for i in range(n)) / n
        cov_xz = sum(dx[i] * dz[i] for i in range(n)) / n
        cov_yz = sum(dy[i] * dz[i] for i in range(n)) / n

        cov_matrix = [
            [cov_xx, cov_xy, cov_xz],
            [cov_xy, cov_yy, cov_yz],
            [cov_xz, cov_yz, cov_zz],
        ]

        eigenvalues, eigenvectors = self._eigen_3x3(cov_matrix)

        sorted_indices = sorted(range(3), key=lambda i: -eigenvalues[i])

        p1 = centroid
        p2 = centroid + Point3D(
            eigenvectors[sorted_indices[0]][0] * math.sqrt(max(eigenvalues[sorted_indices[0]], 0)),
            eigenvectors[sorted_indices[0]][1] * math.sqrt(max(eigenvalues[sorted_indices[0]], 0)),
            eigenvectors[sorted_indices[0]][2] * math.sqrt(max(eigenvalues[sorted_indices[0]], 0)),
        )
        p3 = centroid + Point3D(
            eigenvectors[sorted_indices[1]][0] * math.sqrt(max(eigenvalues[sorted_indices[1]], 0)),
            eigenvectors[sorted_indices[1]][1] * math.sqrt(max(eigenvalues[sorted_indices[1]], 0)),
            eigenvectors[sorted_indices[1]][2] * math.sqrt(max(eigenvalues[sorted_indices[1]], 0)),
        )
        p4 = centroid + Point3D(
            eigenvectors[sorted_indices[2]][0] * math.sqrt(max(eigenvalues[sorted_indices[2]], 0)),
            eigenvectors[sorted_indices[2]][1] * math.sqrt(max(eigenvalues[sorted_indices[2]], 0)),
            eigenvectors[sorted_indices[2]][2] * math.sqrt(max(eigenvalues[sorted_indices[2]], 0)),
        )

        return [p1, p2, p3, p4]

    def _eigen_3x3(
        self, matrix: List[List[float]]
    ) -> Tuple[List[float], List[List[float]]]:
        """Compute eigenvalues and eigenvectors of a 3x3 symmetric matrix using Jacobi iteration."""
        a = [row[:] for row in matrix]
        v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

        for _ in range(100):
            off_diag = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    off_diag += a[i][j] ** 2
            if off_diag < 1e-12:
                break

            for p in range(3):
                for q in range(p + 1, 3):
                    if abs(a[p][q]) < 1e-12:
                        continue
                    theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
                    c = math.cos(theta)
                    s = math.sin(theta)

                    for i in range(3):
                        vip = c * v[i][p] + s * v[i][q]
                        viq = -s * v[i][p] + c * v[i][q]
                        v[i][p] = vip
                        v[i][q] = viq

                    app_new = c * c * a[p][p] + 2 * s * c * a[p][q] + s * s * a[q][q]
                    aqq_new = s * s * a[p][p] - 2 * s * c * a[p][q] + c * c * a[q][q]
                    apq_new = 0.0

                    a[p][p] = app_new
                    a[q][q] = aqq_new
                    a[p][q] = apq_new
                    a[q][p] = apq_new

        eigenvalues = [a[i][i] for i in range(3)]
        eigenvectors = [[v[r][c] for c in range(3)] for r in range(3)]
        return eigenvalues, eigenvectors

    def _compute_alphas(
        self, object_points: List[Point3D], control_points: List[Point3D]
    ) -> List[List[float]]:
        """Compute barycentric coordinates (alphas) for each object point."""
        n = len(object_points)
        alphas: List[List[float]] = []
        for i in range(n):
            p = object_points[i]
            cp = control_points
            v0 = Point3D(cp[1].x - cp[0].x, cp[1].y - cp[0].y, cp[1].z - cp[0].z)
            v1 = Point3D(cp[2].x - cp[0].x, cp[2].y - cp[0].y, cp[2].z - cp[0].z)
            v2 = Point3D(cp[3].x - cp[0].x, cp[3].y - cp[0].y, cp[3].z - cp[0].z)
            vp = Point3D(p.x - cp[0].x, p.y - cp[0].y, p.z - cp[0].z)

            e00 = v0.dot(v0)
            e01 = v0.dot(v1)
            e02 = v0.dot(v2)
            e11 = v1.dot(v1)
            e12 = v1.dot(v2)
            e22 = v2.dot(v2)

            d0 = vp.dot(v0)
            d1 = vp.dot(v1)
            d2 = vp.dot(v2)

            det = e00 * (e11 * e22 - e12 * e12) - e01 * (e01 * e22 - e12 * e02) + e02 * (e01 * e12 - e11 * e02)
            if abs(det) < 1e-12:
                alphas.append([1.0, 0.0, 0.0, 0.0])
                continue

            inv_det = 1.0 / det
            a1 = (d0 * (e11 * e22 - e12 * e12) - e01 * (d1 * e22 - e12 * d2) + e02 * (d1 * e12 - e11 * d2)) * inv_det
            a2 = (e00 * (d1 * e22 - e12 * d2) - d0 * (e01 * e22 - e12 * e02) + e02 * (e01 * d2 - d1 * e02)) * inv_det
            a3 = (e00 * (e11 * d2 - d1 * e12) - e01 * (e01 * d2 - d1 * e02) + d0 * (e01 * e12 - e11 * e02)) * inv_det
            a0 = 1.0 - a1 - a2 - a3

            alphas.append([a0, a1, a2, a3])
        return alphas

    def _solve_betas(
        self,
        alphas: List[List[float]],
        image_points: List[Point2D],
        fu: float,
        fv: float,
        cu: float,
        cv: float,
    ) -> List[List[float]]:
        """Solve for the M matrix (beta coefficients) using the EPnP formulation."""
        n = len(alphas)
        M: List[List[float]] = [[0.0] * 12 for _ in range(2 * n)]

        for i in range(n):
            u = image_points[i].u
            v = image_points[i].v
            a0, a1, a2, a3 = alphas[i]

            ui = u - cu
            vi = v - cv

            M[2 * i][0] = a0 * fu
            M[2 * i][1] = a1 * fu
            M[2 * i][2] = a2 * fu
            M[2 * i][3] = a3 * fu
            M[2 * i][4] = 0.0
            M[2 * i][5] = 0.0
            M[2 * i][6] = 0.0
            M[2 * i][7] = 0.0
            M[2 * i][8] = -a0 * ui
            M[2 * i][9] = -a1 * ui
            M[2 * i][10] = -a2 * ui
            M[2 * i][11] = -a3 * ui

            M[2 * i + 1][0] = 0.0
            M[2 * i + 1][1] = 0.0
            M[2 * i + 1][2] = 0.0
            M[2 * i + 1][3] = 0.0
            M[2 * i + 1][4] = a0 * fv
            M[2 * i + 1][5] = a1 * fv
            M[2 * i + 1][6] = a2 * fv
            M[2 * i + 1][7] = a3 * fv
            M[2 * i + 1][8] = -a0 * vi
            M[2 * i + 1][9] = -a1 * vi
            M[2 * i + 1][10] = -a2 * vi
            M[2 * i + 1][11] = -a3 * vi

        MtM = [[0.0] * 12 for _ in range(12)]
        for i in range(12):
            for j in range(i, 12):
                s = 0.0
                for k in range(2 * n):
                    s += M[k][i] * M[k][j]
                MtM[i][j] = s
                MtM[j][i] = s

        for j in range(12):
            MtM[j][j] += 1e-10

        eigenvalues, eigenvectors = self._eigen_general(MtM, 12)
        min_idx = min(range(12), key=lambda i: eigenvalues[i])
        beta = [eigenvectors[min_idx][j] for j in range(12)]
        return [beta]

    def _eigen_general(
        self, matrix: List[List[float]], size: int
    ) -> Tuple[List[float], List[List[float]]]:
        """Compute eigenvalues/eigenvectors for a general symmetric matrix using QR iteration."""
        n = size
        a = [row[:] for row in matrix]
        v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for iteration in range(200):
            off_diag = sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
            if off_diag < 1e-10:
                break

            for p in range(n):
                for q in range(p + 1, n):
                    if abs(a[p][q]) < 1e-12:
                        continue
                    app = a[p][p]
                    aqq = a[q][q]
                    apq = a[p][q]
                    theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
                    c = math.cos(theta)
                    s = math.sin(theta)

                    a[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq
                    a[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq
                    a[p][q] = 0.0
                    a[q][p] = 0.0

                    for r in range(n):
                        if r != p and r != q:
                            arp = c * a[r][p] + s * a[r][q]
                            arq = -s * a[r][p] + c * a[r][q]
                            a[r][p] = arp
                            a[p][r] = arp
                            a[r][q] = arq
                            a[q][r] = arq

                    for r in range(n):
                        vrp = c * v[r][p] + s * v[r][q]
                        vrq = -s * v[r][p] + c * v[r][q]
                        v[r][p] = vrp
                        v[r][q] = vrq

        eigenvalues = [a[i][i] for i in range(n)]
        eigenvectors = [[v[r][c] for c in range(n)] for r in range(n)]
        return eigenvalues, eigenvectors

    def _compute_control_points_3d(
        self, betas: List[float], control_points: List[Point3D]
    ) -> List[Point3D]:
        """Compute 3D positions of control points from betas."""
        result: List[Point3D] = []
        for j in range(4):
            x = sum(betas[k * 4 + j] * control_points[k].x for k in range(4))
            y = sum(betas[k * 4 + j] * control_points[k].y for k in range(4))
            z = sum(betas[k * 4 + j] * control_points[k].z for k in range(4))
            result.append(Point3D(x, y, z))
        return result

    def _compute_pose_from_control_points(
        self,
        control_points_world: List[Point3D],
        control_points_camera: List[Point3D],
    ) -> Pose6D:
        """Compute R|t from corresponding control point sets using SVD."""
        n = len(control_points_world)
        w_centroid = Point3D(
            sum(p.x for p in control_points_world) / n,
            sum(p.y for p in control_points_world) / n,
            sum(p.z for p in control_points_world) / n,
        )
        c_centroid = Point3D(
            sum(p.x for p in control_points_camera) / n,
            sum(p.y for p in control_points_camera) / n,
            sum(p.z for p in control_points_camera) / n,
        )

        H = [[0.0] * 3 for _ in range(3)]
        for i in range(n):
            wp = Point3D(
                control_points_world[i].x - w_centroid.x,
                control_points_world[i].y - w_centroid.y,
                control_points_world[i].z - w_centroid.z,
            )
            cp = Point3D(
                control_points_camera[i].x - c_centroid.x,
                control_points_camera[i].y - c_centroid.y,
                control_points_camera[i].z - c_centroid.z,
            )
            H[0][0] += wp.x * cp.x
            H[0][1] += wp.x * cp.y
            H[0][2] += wp.x * cp.z
            H[1][0] += wp.y * cp.x
            H[1][1] += wp.y * cp.y
            H[1][2] += wp.y * cp.z
            H[2][0] += wp.z * cp.x
            H[2][1] += wp.z * cp.y
            H[2][2] += wp.z * cp.z

        U, S, Vt = self._svd_3x3(H)

        R = self._mat_mul_3x3(U, Vt)
        det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - \
              R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + \
              R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
        if det < 0:
            U[0][2] *= -1
            U[1][2] *= -1
            U[2][2] *= -1
            R = self._mat_mul_3x3(U, Vt)

        t = Point3D(
            c_centroid.x - (R[0][0] * w_centroid.x + R[0][1] * w_centroid.y + R[0][2] * w_centroid.z),
            c_centroid.y - (R[1][0] * w_centroid.x + R[1][1] * w_centroid.y + R[1][2] * w_centroid.z),
            c_centroid.z - (R[2][0] * w_centroid.x + R[2][1] * w_centroid.y + R[2][2] * w_centroid.z),
        )

        return Pose6D(rotation=RotationMatrix(R), translation=t)

    def _mat_mul_3x3(
        self, a: List[List[float]], b: List[List[float]]
    ) -> List[List[float]]:
        result = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    def _svd_3x3(
        self, matrix: List[List[float]]
    ) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """Compute thin SVD of a 3x3 matrix using Jacobi SVD."""
        HtH = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += matrix[k][i] * matrix[k][j]
                HtH[i][j] = s

        eigenvalues, V = self._eigen_3x3(HtH)

        sorted_idx = sorted(range(3), key=lambda i: -eigenvalues[i])
        eigenvalues = [eigenvalues[sorted_idx[i]] for i in range(3)]
        V = [[V[r][sorted_idx[c]] for c in range(3)] for r in range(3)]

        singular_values = [math.sqrt(max(ev, 0.0)) for ev in eigenvalues]

        U = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            if singular_values[j] > 1e-10:
                for i in range(3):
                    s = 0.0
                    for k in range(3):
                        s += matrix[i][k] * V[k][j]
                    U[i][j] = s / singular_values[j]

        Vt = [[V[r][c] for r in range(3)] for c in range(3)]
        return U, singular_values, Vt

    def solve(
        self,
        object_points: List[Point3D],
        image_points: List[Point2D],
    ) -> Optional[Pose6D]:
        """Solve PnP given 3D-2D correspondences.

        Args:
            object_points: 3D points in object/world frame.
            image_points: Corresponding 2D image points.

        Returns:
            Estimated Pose6D or None if solving fails.
        """
        if len(object_points) < 4 or len(object_points) != len(image_points):
            return None

        fu = self.camera_matrix[0][0]
        fv = self.camera_matrix[1][1]
        cu = self.camera_matrix[0][2]
        cv = self.camera_matrix[1][2]

        control_points = self._select_control_points(object_points)
        alphas = self._compute_alphas(object_points, control_points)
        betas_list = self._solve_betas(alphas, image_points, fu, fv, cu, cv)
        betas = betas_list[0]

        cp_camera = self._compute_control_points_3d(betas, control_points)

        pose = self._compute_pose_from_control_points(control_points, cp_camera)

        if not pose.rotation.is_valid_rotation():
            return None

        return pose

    def compute_reprojection_error(
        self,
        pose: Pose6D,
        object_points: List[Point3D],
        image_points: List[Point2D],
    ) -> float:
        """Compute mean reprojection error for a pose estimate."""
        fu = self.camera_matrix[0][0]
        fv = self.camera_matrix[1][1]
        cu = self.camera_matrix[0][2]
        cv = self.camera_matrix[1][2]

        total_error = 0.0
        for op, ip in zip(object_points, image_points):
            cp = pose.transform_point(op)
            if abs(cp.z) < 1e-10:
                total_error += 1000.0
                continue
            u_proj = fu * cp.x / cp.z + cu
            v_proj = fv * cp.y / cp.z + cv
            error = math.sqrt((u_proj - ip.u) ** 2 + (v_proj - ip.v) ** 2)
            total_error += error

        return total_error / max(len(object_points), 1)


class ICPRefiner:
    """Iterative Closest Point refinement for pose estimation.

    Iteratively refines a pose estimate by finding closest point correspondences
    and minimizing the alignment error.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        max_correspondence_distance: float = 0.1,
        rejection_threshold: float = 3.0,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.rejection_threshold = rejection_threshold

    def find_correspondences(
        self,
        source: List[Point3D],
        target: List[Point3D],
        pose: Pose6D,
    ) -> List[Tuple[int, int, float]]:
        """Find closest point correspondences between transformed source and target."""
        correspondences: List[Tuple[int, int, float]] = []
        for i, sp in enumerate(source):
            transformed = pose.transform_point(sp)
            best_j = -1
            best_dist = float("inf")
            for j, tp in enumerate(target):
                d = transformed.distance_to(tp)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0 and best_dist < self.max_correspondence_distance:
                correspondences.append((i, best_j, best_dist))
        return correspondences

    def compute_transformation(
        self,
        source: List[Point3D],
        target: List[Point3D],
        correspondences: List[Tuple[int, int, float]],
    ) -> Optional[Pose6D]:
        """Compute the optimal rigid transformation from correspondences using SVD."""
        if len(correspondences) < 3:
            return None

        src_pts = [source[i] for i, _, _ in correspondences]
        tgt_pts = [target[j] for _, j, _ in correspondences]

        n = len(src_pts)
        src_c = Point3D(
            sum(p.x for p in src_pts) / n,
            sum(p.y for p in src_pts) / n,
            sum(p.z for p in src_pts) / n,
        )
        tgt_c = Point3D(
            sum(p.x for p in tgt_pts) / n,
            sum(p.y for p in tgt_pts) / n,
            sum(p.z for p in tgt_pts) / n,
        )

        H = [[0.0] * 3 for _ in range(3)]
        for sp, tp in zip(src_pts, tgt_pts):
            ds = Point3D(sp.x - src_c.x, sp.y - src_c.y, sp.z - src_c.z)
            dt = Point3D(tp.x - tgt_c.x, tp.y - tgt_c.y, tp.z - tgt_c.z)
            H[0][0] += ds.x * dt.x
            H[0][1] += ds.x * dt.y
            H[0][2] += ds.x * dt.z
            H[1][0] += ds.y * dt.x
            H[1][1] += ds.y * dt.y
            H[1][2] += ds.y * dt.z
            H[2][0] += ds.z * dt.x
            H[2][1] += ds.z * dt.y
            H[2][2] += ds.z * dt.z

        pnp = PNPSolver()
        U, S, Vt = pnp._svd_3x3(H)
        R = pnp._mat_mul_3x3(U, Vt)

        det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - \
              R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + \
              R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
        if det < 0:
            U[0][2] *= -1
            U[1][2] *= -1
            U[2][2] *= -1
            R = pnp._mat_mul_3x3(U, Vt)

        t = Point3D(
            tgt_c.x - (R[0][0] * src_c.x + R[0][1] * src_c.y + R[0][2] * src_c.z),
            tgt_c.y - (R[1][0] * src_c.x + R[1][1] * src_c.y + R[1][2] * src_c.z),
            tgt_c.z - (R[2][0] * src_c.x + R[2][1] * src_c.y + R[2][2] * src_c.z),
        )

        return Pose6D(rotation=RotationMatrix(R), translation=t)

    def refine(
        self,
        source: List[Point3D],
        target: List[Point3D],
        initial_pose: Pose6D,
    ) -> Tuple[Pose6D, float, int]:
        """Run ICP refinement.

        Args:
            source: Source point cloud.
            target: Target point cloud.
            initial_pose: Initial pose estimate.

        Returns:
            Tuple of (refined_pose, final_error, iterations_used).
        """
        pose = initial_pose.copy()
        prev_error = float("inf")

        for iteration in range(self.max_iterations):
            correspondences = self.find_correspondences(source, target, pose)
            if len(correspondences) < 3:
                break

            mean_dist = sum(d for _, _, d in correspondences) / len(correspondences)
            filtered = [(i, j, d) for i, j, d in correspondences
                        if d < mean_dist * self.rejection_threshold]

            if len(filtered) < 3:
                break

            delta = self.compute_transformation(source, target, filtered)
            if delta is None:
                break

            new_rotation = delta.rotation.multiply(pose.rotation)
            new_translation = delta.rotation.transform_point(pose.translation) + delta.translation
            pose = Pose6D(rotation=new_rotation, translation=new_translation)

            current_error = sum(d for _, _, d in filtered) / len(filtered)
            if abs(prev_error - current_error) < self.convergence_threshold:
                break
            prev_error = current_error

        final_correspondences = self.find_correspondences(source, target, pose)
        final_error = sum(d for _, _, d in final_correspondences) / max(len(final_correspondences), 1)
        return pose, final_error, iteration + 1


class PoseScorer:
    """Score pose hypotheses based on various criteria."""

    def __init__(
        self,
        reprojection_threshold: float = 5.0,
        inlier_ratio_weight: float = 0.4,
        reprojection_weight: float = 0.3,
        smoothness_weight: float = 0.3,
    ) -> None:
        self.reprojection_threshold = reprojection_threshold
        self.inlier_ratio_weight = inlier_ratio_weight
        self.reprojection_weight = reprojection_weight
        self.smoothness_weight = smoothness_weight

    def compute_inlier_ratio(
        self,
        pose: Pose6D,
        object_points: List[Point3D],
        image_points: List[Point2D],
        camera_matrix: List[List[float]],
    ) -> float:
        """Compute the ratio of inlier correspondences."""
        pnp = PNPSolver(camera_matrix=camera_matrix)
        errors = []
        for op, ip in zip(object_points, image_points):
            cp = pose.transform_point(op)
            if abs(cp.z) < 1e-10:
                errors.append(1000.0)
                continue
            fu = camera_matrix[0][0]
            fv = camera_matrix[1][1]
            cu = camera_matrix[0][2]
            cv = camera_matrix[1][2]
            u_proj = fu * cp.x / cp.z + cu
            v_proj = fv * cp.y / cp.z + cv
            err = math.sqrt((u_proj - ip.u) ** 2 + (v_proj - ip.v) ** 2)
            errors.append(err)

        inliers = sum(1 for e in errors if e < self.reprojection_threshold)
        return inliers / max(len(errors), 1)

    def compute_smoothness(
        self,
        pose: Pose6D,
        prev_pose: Optional[Pose6D],
    ) -> float:
        """Compute pose smoothness relative to previous frame."""
        if prev_pose is None:
            return 1.0

        dt = pose.translation.distance_to(prev_pose.translation)
        r_diff = self._rotation_difference(pose.rotation, prev_pose.rotation)

        translation_smooth = math.exp(-dt / 0.1)
        rotation_smooth = math.exp(-r_diff / 0.1)
        return 0.5 * translation_smooth + 0.5 * rotation_smooth

    def _rotation_difference(self, r1: RotationMatrix, r2: RotationMatrix) -> float:
        """Compute angular difference between two rotations in radians."""
        rd = r2.transpose().multiply(r1)
        trace = rd.data[0][0] + rd.data[1][1] + rd.data[2][2]
        trace = max(-1.0, min(3.0, trace))
        angle = math.acos((trace - 1.0) / 2.0)
        return angle

    def score(
        self,
        pose: Pose6D,
        object_points: List[Point3D],
        image_points: List[Point2D],
        camera_matrix: List[List[float]],
        prev_pose: Optional[Pose6D] = None,
    ) -> float:
        """Compute a combined score for a pose hypothesis."""
        inlier_ratio = self.compute_inlier_ratio(pose, object_points, image_points, camera_matrix)
        pnp = PNPSolver(camera_matrix=camera_matrix)
        reproj_error = pnp.compute_reprojection_error(pose, object_points, image_points)
        reproj_score = math.exp(-reproj_error / self.reprojection_threshold)
        smoothness = self.compute_smoothness(pose, prev_pose)

        combined = (
            self.inlier_ratio_weight * inlier_ratio
            + self.reprojection_weight * reproj_score
            + self.smoothness_weight * smoothness
        )
        return max(0.0, min(1.0, combined))


class HypothesisGenerator:
    """Generate multiple pose hypotheses using RANSAC-style sampling."""

    def __init__(
        self,
        num_hypotheses: int = 100,
        min_sample_size: int = 4,
        inlier_threshold: float = 5.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.num_hypotheses = num_hypotheses
        self.min_sample_size = min_sample_size
        self.inlier_threshold = inlier_threshold
        self._rng = random.Random(rng_seed)

    def generate(
        self,
        object_points: List[Point3D],
        image_points: List[Point2D],
        camera_matrix: List[List[float]],
    ) -> List[PoseHypothesis]:
        """Generate pose hypotheses using random sampling."""
        if len(object_points) < self.min_sample_size:
            return []

        pnp = PNPSolver(camera_matrix=camera_matrix)
        hypotheses: List[PoseHypothesis] = []
        n = len(object_points)
        indices = list(range(n))

        for _ in range(self.num_hypotheses):
            sample = self._rng.sample(indices, self.min_sample_size)
            sample_obj = [object_points[i] for i in sample]
            sample_img = [image_points[i] for i in sample]

            pose = pnp.solve(sample_obj, sample_img)
            if pose is None:
                continue

            total_error = 0.0
            inlier_count = 0
            for op, ip in zip(object_points, image_points):
                err = pnp.compute_reprojection_error(
                    Pose6D(pose.rotation, pose.translation), [op], [ip]
                )
                total_error += err
                if err < self.inlier_threshold:
                    inlier_count += 1

            mean_error = total_error / n
            hypothesis = PoseHypothesis(
                pose=pose,
                score=1.0 / (1.0 + mean_error),
                inlier_count=inlier_count,
                reprojection_error=mean_error,
                source="ransac",
            )
            hypotheses.append(hypothesis)

        hypotheses.sort(key=lambda h: -h.score)
        return hypotheses

    def refine_best(
        self,
        hypotheses: List[PoseHypothesis],
        object_points: List[Point3D],
        image_points: List[Point2D],
        camera_matrix: List[List[float]],
        top_k: int = 5,
    ) -> PoseHypothesis:
        """Refine top hypotheses with all inliers and return the best."""
        pnp = PNPSolver(camera_matrix=camera_matrix)
        scorer = PoseScorer()

        for h in hypotheses[:top_k]:
            inlier_obj: List[Point3D] = []
            inlier_img: List[Point2D] = []
            for op, ip in zip(object_points, image_points):
                err = pnp.compute_reprojection_error(h.pose, [op], [ip])
                if err < self.inlier_threshold:
                    inlier_obj.append(op)
                    inlier_img.append(ip)

            if len(inlier_obj) >= 4:
                refined = pnp.solve(inlier_obj, inlier_img)
                if refined is not None:
                    h.pose = refined
                    h.reprojection_error = pnp.compute_reprojection_error(
                        refined, object_points, image_points
                    )
                    h.score = scorer.score(refined, object_points, image_points, camera_matrix)

        hypotheses.sort(key=lambda h: -h.score)
        return hypotheses[0] if hypotheses else PoseHypothesis(pose=Pose6D())


class CovarianceEstimator:
    """Estimate uncertainty (covariance) of a pose estimate."""

    def __init__(
        self,
        noise_std: float = 1.0,
        num_samples: int = 100,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.noise_std = noise_std
        self.num_samples = num_samples
        self._rng = random.Random(rng_seed)

    def estimate_from_pnp(
        self,
        pose: Pose6D,
        object_points: List[Point3D],
        image_points: List[Point2D],
        camera_matrix: List[List[float]],
    ) -> Dict[str, Any]:
        """Estimate pose covariance by perturbing image points and re-solving PnP."""
        pnp = PNPSolver(camera_matrix=camera_matrix)
        translations: List[Point3D] = []
        rotations: List[List[float]] = []

        for _ in range(self.num_samples):
            noisy_img = [
                Point2D(
                    ip.u + self._rng.gauss(0, self.noise_std),
                    ip.v + self._rng.gauss(0, self.noise_std),
                )
                for ip in image_points
            ]

            noisy_pose = pnp.solve(object_points, noisy_img)
            if noisy_pose is not None and noisy_pose.rotation.is_valid_rotation():
                translations.append(noisy_pose.translation)
                rotations.append([
                    noisy_pose.rotation.data[0][0],
                    noisy_pose.rotation.data[1][1],
                    noisy_pose.rotation.data[2][2],
                ])

        if len(translations) < 3:
            return {
                "translation_covariance": [[1e6, 0, 0], [0, 1e6, 0], [0, 0, 1e6]],
                "rotation_std": [math.pi, math.pi, math.pi],
                "confidence": 0.0,
            }

        n = len(translations)
        t_mean = Point3D(
            sum(t.x for t in translations) / n,
            sum(t.y for t in translations) / n,
            sum(t.z for t in translations) / n,
        )

        t_cov = [[0.0] * 3 for _ in range(3)]
        for t in translations:
            d = Point3D(t.x - t_mean.x, t.y - t_mean.y, t.z - t_mean.z)
            t_cov[0][0] += d.x * d.x
            t_cov[0][1] += d.x * d.y
            t_cov[0][2] += d.x * d.z
            t_cov[1][0] += d.y * d.x
            t_cov[1][1] += d.y * d.y
            t_cov[1][2] += d.y * d.z
            t_cov[2][0] += d.z * d.x
            t_cov[2][1] += d.z * d.y
            t_cov[2][2] += d.z * d.z
        for i in range(3):
            for j in range(3):
                t_cov[i][j] /= max(n - 1, 1)

        r_std = [0.0, 0.0, 0.0]
        for k in range(3):
            vals = [r[k] for r in rotations]
            mean_r = sum(vals) / n
            r_std[k] = math.sqrt(sum((v - mean_r) ** 2 for v in vals) / max(n - 1, 1))

        det = t_cov[0][0] * (t_cov[1][1] * t_cov[2][2] - t_cov[1][2] * t_cov[2][1]) - \
              t_cov[0][1] * (t_cov[1][0] * t_cov[2][2] - t_cov[1][2] * t_cov[2][0]) + \
              t_cov[0][2] * (t_cov[1][0] * t_cov[2][1] - t_cov[1][1] * t_cov[2][0])
        confidence = min(1.0, 1.0 / (1.0 + abs(det) * 1e-6))

        return {
            "translation_covariance": t_cov,
            "rotation_std": r_std,
            "confidence": confidence,
            "num_valid_samples": n,
        }

    def compute_confidence_ellipse_axes(
        self, covariance: List[List[float]]
    ) -> List[float]:
        """Compute principal axes of uncertainty ellipse from 2x2 covariance."""
        if len(covariance) < 2:
            return [1.0, 1.0]
        a = covariance[0][0]
        b = covariance[0][1]
        d = covariance[1][1]
        trace = a + d
        det = a * d - b * b
        discriminant = max(trace * trace - 4 * det, 0.0)
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0
        return [math.sqrt(max(lambda1, 0)), math.sqrt(max(lambda2, 0))]


class PoseEstimator6D:
    """Main 6D pose estimation pipeline.

    Combines PnP solving, hypothesis generation, ICP refinement,
    scoring, and uncertainty estimation.
    """

    def __init__(
        self,
        camera_matrix: Optional[List[List[float]]] = None,
        num_hypotheses: int = 100,
        icp_iterations: int = 30,
        icp_max_distance: float = 0.05,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.camera_matrix = camera_matrix or [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
        self.pnp_solver = PNPSolver(camera_matrix=self.camera_matrix)
        self.hypothesis_gen = HypothesisGenerator(
            num_hypotheses=num_hypotheses,
            rng_seed=rng_seed,
        )
        self.icp_refiner = ICPRefiner(
            max_iterations=icp_iterations,
            max_correspondence_distance=icp_max_distance,
        )
        self.scorer = PoseScorer()
        self.covariance_estimator = CovarianceEstimator(rng_seed=rng_seed)
        self._prev_pose: Optional[Pose6D] = None

    def estimate(
        self,
        object_points: List[Point3D],
        image_points: List[Point2D],
        refine_with_icp: bool = True,
        estimate_uncertainty: bool = True,
        target_points: Optional[List[Point3D]] = None,
    ) -> Dict[str, Any]:
        """Estimate 6D pose from 3D-2D correspondences.

        Args:
            object_points: 3D points in object frame.
            image_points: Corresponding 2D image points.
            refine_with_icp: Whether to refine with ICP.
            estimate_uncertainty: Whether to estimate covariance.
            target_points: Optional target point cloud for ICP refinement.

        Returns:
            Dictionary with pose, score, covariance, and metadata.
        """
        if len(object_points) < 4:
            return {
                "pose": Pose6D(),
                "score": 0.0,
                "reprojection_error": float("inf"),
                "inlier_count": 0,
                "success": False,
                "method": "none",
            }

        hypotheses = self.hypothesis_gen.generate(
            object_points, image_points, self.camera_matrix
        )

        if not hypotheses:
            direct_pose = self.pnp_solver.solve(object_points, image_points)
            if direct_pose is None:
                return {
                    "pose": Pose6D(),
                    "score": 0.0,
                    "reprojection_error": float("inf"),
                    "inlier_count": 0,
                    "success": False,
                    "method": "none",
                }
            best = PoseHypothesis(
                pose=direct_pose,
                score=self.scorer.score(direct_pose, object_points, image_points, self.camera_matrix),
                inlier_count=len(object_points),
                reprojection_error=self.pnp_solver.compute_reprojection_error(
                    direct_pose, object_points, image_points
                ),
                source="direct",
            )
        else:
            best = self.hypothesis_gen.refine_best(
                hypotheses, object_points, image_points, self.camera_matrix
            )

        pose = best.pose

        if refine_with_icp and target_points is not None:
            pose, icp_error, icp_iters = self.icp_refiner.refine(
                object_points, target_points, pose
            )
            best.reprojection_error = self.pnp_solver.compute_reprojection_error(
                pose, object_points, image_points
            )
            best.pose = pose

        result: Dict[str, Any] = {
            "pose": pose,
            "score": best.score,
            "reprojection_error": best.reprojection_error,
            "inlier_count": best.inlier_count,
            "success": True,
            "method": best.source,
            "num_hypotheses_tested": len(hypotheses),
        }

        if estimate_uncertainty:
            covariance = self.covariance_estimator.estimate_from_pnp(
                pose, object_points, image_points, self.camera_matrix
            )
            result["covariance"] = covariance

        self._prev_pose = pose.copy()
        return result

    def estimate_with_model(
        self,
        model_points: List[Point3D],
        detected_points_2d: List[Point2D],
        correspondences: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Estimate pose with a 3D model.

        Args:
            model_points: 3D model point cloud.
            detected_points_2d: 2D detected points.
            correspondences: Optional list of (model_idx, detected_idx) pairs.

        Returns:
            Pose estimation result dictionary.
        """
        if correspondences is not None:
            obj_pts = [model_points[i] for i, _ in correspondences]
            img_pts = [detected_points_2d[j] for _, j in correspondences]
        else:
            min_len = min(len(model_points), len(detected_points_2d))
            obj_pts = model_points[:min_len]
            img_pts = detected_points_2d[:min_len]

        return self.estimate(obj_pts, img_pts, target_points=model_points)

    def reset(self) -> None:
        """Reset temporal state (previous pose tracking)."""
        self._prev_pose = None

    def set_camera_matrix(self, camera_matrix: List[List[float]]) -> None:
        """Update the camera intrinsic matrix."""
        self.camera_matrix = camera_matrix
        self.pnp_solver = PNPSolver(camera_matrix=camera_matrix)

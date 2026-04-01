#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URDF-based bimanual closed-chain constraint Jacobian (analytic),
without external robotics libraries (only numpy/scipy are used).

Constraint:
  g(q)      = T_L(q)^{-1} T_R(q)          (relative transform from left frame to right frame)
  g_des     = g(q_ref)                   (default: q_ref = 0)
  T_err(q)  = g_des^{-1} g(q)
  h(q)      = Log_SE3(T_err(q))          (6x1, order [rho; omega] = [v; w])

Analytic Jacobian:
  δ(q)        = (T_err^{-1} dT_err)∨ = (g^{-1} dg)∨
  J_rel(q)    = ∂δ/∂q  (computed from body geometric Jacobians of L and R)
  J_log(q)    = ∂Log/∂δ = J_r^{-1}(xi_err) where xi_err = Log(T_err)
  J_ana(q)    = J_log(q) * J_rel(q)

Notes:
- We compute geometric Jacobians with v = linear velocity of frame origin (NOT the se(3) matrix v = pdot - ω×p).
  For that convention, changing coordinates between world and body uses rotation only (block-diag(R^T, R^T)).
- The SE(3) right-Jacobian inverse J_r^{-1} is computed via a convergent series for J_r and matrix inversion.
  This gives high accuracy across typical joint ranges.

Usage example:
  python bimanual_jacobian_compare_urdf.py \
    --urdf ffw_sg2_follower.urdf \
    --left gripper_l_rh_p12_rn_base \
    --right gripper_r_rh_p12_rn_base \
    --qmode neutral

"""

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from capstone_pkg.utils.config import LEFT_GRIPPER, RIGHT_GRIPPER

import numpy as np
import torch

ROBOT_URDF = "/home/gaga/tb_rrt_ws/src/capstone_pkg/models/ffw_sg2/ffw_sg2_follower.urdf"

# -----------------------
# SE(3) helpers
# -----------------------
def parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.strip().split()], dtype=float)

def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=float)

def rot_x(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, ca, -sa],
                     [0.0, sa,  ca]], dtype=float)

def rot_y(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0.0, sa],
                     [0.0, 1.0, 0.0],
                     [-sa, 0.0, ca]], dtype=float)

def rot_z(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    # URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return rot_z(y) @ rot_y(p) @ rot_x(r)

def T_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T

def axis_angle_to_R(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    K = skew(a)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def joint_motion_T(jtype: str, axis: np.ndarray, q: float) -> np.ndarray:
    T = np.eye(4)
    if jtype in ("revolute", "continuous"):
        T[:3, :3] = axis_angle_to_R(axis, q)
    elif jtype == "prismatic":
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            d = np.zeros(3)
        else:
            d = axis / n * q
        T[:3, 3] = d
    elif jtype == "fixed":
        pass
    else:
        raise NotImplementedError(f"Unsupported joint type: {jtype}")
    return T

def SE3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti

def Adjoint(T: np.ndarray) -> np.ndarray:
    # For twist order [v; w]
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad

def so3_log_robust(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))

    S = 0.5 * (R - R.T)
    v = np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)  # axis*sin(theta)
    sin_theta = float(np.linalg.norm(v))
    theta = math.atan2(sin_theta, cos_theta)

    if sin_theta < 1e-12 and theta < 1e-12:
        return np.zeros(3)

    if abs(theta - math.pi) < 1e-6:
        A = (R + np.eye(3)) / 2.0
        axis = np.array([
            math.sqrt(max(A[0, 0], 0.0)),
            math.sqrt(max(A[1, 1], 0.0)),
            math.sqrt(max(A[2, 2], 0.0)),
        ])
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        return axis * theta

    return v * (theta / sin_theta)

def se3_log_robust(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    omega = so3_log_robust(R)
    theta = float(np.linalg.norm(omega))
    Omega = skew(omega)

    if theta < 1e-9:
        V = np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * (Omega @ Omega)
    else:
        th2 = theta * theta
        A = (1.0 - math.cos(theta)) / th2
        B = (theta - math.sin(theta)) / (theta * th2)
        V = np.eye(3) + A * Omega + B * (Omega @ Omega)

    rho = np.linalg.solve(V, p)
    return np.concatenate([rho, omega])

# -----------------------
# Lie Jacobian: Jr^{-1}(xi)
# -----------------------
def ad_se3(xi: np.ndarray) -> np.ndarray:
    # xi = [v; w]
    v = xi[:3]
    w = xi[3:]
    W = skew(w)
    V = skew(v)
    ad = np.zeros((6, 6))
    ad[:3, :3] = W
    ad[:3, 3:] = V
    ad[3:, 3:] = W
    return ad

def Jr_inv_se3_via_series(xi: np.ndarray, N: int = 60) -> np.ndarray:
    """
    Compute J_r^{-1}(xi) by:
      1) J_r(xi) = sum_{n=0..N} (-1)^n / (n+1)! * ad_xi^n
      2) J_r^{-1} = inv(J_r)
    """
    ad = ad_se3(xi)
    I = np.eye(6)
    Jr = np.zeros((6, 6))
    term = I.copy()
    for n in range(N + 1):
        coeff = ((-1.0) ** n) / math.factorial(n + 1)
        Jr += coeff * term
        term = term @ ad
    return np.linalg.inv(Jr)

# -----------------------
# URDF kinematics
# -----------------------
@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None

class URDFModel:
    def __init__(self, urdf_path: str):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(urdf_path)
        self.urdf_path = urdf_path
        tree = ET.parse(urdf_path)
        self.root = tree.getroot()

        self.links = [l.attrib["name"] for l in self.root.findall("link")]
        self.joints: Dict[str, Joint] = {}
        for je in self.root.findall("joint"):
            name = je.attrib["name"]
            jtype = je.attrib.get("type", "fixed")
            parent = je.find("parent").attrib["link"]
            child = je.find("child").attrib["link"]
            origin_elem = je.find("origin")
            if origin_elem is not None:
                xyz = parse_vec(origin_elem.attrib.get("xyz", "0 0 0"))
                rpy = parse_vec(origin_elem.attrib.get("rpy", "0 0 0"))
            else:
                xyz = np.zeros(3)
                rpy = np.zeros(3)
            T_origin = T_from_xyz_rpy(xyz, rpy)
            axis_elem = je.find("axis")
            axis = parse_vec(axis_elem.attrib.get("xyz", "1 0 0")) if axis_elem is not None else np.array([1.0, 0.0, 0.0])
            lim_elem = je.find("limit")
            lower = upper = None
            if lim_elem is not None:
                if "lower" in lim_elem.attrib:
                    lower = float(lim_elem.attrib["lower"])
                if "upper" in lim_elem.attrib:
                    upper = float(lim_elem.attrib["upper"])
            self.joints[name] = Joint(name=name, type=jtype, parent=parent, child=child, T_origin=T_origin, axis=axis,
                                      limit_lower=lower, limit_upper=upper)

        # child link -> parent joint
        self.child_to_joint: Dict[str, str] = {j.child: j.name for j in self.joints.values()}
        roots = [lnk for lnk in self.links if lnk not in self.child_to_joint]
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly 1 root link, got {roots}")
        self.root_link = roots[0]

    def path_joints(self, link_name: str) -> List[str]:
        path: List[str] = []
        curr = link_name
        while curr != self.root_link:
            jn = self.child_to_joint.get(curr)
            if jn is None:
                raise RuntimeError(f"Link '{curr}' has no parent joint (root is '{self.root_link}')")
            path.append(jn)
            curr = self.joints[jn].parent
        path.reverse()
        return path

    def fk_and_geometric_jacobian_world(self, link_name: str, q_vec: np.ndarray, active_joints: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          T_world_link (4x4)
          J_world (6 x n) geometric Jacobian in world coords:
              [v_world; w_world], where v_world is linear velocity of link origin.
        """
        q_map = {jn: float(q_vec[i]) for i, jn in enumerate(active_joints)}
        active_index = {jn: i for i, jn in enumerate(active_joints)}
        path = self.path_joints(link_name)

        T = np.eye(4)  # world -> current link
        joint_records = []  # (col_idx, type, p_joint_world, axis_world)
        for jn in path:
            j = self.joints[jn]
            T_pre = T @ j.T_origin  # world -> joint frame (before motion)
            if j.type != "fixed":
                idx = active_index.get(jn, None)
                if idx is not None:
                    R_pre = T_pre[:3, :3]
                    axis = j.axis
                    n = float(np.linalg.norm(axis))
                    a = axis / n if n > 1e-12 else np.zeros(3)
                    a_world = R_pre @ a
                    p_joint = T_pre[:3, 3]
                    joint_records.append((idx, j.type, p_joint, a_world))
            qj = q_map.get(jn, 0.0)
            T = T_pre @ joint_motion_T(j.type, j.axis, qj)

        T_end = T
        p_end = T_end[:3, 3]
        J = np.zeros((6, len(active_joints)))
        for idx, jtype, p_joint, a_world in joint_records:
            if jtype in ("revolute", "continuous"):
                w = a_world
                v = np.cross(w, (p_end - p_joint))
            elif jtype == "prismatic":
                w = np.zeros(3)
                v = a_world
            else:
                continue
            J[:3, idx] = v
            J[3:, idx] = w
        return T_end, J

    @staticmethod
    def world_to_body_geometric(J_world: np.ndarray, T_world_link: np.ndarray) -> np.ndarray:
        """
        Convert geometric Jacobian from world coords to body coords at the same link origin:
          v_body = R^T v_world
          w_body = R^T w_world
        """
        R = T_world_link[:3, :3]
        M = np.zeros((6, 6))
        M[:3, :3] = R.T
        M[3:, 3:] = R.T
        return M @ J_world


# -----------------------
# Reusable Jacobian backend (import from planners/projectors)
# -----------------------
@dataclass(frozen=True)
class BimanualResidualJacobian:
    h: np.ndarray  # (N, 6)
    J: np.ndarray  # (N, 6, D)


class BimanualConstraintJacobianBackend:
    """
    Importable backend that provides SE(3) residual and analytic Jacobian:
      h(q) = Log( g_des^{-1} * (T_L(q)^{-1} T_R(q)) )
      J(q) = d h / d q
    """

    def __init__(
        self,
        *,
        urdf_path: str,
        left_link: str,
        right_link: str,
        active_joints: Sequence[str],
        q_ref: Sequence[float],
        jr_order: int = 60,
    ):
        self.model = URDFModel(urdf_path)
        self.left_link = str(left_link)
        self.right_link = str(right_link)
        self.active_joints = [str(x) for x in active_joints]
        self.jr_order = int(jr_order)

        if self.left_link not in self.model.links:
            raise RuntimeError(f"Left frame '{self.left_link}' not found in links")
        if self.right_link not in self.model.links:
            raise RuntimeError(f"Right frame '{self.right_link}' not found in links")
        if len(self.active_joints) == 0:
            raise ValueError("active_joints is empty")

        q_ref_np = np.asarray(q_ref, dtype=np.float64).reshape(-1)
        if q_ref_np.shape[0] != len(self.active_joints):
            raise ValueError(
                f"q_ref dim mismatch: {q_ref_np.shape[0]} vs active_joints={len(self.active_joints)}"
            )

        T_L_ref, _ = self.model.fk_and_geometric_jacobian_world(self.left_link, q_ref_np, self.active_joints)
        T_R_ref, _ = self.model.fk_and_geometric_jacobian_world(self.right_link, q_ref_np, self.active_joints)
        self.g_des = SE3_inv(T_L_ref) @ T_R_ref

    def _to_2d_numpy(self, q: np.ndarray | Sequence[float]) -> np.ndarray:
        if isinstance(q, np.ndarray):
            q_np = q
        else:
            q_np = np.asarray(q, dtype=np.float64)
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        if q_np.ndim != 2:
            raise ValueError(f"q must be (D,) or (N,D), got shape={tuple(q_np.shape)}")
        if q_np.shape[1] != len(self.active_joints):
            raise ValueError(
                f"q dim mismatch: {q_np.shape[1]} vs active_joints={len(self.active_joints)}"
            )
        return q_np.astype(np.float64, copy=False)

    def _one(self, q_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T_L, Jw_L = self.model.fk_and_geometric_jacobian_world(self.left_link, q_vec, self.active_joints)
        T_R, Jw_R = self.model.fk_and_geometric_jacobian_world(self.right_link, q_vec, self.active_joints)

        Jb_L = self.model.world_to_body_geometric(Jw_L, T_L)
        Jb_R = self.model.world_to_body_geometric(Jw_R, T_R)

        g = SE3_inv(T_L) @ T_R
        T_err = SE3_inv(self.g_des) @ g
        xi_err = se3_log_robust(T_err)

        Ad_ginv = Adjoint(SE3_inv(g))
        J_rel = Jb_R - Ad_ginv @ Jb_L

        Jlog = Jr_inv_se3_via_series(xi_err, N=self.jr_order)
        J_ana = Jlog @ J_rel
        return xi_err, J_ana

    def residual(self, q: np.ndarray | Sequence[float]) -> np.ndarray:
        q_np = self._to_2d_numpy(q)
        hs = [self._one(qi)[0] for qi in q_np]
        return np.stack(hs, axis=0)

    def jacobian(self, q: np.ndarray | Sequence[float]) -> np.ndarray:
        q_np = self._to_2d_numpy(q)
        Js = [self._one(qi)[1] for qi in q_np]
        return np.stack(Js, axis=0)

    def residual_and_jacobian(self, q: np.ndarray | Sequence[float]) -> BimanualResidualJacobian:
        q_np = self._to_2d_numpy(q)
        hs: List[np.ndarray] = []
        Js: List[np.ndarray] = []
        for qi in q_np:
            h_i, J_i = self._one(qi)
            hs.append(h_i)
            Js.append(J_i)
        return BimanualResidualJacobian(
            h=np.stack(hs, axis=0),
            J=np.stack(Js, axis=0),
        )


@dataclass(frozen=True)
class JointPathEntryTorch:
    type_code: int
    active_idx: int
    R_origin: np.ndarray
    p_origin: np.ndarray
    axis: np.ndarray


def skew_torch(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    o = torch.zeros_like(x)
    row0 = torch.stack([o, -z, y], dim=-1)
    row1 = torch.stack([z, o, -x], dim=-1)
    row2 = torch.stack([-y, x, o], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def axis_angle_to_R_torch(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    axis = axis.to(device=theta.device, dtype=theta.dtype)
    n = torch.linalg.norm(axis).clamp_min(1e-12)
    a = axis / n
    K = skew_torch(a.view(1, 3)).squeeze(0)
    K2 = K @ K
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).unsqueeze(0).expand(theta.shape[0], 3, 3)
    s = torch.sin(theta).view(-1, 1, 1)
    c = torch.cos(theta).view(-1, 1, 1)
    return eye + s * K + (1.0 - c) * K2


def so3_log_robust_torch(R: torch.Tensor) -> torch.Tensor:
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)

    S = 0.5 * (R - R.transpose(-1, -2))
    v = torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)
    sin_theta = torch.linalg.norm(v, dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)

    omega = torch.zeros_like(v)
    small = (sin_theta < 1e-12) & (theta < 1e-12)
    near_pi = torch.abs(theta - math.pi) < 1e-6
    general = ~(small | near_pi)

    if bool(general.any().item()):
        scale = (theta[general] / sin_theta[general]).unsqueeze(-1)
        omega[general] = v[general] * scale

    if bool(near_pi.any().item()):
        R_pi = R[near_pi]
        theta_pi = theta[near_pi]
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R_pi.shape[0], 3, 3)
        A = (R_pi + I) * 0.5
        axis = torch.sqrt(torch.clamp(torch.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]], dim=-1), min=0.0))

        sx = torch.where(
            (R_pi[:, 2, 1] - R_pi[:, 1, 2]) < 0.0,
            -torch.ones_like(theta_pi),
            torch.ones_like(theta_pi),
        )
        sy = torch.where(
            (R_pi[:, 0, 2] - R_pi[:, 2, 0]) < 0.0,
            -torch.ones_like(theta_pi),
            torch.ones_like(theta_pi),
        )
        sz = torch.where(
            (R_pi[:, 1, 0] - R_pi[:, 0, 1]) < 0.0,
            -torch.ones_like(theta_pi),
            torch.ones_like(theta_pi),
        )
        axis = axis * torch.stack([sx, sy, sz], dim=-1)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
        omega[near_pi] = axis * theta_pi.unsqueeze(-1)

    if bool(small.any().item()):
        omega[small] = v[small]

    return omega


def se3_log_robust_torch(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    omega = so3_log_robust_torch(R)
    theta = torch.linalg.norm(omega, dim=-1)
    Omega = skew_torch(omega)
    I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], 3, 3)

    V = torch.empty_like(I)
    small = theta < 1e-9
    if bool(small.any().item()):
        Om_s = Omega[small]
        V[small] = I[small] + 0.5 * Om_s + (1.0 / 6.0) * (Om_s @ Om_s)
    if bool((~small).any().item()):
        th = theta[~small]
        Om = Omega[~small]
        th2 = th * th
        A = ((1.0 - torch.cos(th)) / th2).view(-1, 1, 1)
        B = ((th - torch.sin(th)) / (th * th2)).view(-1, 1, 1)
        V[~small] = I[~small] + A * Om + B * (Om @ Om)

    rho = torch.linalg.solve(V, p.unsqueeze(-1)).squeeze(-1)
    return torch.cat([rho, omega], dim=-1)


def ad_se3_torch(xi: torch.Tensor) -> torch.Tensor:
    v = xi[..., :3]
    w = xi[..., 3:]
    W = skew_torch(w)
    V = skew_torch(v)
    Z = torch.zeros_like(W)
    top = torch.cat([W, V], dim=-1)
    bottom = torch.cat([Z, W], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def Jr_inv_se3_via_series_torch(xi: torch.Tensor, N: int = 60) -> torch.Tensor:
    ad = ad_se3_torch(xi)
    I = torch.eye(6, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(xi.shape[0], 6, 6)
    Jr = torch.zeros_like(I)
    term = I.clone()
    for n in range(N + 1):
        coeff = ((-1.0) ** n) / math.factorial(n + 1)
        Jr = Jr + coeff * term
        term = term @ ad
    return torch.linalg.inv(Jr)


class BimanualConstraintJacobianBackendTorch:
    """
    Torch batched backend for the same SE(3) residual/Jacobian.

    - No python loop over batch dimension.
    - Still loops over the fixed joint-path length (small, robot-dependent).
    """

    def __init__(
        self,
        *,
        urdf_path: str,
        left_link: str,
        right_link: str,
        active_joints: Sequence[str],
        q_ref: Sequence[float],
        jr_order: int = 60,
    ):
        self.model = URDFModel(urdf_path)
        self.left_link = str(left_link)
        self.right_link = str(right_link)
        self.active_joints = [str(x) for x in active_joints]
        self.jr_order = int(jr_order)
        self.D = len(self.active_joints)

        if self.left_link not in self.model.links:
            raise RuntimeError(f"Left frame '{self.left_link}' not found in links")
        if self.right_link not in self.model.links:
            raise RuntimeError(f"Right frame '{self.right_link}' not found in links")
        if self.D == 0:
            raise ValueError("active_joints is empty")

        q_ref_np = np.asarray(q_ref, dtype=np.float64).reshape(-1)
        if q_ref_np.shape[0] != self.D:
            raise ValueError(
                f"q_ref dim mismatch: {q_ref_np.shape[0]} vs active_joints={self.D}"
            )

        active_index = {jn: i for i, jn in enumerate(self.active_joints)}
        self.path_left = self._build_path_spec(self.left_link, active_index)
        self.path_right = self._build_path_spec(self.right_link, active_index)

        q_ref_t = torch.as_tensor(q_ref_np, dtype=torch.float64).view(1, -1)
        RL_ref, pL_ref, _ = self._fk_and_geometric_jacobian_world(self.path_left, q_ref_t)
        RR_ref, pR_ref, _ = self._fk_and_geometric_jacobian_world(self.path_right, q_ref_t)
        self.g_des_R = (RL_ref.transpose(1, 2) @ RR_ref)[0].contiguous()
        self.g_des_p = (RL_ref.transpose(1, 2) @ (pR_ref - pL_ref).unsqueeze(-1)).squeeze(-1)[0].contiguous()

    def _build_path_spec(
        self,
        link_name: str,
        active_index: Dict[str, int],
    ) -> List[JointPathEntryTorch]:
        path = self.model.path_joints(link_name)
        specs: List[JointPathEntryTorch] = []
        for jn in path:
            j = self.model.joints[jn]
            if j.type == "fixed":
                type_code = 0
            elif j.type in ("revolute", "continuous"):
                type_code = 1
            elif j.type == "prismatic":
                type_code = 2
            else:
                raise NotImplementedError(f"Unsupported joint type: {j.type}")

            axis = np.asarray(j.axis, dtype=np.float64)
            n = float(np.linalg.norm(axis))
            if n > 1e-12:
                axis = axis / n
            else:
                axis = np.zeros(3, dtype=np.float64)

            specs.append(
                JointPathEntryTorch(
                    type_code=type_code,
                    active_idx=active_index.get(jn, -1) if type_code != 0 else -1,
                    R_origin=j.T_origin[:3, :3].astype(np.float64, copy=True),
                    p_origin=j.T_origin[:3, 3].astype(np.float64, copy=True),
                    axis=axis.astype(np.float64, copy=True),
                )
            )
        return specs

    def _to_2d_torch(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 1:
            q = q.view(1, -1)
        if q.ndim != 2:
            raise ValueError(f"q must be (D,) or (N,D), got shape={tuple(q.shape)}")
        if q.shape[1] != self.D:
            raise ValueError(f"q dim mismatch: {q.shape[1]} vs active_joints={self.D}")
        return q

    def _fk_and_geometric_jacobian_world(
        self,
        path_spec: List[JointPathEntryTorch],
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._to_2d_torch(q)
        N = q.shape[0]
        device, dtype = q.device, q.dtype

        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3).clone()
        p = torch.zeros((N, 3), device=device, dtype=dtype)
        zeros_q = torch.zeros((N,), device=device, dtype=dtype)
        records: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []

        for spec in path_spec:
            R0 = torch.as_tensor(spec.R_origin, device=device, dtype=dtype)
            p0 = torch.as_tensor(spec.p_origin, device=device, dtype=dtype)
            axis = torch.as_tensor(spec.axis, device=device, dtype=dtype)

            R_pre = torch.matmul(R, R0)
            p_pre = p + torch.matmul(R, p0.view(1, 3, 1)).squeeze(-1)

            if spec.type_code != 0 and spec.active_idx >= 0:
                a_world = torch.matmul(R_pre, axis.view(1, 3, 1)).squeeze(-1)
                records.append((spec.active_idx, spec.type_code, p_pre, a_world))
                qj = q[:, spec.active_idx]
            else:
                qj = zeros_q

            if spec.type_code == 1:
                R = torch.matmul(R_pre, axis_angle_to_R_torch(axis, qj))
                p = p_pre
            elif spec.type_code == 2:
                d_local = axis.view(1, 3) * qj.unsqueeze(-1)
                d_world = torch.matmul(R_pre, d_local.unsqueeze(-1)).squeeze(-1)
                R = R_pre
                p = p_pre + d_world
            else:
                R = R_pre
                p = p_pre

        J = torch.zeros((N, 6, self.D), device=device, dtype=dtype)
        for idx, type_code, p_joint, a_world in records:
            if type_code == 1:
                w = a_world
                v = torch.cross(w, (p - p_joint), dim=-1)
            elif type_code == 2:
                w = torch.zeros_like(a_world)
                v = a_world
            else:
                continue
            J[:, :3, idx] = v
            J[:, 3:, idx] = w

        return R, p, J

    @staticmethod
    def _world_to_body_geometric(R_world_link: torch.Tensor, J_world: torch.Tensor) -> torch.Tensor:
        Rt = R_world_link.transpose(1, 2)
        Jv = torch.matmul(Rt, J_world[:, :3, :])
        Jw = torch.matmul(Rt, J_world[:, 3:, :])
        return torch.cat([Jv, Jw], dim=1)

    @staticmethod
    def _apply_adjoint_inverse(
        R: torch.Tensor,
        p: torch.Tensor,
        J: torch.Tensor,
    ) -> torch.Tensor:
        Rt = R.transpose(1, 2)
        Jv = torch.matmul(Rt, J[:, :3, :])
        Jw = torch.matmul(Rt, J[:, 3:, :])
        p_inv = -torch.matmul(Rt, p.unsqueeze(-1)).squeeze(-1)
        Jv = Jv + torch.matmul(skew_torch(p_inv), Jw)
        return torch.cat([Jv, Jw], dim=1)

    def residual_and_jacobian(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._to_2d_torch(q)
        device, dtype = q.device, q.dtype

        RL, pL, JwL = self._fk_and_geometric_jacobian_world(self.path_left, q)
        RR, pR, JwR = self._fk_and_geometric_jacobian_world(self.path_right, q)

        JbL = self._world_to_body_geometric(RL, JwL)
        JbR = self._world_to_body_geometric(RR, JwR)

        Rg = torch.matmul(RL.transpose(1, 2), RR)
        pg = torch.matmul(RL.transpose(1, 2), (pR - pL).unsqueeze(-1)).squeeze(-1)

        Rg_des_inv = self.g_des_R.transpose(0, 1).to(device=device, dtype=dtype)
        pg_des = self.g_des_p.to(device=device, dtype=dtype)
        Rerr = torch.matmul(Rg_des_inv, Rg)
        perr = torch.matmul(Rg_des_inv, (pg - pg_des).unsqueeze(-1)).squeeze(-1)
        xi_err = se3_log_robust_torch(Rerr, perr)

        Jrel = JbR - self._apply_adjoint_inverse(Rg, pg, JbL)
        Jlog = Jr_inv_se3_via_series_torch(xi_err, N=self.jr_order)
        Jana = torch.matmul(Jlog, Jrel)
        return xi_err, Jana

    def residual(self, q: torch.Tensor) -> torch.Tensor:
        h, _ = self.residual_and_jacobian(q)
        return h

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        _, J = self.residual_and_jacobian(q)
        return J


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=ROBOT_URDF)
    ap.add_argument("--left", default=LEFT_GRIPPER)
    ap.add_argument("--right", default=RIGHT_GRIPPER)
    ap.add_argument("--qmode", choices=["neutral", "random"], default="neutral")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jr_order", type=int, default=60, help="Series truncation order for J_r^{-1}")
    args = ap.parse_args()

    model = URDFModel(args.urdf)

    if args.left not in model.links:
        raise RuntimeError(f"Left frame '{args.left}' not found in links")
    if args.right not in model.links:
        raise RuntimeError(f"Right frame '{args.right}' not found in links")

    path_L = model.path_joints(args.left)
    path_R = model.path_joints(args.right)

    # active joints = union of non-fixed joints on both paths
    active: List[str] = []
    seen = set()
    for jn in path_L + path_R:
        j = model.joints[jn]
        if j.type == "fixed":
            continue
        if jn not in seen:
            active.append(jn)
            seen.add(jn)

    n = len(active)
    np.random.seed(args.seed)

    # q_ref = zeros
    q_ref = np.zeros(n)

    # desired relative transform at q_ref
    T_L_ref, _ = model.fk_and_geometric_jacobian_world(args.left, q_ref, active)
    T_R_ref, _ = model.fk_and_geometric_jacobian_world(args.right, q_ref, active)
    g_des = SE3_inv(T_L_ref) @ T_R_ref

    # choose q_test
    if args.qmode == "neutral":
        q = q_ref.copy()
    else:
        q = np.zeros(n)
        for i, jn in enumerate(active):
            j = model.joints[jn]
            if j.limit_lower is not None and j.limit_upper is not None:
                lo, up = j.limit_lower, j.limit_upper
                # sample inside limits, avoid boundaries
                q[i] = lo + 0.25 * (up - lo) + np.random.rand() * 0.5 * (up - lo)
            else:
                q[i] = (np.random.rand() - 0.5) * 1.0

    def h(qv: np.ndarray) -> np.ndarray:
        T_L, _ = model.fk_and_geometric_jacobian_world(args.left, qv, active)
        T_R, _ = model.fk_and_geometric_jacobian_world(args.right, qv, active)
        g = SE3_inv(T_L) @ T_R
        T_err = SE3_inv(g_des) @ g
        return se3_log_robust(T_err)

    # analytic Jacobian
    T_L, Jw_L = model.fk_and_geometric_jacobian_world(args.left, q, active)
    T_R, Jw_R = model.fk_and_geometric_jacobian_world(args.right, q, active)
    Jb_L = model.world_to_body_geometric(Jw_L, T_L)
    Jb_R = model.world_to_body_geometric(Jw_R, T_R)
    g = SE3_inv(T_L) @ T_R
    T_err = SE3_inv(g_des) @ g
    xi_err = se3_log_robust(T_err)

    Ad_ginv = Adjoint(SE3_inv(g))
    J_rel = Jb_R - Ad_ginv @ Jb_L

    Jlog = Jr_inv_se3_via_series(xi_err, N=args.jr_order)
    J_ana = Jlog @ J_rel

    # Pretty print
    np.set_printoptions(precision=6, suppress=True)
    print("=== Model info ===")
    print(f"robot             : {model.root.attrib.get('name','(unnamed)')}")
    print(f"root link         : {model.root_link}")
    print(f"left link         : {args.left}")
    print(f"right link        : {args.right}")
    print(f"active dofs (n)    : {n}")
    print(f"active joints      : {active}")
    print(f"qmode              : {args.qmode}")
    print(f"jr_order           : {args.jr_order}")
    print("")
    print("=== h(q) ===")
    print(h(q).reshape(6,1))
    print("")
    print("=== J_ana (6 x n) ===")
    print(J_ana)

if __name__ == "__main__":
    main()

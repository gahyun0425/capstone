from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import yaml

from capstone_pkg.constraint_projection.bimanual_jacobian_compare_urdf import (
    URDFModel,
    so3_log_robust,
)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _rot_z(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rot(q_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(xyz: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat_wxyz)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


def _base_pose_to_matrix(base_pose_xyyaw: Sequence[float]) -> np.ndarray:
    x, y, yaw = [float(v) for v in base_pose_xyyaw]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot_z(yaw)
    T[:3, 3] = [x, y, 0.0]
    return T


def _base_body_twist_between(
    a: Sequence[float],
    b: Sequence[float],
    *,
    dt: float,
) -> np.ndarray:
    dt = max(1.0e-6, float(dt))
    ax, ay, ayaw = [float(v) for v in a]
    bx, by, byaw = [float(v) for v in b]
    c = math.cos(-ayaw)
    s = math.sin(-ayaw)
    dx = bx - ax
    dy = by - ay
    return np.array(
        [
            (c * dx - s * dy) / dt,
            (s * dx + c * dy) / dt,
            _wrap_pi(byaw - ayaw) / dt,
        ],
        dtype=np.float64,
    )


def _integrate_base_body_twist(
    base_pose: Sequence[float],
    twist_body: Sequence[float],
    *,
    dt: float,
) -> list[float]:
    x, y, yaw = [float(v) for v in base_pose]
    vx, vy, wz = [float(v) for v in twist_body]
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [
        x + (c * vx - s * vy) * dt,
        y + (s * vx + c * vy) * dt,
        _wrap_pi(yaw + wz * dt),
    ]


def _load_named_joint_limits(
    joint_limit_yml: str,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    with open(str(joint_limit_yml), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    jl = data.get("joint_limits", data)
    names = [str(v) for v in jl.get("joint_names", [])]
    lower_all = [float(v) for v in jl["lower"]]
    upper_all = [float(v) for v in jl["upper"]]
    if names and len(names) == len(lower_all) == len(upper_all):
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        lower = []
        upper = []
        for name in joint_names:
            idx = name_to_idx.get(str(name))
            if idx is None:
                raise RuntimeError(f"joint limit for '{name}' not found in {joint_limit_yml}")
            lower.append(lower_all[idx])
            upper.append(upper_all[idx])
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    if len(lower_all) != len(joint_names) or len(upper_all) != len(joint_names):
        raise RuntimeError(
            "joint_limits.yaml has no matching joint_names and limit dimension "
            f"does not match selected joints: {len(lower_all)} vs {len(joint_names)}"
        )
    return np.asarray(lower_all, dtype=np.float64), np.asarray(upper_all, dtype=np.float64)


@dataclass(frozen=True)
class DoorWBCQPConfig:
    urdf_path: str
    ee_frame: str
    joint_limit_yml: str
    dt: float
    max_base_linear_mps: float
    max_base_angular_rps: float
    max_joint_velocity: float
    kp_pos: float = 2.0
    kp_rot: float = 2.0
    task_weight: float = 100.0
    base_ref_weight: float = 1.0
    joint_ref_weight: float = 1.0
    joint_reg_weight: float = 1.0e-3
    hard_task_constraint: bool = False
    backend: str = "auto"


@dataclass(frozen=True)
class DoorWBCRolloutResult:
    joint_path: list[list[float]]
    base_poses: list[list[float]]
    task_error_norms: list[float] = field(default_factory=list)
    solver_statuses: list[str] = field(default_factory=list)
    used_backend: str = ""


class DoorOpeningWBCQP:
    """Velocity-level QP rollout for door opening base+arm references."""

    def __init__(self, cfg: DoorWBCQPConfig):
        self.cfg = cfg
        self.model = URDFModel(str(cfg.urdf_path))
        if str(cfg.ee_frame) not in self.model.links:
            raise RuntimeError(f"EE frame '{cfg.ee_frame}' not found in URDF links")

    def rollout(
        self,
        *,
        joint_names: Sequence[str],
        joint_path: Sequence[Sequence[float]],
        base_poses: Sequence[Sequence[float]],
        desired_ee_poses: Sequence[tuple[Sequence[float], Sequence[float]]],
    ) -> DoorWBCRolloutResult:
        names = [str(name) for name in joint_names]
        q_ref = np.asarray(joint_path, dtype=np.float64)
        b_ref = np.asarray(base_poses, dtype=np.float64)
        if q_ref.ndim != 2:
            raise RuntimeError("joint_path must be a 2D numeric array")
        if b_ref.shape != (q_ref.shape[0], 3):
            raise RuntimeError("base_poses must have shape (N, 3)")
        if len(desired_ee_poses) != q_ref.shape[0]:
            raise RuntimeError("desired_ee_poses length must match joint_path")
        if q_ref.shape[0] < 2:
            raise RuntimeError("QP rollout needs at least two waypoints")
        if len(names) != q_ref.shape[1]:
            raise RuntimeError("joint_names length must match joint_path dimension")

        q_lower, q_upper = _load_named_joint_limits(self.cfg.joint_limit_yml, names)
        dt = max(1.0e-6, float(self.cfg.dt))
        q_cur = np.clip(q_ref[0].copy(), q_lower, q_upper)
        b_cur = b_ref[0].copy()
        out_q = [q_cur.tolist()]
        out_b = [b_cur.tolist()]
        errors: list[float] = []
        statuses: list[str] = []
        backend_used = ""

        desired_T = [_pose_matrix(xyz, quat) for xyz, quat in desired_ee_poses]
        for idx in range(q_ref.shape[0] - 1):
            J, T_cur = self._world_jacobian(q_cur, b_cur, names)
            target_twist = self._target_twist(
                T_cur=T_cur,
                T_des=desired_T[idx],
                T_des_next=desired_T[idx + 1],
                dt=dt,
            )
            base_ref = _base_body_twist_between(b_ref[idx], b_ref[idx + 1], dt=dt)
            qdot_ref = (q_ref[idx + 1] - q_ref[idx]) / dt
            lower, upper = self._velocity_bounds(
                q_cur=q_cur,
                q_lower=q_lower,
                q_upper=q_upper,
                dt=dt,
            )
            u, status, backend = self._solve_velocity_qp(
                J=J,
                target_twist=target_twist,
                base_ref=base_ref,
                qdot_ref=qdot_ref,
                lower=lower,
                upper=upper,
            )
            backend_used = backend_used or backend
            statuses.append(status)
            task_err = J @ u - target_twist
            errors.append(float(np.linalg.norm(task_err)))
            b_cur = np.asarray(
                _integrate_base_body_twist(b_cur, u[:3], dt=dt),
                dtype=np.float64,
            )
            q_cur = np.clip(q_cur + u[3:] * dt, q_lower, q_upper)
            out_b.append(b_cur.tolist())
            out_q.append(q_cur.tolist())

        return DoorWBCRolloutResult(
            joint_path=out_q,
            base_poses=out_b,
            task_error_norms=errors,
            solver_statuses=statuses,
            used_backend=backend_used,
        )

    def _world_jacobian(
        self,
        q: np.ndarray,
        base_pose: Sequence[float],
        joint_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        T_base_ee, J_base_arm = self.model.fk_and_geometric_jacobian_world(
            str(self.cfg.ee_frame),
            np.asarray(q, dtype=np.float64).reshape(-1),
            list(joint_names),
        )
        T_world_base = _base_pose_to_matrix(base_pose)
        T_world_ee = T_world_base @ T_base_ee
        R_world_base = T_world_base[:3, :3]
        p_world_from_base = R_world_base @ T_base_ee[:3, 3]

        n = len(joint_names)
        J = np.zeros((6, 3 + n), dtype=np.float64)
        J[:3, 0] = R_world_base[:, 0]
        J[:3, 1] = R_world_base[:, 1]
        J[:3, 2] = np.cross(np.array([0.0, 0.0, 1.0]), p_world_from_base)
        J[3:, 2] = [0.0, 0.0, 1.0]

        J[:3, 3:] = R_world_base @ J_base_arm[:3, :]
        J[3:, 3:] = R_world_base @ J_base_arm[3:, :]
        return J, T_world_ee

    def _target_twist(
        self,
        *,
        T_cur: np.ndarray,
        T_des: np.ndarray,
        T_des_next: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        p_ref = (T_des_next[:3, 3] - T_des[:3, 3]) / dt
        w_ref = so3_log_robust(T_des_next[:3, :3] @ T_des[:3, :3].T) / dt
        pos_err = T_des[:3, 3] - T_cur[:3, 3]
        rot_err = so3_log_robust(T_des[:3, :3] @ T_cur[:3, :3].T)
        return np.concatenate(
            [
                p_ref + float(self.cfg.kp_pos) * pos_err,
                w_ref + float(self.cfg.kp_rot) * rot_err,
            ]
        )

    def _velocity_bounds(
        self,
        *,
        q_cur: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = q_cur.shape[0]
        lower = np.empty(3 + n, dtype=np.float64)
        upper = np.empty(3 + n, dtype=np.float64)
        lower[:3] = [
            -abs(float(self.cfg.max_base_linear_mps)),
            -abs(float(self.cfg.max_base_linear_mps)),
            -abs(float(self.cfg.max_base_angular_rps)),
        ]
        upper[:3] = [
            abs(float(self.cfg.max_base_linear_mps)),
            abs(float(self.cfg.max_base_linear_mps)),
            abs(float(self.cfg.max_base_angular_rps)),
        ]
        vlim = abs(float(self.cfg.max_joint_velocity))
        lower[3:] = np.maximum(-vlim, (q_lower - q_cur) / dt)
        upper[3:] = np.minimum(vlim, (q_upper - q_cur) / dt)
        return lower, upper

    def _solve_velocity_qp(
        self,
        *,
        J: np.ndarray,
        target_twist: np.ndarray,
        base_ref: np.ndarray,
        qdot_ref: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, str, str]:
        nvar = lower.shape[0]
        rows = [
            math.sqrt(float(self.cfg.task_weight)) * J,
            math.sqrt(float(self.cfg.base_ref_weight)) * np.hstack(
                [np.eye(3), np.zeros((3, nvar - 3))]
            ),
            math.sqrt(float(self.cfg.joint_ref_weight)) * np.hstack(
                [np.zeros((nvar - 3, 3)), np.eye(nvar - 3)]
            ),
            math.sqrt(float(self.cfg.joint_reg_weight)) * np.hstack(
                [np.zeros((nvar - 3, 3)), np.eye(nvar - 3)]
            ),
        ]
        rhs = [
            math.sqrt(float(self.cfg.task_weight)) * target_twist,
            math.sqrt(float(self.cfg.base_ref_weight)) * base_ref,
            math.sqrt(float(self.cfg.joint_ref_weight)) * qdot_ref,
            np.zeros(nvar - 3, dtype=np.float64),
        ]
        A_cost = np.vstack(rows)
        b_cost = np.concatenate(rhs)
        P = 2.0 * (A_cost.T @ A_cost)
        q = -2.0 * (A_cost.T @ b_cost)
        P = 0.5 * (P + P.T) + 1.0e-9 * np.eye(nvar)

        backend = str(self.cfg.backend).strip().lower()
        if backend in ("auto", "osqp"):
            try:
                return self._solve_osqp(
                    P=P,
                    q=q,
                    lower=lower,
                    upper=upper,
                    J=J,
                    target_twist=target_twist,
                    hard_task_constraint=bool(self.cfg.hard_task_constraint),
                )
            except ModuleNotFoundError:
                if backend == "osqp":
                    raise
            except RuntimeError:
                if backend == "osqp" or not bool(self.cfg.hard_task_constraint):
                    raise

        return self._solve_scipy_slsqp(
            P=P,
            q=q,
            lower=lower,
            upper=upper,
            J=J,
            target_twist=target_twist,
            hard_task_constraint=bool(self.cfg.hard_task_constraint),
        )

    @staticmethod
    def _solve_osqp(
        *,
        P: np.ndarray,
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        J: np.ndarray,
        target_twist: np.ndarray,
        hard_task_constraint: bool,
    ) -> tuple[np.ndarray, str, str]:
        import osqp
        import scipy.sparse as sp

        A_parts = [sp.eye(lower.shape[0], format="csc")]
        l_parts = [lower]
        u_parts = [upper]
        if hard_task_constraint:
            A_parts.append(sp.csc_matrix(J))
            l_parts.append(target_twist)
            u_parts.append(target_twist)
        solver = osqp.OSQP()
        solver.setup(
            P=sp.csc_matrix(P),
            q=q,
            A=sp.vstack(A_parts, format="csc"),
            l=np.concatenate(l_parts),
            u=np.concatenate(u_parts),
            verbose=False,
            polish=False,
            warm_start=True,
        )
        res = solver.solve()
        status = str(res.info.status)
        if res.x is None or status.lower() not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"OSQP failed: {status}")
        return np.asarray(res.x, dtype=np.float64), status, "osqp"

    @staticmethod
    def _solve_scipy_slsqp(
        *,
        P: np.ndarray,
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        J: np.ndarray,
        target_twist: np.ndarray,
        hard_task_constraint: bool,
    ) -> tuple[np.ndarray, str, str]:
        from scipy.optimize import LinearConstraint, minimize

        def fun(x: np.ndarray) -> float:
            return float(0.5 * x @ P @ x + q @ x)

        def jac(x: np.ndarray) -> np.ndarray:
            return P @ x + q

        bounds = list(zip(lower.tolist(), upper.tolist()))
        constraints: list[Any] = []
        if hard_task_constraint:
            constraints.append(LinearConstraint(J, target_twist, target_twist))
        x0 = np.clip(np.zeros_like(lower), lower, upper)
        result = minimize(
            fun,
            x0,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1.0e-8, "maxiter": 100, "disp": False},
        )
        if not result.success and hard_task_constraint:
            result = minimize(
                fun,
                x0,
                jac=jac,
                bounds=bounds,
                method="SLSQP",
                options={"ftol": 1.0e-8, "maxiter": 100, "disp": False},
            )
        if result.x is None:
            raise RuntimeError(f"SLSQP failed: {result.message}")
        return np.asarray(result.x, dtype=np.float64), str(result.message), "scipy-slsqp"

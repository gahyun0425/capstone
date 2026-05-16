from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.constraint_projection.projection import ManifoldProjectorTorch
from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.types import PlanResult
from capstone_pkg.utils.config import JOINT_LIMIT, LEFT_JOINTS, RIGHT_JOINTS
from capstone_pkg.utils.joint_limit import load_joint_limits_torch

from .planner import plan_tbrrt_extcon_batch_conext


def _normalize_arm_name(arm: str) -> str:
    raw = str(arm).strip().lower()
    aliases = {
        "l": "left",
        "left": "left",
        "left_arm": "left",
        "left-arm": "left",
        "r": "right",
        "right": "right",
        "right_arm": "right",
        "right-arm": "right",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        raise ValueError("arm must be one of: left, right")
    return normalized


@dataclass
class FixedJointConstraint:
    fixed_indices: torch.Tensor
    fixed_values: torch.Tensor

    @torch.no_grad()
    def residual_and_jacobian_torch(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim == 1:
            q = q.view(1, -1)

        q = q.contiguous()
        n_batch, dof = q.shape
        idx = self.fixed_indices.to(device=q.device, dtype=torch.long)
        m = int(idx.numel())

        if m == 0:
            h = torch.zeros((n_batch, 0), device=q.device, dtype=q.dtype)
            j = torch.zeros((n_batch, 0, dof), device=q.device, dtype=q.dtype)
            return h, j

        ref = self.fixed_values.to(device=q.device, dtype=q.dtype).view(1, m)
        h = q.index_select(1, idx) - ref

        j = torch.zeros((n_batch, m, dof), device=q.device, dtype=q.dtype)
        rows = torch.arange(m, device=q.device, dtype=torch.long)
        j[:, rows, idx] = 1.0
        return h, j

    @torch.no_grad()
    def h(self, q: torch.Tensor) -> torch.Tensor:
        h, _ = self.residual_and_jacobian_torch(q)
        return h

    @torch.no_grad()
    def residual_torch(self, q: torch.Tensor) -> torch.Tensor:
        return self.h(q)

    @torch.no_grad()
    def residual_and_jacobian_if_available(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.residual_and_jacobian_torch(q)

    @torch.no_grad()
    def jacobian_torch(self, q: torch.Tensor) -> torch.Tensor:
        _h, j = self.residual_and_jacobian_torch(q)
        return j

    @torch.no_grad()
    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        _h, j = self.residual_and_jacobian_torch(q)
        return j

    @torch.no_grad()
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        return self.h(q)

    @torch.no_grad()
    def residual_norm(self, q: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.h(q), dim=-1)


@dataclass
class FixedJointEEZConstraint:
    fixed_indices: torch.Tensor
    fixed_values: torch.Tensor
    active_indices: torch.Tensor
    target_z: float
    ik_solver: object
    fd_eps: float = 1.0e-3

    @torch.no_grad()
    def _ee_z_from_active(self, q_active: torch.Tensor) -> torch.Tensor:
        if q_active.ndim == 1:
            q_active = q_active.view(1, -1)
        kin = self.ik_solver.solver.fk(q_active.contiguous())
        return kin.ee_position[:, 2:3].contiguous()

    @torch.no_grad()
    def _ee_z_from_cspace(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 1:
            q = q.view(1, -1)
        active_idx = self.active_indices.to(device=q.device, dtype=torch.long)
        q_active = q.contiguous().index_select(1, active_idx)
        return self._ee_z_from_active(q_active)

    @torch.no_grad()
    def residual_and_jacobian_torch(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim == 1:
            q = q.view(1, -1)

        q = q.contiguous()
        n_batch, dof = q.shape
        fixed_idx = self.fixed_indices.to(device=q.device, dtype=torch.long)
        active_idx = self.active_indices.to(device=q.device, dtype=torch.long)
        fixed_dim = int(fixed_idx.numel())

        if fixed_dim > 0:
            ref = self.fixed_values.to(device=q.device, dtype=q.dtype).view(1, fixed_dim)
            fixed_h = q.index_select(1, fixed_idx) - ref
        else:
            fixed_h = torch.zeros((n_batch, 0), device=q.device, dtype=q.dtype)

        ee_z = self._ee_z_from_cspace(q)
        z_h = ee_z - float(self.target_z)
        h = torch.cat([fixed_h, z_h], dim=1)

        j = torch.zeros((n_batch, fixed_dim + 1, dof), device=q.device, dtype=q.dtype)
        if fixed_dim > 0:
            rows = torch.arange(fixed_dim, device=q.device, dtype=torch.long)
            j[:, rows, fixed_idx] = 1.0

        active_dim = int(active_idx.numel())
        if active_dim > 0:
            qa = q.index_select(1, active_idx)
            diag = torch.arange(active_dim, device=q.device, dtype=torch.long)
            qa_pert = qa.unsqueeze(1).expand(n_batch, active_dim, active_dim).clone()
            qa_pert[:, diag, diag] += float(self.fd_eps)
            ee_z_pert = self._ee_z_from_active(qa_pert.reshape(n_batch * active_dim, active_dim))
            ee_z_pert = ee_z_pert.view(n_batch, active_dim)
            dz_dqa = (ee_z_pert - ee_z.view(n_batch, 1)) / float(self.fd_eps)
            j[:, fixed_dim, active_idx] = dz_dqa
        return h, j

    @torch.no_grad()
    def h(self, q: torch.Tensor) -> torch.Tensor:
        h, _ = self.residual_and_jacobian_torch(q)
        return h

    @torch.no_grad()
    def residual_torch(self, q: torch.Tensor) -> torch.Tensor:
        return self.h(q)

    @torch.no_grad()
    def residual_and_jacobian_if_available(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.residual_and_jacobian_torch(q)

    @torch.no_grad()
    def jacobian_torch(self, q: torch.Tensor) -> torch.Tensor:
        _h, j = self.residual_and_jacobian_torch(q)
        return j

    @torch.no_grad()
    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        _h, j = self.residual_and_jacobian_torch(q)
        return j

    @torch.no_grad()
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        return self.h(q)

    @torch.no_grad()
    def residual_norm(self, q: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.h(q), dim=-1)


def plan_single_arm_tbrrt_batch_conext(
    *,
    robot_yml: str,
    arm: str,
    q_start: Sequence[float],
    q_goals: Sequence[Sequence[float]],
    world_yml: str | None,
    cpu: bool,
    cfg: TBRRTConfig,
    joint_limit_yml: str = JOINT_LIMIT,
    block_k: int = 32,
    proj_iters: int = 60,
    proj_tol: float = 1.0e-3,
    proj_damping: float = 0.0,
    proj_step: float = 1.0,
    proj_fd_eps: float = 1.0e-3,
) -> PlanResult:
    normalized_arm = _normalize_arm_name(arm)
    checker = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
    device = checker.tensor_args.device
    dtype = torch.float32

    cspace_joint_names = list(checker.cspace_names)
    dof = len(cspace_joint_names)
    q_start_list = [float(v) for v in q_start]
    if len(q_start_list) != dof:
        raise ValueError(f"q_start dim mismatch: got={len(q_start_list)} expected={dof}")
    if not q_goals:
        raise ValueError("q_goals must not be empty")

    active_joint_names = list(LEFT_JOINTS if normalized_arm == "left" else RIGHT_JOINTS)
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    active_indices = {name_to_idx[name] for name in active_joint_names if name in name_to_idx}
    if not active_indices:
        raise RuntimeError(
            f"{normalized_arm} arm joints were not found in cspace joint names: {cspace_joint_names}"
        )
    fixed_indices_list = [idx for idx in range(dof) if idx not in active_indices]
    fixed_indices = torch.tensor(fixed_indices_list, device=device, dtype=torch.long)

    q_goal_rows: list[list[float]] = []
    for q_goal in q_goals:
        row = [float(v) for v in q_goal]
        if len(row) != dof:
            raise ValueError(f"q_goal dim mismatch: got={len(row)} expected={dof}")
        for idx in fixed_indices_list:
            row[idx] = q_start_list[idx]
        q_goal_rows.append(row)

    q_ref = torch.tensor(q_start_list, device=device, dtype=dtype)
    fixed_values = q_ref.index_select(0, fixed_indices)
    constraint = FixedJointConstraint(
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
    )
    joint_limits = load_joint_limits_torch(str(joint_limit_yml), device=device, dtype=dtype)
    projector = ManifoldProjectorTorch(
        constraint=constraint,
        limits=joint_limits,
        max_iters=int(proj_iters),
        tol=float(proj_tol),
        fd_eps=float(proj_fd_eps),
        damping=float(proj_damping),
        step_size=float(proj_step),
    )

    return plan_tbrrt_extcon_batch_conext(
        q_start=q_start_list,
        q_goals=q_goal_rows,
        cfg=cfg,
        checker=checker,
        projector=projector,
        joint_limits=joint_limits,
        device=device,
        block_K=int(block_k),
    )


def plan_single_arm_tbrrt_batch_conext_fixed_ee_z(
    *,
    robot_yml: str,
    arm: str,
    q_start: Sequence[float],
    q_goals: Sequence[Sequence[float]],
    world_yml: str | None,
    cpu: bool,
    cfg: TBRRTConfig,
    joint_limit_yml: str = JOINT_LIMIT,
    block_k: int = 32,
    proj_iters: int = 60,
    proj_tol: float = 1.0e-3,
    proj_damping: float = 0.0,
    proj_step: float = 1.0,
    proj_fd_eps: float = 1.0e-3,
    target_ee_z: float | None = None,
) -> PlanResult:
    normalized_arm = _normalize_arm_name(arm)
    checker = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
    device = checker.tensor_args.device
    dtype = torch.float32

    cspace_joint_names = list(checker.cspace_names)
    dof = len(cspace_joint_names)
    q_start_list = [float(v) for v in q_start]
    if len(q_start_list) != dof:
        raise ValueError(f"q_start dim mismatch: got={len(q_start_list)} expected={dof}")
    if not q_goals:
        raise ValueError("q_goals must not be empty")

    active_joint_names = list(LEFT_JOINTS if normalized_arm == "left" else RIGHT_JOINTS)
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    active_indices = {name_to_idx[name] for name in active_joint_names if name in name_to_idx}
    if not active_indices:
        raise RuntimeError(
            f"{normalized_arm} arm joints were not found in cspace joint names: {cspace_joint_names}"
        )
    fixed_indices_list = [idx for idx in range(dof) if idx not in active_indices]
    fixed_indices = torch.tensor(fixed_indices_list, device=device, dtype=torch.long)

    q_goal_rows: list[list[float]] = []
    for q_goal in q_goals:
        row = [float(v) for v in q_goal]
        if len(row) != dof:
            raise ValueError(f"q_goal dim mismatch: got={len(row)} expected={dof}")
        for idx in fixed_indices_list:
            row[idx] = q_start_list[idx]
        q_goal_rows.append(row)

    ik = get_single_arm_ik(robot_yml, arm=normalized_arm, cpu=cpu, world_yml=world_yml)
    fk_active_indices_list = [name_to_idx[name] for name in ik.active_joint_names if name in name_to_idx]
    if not fk_active_indices_list:
        raise RuntimeError(
            f"{normalized_arm} arm FK joints were not found in cspace joint names: {cspace_joint_names}"
        )
    fk_active_indices = torch.tensor(fk_active_indices_list, device=device, dtype=torch.long)

    q_start_t = torch.tensor(q_start_list, device=device, dtype=dtype).view(1, -1)
    q_start_active = q_start_t.index_select(1, fk_active_indices)
    ee_z_start = float(ik.solver.fk(q_start_active).ee_position[0, 2].item())

    q_ref = torch.tensor(q_start_list, device=device, dtype=dtype)
    fixed_values = q_ref.index_select(0, fixed_indices)
    constraint = FixedJointEEZConstraint(
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        active_indices=fk_active_indices,
        target_z=ee_z_start if target_ee_z is None else float(target_ee_z),
        ik_solver=ik,
        fd_eps=float(proj_fd_eps),
    )
    joint_limits = load_joint_limits_torch(str(joint_limit_yml), device=device, dtype=dtype)
    projector = ManifoldProjectorTorch(
        constraint=constraint,
        limits=joint_limits,
        max_iters=int(proj_iters),
        tol=float(proj_tol),
        fd_eps=float(proj_fd_eps),
        damping=float(proj_damping),
        step_size=float(proj_step),
    )

    return plan_tbrrt_extcon_batch_conext(
        q_start=q_start_list,
        q_goals=q_goal_rows,
        cfg=cfg,
        checker=checker,
        projector=projector,
        joint_limits=joint_limits,
        device=device,
        block_K=int(block_k),
    )

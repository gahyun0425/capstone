# capstone_pkg/constraint_projection/projection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from capstone_pkg.utils.joint_limit import JointLimitsTorch
from capstone_pkg.constraint_projection.constraint import RigidConstraint

# -----------------------------
# Results
# -----------------------------
@dataclass
class ProjectionResult:
    q_proj: torch.Tensor
    success: bool
    iters: int
    final_residual: float


@dataclass
class BatchProjectionResult:
    q_proj: torch.Tensor
    success_mask: torch.Tensor
    iters: torch.Tensor
    final_residual: torch.Tensor


# ============================================================
# Non-batched (non-torch-linear-algebra) projector
#   - Intended for "basic / non-batch TB-RRT"
#   - Implements the paper's Newton–Raphson style projection:
#       q <- q - J(q)^† f(q),
#     with J^† = J^T (J J^T)^(-1)
# ============================================================
class ManifoldProjector:
    """
    Paper-style ProjectionNewtonRaphson (single-point).

    NOTE:
      - Constraint evaluation h(q) is performed via the provided RigidConstraint (torch).
      - Jacobian + linear algebra are computed in numpy (CPU).

    Update:
      dq = - J^T (J J^T)^(-1) h

    where:
      q: (D,)
      h: (m,)
      J: (m, D)
    """

    def __init__(
        self,
        *,
        constraint: RigidConstraint,
        limits: Optional[JointLimitsTorch] = None,
        max_iters: int = 25,
        tol: float = 1e-4,
        fd_eps: float = 1e-4,
        damping: float = 0.0,
        step_size: float = 1.0,
    ):
        self.c = constraint
        self.limits = limits
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)
        if float(damping) < 0.0:
            raise ValueError("damping must be >= 0")
        if float(step_size) <= 0.0:
            raise ValueError("step_size must be > 0")
        self.damping = float(damping)
        self.step_size = float(step_size)

    @torch.no_grad()
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        return self.c.h(q)

    @torch.no_grad()
    def residual_and_jacobian_if_available(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        analytic_pair = self.c.residual_and_jacobian_torch(q)
        if analytic_pair is not None:
            h, J = analytic_pair
            return h, J.contiguous()
        return self.residual(q), None

    def _h_torch(self, q_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
        q = torch.as_tensor(q_np, device=device, dtype=dtype).view(1, -1)
        h = self.residual(q)  # (1,m)
        return h.detach().cpu().numpy().reshape(-1)

    def _jacobian_fd_np(
        self, q_np: np.ndarray, h0_np: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> np.ndarray:
        """
        Effective Jacobian on CPU numpy.

        Prefer the analytic URDF backend when the constraint provides one;
        otherwise fall back to finite differences.

        q_np : (D,)
        h0_np: (m,)
        return: J_np (m,D)
        """
        q = torch.as_tensor(q_np, device=device, dtype=dtype).view(1, -1)
        J_analytic = self.c.jacobian(q)
        if J_analytic is not None:
            return J_analytic.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)

        D = int(q_np.shape[0])
        m = int(h0_np.shape[0])
        eps = float(self.fd_eps)
        J = np.empty((m, D), dtype=np.float64)
        for j in range(D):
            qp = q_np.copy()
            qp[j] += eps
            hp = self._h_torch(qp, device, dtype)
            J[:, j] = (hp - h0_np) / eps
        return J

    # tangent_space.py compatibility wrapper
    def _jacobian_fd(self, q: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Compatibility helper for tangent_space.py.

        q  : (1,D) torch
        h0 : (1,m) torch
        returns: (1,m,D) torch
        """
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError("q must be (1,D)")
        if h0.ndim != 2 or h0.shape[0] != 1:
            raise ValueError("h0 must be (1,m)")

        device, dtype = q.device, q.dtype

        q_np = q.detach().cpu().numpy().reshape(-1).astype(np.float64)
        h0_np = h0.detach().cpu().numpy().reshape(-1).astype(np.float64)

        J_np = self._jacobian_fd_np(q_np, h0_np, device, dtype)  # (m,D) numpy
        J = torch.as_tensor(J_np, device=device, dtype=dtype).unsqueeze(0)  # (1,m,D)
        return J

    def project(self, q_in: torch.Tensor) -> ProjectionResult:
        if q_in.ndim != 1:
            raise ValueError("q_in must be (D,)")

        device = q_in.device
        dtype = q_in.dtype

        # NR iterations on CPU numpy (linear algebra), constraint eval via torch
        q_np = q_in.detach().cpu().numpy().astype(np.float64).copy()

        it_done = 0
        success = False

        for k in range(self.max_iters):
            h0 = self._h_torch(q_np, device, dtype)  # (m,)
            r0 = float(np.linalg.norm(h0))
            it_done = k

            if r0 <= self.tol:
                success = True
                break

            J = self._jacobian_fd_np(q_np, h0, device, dtype)  # (m,D)
            A = J @ J.T  # (m,m)
            if self.damping > 0.0:
                A = A + (self.damping * np.eye(A.shape[0], dtype=A.dtype))

            # Solve A x = h0, dq = -J^T x
            try:
                x = np.linalg.solve(A, h0.reshape(-1, 1))  # (m,1)
            except np.linalg.LinAlgError:
                x = np.linalg.pinv(A) @ h0.reshape(-1, 1)

            dq = -(J.T @ x).reshape(-1)  # (D,)
            q_np = q_np + (self.step_size * dq)

            if self.limits is not None:
                q_t = torch.as_tensor(q_np, device=device, dtype=dtype)
                q_t = self.limits.clamp(q_t)
                q_np = q_t.detach().cpu().numpy().astype(np.float64)

        q_out = torch.as_tensor(q_np, device=device, dtype=dtype)
        if self.limits is not None:
            q_out = self.limits.clamp(q_out)

        r_final = float(self.c.residual_norm(q_out.view(1, -1)).item())
        success = bool(r_final <= self.tol)

        return ProjectionResult(
            q_proj=q_out,
            success=success,
            iters=int(it_done),
            final_residual=float(r_final),
        )


# ============================================================
# Batched (torch) projector
#   - Intended for "batch TB-RRT"
#   - Implements the same paper-style NR update, but in batch:
#       dq = - J^T (J J^T)^(-1) h
# ============================================================
class ManifoldProjectorTorch:
    """
    Paper-style ProjectionNewtonRaphson (batched torch).

    Inputs:
      qs: (N,D) torch tensor

    Update (per point):
      dq = - J^T (J J^T)^(-1) h
    """

    def __init__(
        self,
        *,
        constraint: RigidConstraint,
        limits: Optional[JointLimitsTorch] = None,
        max_iters: int = 25,
        tol: float = 1e-4,
        fd_eps: float = 1e-4,
        damping: float = 0.0,
        step_size: float = 1.0,
    ):
        self.c = constraint
        self.limits = limits
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)
        if float(damping) < 0.0:
            raise ValueError("damping must be >= 0")
        if float(step_size) <= 0.0:
            raise ValueError("step_size must be > 0")
        self.damping = float(damping)
        self.step_size = float(step_size)
        self._fixed_tensor_cache: Dict[Tuple[str, Optional[int], torch.dtype, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _fixed_cache_key(
        *,
        device: torch.device,
        dtype: torch.dtype,
        q_dim: int,
        residual_dim: int,
    ) -> Tuple[str, Optional[int], torch.dtype, int, int]:
        return (device.type, device.index, dtype, int(q_dim), int(residual_dim))

    def _get_fixed_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        q_dim: int,
        residual_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._fixed_cache_key(
            device=device,
            dtype=dtype,
            q_dim=q_dim,
            residual_dim=residual_dim,
        )
        cached = self._fixed_tensor_cache.get(key)
        if cached is None:
            cached = (
                torch.arange(int(q_dim), device=device, dtype=torch.long),
                torch.eye(int(residual_dim), device=device, dtype=dtype),
            )
            self._fixed_tensor_cache[key] = cached
        return cached

    @torch.no_grad()
    def prepare_fixed_tensors(
        self,
        *,
        q_dim: int,
        residual_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._get_fixed_tensors(
            device=device,
            dtype=dtype,
            q_dim=q_dim,
            residual_dim=residual_dim,
        )

    @torch.no_grad()
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        h = self.c.residual_torch(q)
        if h is not None:
            return h
        return self.c.h(q)

    @torch.no_grad()
    def residual_and_jacobian_if_available(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        analytic_pair = self.c.residual_and_jacobian_torch(q)
        if analytic_pair is not None:
            h, J = analytic_pair
            return h, J.contiguous()
        return self.residual(q), None

    @torch.no_grad()
    def _jacobian_fd(self, q: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Batched effective Jacobian.

        Prefer the analytic URDF backend when the constraint provides one;
        otherwise fall back to finite differences.

        q : (N, D)
        h0: (N, m)
        return J: (N, m, D)
        """
        if q.ndim != 2:
            raise ValueError("q must be (N,D)")
        if h0.ndim != 2:
            raise ValueError("h0 must be (N,m)")

        J_analytic = self.c.jacobian_torch(q)
        if J_analytic is None:
            J_analytic = self.c.jacobian(q)
        if J_analytic is not None:
            return J_analytic.contiguous()

        N, D = q.shape
        m = h0.shape[-1]
        eps = float(self.fd_eps)
        diag, _eye = self._get_fixed_tensors(
            device=q.device,
            dtype=q.dtype,
            q_dim=D,
            residual_dim=m,
        )

        # q_pert: (N, D, D) where for each j we perturb joint j
        q_pert = q.unsqueeze(1).expand(N, D, D).clone()
        q_pert[:, diag, diag] += eps

        q_pert_flat = q_pert.reshape(N * D, D)
        hp_flat = self.residual(q_pert_flat)  # (N*D, m)
        hp = hp_flat.reshape(N, D, m)    # (N, D, m)

        # J[n, :, j] = (h(q+eps e_j) - h0)/eps
        J = ((hp - h0.unsqueeze(1)) / eps).permute(0, 2, 1).contiguous()  # (N,m,D)
        return J

    @torch.no_grad()
    def project_batch(
        self,
        qs: torch.Tensor,
        h0_init: Optional[torch.Tensor] = None,
        J0_init: Optional[torch.Tensor] = None,
    ) -> BatchProjectionResult:
        if qs.ndim != 2:
            raise ValueError("qs must be (N,D)")

        q = qs.clone()
        device = q.device
        dtype = q.dtype
        N, _ = q.shape

        if h0_init is not None:
            if h0_init.ndim != 2 or h0_init.shape[0] != N:
                raise ValueError(f"h0_init must be (N,m), got {tuple(h0_init.shape)}")
            h0_init = h0_init.to(device=device, dtype=dtype)
        if J0_init is not None:
            if h0_init is None:
                raise ValueError("J0_init requires h0_init")
            if J0_init.ndim != 3 or J0_init.shape[0] != N or J0_init.shape[2] != qs.shape[1]:
                raise ValueError(f"J0_init must be (N,m,D), got {tuple(J0_init.shape)}")
            if J0_init.shape[1] != h0_init.shape[1]:
                raise ValueError(
                    f"J0_init constraint dim {J0_init.shape[1]} != h0_init constraint dim {h0_init.shape[1]}"
                )
            J0_init = J0_init.to(device=device, dtype=dtype)

        success = torch.zeros((N,), device=device, dtype=torch.bool)
        iters = torch.zeros((N,), device=device, dtype=torch.int32)
        final_res = torch.full((N,), float("inf"), device=device, dtype=dtype)
        active_idx = torch.arange(N, device=device, dtype=torch.long)

        for k in range(self.max_iters):
            if active_idx.numel() == 0:
                break

            qa = q.index_select(0, active_idx)
            analytic_pair = None
            J0 = None
            # Reuse caller-provided residual for the first iteration when available.
            if k == 0 and h0_init is not None:
                h0 = h0_init.index_select(0, active_idx)
                if J0_init is not None:
                    J0 = J0_init.index_select(0, active_idx).contiguous()
            else:
                analytic_pair = self.c.residual_and_jacobian_torch(qa)
                if analytic_pair is not None:
                    h0, _J_full = analytic_pair
                else:
                    h0 = self.residual(qa)  # (Na,m)
            r0 = torch.linalg.norm(h0, dim=-1)

            final_res.index_copy_(0, active_idx, r0)
            newly_ok = (r0 <= self.tol)

            idx_new_ok = active_idx[newly_ok]
            if idx_new_ok.numel() > 0:
                iters[idx_new_ok] = k
                success[idx_new_ok] = True

            still_active_local = ~newly_ok
            active_idx = active_idx[still_active_local]
            if active_idx.numel() == 0:
                break

            # active2 is a subset of active before q update; reuse h0 computed above.
            qa2 = qa[still_active_local]
            h0_2 = h0[still_active_local]
            if analytic_pair is not None:
                _h_full, J_full = analytic_pair
                J = J_full[still_active_local].contiguous()
            elif J0 is not None:
                J = J0[still_active_local].contiguous()
            else:
                J = self._jacobian_fd(qa2, h0_2)  # (Na2,m,D)

            A = J @ J.transpose(1, 2)      # (Na2,m,m)
            if self.damping > 0.0:
                _diag, eye2d = self._get_fixed_tensors(
                    device=A.device,
                    dtype=A.dtype,
                    q_dim=qs.shape[1],
                    residual_dim=A.shape[-1],
                )
                eye = eye2d.unsqueeze(0)
                A = A + (self.damping * eye)
            b = h0_2.unsqueeze(-1)         # (Na2,m,1)

            # Solve A x = b; dq = -J^T x
            x, info = torch.linalg.solve_ex(A, b)
            if (info != 0).any():
                fail = info != 0
                x2 = torch.linalg.pinv(A[fail]) @ b[fail]
                x = x.clone()
                x[fail] = x2

            dq = -(J.transpose(1, 2) @ x).squeeze(-1)  # (Na2,D)
            q_next = qa2 + (self.step_size * dq)

            if self.limits is not None:
                q_next = self.limits.clamp(q_next)

            q.index_copy_(0, active_idx, q_next)

        r_final = final_res.clone()
        unresolved_idx = active_idx
        if unresolved_idx.numel() > 0:
            q_unresolved = q.index_select(0, unresolved_idx)
            r_unresolved = torch.linalg.norm(self.residual(q_unresolved), dim=-1)
            r_final.index_copy_(0, unresolved_idx, r_unresolved)
            success = success.clone()
            success[unresolved_idx] = (r_unresolved <= self.tol)

        return BatchProjectionResult(
            q_proj=q,
            success_mask=success,
            iters=iters,
            final_residual=r_final,
        )

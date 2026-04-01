from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector
from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.tangent_space import TangentSpace


@dataclass
class ConnectionTestResult:
    ok: bool
    reason: str = ""
    tangent_err_A: float = 0.0
    tangent_err_B: float = 0.0
    residual_max: float = 0.0
    first_bad_alpha: Optional[float] = None


@torch.no_grad()
def connection_test_paper(
    *,
    qA: torch.Tensor,                # (D,)
    tsA: TangentSpace,
    qB: torch.Tensor,                # (D,)
    tsB: TangentSpace,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
) -> ConnectionTestResult:
    """Paper-style 'Connection test' (Tangent Bundle RRT, Sec. 3.8).

    Accept a connection between two nodes only if:
      1) the connecting direction is approximately tangent at BOTH ends:
            || J(q_end) * v_hat || <= conn_tangent_tol
      2) the connecting segment is collision-free (edge collision check)
      3) sampling points along the segment satisfy:
            - point collision-free
            - equality residual ||h(q)|| <= E_conn
    """

    # ---- 0) direction ----
    v = (qB - qA).view(-1)
    v_norm = float(torch.linalg.norm(v).item())
    if v_norm < 1e-12:
        return ConnectionTestResult(ok=True, reason="degenerate")

    v_hat = v / v_norm  # (D,)

    # ---- 1) tangent compatibility at both ends ----
    tol = float(getattr(cfg, "conn_tangent_tol", 0.05))
    # J: (m, D) -> (m,)
    errA = float(torch.linalg.norm(tsA.J.to(v_hat.device) @ v_hat).item())
    if errA > tol:
        return ConnectionTestResult(ok=False, reason="tangent_incompatible_A", tangent_err_A=errA, tangent_err_B=0.0)

    errB = float(torch.linalg.norm(tsB.J.to(v_hat.device) @ v_hat).item())
    if errB > tol:
        return ConnectionTestResult(ok=False, reason="tangent_incompatible_B", tangent_err_A=errA, tangent_err_B=errB)

    # ---- 2) edge collision ----
    e = edge_checker.check_edge(qA, qB)
    if bool(e.edge_in_collision):
        return ConnectionTestResult(
            ok=False,
            reason="edge_in_collision",
            tangent_err_A=errA,
            tangent_err_B=errB,
            first_bad_alpha=e.first_collision_alpha,
        )

    # ---- 3) sample points on the segment ----
    num = int(getattr(cfg, "conn_num_samples", 15))
    num = max(3, num)
    alphas = torch.linspace(0.0, 1.0, steps=num, device=qA.device, dtype=qA.dtype)

    # collision-free mask
    qs = (1.0 - alphas).unsqueeze(1) * qA.view(1, -1) + alphas.unsqueeze(1) * qB.view(1, -1)  # (num, D)
    free_mask = checker.get_collision_free_mask(qs)
    if not bool(torch.all(free_mask).item()):
        bad = int(torch.where(~free_mask)[0][0].item())
        return ConnectionTestResult(
            ok=False,
            reason="point_in_collision",
            tangent_err_A=errA,
            tangent_err_B=errB,
            first_bad_alpha=float(alphas[bad].item()),
        )

    # equality residual bound
    # projector.c.h expects (B,D) and returns (B,m)
    h = projector.residual(qs)  # (num, m)
    res = torch.linalg.norm(h, dim=1)  # (num,)
    res_max = float(torch.max(res).item())
    if res_max > float(cfg.E_conn):
        bad = int(torch.argmax(res).item())
        return ConnectionTestResult(
            ok=False,
            reason="residual_too_large",
            tangent_err_A=errA,
            tangent_err_B=errB,
            residual_max=res_max,
            first_bad_alpha=float(alphas[bad].item()),
        )

    return ConnectionTestResult(ok=True, reason="ok", tangent_err_A=errA, tangent_err_B=errB, residual_max=res_max)

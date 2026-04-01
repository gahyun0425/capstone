from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector

from .config import TBRRTConfig
from .extend_extcon import ExtendStatus, extend_extcon_once
from .tree import Tree
from .ts_bank import TSBank


@dataclass
class ConnectResult:
    status: ExtendStatus
    last_idx: Optional[int]
    created_ts: int


@torch.no_grad()
def connect_extcon(
    *,
    tree: Tree,
    bank: TSBank,
    q_target: torch.Tensor,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
) -> ConnectResult:
    """RRT-Connect (paper-style):

    1) Find the nearest node ONCE (q_near) in the current tree to q_target.
    2) Keep extending from the LAST added node along that branch toward q_target
       until TRAPPED or REACHED (or max steps).

    This avoids recomputing NN at every step (which can cause the "connect" to
    jump between different branches), and matches the intended TB-RRT ExtCon
    connect behavior.
    """

    last_idx: Optional[int] = None
    created_ts = 0
    status = ExtendStatus.ADVANCED

    # Start from the nearest node once, then advance along the same branch.
    cur_idx, _ = tree.nearest(q_target)

    for _ in range(int(cfg.connect_max_steps)):
        out = extend_extcon_once(
            tree=tree,
            bank=bank,
            q_target=q_target,
            cfg=cfg,
            checker=checker,
            edge_checker=edge_checker,
            projector=projector,
            iter_idx=iter_idx,
            nn_idx_override=cur_idx,
        )
        if out.new_ts_id is not None:
            created_ts += 1
        if out.status == ExtendStatus.TRAPPED:
            status = ExtendStatus.TRAPPED
            break

        last_idx = out.new_idx
        # Continue from the node we just added (same branch).
        cur_idx = int(last_idx)
        status = out.status
        if out.status == ExtendStatus.REACHED:
            break

    return ConnectResult(status=status, last_idx=last_idx, created_ts=created_ts)

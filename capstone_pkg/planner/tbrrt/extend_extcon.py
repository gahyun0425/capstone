from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector

from .config import TBRRTConfig
from .tree import Tree
from .ts_bank import TSBank
from .tangent_space import build_tangent_space_fd, project_vector_to_tangent
from .stats import get_stats


class ExtendStatus(str, Enum):
    TRAPPED = "TRAPPED"
    ADVANCED = "ADVANCED"
    REACHED = "REACHED"


@dataclass
class ExtendResult:
    status: ExtendStatus
    new_idx: Optional[int]
    new_ts_id: Optional[int]
    dist_to_target: float


@torch.no_grad()
def _point_is_free(checker: SelfCollisionChecker, q: torch.Tensor, *, margin: float = 0.0) -> bool:
    m = checker.get_collision_free_mask(q.view(1, -1), margin=float(margin))
    return bool(m.item())


@torch.no_grad()
def extend_extcon_once(
    *,
    tree: Tree,
    bank: TSBank,
    q_target: torch.Tensor,  # (D,)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    nn_idx_override: Optional[int] = None,
) -> ExtendResult:
    """One RRT-ExtCon extend step (paper Algorithm 3 style).

    - Find nearest node.
    - Take ONE step in its associated tangent space toward q_target.
    - Collision check.
    - If residual exceeds EM: project (to create new tangent space) and replace q_new.
    - Add node.

    Projection is only done in the EM-triggered branch.
    """

    # NOTE (RRT-Connect / TB-RRT ExtCon):
    # - For a normal EXTEND call, we compute nn_idx from the full tree.
    # - For a CONNECT call, paper-style behavior is to start from the NN once,
    #   then keep extending from the last added node along the same branch.
    #   In that case, the caller passes nn_idx_override.
    if nn_idx_override is None:
        nn_idx, _ = tree.nearest(q_target)
    else:
        nn_idx = int(nn_idx_override)
    q_near = tree.get_node(nn_idx)
    stats = get_stats()
    stats.inc("extend_calls", 1)
    if nn_idx_override is None:
        stats.inc("extend_mode_extend", 1)
    else:
        stats.inc("extend_mode_connect", 1)
    ts = bank.get(int(tree.ts_id[nn_idx].item()))
    ts_used_id = int(ts.ts_id)

    # direction in tangent space
    v = q_target - q_near
    v_tan = project_vector_to_tangent(ts, v)
    n = torch.linalg.norm(v_tan).clamp_min(1e-12)
    dir_tan = v_tan / n

    q_new = (q_near + float(cfg.step_size) * dir_tan).contiguous()

    # basic reached check uses config-space distance
    dist_to_target = float(torch.linalg.norm(q_target - q_new).item())

    # point collision
    if not _point_is_free(checker, q_new):
        stats.inc("trapped_point_collision", 1)
        return ExtendResult(ExtendStatus.TRAPPED, None, None, dist_to_target)

    # edge collision (q_near -> q_new)
    e = edge_checker.check_edge(q_near, q_new, steps=None, return_first_hit=False)
    if e.edge_in_collision:
        stats.inc("trapped_edge_collision", 1)
        return ExtendResult(ExtendStatus.TRAPPED, None, None, dist_to_target)

    # tangent-space approximation error
    res = float(projector.c.residual_norm(q_new.view(1, -1)).item())

    new_ts_id: Optional[int] = None
    projected = False
    if res > float(cfg.EM):
        projected = True
        stats.inc("projection_triggered", 1)
        # project to manifold and create a NEW tangent space at projected point
        pr = projector.project(q_new)
        if not pr.success:
            stats.inc("trapped_projection_fail", 1)
            return ExtendResult(ExtendStatus.TRAPPED, None, None, dist_to_target)
        q_proj = pr.q_proj.view(-1).contiguous()

        # collision re-check at projected point
        if not _point_is_free(checker, q_proj):
            stats.inc("trapped_point_collision_after_proj", 1)
            return ExtendResult(ExtendStatus.TRAPPED, None, None, dist_to_target)
        e2 = edge_checker.check_edge(q_near, q_proj, steps=None, return_first_hit=False)
        if e2.edge_in_collision:
            stats.inc("trapped_edge_collision_after_proj", 1)
            return ExtendResult(ExtendStatus.TRAPPED, None, None, dist_to_target)

        # new tangent space
        new_ts_id = len(bank)
        ts_new = build_tangent_space_fd(
            q_root=q_proj,
            projector=projector,
            svd_tol=float(cfg.svd_tol),
            ts_id=new_ts_id,
            created_iter=iter_idx,
        )
        bank.add(ts_new)
        q_new = q_proj

        # update distance after projection
        dist_to_target = float(torch.linalg.norm(q_target - q_new).item())

    # add node
    use_ts_id = int(new_ts_id) if new_ts_id is not None else int(ts.ts_id)
    new_idx = tree.add_node(q_new, parent=nn_idx, ts_id=use_ts_id, is_proj_root=(new_ts_id is not None))
    bank.increment_count(use_ts_id, inc=1)

    # ---- Dynamic domain sizing (paper Sec. 3.6 / Algorithm 9) ----
    # Adjust the *used* tangent space domain (the TS at q_near) based on whether
    # the new node required projection and how far it is from the TS root.
    if bool(getattr(cfg, "dynamic_domain_enable", False)):
        try:
            before_dom = float(bank.get_domain(ts_used_id))
            dom = before_dom
            dist = float(torch.linalg.norm(q_new - ts.root).item())
            expanded = False
            shrunk = False
            if (not projected) and (dist > float(cfg.ts_domain_expand_frac) * dom):
                dom = dom * float(cfg.ts_domain_expand_ratio)
                expanded = True
            elif projected and (dist < float(cfg.ts_domain_shrink_frac) * dom):
                dom = dom * float(cfg.ts_domain_shrink_ratio)
                shrunk = True

            dom = float(max(float(cfg.ts_domain_min), min(float(cfg.ts_domain_max), dom)))
            bank.set_domain(ts_used_id, dom)

            # stats
            if expanded:
                stats.inc("domain_expand", 1)
            if shrunk:
                stats.inc("domain_shrink", 1)
            if dom <= float(cfg.ts_domain_min) + 1e-12 and before_dom > dom:
                stats.inc("domain_hit_min", 1)
            if dom >= float(cfg.ts_domain_max) - 1e-12 and before_dom < dom:
                stats.inc("domain_hit_max", 1)
        except Exception:
            stats.inc("domain_update_error", 1)

    status = ExtendStatus.REACHED if dist_to_target <= float(cfg.goal_threshold) else ExtendStatus.ADVANCED
    return ExtendResult(status, new_idx, new_ts_id, dist_to_target)

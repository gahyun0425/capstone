from __future__ import annotations

from typing import List

import torch

from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector

from .tree import Tree


@torch.no_grad()
def extract_path(treeA: Tree, treeB: Tree, idxA: int, idxB: int) -> torch.Tensor:
    """Extract a full path (L,D) from start(root of A) to goal(root of B).

    idxA is in treeA, idxB is in treeB. The connection is assumed between those nodes.
    """
    pA = treeA.backtrack_path(idxA)              # (LA,D) start->...
    pB = treeB.backtrack_path(idxB)              # (LB,D) goal->... but in B-root frame (root is goal)

    # treeB root is goal, backtrack_path returns root->idxB. We need idxB->root then reverse.
    pB_rev = torch.flip(pB, dims=[0])            # idxB->root(goal)

    # combine: start->...idxA, then connection to idxB, then ...->goal
    # include idxB point (first of pB_rev) explicitly.
    full = torch.cat([pA, pB_rev], dim=0)
    return full


@torch.no_grad()
def lazy_project_path(
    path: torch.Tensor,
    *,
    projector: ManifoldProjector,
    edge_checker: EdgeCollisionChecker | None = None,
) -> torch.Tensor:
    """Project the final extracted path onto the manifold.

    This implements the paper's 'Lazy Projection' concept: apply projection only on
    the final path (not during tree expansion except for EM-triggered TS creation).

    If edge_checker is provided, we additionally verify that each edge remains collision-free.
    """

    if path.ndim != 2:
        raise ValueError("path must be (L,D)")

    # non-batch lazy projection: project each node independently (paper-style)
    q_out = []
    bad = []
    for i in range(int(path.shape[0])):
        pr_i = projector.project(path[i])
        if not pr_i.success:
            bad.append(i)
            q_out.append(path[i])
        else:
            q_out.append(pr_i.q_proj)
    if bad:
        raise RuntimeError(f"Lazy projection failed at indices: {bad}")
    q_proj = torch.stack(q_out, dim=0).contiguous()

    if edge_checker is not None and q_proj.shape[0] >= 2:
        q0 = q_proj[:-1]
        q1 = q_proj[1:]
        out = edge_checker.check_edges_batch(q0, q1)
        if bool(out.edge_in_collision.any().item()):
            idx = int(out.edge_in_collision.to(torch.int32).nonzero(as_tuple=False)[0].item())
            raise RuntimeError(f"Edge collision after lazy projection at segment {idx}")

    return q_proj


def path_to_list(path: torch.Tensor) -> List[List[float]]:
    return [[float(x) for x in row.tolist()] for row in path.detach().cpu()]

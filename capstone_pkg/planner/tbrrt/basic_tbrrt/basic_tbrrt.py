from __future__ import annotations

import time
from typing import Optional

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector
from capstone_pkg.utils.joint_limit import JointLimitsTorch

from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.extend_extcon import ExtendStatus, extend_extcon_once
from capstone_pkg.planner.tbrrt.connect import connect_extcon
from capstone_pkg.planner.tbrrt.connection_test import connection_test_paper
from capstone_pkg.planner.tbrrt.postprocess import extract_path, lazy_project_path, path_to_list
from capstone_pkg.planner.tbrrt.tangent_space import build_tangent_space_fd
from capstone_pkg.planner.tbrrt.ts_bank import TSBank
from capstone_pkg.planner.tbrrt.tree import Tree
from capstone_pkg.planner.tbrrt.types import PlanResult, PlanStats
from capstone_pkg.planner.tbrrt.sampling import sample_qrand
from capstone_pkg.planner.tbrrt.goal_region import GoalStates
from capstone_pkg.planner.tbrrt.stats import get_stats


@torch.no_grad()
def plan_tbrrt_extcon(
    *,
    q_start: list[float],
    q_goals: list[list[float]] | list[float],
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    projector: ManifoldProjector,
    joint_limits: JointLimitsTorch,
    device: torch.device,
) -> PlanResult:
    """Bi-directional extcon TB-RRT planner.

    This is a faithful engineering implementation of the paper's extcon variant:
    - Expansion happens on tangent spaces.
    - Projection is performed ONLY when EM is violated (new TS) and on final path.
    """

    t0 = time.time()

    # reset heuristic stats for this planning call
    stats_h = get_stats(reset=True)

    if cfg.seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(cfg.seed))
    else:
        g = None

    q_start_t = torch.tensor(q_start, device=device, dtype=torch.float32)
    q_goals_t = torch.tensor(q_goals, device=device, dtype=torch.float32)
    if q_goals_t.ndim == 1:
        q_goals_t = q_goals_t.view(1, -1)

    # Root projection is allowed (we are defining the manifold roots).
    pr_s = projector.project(q_start_t)
    if not pr_s.success:
        return PlanResult(
            success=False,
            path=None,
            stats=PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "root_projection_failed"}),
        )
    q_start_t = pr_s.q_proj

    proj_goals = []
    for i in range(int(q_goals_t.shape[0])):
        pr_g = projector.project(q_goals_t[i])
        if pr_g.success:
            proj_goals.append(pr_g.q_proj.view(-1))
    if len(proj_goals) == 0:
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "goal_projection_failed"}))
    q_goal_states_t = torch.stack(proj_goals, dim=0).contiguous()

    # Collision check roots
    if not bool(checker.get_collision_free_mask(q_start_t.view(1, -1)).item()):
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "start_in_collision"}))

    free_goal_mask = checker.get_collision_free_mask(q_goal_states_t, margin=0.0)
    if not bool(free_goal_mask.any().item()):
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "goal_in_collision"}))
    q_goal_states_t = q_goal_states_t[free_goal_mask]
    goal_region = GoalStates(q_goal_states_t, threshold=float(cfg.goal_threshold))

    # Edge checker
    edge_checker = EdgeCollisionChecker(
        robot_yml=checker.robot_yml,
        cpu=(device.type == "cpu"),
        world_yml=checker.world_yml,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )
    conn_fail_counts: dict[str, int] = {}  # connection-test failure counters


    D = int(q_start_t.numel())

    # Tangent spaces (bundle)
    bank = TSBank(
        spaces=[],
        ts_radius=float(cfg.ts_radius),
        bias_volume=float(getattr(cfg, "ts_bias_volume", 1.0)),
        bias_curvature=float(getattr(cfg, "ts_bias_curvature", 1.0)),
        bias_nodecount=float(getattr(cfg, "ts_bias_nodecount", 1.0)),
        curv_eps=float(getattr(cfg, "ts_curv_eps", 1e-3)),
    )
    ts0 = build_tangent_space_fd(q_root=q_start_t, projector=projector, svd_tol=float(cfg.svd_tol), ts_id=0, created_iter=0)
    bank.add(ts0)

    # Trees
    tree_start = Tree(device=device, dtype=torch.float32, D=D)
    tree_goal  = Tree(device=device, dtype=torch.float32, D=D)
    rootA = tree_start.add_node(q_start_t, parent=-1, ts_id=0)
    bank.increment_count(0, inc=1)
    assert rootA == 0

    goal_ts_start = len(bank)
    for gi in range(int(goal_region.get_state_count())):
        ts_id = goal_ts_start + gi
        q_goal_i = goal_region.get_state(gi)
        ts_i = build_tangent_space_fd(q_root=q_goal_i, projector=projector, svd_tol=float(cfg.svd_tol), ts_id=ts_id, created_iter=0)
        bank.add(ts_i)
        tree_goal.add_node(q_goal_i, parent=-1, ts_id=ts_id)
        bank.increment_count(ts_id, inc=1)

    ta, tb = tree_start, tree_goal
    swapped = False

    for it in range(int(cfg.max_iters)):
        if (time.time() - t0) > float(cfg.time_limit_sec):
            break

        q_bias_target = q_start_t if swapped else goal_region.nearest_goal_state(ta.get_node(max(0, len(ta) - 1)))
        q_rand = sample_qrand(
            bank=bank,
            joint_limits=joint_limits,
            p_uniform=float(cfg.p_uniform),
            goal_bias=float(cfg.goal_bias),
            q_goal=q_bias_target,
            enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
            generator=g,
        )

        # Paper Sec. 3.5.2: prevent overlapping tangent spaces by discarding samples
        # whose nearest neighbor is (a) a projected TS-root node or (b) the parent of a projected TS-root.
        if bool(getattr(cfg, "enable_overlap_discard", True)):
            _discard_tries = 0
            while True:
                nn_idx, _ = ta.nearest(q_rand)
                if bool(ta.is_proj_root[int(nn_idx)].item()) or bool(ta.is_parent_of_proj_root[int(nn_idx)].item()):
                    stats_h.inc("overlap_discard", 1)
                    _discard_tries += 1
                    if _discard_tries >= int(getattr(cfg, "discard_overlap_max_tries", 20)):
                        stats_h.inc("overlap_discard_maxed", 1)
                        break
                    q_rand = sample_qrand(
                        bank=bank,
                        joint_limits=joint_limits,
                        p_uniform=float(cfg.p_uniform),
                        goal_bias=float(cfg.goal_bias),
                        q_goal=q_bias_target,
                        enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
                        generator=g,
                    )
                    continue
                break


        ext = extend_extcon_once(
            tree=ta,
            bank=bank,
            q_target=q_rand,
            cfg=cfg,
            checker=checker,
            edge_checker=edge_checker,
            projector=projector,
            iter_idx=it,
        )

        if ext.status != ExtendStatus.TRAPPED and ext.new_idx is not None:
            q_new = ta.get_node(ext.new_idx)

            conn = connect_extcon(
                tree=tb,
                bank=bank,
                q_target=q_new,
                cfg=cfg,
                checker=checker,
                edge_checker=edge_checker,
                projector=projector,
                iter_idx=it,
            )

            if conn.status == ExtendStatus.REACHED and conn.last_idx is not None:
                # Paper-style connection test (Sec. 3.8):
                # accept only if the connecting segment is (a) tangent-compatible at both ends,
                # (b) collision-free, and (c) satisfies residual bound along the segment.
                if not swapped:
                    treeA, treeB = ta, tb
                    idxA, idxB = ext.new_idx, conn.last_idx
                else:
                    treeA, treeB = tb, ta
                    idxA, idxB = conn.last_idx, ext.new_idx

                qA = treeA.get_node(int(idxA))
                qB = treeB.get_node(int(idxB))
                tsA = bank.get(int(treeA.ts_id[int(idxA)].item()))
                tsB = bank.get(int(treeB.ts_id[int(idxB)].item()))

                ct = connection_test_paper(
                    qA=qA,
                    tsA=tsA,
                    qB=qB,
                    tsB=tsB,
                    cfg=cfg,
                    checker=checker,
                    edge_checker=edge_checker,
                    projector=projector,
                )
                if not ct.ok:
                    conn_fail_counts[ct.reason] = conn_fail_counts.get(ct.reason, 0) + 1
                    continue

                # Connection accepted -> build path
                if not swapped:
                    path = extract_path(ta, tb, idxA, idxB)   # ta=start, tb=goal
                else:
                    path = extract_path(tb, ta, idxA, idxB)   # tb=start, ta=goal

                path_lp = lazy_project_path(path, projector=projector, edge_checker=edge_checker)

                stats = PlanStats(
                    iters=it + 1,
                    nodes_A=len(tree_start),
                    nodes_B=len(tree_goal),
                    ts_count=len(bank),
                    time_sec=time.time() - t0,
                    extra={
                        "swapped": swapped,
                        "tree_start": tree_start,
                        "tree_goal": tree_goal,
                        "bank": bank,
            "conn_test_fail_counts": dict(conn_fail_counts),
            "heuristic_stats": stats_h.summary(),
                        "q_start_proj": q_start_t,
                        "q_goal_proj": q_goal_states_t,
                        "goal_region": goal_region,
                        "matched_goal_idx": goal_region.nearest_goal_index(path_lp[-1]),
                    },
                )

                return PlanResult(
                    success=True,
                    path=path_to_list(path_lp),
                    stats=stats,
                    conn_idx_A=int(idxA),
                    conn_idx_B=int(idxB),
                )

        # swap only pointers
        ta, tb = tb, ta
        swapped = not swapped

    # 실패 stats도 tree를 남겨서 plot/debug 가능
    stats = PlanStats(
        iters=int(cfg.max_iters),
        nodes_A=len(tree_start),
        nodes_B=len(tree_goal),
        ts_count=len(bank),
        time_sec=time.time() - t0,
        extra={
            "reason": "timeout_or_maxiters",
            "swapped": swapped,
            "tree_start": tree_start,
            "tree_goal": tree_goal,
            "bank": bank,
            "conn_test_fail_counts": dict(conn_fail_counts),
            "heuristic_stats": stats_h.summary(),
            "q_start_proj": q_start_t,
            "q_goal_proj": q_goal_states_t,
            "goal_region": goal_region,
        },
    )
    return PlanResult(False, None, stats)



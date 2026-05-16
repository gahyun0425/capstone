from __future__ import annotations

"""Batch-Conext TB-RRT (no-waypoint style) with TS-heuristic escape.

Goal
----
Create a planner that keeps exactly two trees (A=start, B=goal) per IK instance (batch b)
and expands both trees concurrently.

Core loop
---------
1) CONNECT: each active block in each tree performs 1-step connect toward the nearest
   node in the opposite tree (target updated every `connect_max_steps`).
2) If connect is TRAPPED (collision / edge collision / projection failure):
   - choose a tangent space id using TSBank's heuristic (paper Sec. 3.7)
   - sample `escape_spawn_blocks` q_rand in that TS
   - for each q_rand: find q_near in the trapped tree and EXTEND 1-step
   - the new nodes become new blocks of that same tree
3) Newly extended blocks resume CONNECT.

Implementation notes
--------------------
* Code structure mirrors `tbrrt/no_waypoint/no_waypoint.py`.
* Roots start with 2 TS (start/goal), and projected nodes can spawn new TS.
* Max blocks per tree is fixed to `cfg.block_k` (with an internal floor large enough
  for escape branching). Escape spawns up to `cfg.escape_spawn_blocks` blocks per
  trapped block; if insufficient free slots exist, we truncate.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjectorTorch as ManifoldProjector
from capstone_pkg.utils.joint_limit import JointLimitsTorch

from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.planning_trace import PlanningTraceRecorder
from capstone_pkg.planner.tbrrt.postprocess import (
    cubic_spline_interpolate_and_validate_path,
    path_to_list,
    shortcut_smooth_path,
    topp_retime_path,
)
from capstone_pkg.planner.tbrrt.ts_bank import TSBank
from capstone_pkg.planner.tbrrt.types import PlanResult, PlanStats

from capstone_pkg.planner.tbrrt.batch.tree_batch import TreeBatchGPU
from capstone_pkg.planner.tbrrt.tangent_space import TangentSpace
from capstone_pkg.planner.tbrrt.batch.connect_batch import (
    _build_tangent_spaces_batch,
    connect_two_trees_one_step_with_state as _connect_two_trees_one_step_with_state,
)
from capstone_pkg.planner.tbrrt.batch.escape import _escape_one_tree, _escape_two_trees
from capstone_pkg.planner.tbrrt.batch.extend_batch import (
    _extend_one_step_from_parent,
    extend_two_trees_one_step_from_parent,
)
from capstone_pkg.planner.tbrrt.batch.heuristic_batch import (
    _apply_overlap_discard_qrand_batch,
    _sample_in_selected_ts_ball_batch,
    _sample_ts_ids_from_banks_batch,
    _select_connect_target_idx_batch,
    _select_escape_q_hint,
)
from capstone_pkg.planner.tbrrt.batch.nn_batch import nn_1_tree_all_candidates_cdist
from capstone_pkg.planner.tbrrt.batch.projector_stack import ProjectorStackCache
from capstone_pkg.planner.tbrrt.batch.prealloc import BatchConextPrealloc


@dataclass
class ConnectionCandidate:
    path: torch.Tensor
    b: int
    k: int
    idxA: int
    idxB: int
    nodes_a: List[int]
    nodes_b_rev: List[int]


@torch.no_grad()
def _backtrack_indices_one(tree: TreeBatchGPU, b: int, idx: int) -> List[int]:
    if idx < 0:
        raise ValueError("idx must be >=0")
    n = int(tree.n_nodes[b].item())
    if idx >= n:
        raise ValueError("idx out of range")
    cur = int(idx)
    out: List[int] = []
    while cur >= 0:
        out.append(cur)
        cur = int(tree.parent[b, cur].item())
    out.reverse()
    return out


@torch.no_grad()
def _extract_connection_candidate(
    tree_start: TreeBatchGPU,
    tree_goal: TreeBatchGPU,
    b: int,
    k: int,
    idxA: int,
    idxB: int,
) -> ConnectionCandidate:
    nodes_a = _backtrack_indices_one(tree_start, int(b), int(idxA))
    nodes_b = _backtrack_indices_one(tree_goal, int(b), int(idxB))
    nodes_b_rev = list(reversed(nodes_b))
    idx_a_t = torch.tensor(nodes_a, device=tree_start.device, dtype=torch.long)
    idx_b_t = torch.tensor(nodes_b_rev, device=tree_goal.device, dtype=torch.long)
    pA = tree_start.q[int(b), idx_a_t, :].detach().clone()
    pB_rev = tree_goal.q[int(b), idx_b_t, :].detach().clone()
    return ConnectionCandidate(
        path=torch.cat([pA, pB_rev], dim=0),
        b=int(b),
        k=int(k),
        idxA=int(idxA),
        idxB=int(idxB),
        nodes_a=nodes_a,
        nodes_b_rev=nodes_b_rev,
    )


@torch.no_grad()
def _gather_selected_ts_tensors(
    *,
    banks: List[TSBank],
    ts_ids: torch.Tensor,  # (B,) long, -1 allowed
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_ids = ts_ids.clamp_min(0).to(torch.long)
    safe_id_list = safe_ids.tolist()

    ts_sel = [bank.get(int(ts_id)) for bank, ts_id in zip(banks, safe_id_list)]
    roots = torch.stack([ts.root.to(device=device, dtype=dtype) for ts in ts_sel], dim=0)
    dim = torch.tensor([int(ts.dim) for ts in ts_sel], device=device, dtype=torch.long)
    domain = torch.tensor(
        [float(bank.get_domain(int(ts_id))) for bank, ts_id in zip(banks, safe_id_list)],
        device=device,
        dtype=dtype,
    )

    kmax = int(dim.max().item()) if dim.numel() > 0 else 0
    if kmax <= 0:
        basis = torch.zeros((len(ts_sel), roots.shape[1], 0), device=device, dtype=dtype)
    else:
        basis_kd = [ts.basis.transpose(0, 1).to(device=device, dtype=dtype) for ts in ts_sel]
        basis_kd_pad = torch.nn.utils.rnn.pad_sequence(basis_kd, batch_first=True)  # (B,kmax,D)
        basis = basis_kd_pad.transpose(1, 2).contiguous()  # (B,D,kmax)
    return roots, basis, dim, domain


@torch.no_grad()
def _target_hits_overlap_nodes(
    *,
    tree: TreeBatchGPU,
    target_idx: torch.Tensor,  # (B,K)
    prealloc: Optional[BatchConextPrealloc] = None,
) -> torch.Tensor:
    if target_idx.ndim != 2:
        raise ValueError("target_idx must be (B,K)")

    B, K = target_idx.shape
    cap = int(tree.is_proj_root.shape[1])
    safe_idx = target_idx.clamp(min=0, max=max(0, cap - 1))
    if prealloc is not None and prealloc.supports(B=B, K=K, D=tree.D, device=target_idx.device, dtype=tree.dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=target_idx.device).view(B, 1).expand(B, K)
    valid = (target_idx >= 0) & (target_idx < tree.n_nodes.view(B, 1))
    overlap = tree.is_proj_root[b_idx, safe_idx] | tree.is_parent_of_proj_root[b_idx, safe_idx]
    return valid & overlap


@torch.no_grad()
def _lazy_project_paths_with_edge_check_batch(
    paths: List[torch.Tensor],
    projector: ManifoldProjector,
    edge_checker: EdgeCollisionChecker,
    prealloc: Optional[BatchConextPrealloc] = None,
) -> Tuple[List[Optional[torch.Tensor]], List[str], List[int], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Project candidate paths together and batch-check all projected edges."""
    n_paths = len(paths)
    results: List[Optional[torch.Tensor]] = [None] * n_paths
    fail_reasons: List[str] = ["invalid"] * n_paths
    fail_edge_indices: List[int] = [-1] * n_paths
    fail_edge_pairs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_paths
    if n_paths == 0:
        return results, fail_reasons, fail_edge_indices, fail_edge_pairs

    starts: List[int] = []
    lengths: List[int] = []
    path_index_for_valid_flat: List[int] = []
    flat_parts: List[torch.Tensor] = []
    cursor = 0
    for i, path in enumerate(paths):
        if path.ndim != 2 or path.shape[0] == 0:
            continue
        starts.append(cursor)
        length = int(path.shape[0])
        lengths.append(length)
        path_index_for_valid_flat.append(i)
        cursor += length
        fail_reasons[i] = "projection"

    total_points = cursor
    if total_points <= 0:
        return results, fail_reasons, fail_edge_indices, fail_edge_pairs

    total_edges = sum(max(0, length - 1) for length in lengths)
    sample = paths[path_index_for_valid_flat[0]]
    use_scratch = (
        prealloc is not None
        and prealloc.lazy_scratch_fits(
            n_paths=n_paths,
            total_points=total_points,
            total_edges=total_edges,
            device=sample.device,
            dtype=sample.dtype,
        )
    )

    if use_scratch:
        assert prealloc is not None
        assert prealloc.lazy_flat_q is not None
        flat = prealloc.lazy_flat_q[:total_points, :]
        flat_parts = [paths[path_i] for path_i in path_index_for_valid_flat]
        torch.cat(flat_parts, dim=0, out=flat)
    else:
        for path_i in path_index_for_valid_flat:
            flat_parts.append(paths[path_i])
        flat = torch.cat(flat_parts, dim=0).contiguous()

    pr = projector.project_batch(flat)
    q_proj_flat = pr.q_proj.contiguous()
    success_flat = pr.success_mask.bool()

    projected_ok = [False] * n_paths
    for local_i, path_i in enumerate(path_index_for_valid_flat):
        st = starts[local_i]
        ed = st + lengths[local_i]
        if bool(success_flat[st:ed].all()):
            projected_ok[path_i] = True
            fail_reasons[path_i] = "edge"

    edge_q0_parts: List[torch.Tensor] = []
    edge_q1_parts: List[torch.Tensor] = []
    edge_owner: List[int] = []
    edge_path_idx: List[int] = []
    edge_cursor = 0
    q0 = None
    q1 = None
    owner_t = None
    edge_idx_t = None
    if use_scratch:
        assert prealloc is not None
        assert prealloc.lazy_edge_q0 is not None
        assert prealloc.lazy_edge_q1 is not None
        assert prealloc.lazy_edge_owner is not None
        assert prealloc.lazy_edge_path_idx is not None
        q0 = prealloc.lazy_edge_q0[:total_edges, :]
        q1 = prealloc.lazy_edge_q1[:total_edges, :]
        owner_t = prealloc.lazy_edge_owner[:total_edges]
        edge_idx_t = prealloc.lazy_edge_path_idx[:total_edges]

    for local_i, path_i in enumerate(path_index_for_valid_flat):
        if not projected_ok[path_i]:
            continue
        st = starts[local_i]
        length = lengths[local_i]
        path_proj = q_proj_flat[st : st + length]
        if length < 2:
            results[path_i] = path_proj.contiguous()
            fail_reasons[path_i] = ""
            continue
        n_edges = length - 1
        if use_scratch:
            assert q0 is not None
            assert q1 is not None
            assert owner_t is not None
            assert edge_idx_t is not None
            edge_q0_parts.append(path_proj[:-1])
            edge_q1_parts.append(path_proj[1:])
            owner_t[edge_cursor : edge_cursor + n_edges].fill_(int(path_i))
            edge_idx_t[edge_cursor : edge_cursor + n_edges].copy_(
                torch.arange(n_edges, device=edge_idx_t.device, dtype=torch.long)
            )
            edge_cursor += n_edges
        else:
            edge_q0_parts.append(path_proj[:-1])
            edge_q1_parts.append(path_proj[1:])
            edge_owner.extend([path_i] * n_edges)
            edge_path_idx.extend(range(n_edges))

    if use_scratch:
        assert prealloc is not None
        assert prealloc.lazy_edge_ok is not None
        edge_ok = prealloc.lazy_edge_ok[:n_paths]
        edge_ok.fill_(True)
    else:
        edge_ok = torch.ones((n_paths,), device=flat.device, dtype=torch.bool)

    if use_scratch and edge_cursor > 0:
        assert q0 is not None
        assert q1 is not None
        assert owner_t is not None
        assert edge_idx_t is not None
        torch.cat(edge_q0_parts, dim=0, out=q0[:edge_cursor, :])
        torch.cat(edge_q1_parts, dim=0, out=q1[:edge_cursor, :])
        edge_res = edge_checker.check_edges_batch(q0[:edge_cursor, :], q1[:edge_cursor, :])
        edge_in_col = edge_res.bool() if torch.is_tensor(edge_res) else edge_res.edge_in_collision.bool()
        if bool(edge_in_col.any()):
            failed_owner_all = owner_t[:edge_cursor][edge_in_col]
            failed_edge_all = edge_idx_t[:edge_cursor][edge_in_col]
            failed_owner = failed_owner_all.unique()
            edge_ok[failed_owner] = False
            for owner_i in failed_owner.tolist():
                owner_mask = failed_owner_all == int(owner_i)
                edge_i = int(failed_edge_all[owner_mask].min().item())
                fail_edge_indices[int(owner_i)] = edge_i
                first_edge = torch.nonzero(
                    edge_in_col
                    & (owner_t[:edge_cursor] == int(owner_i))
                    & (edge_idx_t[:edge_cursor] == edge_i),
                    as_tuple=False,
                ).view(-1)
                if first_edge.numel() > 0:
                    flat_i = int(first_edge[0].item())
                    fail_edge_pairs[int(owner_i)] = (
                        q0[flat_i].detach().clone(),
                        q1[flat_i].detach().clone(),
                    )
    elif edge_q0_parts:
        q0_dyn = torch.cat(edge_q0_parts, dim=0).contiguous()
        q1_dyn = torch.cat(edge_q1_parts, dim=0).contiguous()
        edge_res = edge_checker.check_edges_batch(q0_dyn, q1_dyn)
        edge_in_col = edge_res.bool() if torch.is_tensor(edge_res) else edge_res.edge_in_collision.bool()
        if bool(edge_in_col.any()):
            owner = torch.tensor(edge_owner, device=edge_in_col.device, dtype=torch.long)
            path_idx = torch.tensor(edge_path_idx, device=edge_in_col.device, dtype=torch.long)
            failed_owner_all = owner[edge_in_col]
            failed_edge_all = path_idx[edge_in_col]
            failed_owner = failed_owner_all.unique()
            edge_ok[failed_owner] = False
            for owner_i in failed_owner.tolist():
                owner_mask = failed_owner_all == int(owner_i)
                edge_i = int(failed_edge_all[owner_mask].min().item())
                fail_edge_indices[int(owner_i)] = edge_i
                first_edge = torch.nonzero(
                    edge_in_col
                    & (owner == int(owner_i))
                    & (path_idx == edge_i),
                    as_tuple=False,
                ).view(-1)
                if first_edge.numel() > 0:
                    flat_i = int(first_edge[0].item())
                    fail_edge_pairs[int(owner_i)] = (
                        q0_dyn[flat_i].detach().clone(),
                        q1_dyn[flat_i].detach().clone(),
                    )

    for local_i, path_i in enumerate(path_index_for_valid_flat):
        if not projected_ok[path_i]:
            continue
        if not bool(edge_ok[path_i].item()):
            fail_reasons[path_i] = "edge"
            continue
        st = starts[local_i]
        length = lengths[local_i]
        results[path_i] = q_proj_flat[st : st + length].contiguous()
        fail_reasons[path_i] = ""

    return results, fail_reasons, fail_edge_indices, fail_edge_pairs


@torch.no_grad()
def plan_tbrrt_extcon_batch_conext(
    *,
    q_start: List[float],
    q_goals: List[List[float]],  # length B
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    projector: ManifoldProjector,
    joint_limits: JointLimitsTorch,
    device: torch.device,
    block_K: int = 50,
    trace_recorder: Optional[PlanningTraceRecorder] = None,
) -> PlanResult:
    t0 = time.time()

    if len(q_goals) == 0:
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "no_goals"}))

    dtype = torch.float32

    B = len(q_goals)
    D = len(q_start)

    q_start_t = torch.tensor(q_start, device=device, dtype=dtype).view(1, -1).expand(B, -1).contiguous()
    q_goals_t = torch.tensor(q_goals, device=device, dtype=dtype).contiguous()
    if hasattr(projector, "prepare_fixed_tensors"):
        residual_dim = int(projector.residual(q_start_t[:1]).shape[1])
        projector.prepare_fixed_tensors(
            q_dim=D,
            residual_dim=residual_dim,
            device=device,
            dtype=dtype,
        )

    # escape_spawn_blocks = max(1, int(getattr(cfg, "escape_spawn_blocks", 3)))

    Kmax = int(block_K)
    prealloc = BatchConextPrealloc.create(
        B=B,
        Kmax=Kmax,
        D=D,
        device=device,
        dtype=dtype,
    )
    lazy_prealloc_enable = bool(getattr(cfg, "connection_lazy_prealloc_enable", True))
    lazy_prealloc_max_candidates = int(getattr(cfg, "connection_lazy_prealloc_max_candidates", 0))
    if lazy_prealloc_max_candidates <= 0:
        lazy_prealloc_max_candidates = max(1, B * Kmax)
    lazy_prealloc_max_path_points = max(0, int(getattr(cfg, "connection_lazy_prealloc_max_path_points", 512)))
    if lazy_prealloc_enable and lazy_prealloc_max_path_points > 0:
        prealloc.ensure_lazy_scratch(
            max_paths=lazy_prealloc_max_candidates,
            max_points_per_path=lazy_prealloc_max_path_points,
        )

    seed = 0 if getattr(cfg, "seed", None) is None else int(cfg.seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    roots0 = torch.cat([q_start_t, q_goals_t], dim=0)
    pr0 = projector.project_batch(roots0)
    if not bool(pr0.success_mask.all()):
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "root_projection_failed"}))
    roots = pr0.q_proj
    q_start_proj = roots[:B]
    q_goal_proj = roots[B:]

    free_start = checker.get_collision_free_mask(q_start_proj, margin=0.0)
    free_goal = checker.get_collision_free_mask(q_goal_proj, margin=0.0)
    valid_b = free_start & free_goal
    if not bool(valid_b.any()):
        return PlanResult(False, None, PlanStats(0, 0, 0, 0, time.time() - t0, {"reason": "all_roots_in_collision"}))

    edge_checker = EdgeCollisionChecker(
        robot_yml=checker.robot_yml,
        cpu=(device.type == "cpu"),
        world_yml=checker.world_yml,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )

    ts0_batch = _build_tangent_spaces_batch(
        q_roots=q_start_proj,
        projector=projector,
        svd_tol=float(cfg.svd_tol),
        ts_ids=prealloc.zero_long_b,
        created_iter=0,
        debug_label="initial",
    )
    ts1_batch = _build_tangent_spaces_batch(
        q_roots=q_goal_proj,
        projector=projector,
        svd_tol=float(cfg.svd_tol),
        ts_ids=prealloc.one_long_b,
        created_iter=0,
        debug_label="initial",
    )

    def _make_bank(ts0: TangentSpace, ts1: TangentSpace) -> TSBank:
        bank = TSBank(
            spaces=[ts0, ts1],
            ts_radius=float(cfg.ts_radius),
            bias_volume=float(getattr(cfg, "ts_bias_volume", 1.0)),
            bias_curvature=float(getattr(cfg, "ts_bias_curvature", 1.0)),
            bias_nodecount=float(getattr(cfg, "ts_bias_nodecount", 1.0)),
            bias_collision=float(getattr(cfg, "ts_bias_collision", 1.0)),
            curv_eps=float(getattr(cfg, "ts_curv_eps", 1e-3)),
        )
        bank.increment_count(0, inc=1)
        bank.increment_count(1, inc=1)
        return bank

    banks: List[TSBank] = [_make_bank(ts0, ts1) for ts0, ts1 in zip(ts0_batch, ts1_batch)]
    # P_flat = 접공간 projector 모음
    # J_flat = constraint Jacobian 모음
    # offsets = 배치별 local ts_id를 flat 텐서의 global index로 바꾸는 기준점
    projector_cache = ProjectorStackCache() # P_flat, J_flat, offsets 저장 

    treeA = TreeBatchGPU(
        device=device,
        dtype=torch.float32,
        B=B,
        D=D,
        capacity=32768,
        trace_recorder=trace_recorder,
        trace_tree_name="start",
    )
    treeB = TreeBatchGPU(
        device=device,
        dtype=torch.float32,
        B=B,
        D=D,
        capacity=32768,
        trace_recorder=trace_recorder,
        trace_tree_name="goal",
    )
    treeA.init_roots(q_start_proj, prealloc.zero_long_b)
    treeB.init_roots(q_goal_proj, prealloc.one_long_b)

    # 블록 상태 텐서들:
    # - 첫 번째 축 2: side batch. 0은 start/treeA, 1은 goal/treeB다.
    # - 두 번째 축 B: IK goal 배치 차원.
    # - 세 번째 축 Kmax: 각 배치 항목 안에서 동시에 유지하는 block/branch 슬롯 개수다.
    #   slot 0은 루트에서 시작하고, 나머지 슬롯은 ESCAPE 단계에서 추가로 활성화될 수 있다.
    activeAB = prealloc.state_active
    curAB = prealloc.state_cur
    modeAB = prealloc.state_mode
    segAB = prealloc.state_seg
    targetAB = prealloc.state_target
    stagnAB = prealloc.state_stagn
    banTargetAB = prealloc.state_ban_target
    banCdAB = prealloc.state_ban_cd
    activeAB.zero_()
    curAB.zero_()
    modeAB.zero_()
    segAB.zero_()
    targetAB.fill_(-1)
    stagnAB.zero_()
    banTargetAB.fill_(-1)
    banCdAB.zero_()
    edgeRegionHitAB = torch.zeros_like(stagnAB)

    activeA = activeAB[0]
    activeB = activeAB[1]
    # 각 트리의 slot 0만 루트 블록으로 시작한다. valid_b=False인 배치는 처음부터 비활성.
    # A트리/B트리의 각 slot이 현재 살아 있는가
    activeA[:, 0] = valid_b
    activeB[:, 0] = valid_b

    # 각 block이 현재 가리키는 "트리 내부 현재 노드 인덱스".
    # connect/escape 모두 이 노드에서 다음 확장을 시작한다.
    curA = curAB[0]
    curB = curAB[1]

    # 각 block의 동작 모드.
    # 0: CONNECT 모드 (반대편 트리의 목표 노드를 향해 전진)
    # 1: ESCAPE 모드 (막혔을 때 새 샘플을 뽑아 탈출 시도)
    modeA = modeAB[0]
    modeB = modeAB[1]

    # 현재 connect target을 향해 연속으로 전진한 step 수.
    # 타깃을 다시 잡거나 escape branch를 새로 만들면 0으로 리셋된다.
    segA = segAB[0]
    segB = segAB[1]
    # 반대편 트리에서 현재 connect 대상으로 삼고 있는 노드 인덱스.
    # -1이면 아직 타깃이 없어서 nearest-node 재선정이 필요하다는 뜻이다.
    tgtB_for_A = targetAB[0]
    tgtA_for_B = targetAB[1]
    stagnA = stagnAB[0]
    stagnB = stagnAB[1]
    ban_tgtB_for_A = banTargetAB[0]
    ban_tgtA_for_B = banTargetAB[1]
    ban_cdA = banCdAB[0]
    ban_cdB = banCdAB[1]
    edge_region_hitA = edgeRegionHitAB[0]
    edge_region_hitB = edgeRegionHitAB[1]
    curvature_pad_cache: Optional[torch.Tensor] = None
    curvature_pad_counts: Optional[Tuple[int, ...]] = None

    def _record_iter_summary(iter_idx: int) -> None:
        if trace_recorder is None or not trace_recorder.wants("summary"):
            return
        connect_start = activeA & (modeA == 0)
        connect_goal = activeB & (modeB == 0)
        escape_start = activeA & (modeA == 1)
        escape_goal = activeB & (modeB == 1)
        trace_recorder.record_iter_summary(
            iter_idx=int(iter_idx),
            active_start=int(activeA.sum().item()),
            active_goal=int(activeB.sum().item()),
            connect_start=int(connect_start.sum().item()),
            connect_goal=int(connect_goal.sum().item()),
            escape_start=int(escape_start.sum().item()),
            escape_goal=int(escape_goal.sum().item()),
            nodes_start_max=int(treeA.n_nodes.max().item()),
            nodes_goal_max=int(treeB.n_nodes.max().item()),
            nodes_start_sum=int(treeA.n_nodes.sum().item()),
            nodes_goal_sum=int(treeB.n_nodes.sum().item()),
            ts_count_max=max(len(bk.spaces) for bk in banks),
            elapsed_sec=float(time.time() - t0),
        )

    connection_attempt_total = 0
    connection_attempt_success = 0
    connection_lazy_failures = 0
    connection_region_skips = 0
    connection_edge_region_skips = 0
    connection_segment_precheck_failures = 0
    connection_lazy_projection_failures = 0
    connection_lazy_edge_failures = 0
    connection_lazy_batch_calls = 0
    connection_lazy_batch_candidates = 0
    connection_lazy_prealloc_overflows = 0
    connection_banned_branch_skips = 0
    connection_lazy_edge_banned_nodes = 0
    connection_lazy_edge_single_banned_nodes = 0
    connection_lazy_edge_single_ban_fallbacks = 0
    connection_lazy_edge_connector_failures = 0
    connection_banned_current_resets = 0
    connection_edge_region_retargets = 0
    connection_edge_region_escapes = 0
    failed_connection_regions_recorded = 0
    failed_edge_collision_regions_recorded = 0
    connection_attempts_by_reason: dict[str, int] = {}
    connection_lazy_failures_by_reason: dict[str, int] = {}
    fail_region_enable = bool(getattr(cfg, "failed_connection_region_enable", True))
    fail_region_max = max(0, int(getattr(cfg, "failed_connection_region_max", 2048)))
    fail_region_radius = max(0.0, float(getattr(cfg, "failed_connection_region_radius", 0.05)))
    fail_region_qA = (
        torch.empty((B, fail_region_max, D), device=device, dtype=dtype)
        if fail_region_enable and fail_region_max > 0 and fail_region_radius > 0.0
        else None
    )
    fail_region_qB = (
        torch.empty((B, fail_region_max, D), device=device, dtype=dtype)
        if fail_region_qA is not None
        else None
    )
    fail_region_count = (
        torch.zeros((B,), device=device, dtype=torch.long)
        if fail_region_qA is not None
        else None
    )
    fail_region_write = (
        torch.zeros((B,), device=device, dtype=torch.long)
        if fail_region_qA is not None
        else None
    )
    edge_fail_region_enable = bool(getattr(cfg, "failed_edge_region_enable", True))
    edge_fail_region_max = max(0, int(getattr(cfg, "failed_edge_region_max", 4096)))
    edge_fail_region_radius = max(0.0, float(getattr(cfg, "failed_edge_region_radius", fail_region_radius)))
    edge_region_retarget_threshold = max(0, int(getattr(cfg, "failed_edge_region_retarget_threshold", 4)))
    edge_region_escape_threshold = max(0, int(getattr(cfg, "failed_edge_region_escape_threshold", 16)))
    edge_region_cooldown = max(
        0,
        int(getattr(cfg, "failed_edge_region_cooldown", getattr(cfg, "failed_connection_cooldown", 8))),
    )
    edge_fail_region_q0 = (
        torch.empty((B, edge_fail_region_max, D), device=device, dtype=dtype)
        if edge_fail_region_enable and edge_fail_region_max > 0 and edge_fail_region_radius > 0.0
        else None
    )
    edge_fail_region_q1 = (
        torch.empty((B, edge_fail_region_max, D), device=device, dtype=dtype)
        if edge_fail_region_q0 is not None
        else None
    )
    edge_fail_region_count = (
        torch.zeros((B,), device=device, dtype=torch.long)
        if edge_fail_region_q0 is not None
        else None
    )
    edge_fail_region_write = (
        torch.zeros((B,), device=device, dtype=torch.long)
        if edge_fail_region_q0 is not None
        else None
    )

    edge_branch_ban_enable = bool(getattr(cfg, "connection_lazy_edge_branch_ban_enable", True))
    edge_subtree_ban_max_nodes = int(getattr(cfg, "connection_lazy_edge_subtree_ban_max_nodes", 256))

    def _node_banned_mask(tree: TreeBatchGPU, node_idx: torch.Tensor) -> torch.Tensor:
        B_m, K_m = node_idx.shape
        safe_idx = node_idx.clamp(min=0, max=max(0, int(tree.banned_node.shape[1]) - 1))
        b_idx_m = prealloc.b_idx(K_m)
        valid = (node_idx >= 0) & (node_idx < tree.n_nodes.view(B_m, 1))
        return valid & tree.banned_node[b_idx_m, safe_idx]

    def _node_blocked_mask(tree: TreeBatchGPU, node_idx: torch.Tensor) -> torch.Tensor:
        blocked = getattr(tree, "blocked_node", tree.banned_node)
        B_m, K_m = node_idx.shape
        safe_idx = node_idx.clamp(min=0, max=max(0, int(blocked.shape[1]) - 1))
        b_idx_m = prealloc.b_idx(K_m)
        valid = (node_idx >= 0) & (node_idx < tree.n_nodes.view(B_m, 1))
        return valid & blocked[b_idx_m, safe_idx]

    def _candidate_hits_banned_branch(cand: ConnectionCandidate) -> bool:
        if not edge_branch_ban_enable:
            return False
        b_i = int(cand.b)
        blocked_a = getattr(treeA, "blocked_node", treeA.banned_node)
        blocked_b = getattr(treeB, "blocked_node", treeB.banned_node)
        if any(bool(blocked_a[b_i, int(node)].item()) for node in cand.nodes_a[1:]):
            return True
        if any(bool(blocked_b[b_i, int(node)].item()) for node in cand.nodes_b_rev):
            return True
        return False

    def _record_lazy_edge_branch_ban(cand: ConnectionCandidate, fail_edge_idx: int) -> str:
        nonlocal connection_lazy_edge_banned_nodes, connection_lazy_edge_single_banned_nodes
        nonlocal connection_lazy_edge_single_ban_fallbacks, connection_lazy_edge_connector_failures
        if not edge_branch_ban_enable or fail_edge_idx < 0:
            return "none"

        def _ban_failed_child(tree: TreeBatchGPU, node: int) -> str:
            nonlocal connection_lazy_edge_banned_nodes, connection_lazy_edge_single_banned_nodes
            nonlocal connection_lazy_edge_single_ban_fallbacks
            if edge_subtree_ban_max_nodes > 0:
                n_banned, subtree_banned = tree.ban_subtree_limited(
                    int(cand.b),
                    int(node),
                    edge_subtree_ban_max_nodes,
                )
                if subtree_banned:
                    connection_lazy_edge_banned_nodes += int(n_banned)
                    return "subtree"
                connection_lazy_edge_single_banned_nodes += int(n_banned)
                connection_lazy_edge_single_ban_fallbacks += 1
                return "single"
            connection_lazy_edge_banned_nodes += tree.ban_subtree(int(cand.b), int(node))
            return "subtree"

        len_a = len(cand.nodes_a)
        if fail_edge_idx < max(0, len_a - 1):
            node = int(cand.nodes_a[int(fail_edge_idx) + 1])
            return f"treeA_{_ban_failed_child(treeA, node)}"
        if fail_edge_idx == len_a - 1:
            connection_lazy_edge_connector_failures += 1
            return "connector"

        b_edge_i = int(fail_edge_idx) - len_a
        if 0 <= b_edge_i < max(0, len(cand.nodes_b_rev) - 1):
            node = int(cand.nodes_b_rev[b_edge_i])
            return f"treeB_{_ban_failed_child(treeB, node)}"
        return "none"

    def _reset_slots_on_banned_current(
        *,
        tree: TreeBatchGPU,
        active: torch.Tensor,
        cur: torch.Tensor,
        mode: torch.Tensor,
        seg: torch.Tensor,
        stagn: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        nonlocal connection_banned_current_resets
        if not edge_branch_ban_enable:
            return
        banned_cur = active & _node_banned_mask(tree, cur)
        if not bool(banned_cur.any()):
            return
        for b_i, k_i in torch.nonzero(banned_cur, as_tuple=False).tolist():
            node = int(cur[b_i, k_i].item())
            while node > 0 and bool(tree.banned_node[b_i, node].item()):
                node = int(tree.parent[b_i, node].item())
            if node < 0:
                node = 0
            cur[b_i, k_i] = node
        mode.copy_(torch.where(banned_cur, prealloc.one_long(Kmax), mode))
        seg.copy_(torch.where(banned_cur, prealloc.zero_long(Kmax), seg))
        stagn.copy_(torch.where(banned_cur, prealloc.zero_long(Kmax), stagn))
        target.copy_(torch.where(banned_cur, prealloc.minus_one_long(Kmax), target))
        connection_banned_current_resets += int(banned_cur.sum().item())

    def _record_failed_connection_region(*, b: int, idxA: int, idxB: int) -> None:
        nonlocal failed_connection_regions_recorded
        if fail_region_qA is None or fail_region_qB is None or fail_region_count is None or fail_region_write is None:
            return
        if idxA < 0 or idxB < 0:
            return
        n_a = int(treeA.n_nodes[b].item())
        n_b = int(treeB.n_nodes[b].item())
        if idxA >= n_a or idxB >= n_b:
            return

        pos = int(fail_region_write[b].item())
        fail_region_qA[b, pos, :].copy_(treeA.q[b, idxA, :])
        fail_region_qB[b, pos, :].copy_(treeB.q[b, idxB, :])
        fail_region_write[b] = (pos + 1) % fail_region_max
        if int(fail_region_count[b].item()) < fail_region_max:
            fail_region_count[b] += 1
        failed_connection_regions_recorded += 1

    def _hits_failed_connection_region(*, b: int, idxA: int, idxB: int) -> bool:
        if fail_region_qA is None or fail_region_qB is None or fail_region_count is None:
            return False
        if idxA < 0 or idxB < 0:
            return False
        count = int(fail_region_count[b].item())
        if count <= 0:
            return False
        q_a = treeA.q[b, idxA, :].view(1, D)
        q_b = treeB.q[b, idxB, :].view(1, D)
        d_a = torch.linalg.norm(fail_region_qA[b, :count, :] - q_a, dim=-1)
        d_b = torch.linalg.norm(fail_region_qB[b, :count, :] - q_b, dim=-1)
        hit = (d_a <= fail_region_radius) & (d_b <= fail_region_radius)
        return bool(hit.any().item())

    def _record_failed_edge_collision_region(*, b: int, q0: torch.Tensor | None, q1: torch.Tensor | None) -> None:
        nonlocal failed_edge_collision_regions_recorded
        if (
            edge_fail_region_q0 is None
            or edge_fail_region_q1 is None
            or edge_fail_region_count is None
            or edge_fail_region_write is None
            or q0 is None
            or q1 is None
        ):
            return
        if b < 0 or b >= B:
            return

        pos = int(edge_fail_region_write[b].item())
        edge_fail_region_q0[b, pos, :].copy_(q0.to(device=device, dtype=dtype))
        edge_fail_region_q1[b, pos, :].copy_(q1.to(device=device, dtype=dtype))
        edge_fail_region_write[b] = (pos + 1) % edge_fail_region_max
        if int(edge_fail_region_count[b].item()) < edge_fail_region_max:
            edge_fail_region_count[b] += 1
        failed_edge_collision_regions_recorded += 1

    def _failed_edge_collision_region_hit_edge(cand: ConnectionCandidate) -> int:
        if edge_fail_region_q0 is None or edge_fail_region_q1 is None or edge_fail_region_count is None:
            return -1
        b = int(cand.b)
        count = int(edge_fail_region_count[b].item())
        if count <= 0 or int(cand.path.shape[0]) < 2:
            return -1

        reg0 = edge_fail_region_q0[b, :count, :]
        reg1 = edge_fail_region_q1[b, :count, :]
        r = float(edge_fail_region_radius)
        q0_edges = cand.path[:-1, :]
        q1_edges = cand.path[1:, :]
        chunk = 64
        for start_i in range(0, int(q0_edges.shape[0]), chunk):
            end_i = min(start_i + chunk, int(q0_edges.shape[0]))
            edge0 = q0_edges[start_i:end_i, :]
            edge1 = q1_edges[start_i:end_i, :]
            fwd = (torch.cdist(edge0, reg0) <= r) & (torch.cdist(edge1, reg1) <= r)
            rev = (torch.cdist(edge0, reg1) <= r) & (torch.cdist(edge1, reg0) <= r)
            hit = fwd | rev
            if bool(hit.any().item()):
                hit_rows = torch.nonzero(hit.any(dim=1), as_tuple=False).view(-1)
                if hit_rows.numel() > 0:
                    return int(start_i + int(hit_rows[0].item()))
        return -1

    def _candidate_slot_side(cand: ConnectionCandidate) -> Optional[str]:
        b = int(cand.b)
        k = int(cand.k)
        if b < 0 or b >= B or k < 0 or k >= Kmax:
            return None

        cur_a = int(curA[b, k].item())
        cur_b = int(curB[b, k].item())
        tgt_a = int(tgtB_for_A[b, k].item())
        tgt_b = int(tgtA_for_B[b, k].item())

        from_a = cur_a == int(cand.idxA)
        from_b = cur_b == int(cand.idxB)
        if from_a and not from_b:
            return "A"
        if from_b and not from_a:
            return "B"
        if from_a and tgt_a == int(cand.idxB):
            return "A"
        if from_b and tgt_b == int(cand.idxA):
            return "B"
        if from_a:
            return "A"
        if from_b:
            return "B"
        return None

    def _cooldown_edge_region_target(cand: ConnectionCandidate, side: str) -> None:
        if edge_region_cooldown <= 0:
            return
        b = int(cand.b)
        if side == "A" and int(cand.idxB) >= 0:
            ban_tgtB_for_A[b] = int(cand.idxB)
            ban_cdA[b] = edge_region_cooldown + 1
        elif side == "B" and int(cand.idxA) >= 0:
            ban_tgtA_for_B[b] = int(cand.idxA)
            ban_cdB[b] = edge_region_cooldown + 1

    def _clear_edge_region_slot_hit(cand: ConnectionCandidate) -> None:
        side = _candidate_slot_side(cand)
        if side is None:
            return
        b = int(cand.b)
        k = int(cand.k)
        if side == "A":
            edge_region_hitA[b, k] = 0
        else:
            edge_region_hitB[b, k] = 0

    def _handle_edge_region_skip(cand: ConnectionCandidate) -> None:
        nonlocal connection_edge_region_retargets, connection_edge_region_escapes
        if edge_region_retarget_threshold <= 0 and edge_region_escape_threshold <= 0:
            return
        side = _candidate_slot_side(cand)
        if side is None:
            return

        b = int(cand.b)
        k = int(cand.k)
        hit = edge_region_hitA if side == "A" else edge_region_hitB      # 이 slot이 failed edge region에 걸린 누적 횟수
        target = tgtB_for_A if side == "A" else tgtA_for_B               # 현재 CONNECT 중인 반대편 트리의 target node idx
        seg = segA if side == "A" else segB                              # 현재 target을 향해 연속으로 진행한 connect step 수
        stagn = stagnA if side == "A" else stagnB                        # 이 slot의 정체(stagnation) 카운터
        mode = modeA if side == "A" else modeB                           # slot 동작 모드: 0=CONNECT, 1=ESCAPE

        hit[b, k] += 1
        n_hits = int(hit[b, k].item())
        _cooldown_edge_region_target(cand, side)

        if edge_region_escape_threshold > 0 and n_hits >= edge_region_escape_threshold:
            target[b, k] = -1
            seg[b, k] = 0
            stagn[b, k] = 0
            mode[b, k] = 1
            hit[b, k] = 0
            connection_edge_region_escapes += 1
            return

        if edge_region_retarget_threshold > 0 and (n_hits % edge_region_retarget_threshold) == 0:
            target[b, k] = -1
            seg[b, k] = 0
            stagn[b, k] = 0
            connection_edge_region_retargets += 1

    def _make_success_result(
        *,
        path_lp: torch.Tensor,
        b: int,
        k: int,
        idxA: int,
        idxB: int,
        iter_idx: int,
        connection_reason: str,
    ) -> Optional[PlanResult]:
        if trace_recorder is not None:
            trace_recorder.record_path(
                stage="lazy_projected",
                iter_idx=iter_idx,
                batch_idx=b,
                q=path_lp,
                reason=connection_reason,
            )

        planning_time_sec = time.time() - t0
        smoothing_time_sec = 0.0
        shortcut_stats = None
        if bool(getattr(cfg, "shortcut_smoothing", False)):
            t_smooth0 = time.time()
            smooth_seed = None if getattr(cfg, "seed", None) is None else int(cfg.seed) + int(b)
            path_lp, shortcut_stats = shortcut_smooth_path(
                path_lp,
                projector=projector,
                checker=checker,
                edge_checker=edge_checker,
                iters=int(getattr(cfg, "shortcut_smoothing_iters", 0)),
                step_q=float(cfg.edge_step_q),
                max_steps=int(cfg.edge_max_steps),
                min_skip=int(getattr(cfg, "shortcut_smoothing_min_skip", 1)),
                seed=smooth_seed,
                use_batch_projection=True,
            )
            smoothing_time_sec += time.time() - t_smooth0

        spline_stats = None
        if bool(getattr(cfg, "spline_interpolation", False)):
            t_spline0 = time.time()
            path_lp, spline_stats = cubic_spline_interpolate_and_validate_path(
                path_lp,
                projector=projector,
                checker=checker,
                edge_checker=edge_checker,
                joint_limits=joint_limits,
                step_q=float(getattr(cfg, "spline_step_q", cfg.edge_step_q)),
                max_steps_per_segment=int(getattr(cfg, "spline_max_steps_per_segment", cfg.edge_max_steps)),
                max_points=int(getattr(cfg, "spline_max_points", 2048)),
                fallback_to_input=bool(getattr(cfg, "spline_fallback_to_input", True)),
                use_batch_projection=True,
            )
            smoothing_time_sec += time.time() - t_spline0

        topp_time_sec = 0.0
        trajectory = None
        if bool(getattr(cfg, "topp_enable", False)):
            t_topp0 = time.time()
            trajectory = topp_retime_path(
                path_lp,
                max_velocity=getattr(cfg, "topp_max_velocity", 1.0),
                max_acceleration=getattr(cfg, "topp_max_acceleration", 2.0),
                output_dt=float(getattr(cfg, "topp_output_dt", 0.2)),
                max_duration_sec=getattr(cfg, "topp_max_duration_sec", 10.0),
                safety_scale=float(getattr(cfg, "topp_safety_scale", 1.05)),
                max_iterations=int(getattr(cfg, "topp_max_iterations", 20)),
            )
            path_lp = trajectory.q
            topp_time_sec = time.time() - t_topp0
        total_time_sec = time.time() - t0

        if trace_recorder is not None:
            trace_recorder.record_path(
                stage="final",
                iter_idx=iter_idx,
                batch_idx=b,
                q=path_lp,
                reason=connection_reason,
            )
            trace_recorder.record_connection(
                iter_idx=iter_idx,
                batch_idx=int(b),
                idx_a=int(idxA),
                idx_b=int(idxB),
            )

        extra = {
            "reason": "success",
            "connection_reason": str(connection_reason),
            "escape_fuse_trees": bool(getattr(cfg, "escape_fuse_trees", False)),
            "tree_storage_side_batch": False,
            "state_batch_shape": [2, B, Kmax],
            "connection_attempt_total": int(connection_attempt_total),
            "connection_attempt_success": int(connection_attempt_success),
            "connection_lazy_failures": int(connection_lazy_failures),
            "connection_region_skips": int(connection_region_skips),
            "connection_edge_region_skips": int(connection_edge_region_skips),
            "connection_segment_precheck_failures": int(connection_segment_precheck_failures),
            "connection_lazy_projection_failures": int(connection_lazy_projection_failures),
            "connection_lazy_edge_failures": int(connection_lazy_edge_failures),
            "connection_lazy_batch_calls": int(connection_lazy_batch_calls),
            "connection_lazy_batch_candidates": int(connection_lazy_batch_candidates),
            "connection_lazy_prealloc_overflows": int(connection_lazy_prealloc_overflows),
            "connection_lazy_prealloc_max_candidates": int(prealloc.lazy_max_paths),
            "connection_lazy_prealloc_max_path_points": int(prealloc.lazy_max_points_per_path),
            "connection_banned_branch_skips": int(connection_banned_branch_skips),
            "connection_lazy_edge_banned_nodes": int(connection_lazy_edge_banned_nodes),
            "connection_lazy_edge_single_banned_nodes": int(connection_lazy_edge_single_banned_nodes),
            "connection_lazy_edge_single_ban_fallbacks": int(connection_lazy_edge_single_ban_fallbacks),
            "connection_lazy_edge_subtree_ban_max_nodes": int(edge_subtree_ban_max_nodes),
            "connection_lazy_edge_connector_failures": int(connection_lazy_edge_connector_failures),
            "connection_banned_current_resets": int(connection_banned_current_resets),
            "connection_edge_region_retargets": int(connection_edge_region_retargets),
            "connection_edge_region_escapes": int(connection_edge_region_escapes),
            "failed_connection_regions_recorded": int(failed_connection_regions_recorded),
            "failed_edge_collision_regions_recorded": int(failed_edge_collision_regions_recorded),
            "failed_connection_region_radius": float(fail_region_radius),
            "failed_connection_region_max": int(fail_region_max),
            "failed_edge_region_radius": float(edge_fail_region_radius),
            "failed_edge_region_max": int(edge_fail_region_max),
            "connection_attempts_by_reason": dict(connection_attempts_by_reason),
            "connection_lazy_failures_by_reason": dict(connection_lazy_failures_by_reason),
            "batch_B": B,
            "block_K": Kmax,
            "winner_b": int(b),
            "winner_k": int(k),
            "tree_start": treeA,
            "tree_goal": treeB,
            "q_start_proj": q_start_proj,
            "q_goal_proj": q_goal_proj,
        }
        if shortcut_stats is not None:
            extra["shortcut_smoothing"] = shortcut_stats.to_dict()
        if spline_stats is not None:
            extra["spline_interpolation"] = spline_stats.to_dict()
        if trajectory is not None:
            extra["trajectory"] = {
                "t": trajectory.t,
                "qdot": trajectory.qdot,
                "qddot": trajectory.qddot,
                "stats": trajectory.stats_dict(),
            }
            extra["topp_time_sec"] = float(topp_time_sec)

        stats = PlanStats(
            iters=iter_idx + 1,
            nodes_A=int(treeA.n_nodes.max().item()),
            nodes_B=int(treeB.n_nodes.max().item()),
            ts_count=max(len(bk.spaces) for bk in banks),
            time_sec=planning_time_sec,
            extra=extra,
            smoothing_time_sec=smoothing_time_sec,
            total_time_sec=total_time_sec,
        )
        return PlanResult(
            success=True,
            path=path_to_list(path_lp),
            stats=stats,
            conn_idx_A=int(idxA),
            conn_idx_B=int(idxB),
        )

    def _try_finish_paths_batch(
        *,
        candidates: List[ConnectionCandidate],
        iter_idx: int,
        connection_reason: str,
    ) -> Optional[PlanResult]:
        nonlocal connection_attempt_total, connection_attempt_success, connection_lazy_failures
        nonlocal connection_region_skips, connection_segment_precheck_failures
        nonlocal connection_edge_region_skips
        nonlocal connection_lazy_projection_failures, connection_lazy_edge_failures
        nonlocal connection_lazy_batch_calls, connection_lazy_batch_candidates
        nonlocal connection_lazy_prealloc_overflows
        nonlocal connection_banned_branch_skips
        if not candidates:
            return None

        connection_attempt_total += len(candidates)
        connection_attempts_by_reason[connection_reason] = (
            connection_attempts_by_reason.get(connection_reason, 0) + len(candidates)
        )

        region_candidates: List[ConnectionCandidate] = []
        for cand in candidates:
            if trace_recorder is not None:
                trace_recorder.record_path(
                    stage="raw",
                    iter_idx=iter_idx,
                    batch_idx=cand.b,
                    q=cand.path,
                    reason=connection_reason,
                )
            if _candidate_hits_banned_branch(cand):
                connection_banned_branch_skips += 1
                if trace_recorder is not None:
                    trace_recorder.record_connection_attempt(
                        iter_idx=iter_idx,
                        batch_idx=int(cand.b),
                        slot_idx=int(cand.k),
                        idxA=int(cand.idxA),
                        idxB=int(cand.idxB),
                        raw_path_len=int(cand.path.shape[0]),
                        lazy_projection_success=False,
                        reason=f"{connection_reason}_banned_branch_skipped",
                    )
                continue
            if _hits_failed_connection_region(b=int(cand.b), idxA=int(cand.idxA), idxB=int(cand.idxB)):
                connection_region_skips += 1
                if trace_recorder is not None:
                    trace_recorder.record_connection_attempt(
                        iter_idx=iter_idx,
                        batch_idx=int(cand.b),
                        slot_idx=int(cand.k),
                        idxA=int(cand.idxA),
                        idxB=int(cand.idxB),
                        raw_path_len=int(cand.path.shape[0]),
                        lazy_projection_success=False,
                        reason=f"{connection_reason}_failed_region_skipped",
                    )
                continue
            edge_region_hit_idx = _failed_edge_collision_region_hit_edge(cand)
            if edge_region_hit_idx >= 0:
                connection_edge_region_skips += 1
                _record_lazy_edge_branch_ban(cand, int(edge_region_hit_idx))
                _handle_edge_region_skip(cand)
                if trace_recorder is not None:
                    trace_recorder.record_connection_attempt(
                        iter_idx=iter_idx,
                        batch_idx=int(cand.b),
                        slot_idx=int(cand.k),
                        idxA=int(cand.idxA),
                        idxB=int(cand.idxB),
                        raw_path_len=int(cand.path.shape[0]),
                        lazy_projection_success=False,
                        reason=f"{connection_reason}_failed_edge_region_skipped_edge={int(edge_region_hit_idx)}",
                    )
                continue
            _clear_edge_region_slot_hit(cand)
            region_candidates.append(cand)

        lazy_candidates = region_candidates
        if bool(getattr(cfg, "connection_segment_precheck", True)) and lazy_candidates:
            n_lazy = len(lazy_candidates)
            use_pair_scratch = prealloc.lazy_scratch_fits(
                n_paths=n_lazy,
                total_points=2 * n_lazy,
                total_edges=0,
                device=device,
                dtype=dtype,
            )
            endpoint_parts: List[torch.Tensor] = []
            for cand in lazy_candidates:
                endpoint_parts.append(treeA.q[cand.b, cand.idxA, :])
                endpoint_parts.append(treeB.q[cand.b, cand.idxB, :])
            if use_pair_scratch:
                assert prealloc.lazy_pair_q is not None
                q_pair_flat = prealloc.lazy_pair_q[: 2 * n_lazy, :]
                torch.stack(endpoint_parts, dim=0, out=q_pair_flat)
            else:
                if prealloc.lazy_pair_q is not None:
                    connection_lazy_prealloc_overflows += 1
                q_pair_flat = torch.stack(endpoint_parts, dim=0).contiguous()
            pr = projector.project_batch(q_pair_flat)
            pair_ok = pr.success_mask.bool().view(-1, 2).all(dim=1)
            if use_pair_scratch:
                assert prealloc.lazy_pair_ok is not None
                precheck_ok = prealloc.lazy_pair_ok[:n_lazy]
                precheck_ok.copy_(pair_ok)
            else:
                precheck_ok = pair_ok.clone()
            q_pair_proj = None
            if bool(pair_ok.any()):
                ok_idx = torch.nonzero(pair_ok, as_tuple=False).view(-1)
                q_pair_proj = pr.q_proj.contiguous().view(-1, 2, D)
                edge_res = edge_checker.check_edges_batch(
                    q_pair_proj[ok_idx, 0, :],
                    q_pair_proj[ok_idx, 1, :],
                    step_q=float(cfg.edge_step_q),
                    max_steps=int(cfg.edge_max_steps),
                )
                edge_in_col = edge_res.bool() if torch.is_tensor(edge_res) else edge_res.edge_in_collision.bool()
                precheck_ok[ok_idx] = ~edge_in_col

            next_candidates: List[ConnectionCandidate] = []
            for cand_i, cand in enumerate(lazy_candidates):
                if bool(precheck_ok[cand_i].item()):
                    next_candidates.append(cand)
                    continue
                connection_segment_precheck_failures += 1
                if q_pair_proj is not None and bool(pair_ok[cand_i].item()):
                    _record_failed_edge_collision_region(
                        b=int(cand.b),
                        q0=q_pair_proj[cand_i, 0, :],
                        q1=q_pair_proj[cand_i, 1, :],
                    )
                _record_failed_connection_region(b=int(cand.b), idxA=int(cand.idxA), idxB=int(cand.idxB))
                if trace_recorder is not None:
                    trace_recorder.record_connection_attempt(
                        iter_idx=iter_idx,
                        batch_idx=int(cand.b),
                        slot_idx=int(cand.k),
                        idxA=int(cand.idxA),
                        idxB=int(cand.idxB),
                        raw_path_len=int(cand.path.shape[0]),
                        lazy_projection_success=False,
                        reason=f"{connection_reason}_segment_precheck_failed",
                    )
            lazy_candidates = next_candidates

        if not lazy_candidates:
            return None

        connection_lazy_batch_calls += 1
        connection_lazy_batch_candidates += len(lazy_candidates)
        lazy_total_points = sum(int(cand.path.shape[0]) for cand in lazy_candidates)
        lazy_total_edges = sum(max(0, int(cand.path.shape[0]) - 1) for cand in lazy_candidates)
        if not prealloc.lazy_scratch_fits(
            n_paths=len(lazy_candidates),
            total_points=lazy_total_points,
            total_edges=lazy_total_edges,
            device=device,
            dtype=dtype,
        ):
            if prealloc.lazy_flat_q is not None:
                connection_lazy_prealloc_overflows += 1
        path_lps, fail_reasons, fail_edge_indices, fail_edge_pairs = _lazy_project_paths_with_edge_check_batch(
            [cand.path for cand in lazy_candidates],
            projector,
            edge_checker,
            prealloc=prealloc,
        )

        selected: Optional[Tuple[torch.Tensor, ConnectionCandidate]] = None
        selected_path_len: Optional[int] = None
        for cand, path_lp, fail_reason, fail_edge_idx, fail_edge_pair in zip(
            lazy_candidates,
            path_lps,
            fail_reasons,
            fail_edge_indices,
            fail_edge_pairs,
        ):
            if path_lp is None:
                connection_lazy_failures += 1
                connection_lazy_failures_by_reason[connection_reason] = (
                    connection_lazy_failures_by_reason.get(connection_reason, 0) + 1
                )
                if fail_reason == "edge":
                    connection_lazy_edge_failures += 1
                    _record_lazy_edge_branch_ban(cand, int(fail_edge_idx))
                    edge_pair = fail_edge_pair
                    if edge_pair is not None:
                        _record_failed_edge_collision_region(
                            b=int(cand.b),
                            q0=edge_pair[0],
                            q1=edge_pair[1],
                        )
                else:
                    connection_lazy_projection_failures += 1
                _record_failed_connection_region(b=int(cand.b), idxA=int(cand.idxA), idxB=int(cand.idxB))
                if trace_recorder is not None:
                    trace_recorder.record_connection_attempt(
                        iter_idx=iter_idx,
                        batch_idx=int(cand.b),
                        slot_idx=int(cand.k),
                        idxA=int(cand.idxA),
                        idxB=int(cand.idxB),
                        raw_path_len=int(cand.path.shape[0]),
                        lazy_projection_success=False,
                        reason=f"{connection_reason}_lazy_{fail_reason}_failed_edge={int(fail_edge_idx)}",
                    )
                continue

            connection_attempt_success += 1
            if trace_recorder is not None:
                trace_recorder.record_connection_attempt(
                    iter_idx=iter_idx,
                    batch_idx=int(cand.b),
                    slot_idx=int(cand.k),
                    idxA=int(cand.idxA),
                    idxB=int(cand.idxB),
                    raw_path_len=int(cand.path.shape[0]),
                    lazy_projection_success=True,
                    reason=connection_reason,
                )
            path_len = int(path_lp.shape[0])
            if selected is None or selected_path_len is None or path_len < selected_path_len:
                selected = (path_lp, cand)
                selected_path_len = path_len

        if selected is None:
            return None

        path_lp, cand = selected
        return _make_success_result(
            path_lp=path_lp,
            b=int(cand.b),
            k=int(cand.k),
            idxA=int(cand.idxA),
            idxB=int(cand.idxB),
            iter_idx=int(iter_idx),
            connection_reason=connection_reason,
        )

    def _distance_batch(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
        diff = prealloc.connect_diff[:, :Kmax, :]
        torch.sub(q_to, q_from, out=diff)
        dist = prealloc.connect_dist[:, :Kmax]
        torch.linalg.vector_norm(diff, dim=-1, out=dist)
        return dist

    def _get_curvature_pad() -> torch.Tensor:
        nonlocal curvature_pad_cache, curvature_pad_counts
        counts = tuple(len(bank.spaces) for bank in banks)
        if (
            curvature_pad_cache is not None
            and curvature_pad_counts == counts
            and curvature_pad_cache.device == device
            and curvature_pad_cache.dtype == dtype
        ):
            return curvature_pad_cache

        smax = max(1, max(counts, default=1))
        curv_pad = torch.full((B, smax), float("inf"), device=device, dtype=dtype)
        for b_t, bank in enumerate(banks):
            curv = bank.get_curvatures_tensor(device=device, dtype=dtype)
            if curv.numel() > 0:
                curv_pad[b_t, : int(curv.numel())] = curv
        curvature_pad_cache = curv_pad
        curvature_pad_counts = counts
        return curv_pad

    def _node_curvature_batch(*, tree_from: TreeBatchGPU, node_idx: torch.Tensor) -> torch.Tensor:
        b_idx = prealloc.b_idx(Kmax)
        ts_idx = tree_from.ts_id[b_idx, node_idx.clamp_min(0)]
        curv_pad = _get_curvature_pad()
        smax = int(curv_pad.shape[1])
        out = prealloc.connect_node_curvature[:, :Kmax]
        out.fill_(float("inf"))
        valid = (ts_idx >= 0) & (ts_idx < smax)
        safe_ts_idx = ts_idx.clamp(min=0, max=max(0, smax - 1))
        out.copy_(curv_pad.gather(1, safe_ts_idx))
        out.masked_fill_(~valid, float("inf"))
        return out

    def _try_bridge_candidates(
        *,
        tree_from: TreeBatchGPU,
        cur_from: torch.Tensor,
        target_other: torch.Tensor,
        q_target_other: torch.Tensor,
        not_reached: torch.Tensor,
        from_start_tree: bool,
        iter_idx: int,
    ) -> Optional[PlanResult]:
        if not bool(not_reached.any()):
            return None

        near_threshold = float(getattr(cfg, "connect_bridge_near_threshold", 0.0))
        if near_threshold <= 0.0:
            near_threshold = 3.0 * float(cfg.step_size)
        curvature_threshold = float(getattr(cfg, "connect_bridge_curvature_threshold", 2.5))
        max_attempts = max(1, int(getattr(cfg, "connect_bridge_max_attempts_per_iter", 4)))

        tree_q, _ = tree_from.get_nodes()
        b_idx = prealloc.b_idx(Kmax)
        cur_safe = cur_from.clamp_min(0)
        q_cur = tree_q[b_idx, cur_safe, :]
        dist = _distance_batch(q_cur, q_target_other)
        curv = _node_curvature_batch(tree_from=tree_from, node_idx=cur_from)
        close = prealloc.connect_close[:, :Kmax]
        aux = prealloc.connect_aux_bool[:, :Kmax]
        torch.le(dist, near_threshold, out=close)
        close.logical_and_(not_reached)
        torch.ge(target_other, 0, out=aux)
        close.logical_and_(aux)
        torch.le(curv, curvature_threshold, out=aux)
        close.logical_and_(aux)
        hits = torch.nonzero(close, as_tuple=False)
        if hits.numel() == 0:
            return None

        attempts = 0
        candidates: List[ConnectionCandidate] = []
        for b_t, k_t in hits.tolist():
            idx_from = int(cur_from[b_t, k_t].item())
            idx_other = int(target_other[b_t, k_t].item())
            if idx_from < 0 or idx_other < 0:
                continue
            if attempts >= max_attempts:
                break
            attempts += 1

            if from_start_tree:
                idxA = idx_from
                idxB = idx_other
            else:
                idxA = idx_other
                idxB = idx_from

            candidates.append(_extract_connection_candidate(treeA, treeB, int(b_t), int(k_t), int(idxA), int(idxB)))

        return _try_finish_paths_batch(
            candidates=candidates,
            iter_idx=int(iter_idx),
            connection_reason="direct_bridge",
        )

    def _try_new_node_connection_candidates(
        *,
        tree_from: TreeBatchGPU,
        other: TreeBatchGPU,
        node_idx: torch.Tensor,
        new_mask: torch.Tensor,
        from_start_tree: bool,
        discard_overlap: bool,
        iter_idx: int,
        reached_reason: str,
        bridge_reason: str,
    ) -> Optional[PlanResult]:
        if not bool(new_mask.any()):
            return None

        tree_q, _ = tree_from.get_nodes()
        b_idx = prealloc.b_idx(Kmax)
        q_cur = tree_q[b_idx, node_idx.clamp_min(0), :]
        target_idx = _select_connect_target_idx_batch(
            other=other,
            q_query=q_cur,
            active_mask=new_mask,
            discard_overlap=discard_overlap,
            prealloc=prealloc,
        )

        other_q, _ = other.get_nodes()
        q_target = other_q[b_idx, target_idx.clamp_min(0), :]
        dist = _distance_batch(q_cur, q_target)
        target_valid = prealloc.connect_target_valid[:, :Kmax]
        torch.ge(target_idx, 0, out=target_valid)
        target_valid.logical_and_(new_mask)
        reached = prealloc.connect_reached[:, :Kmax]
        torch.le(dist, float(cfg.goal_threshold), out=reached)
        reached.logical_and_(target_valid)
        hits = torch.nonzero(reached, as_tuple=False)
        candidates: List[ConnectionCandidate] = []
        for b_t, k_t in hits.tolist():
            idx_from = int(node_idx[b_t, k_t].item())
            idx_other = int(target_idx[b_t, k_t].item())
            if idx_from < 0 or idx_other < 0:
                continue

            if from_start_tree:
                idxA = idx_from
                idxB = idx_other
            else:
                idxA = idx_other
                idxB = idx_from

            candidates.append(_extract_connection_candidate(treeA, treeB, int(b_t), int(k_t), int(idxA), int(idxB)))

        result = _try_finish_paths_batch(
            candidates=candidates,
            iter_idx=int(iter_idx),
            connection_reason=reached_reason,
        )
        if result is not None:
            return result

        if bool(getattr(cfg, "connect_bridge_enable", True)):
            near_threshold = float(getattr(cfg, "connect_bridge_near_threshold", 0.0))
            if near_threshold <= 0.0:
                near_threshold = 3.0 * float(cfg.step_size)
            curvature_threshold = float(getattr(cfg, "connect_bridge_curvature_threshold", 2.5))
            max_attempts = max(1, int(getattr(cfg, "connect_bridge_max_attempts_per_iter", 4)))
            curv = _node_curvature_batch(tree_from=tree_from, node_idx=node_idx)
            close = prealloc.connect_close[:, :Kmax]
            aux = prealloc.connect_aux_bool[:, :Kmax]
            torch.le(dist, near_threshold, out=close)
            close.logical_and_(target_valid)
            torch.logical_not(reached, out=aux)
            close.logical_and_(aux)
            torch.le(curv, curvature_threshold, out=aux)
            close.logical_and_(aux)
            hits = torch.nonzero(close, as_tuple=False)
            attempts = 0
            candidates: List[ConnectionCandidate] = []
            for b_t, k_t in hits.tolist():
                idx_from = int(node_idx[b_t, k_t].item())
                idx_other = int(target_idx[b_t, k_t].item())
                if idx_from < 0 or idx_other < 0:
                    continue
                if attempts >= max_attempts:
                    break
                attempts += 1

                if from_start_tree:
                    idxA = idx_from
                    idxB = idx_other
                else:
                    idxA = idx_other
                    idxB = idx_from

                candidates.append(_extract_connection_candidate(treeA, treeB, int(b_t), int(k_t), int(idxA), int(idxB)))

            result = _try_finish_paths_batch(
                candidates=candidates,
                iter_idx=int(iter_idx),
                connection_reason=bridge_reason,
            )
            if result is not None:
                return result

        return None

    def _reset_connect_targets(
        *,
        reset: torch.Tensor,
        target: torch.Tensor,
        seg: torch.Tensor,
        stagn: torch.Tensor,
        ban_target: torch.Tensor,
        ban_cd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(reset.any()):
            return target, seg, stagn

        cooldown = max(0, int(getattr(cfg, "connect_bridge_reset_cooldown", 1)))
        if cooldown > 0:
            rows = torch.nonzero(reset.any(dim=1), as_tuple=False).view(-1)
            for b_t in rows.tolist():
                slots = torch.nonzero(reset[b_t], as_tuple=False).view(-1)
                if slots.numel() == 0:
                    continue
                tgt = int(target[b_t, int(slots[0].item())].item())
                if tgt >= 0:
                    ban_target[b_t] = tgt
                    ban_cd[b_t] = cooldown + 1

        target = torch.where(reset, prealloc.minus_one_long(Kmax), target)
        seg = torch.where(reset, prealloc.zero_long(Kmax), seg)
        stagn = torch.where(reset, prealloc.zero_long(Kmax), stagn)
        return target, seg, stagn

    for it in range(int(cfg.max_iters)):
        if (time.time() - t0) > float(cfg.time_limit_sec):
            break

        if (not bool(activeA.any())) and (not bool(activeB.any())):
            break

        ban_cdA.sub_(1).clamp_min_(0)
        ban_cdB.sub_(1).clamp_min_(0)
        ban_tgtB_for_A.copy_(torch.where(ban_cdA > 0, ban_tgtB_for_A, prealloc.minus_one_long_b))
        ban_tgtA_for_B.copy_(torch.where(ban_cdB > 0, ban_tgtA_for_B, prealloc.minus_one_long_b))

        _reset_slots_on_banned_current(
            tree=treeA,
            active=activeA,
            cur=curA,
            mode=modeA,
            seg=segA,
            stagn=stagnA,
            target=tgtB_for_A,
        )
        _reset_slots_on_banned_current(
            tree=treeB,
            active=activeB,
            cur=curB,
            mode=modeB,
            seg=segB,
            stagn=stagnB,
            target=tgtA_for_B,
        )

        connA = activeA & (modeA == 0)
        connB = activeB & (modeB == 0)
        use_overlap_discard = bool(getattr(cfg, "enable_overlap_discard", True))
        ban_activeA = (ban_cdA > 0).view(B, 1)
        ban_activeB = (ban_cdB > 0).view(B, 1)

        retarget_overlapA = connA & _target_hits_overlap_nodes(tree=treeB, target_idx=tgtB_for_A, prealloc=prealloc) if use_overlap_discard else prealloc.zero_bool(Kmax)
        retarget_overlapB = connB & _target_hits_overlap_nodes(tree=treeA, target_idx=tgtA_for_B, prealloc=prealloc) if use_overlap_discard else prealloc.zero_bool(Kmax)
        retarget_branch_bannedA = connA & _node_blocked_mask(treeB, tgtB_for_A)
        retarget_branch_bannedB = connB & _node_blocked_mask(treeA, tgtA_for_B)
        retarget_bannedA = connA & ban_activeA & (tgtB_for_A == ban_tgtB_for_A.view(B, 1))
        retarget_bannedB = connB & ban_activeB & (tgtA_for_B == ban_tgtA_for_B.view(B, 1))

        need_retA = connA & (
            (tgtB_for_A < 0)
            | (segA >= int(cfg.connect_max_steps))
            | retarget_overlapA
            | retarget_bannedA
            | retarget_branch_bannedA
        )
        need_retB = connB & (
            (tgtA_for_B < 0)
            | (segB >= int(cfg.connect_max_steps))
            | retarget_overlapB
            | retarget_bannedB
            | retarget_branch_bannedB
        )

        if bool(need_retA.any()):
            qA_all, _ = treeA.get_nodes()
            b_idx = prealloc.b_idx(Kmax)
            q_curA = qA_all[b_idx, curA.clamp_min(0), :]
            nn_idx = _select_connect_target_idx_batch(
                other=treeB,
                q_query=q_curA,
                active_mask=need_retA,
                discard_overlap=use_overlap_discard,
                exclude_idx=torch.where(
                    ban_activeA,
                    ban_tgtB_for_A.view(B, 1).expand(B, Kmax),
                    prealloc.minus_one_long(Kmax),
                ),
                prealloc=prealloc,
            )
            blockedA = need_retA & ban_activeA & (nn_idx == ban_tgtB_for_A.view(B, 1))
            no_targetA = need_retA & (nn_idx < 0)
            tgtB_for_A.copy_(torch.where(blockedA, prealloc.minus_one_long(Kmax), torch.where(need_retA, nn_idx, tgtB_for_A)))
            segA.copy_(torch.where(need_retA, prealloc.zero_long(Kmax), segA)) # con step reset
            stagnA.copy_(torch.where(need_retA, prealloc.zero_long(Kmax), stagnA))
            modeA.copy_(torch.where(blockedA | no_targetA, prealloc.one_long(Kmax), modeA))

        if bool(need_retB.any()):
            qB_all, _ = treeB.get_nodes()
            b_idx = prealloc.b_idx(Kmax)
            q_curB = qB_all[b_idx, curB.clamp_min(0), :]
            nn_idx = _select_connect_target_idx_batch(
                other=treeA,
                q_query=q_curB,
                active_mask=need_retB,
                discard_overlap=use_overlap_discard,
                exclude_idx=torch.where(
                    ban_activeB,
                    ban_tgtA_for_B.view(B, 1).expand(B, Kmax),
                    prealloc.minus_one_long(Kmax),
                ),
                prealloc=prealloc,
            )
            blockedB = need_retB & ban_activeB & (nn_idx == ban_tgtA_for_B.view(B, 1))
            no_targetB = need_retB & (nn_idx < 0)
            tgtA_for_B.copy_(torch.where(blockedB, prealloc.minus_one_long(Kmax), torch.where(need_retB, nn_idx, tgtA_for_B)))
            segB.copy_(torch.where(need_retB, prealloc.zero_long(Kmax), segB))
            stagnB.copy_(torch.where(need_retB, prealloc.zero_long(Kmax), stagnB))
            modeB.copy_(torch.where(blockedB | no_targetB, prealloc.one_long(Kmax), modeB))

        connA = activeA & (modeA == 0)
        connB = activeB & (modeB == 0)

        qB_all, _ = treeB.get_nodes()
        qA_all, _ = treeA.get_nodes()
        b_idx = prealloc.b_idx(Kmax)
        q_tgt_for_A = qB_all[b_idx, tgtB_for_A.clamp_min(0), :]
        q_tgt_for_B = qA_all[b_idx, tgtA_for_B.clamp_min(0), :]

        stepA, stepB = _connect_two_trees_one_step_with_state(
            tree_a=treeA,
            tree_b=treeB,
            banks=banks,
            cur_idx_a=curA,
            cur_idx_b=curB,
            q_target_a=q_tgt_for_A,
            q_target_b=q_tgt_for_B,
            target_idx_other_a=tgtB_for_A,
            target_idx_other_b=tgtA_for_B,
            cfg=cfg,
            checker=checker,
            edge_checker=edge_checker,
            projector=projector,
            iter_idx=it,
            mask_a=connA,
            mask_b=connB,
            projector_cache=projector_cache,
            prealloc=prealloc,
        )

        curA.copy_(torch.where(stepA.advanced, stepA.new_idx, curA))
        curB.copy_(torch.where(stepB.advanced, stepB.new_idx, curB))
        segA.copy_(torch.where(stepA.advanced, segA + 1, segA))
        segB.copy_(torch.where(stepB.advanced, segB + 1, segB))

        stagnation_steps = int(getattr(cfg, "connect_stagnation_steps", 0))
        stagnation_progress = float(getattr(cfg, "connect_stagnation_progress_ratio", 0.0)) * float(cfg.step_size)
        stagnation_escape = bool(getattr(cfg, "connect_stagnation_escape", True))
        if stagnation_steps > 0 and stagnation_progress > 0.0:
            conn_advA = connA & stepA.advanced
            conn_advB = connB & stepB.advanced
            low_progA = conn_advA & (stepA.progress < stagnation_progress)
            low_progB = conn_advB & (stepB.progress < stagnation_progress)
            stagnA_next = torch.where(conn_advA, torch.where(low_progA, stagnA + 1, prealloc.zero_long(Kmax)), stagnA)
            stagnB_next = torch.where(conn_advB, torch.where(low_progB, stagnB + 1, prealloc.zero_long(Kmax)), stagnB)
            stagn_hitA = conn_advA & (stagnA_next >= stagnation_steps)
            stagn_hitB = conn_advB & (stagnB_next >= stagnation_steps)
            stagnA.copy_(torch.where(stagn_hitA, prealloc.zero_long(Kmax), stagnA_next))
            stagnB.copy_(torch.where(stagn_hitB, prealloc.zero_long(Kmax), stagnB_next))
            segA.copy_(torch.where(stagn_hitA, prealloc.zero_long(Kmax), segA))
            segB.copy_(torch.where(stagn_hitB, prealloc.zero_long(Kmax), segB))
            tgtB_for_A.copy_(torch.where(stagn_hitA, prealloc.minus_one_long(Kmax), tgtB_for_A))
            tgtA_for_B.copy_(torch.where(stagn_hitB, prealloc.minus_one_long(Kmax), tgtA_for_B))
            if stagnation_escape:
                modeA.copy_(torch.where(stagn_hitA, prealloc.one_long(Kmax), modeA))
                modeB.copy_(torch.where(stagn_hitB, prealloc.one_long(Kmax), modeB))

        trappedA = connA & (~stepA.advanced)
        trappedB = connB & (~stepB.advanced)
        modeA.copy_(torch.where(trappedA, prealloc.one_long(Kmax), modeA))
        modeB.copy_(torch.where(trappedB, prealloc.one_long(Kmax), modeB))
        stagnA.copy_(torch.where(trappedA, prealloc.zero_long(Kmax), stagnA))
        stagnB.copy_(torch.where(trappedB, prealloc.zero_long(Kmax), stagnB))

        advancedA = connA & stepA.advanced
        advancedB = connB & stepB.advanced

        result = _try_new_node_connection_candidates(
            tree_from=treeA,
            other=treeB,
            node_idx=stepA.new_idx,
            new_mask=advancedA,
            from_start_tree=True,
            discard_overlap=use_overlap_discard,
            iter_idx=it,
            reached_reason="connect_reached",
            bridge_reason="connect_direct_bridge",
        )
        if result is not None:
            return result

        result = _try_new_node_connection_candidates(
            tree_from=treeB,
            other=treeA,
            node_idx=stepB.new_idx,
            new_mask=advancedB,
            from_start_tree=False,
            discard_overlap=use_overlap_discard,
            iter_idx=it,
            reached_reason="connect_reached",
            bridge_reason="connect_direct_bridge",
        )
        if result is not None:
            return result

        reached_any = (stepA.reached & connA) | (stepB.reached & connB)
        if bool(reached_any.any()):
            reached_hits = torch.nonzero(reached_any, as_tuple=False)
            candidates: List[ConnectionCandidate] = []
            for b_t, k_t in reached_hits.tolist():
                b = int(b_t)
                k = int(k_t)
                if bool(stepA.reached[b, k].item()):
                    idxA = int(stepA.reached_idx[b, k].item())
                    idxB = int(stepA.target_idx_other[b, k].item())
                else:
                    idxB = int(stepB.reached_idx[b, k].item())
                    idxA = int(stepB.target_idx_other[b, k].item())
                if idxA < 0 or idxB < 0:
                    continue
                candidates.append(_extract_connection_candidate(treeA, treeB, b, k, idxA, idxB))
            result = _try_finish_paths_batch(
                candidates=candidates,
                iter_idx=it,
                connection_reason="success",
            )
            if result is None:
                cooldown = max(0, int(getattr(cfg, "failed_connection_cooldown", 0)))
                for cand in candidates:
                    if cooldown > 0:
                        ban_tgtB_for_A[cand.b] = cand.idxB
                        ban_tgtA_for_B[cand.b] = cand.idxA
                        ban_cdA[cand.b] = cooldown + 1
                        ban_cdB[cand.b] = cooldown + 1
                    modeA[cand.b, cand.k] = 1
                    modeB[cand.b, cand.k] = 1
                    segA[cand.b, cand.k] = 0
                    segB[cand.b, cand.k] = 0
                    stagnA[cand.b, cand.k] = 0
                    stagnB[cand.b, cand.k] = 0
                    tgtB_for_A[cand.b, cand.k] = -1
                    tgtA_for_B[cand.b, cand.k] = -1
                _record_iter_summary(it)
                continue
            return result

        if bool(getattr(cfg, "connect_bridge_enable", True)):
            not_reachedA = advancedA & (~stepA.reached) & (modeA == 0)
            not_reachedB = advancedB & (~stepB.reached) & (modeB == 0)

            result = _try_bridge_candidates(
                tree_from=treeA,
                cur_from=curA,
                target_other=tgtB_for_A,
                q_target_other=q_tgt_for_A,
                not_reached=not_reachedA,
                from_start_tree=True,
                iter_idx=it,
            )
            if result is not None:
                return result

            result = _try_bridge_candidates(
                tree_from=treeB,
                cur_from=curB,
                target_other=tgtA_for_B,
                q_target_other=q_tgt_for_B,
                not_reached=not_reachedB,
                from_start_tree=False,
                iter_idx=it,
            )
            if result is not None:
                return result

            tgt_next, seg_next, stagn_next = _reset_connect_targets(
                reset=not_reachedA,
                target=tgtB_for_A,
                seg=segA,
                stagn=stagnA,
                ban_target=ban_tgtB_for_A,
                ban_cd=ban_cdA,
            )
            tgtB_for_A.copy_(tgt_next)
            segA.copy_(seg_next)
            stagnA.copy_(stagn_next)
            tgt_next, seg_next, stagn_next = _reset_connect_targets(
                reset=not_reachedB,
                target=tgtA_for_B,
                seg=segB,
                stagn=stagnB,
                ban_target=ban_tgtA_for_B,
                ban_cd=ban_cdB,
            )
            tgtA_for_B.copy_(tgt_next)
            segB.copy_(seg_next)
            stagnB.copy_(stagn_next)

        needEscA = activeA & (modeA == 1)
        needEscB = activeB & (modeB == 1)

        if bool(getattr(cfg, "escape_fuse_trees", False)):
            (
                activeA_next, modeA_next, curA_next, segA_next, tgtB_for_A_next, _escape_okA,
                activeB_next, modeB_next, curB_next, segB_next, tgtA_for_B_next, _escape_okB,
                escape_result,
            ) = _escape_two_trees(
                tree_a=treeA,
                tree_b=treeB,
                active_a=activeA,
                active_b=activeB,
                mode_a=modeA,
                mode_b=modeB,
                cur_a=curA,
                cur_b=curB,
                seg_a=segA,
                seg_b=segB,
                tgt_other_a=tgtB_for_A,
                tgt_other_b=tgtA_for_B,
                need_esc_a=needEscA,
                need_esc_b=needEscB,
                valid_b=valid_b,
                banks=banks,
                joint_limits=joint_limits,
                cfg=cfg,
                checker=checker,
                edge_checker=edge_checker,
                projector=projector,
                iter_idx=it,
                _sample_ts_ids_from_banks_batch=_sample_ts_ids_from_banks_batch,
                _gather_selected_ts_tensors=_gather_selected_ts_tensors,
                _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
                _apply_overlap_discard_qrand_batch=_apply_overlap_discard_qrand_batch,
                _select_escape_q_hint=_select_escape_q_hint,
                _extend_two_trees_one_step_from_parent=extend_two_trees_one_step_from_parent,
                nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
                generator=g,
                projector_cache=projector_cache,
                prealloc=prealloc,
                after_step_connect_a=lambda step_idx, step_ok, escape_step_idx: _try_new_node_connection_candidates(
                    tree_from=treeA,
                    other=treeB,
                    node_idx=step_idx,
                    new_mask=step_ok,
                    from_start_tree=True,
                    discard_overlap=use_overlap_discard,
                    iter_idx=it,
                    reached_reason="escape_reached",
                    bridge_reason="escape_direct_bridge",
                ),
                after_step_connect_b=lambda step_idx, step_ok, escape_step_idx: _try_new_node_connection_candidates(
                    tree_from=treeB,
                    other=treeA,
                    node_idx=step_idx,
                    new_mask=step_ok,
                    from_start_tree=False,
                    discard_overlap=use_overlap_discard,
                    iter_idx=it,
                    reached_reason="escape_reached",
                    bridge_reason="escape_direct_bridge",
                ),
            )
            activeA.copy_(activeA_next)
            modeA.copy_(modeA_next)
            curA.copy_(curA_next)
            segA.copy_(segA_next)
            tgtB_for_A.copy_(tgtB_for_A_next)
            activeB.copy_(activeB_next)
            modeB.copy_(modeB_next)
            curB.copy_(curB_next)
            segB.copy_(segB_next)
            tgtA_for_B.copy_(tgtA_for_B_next)
            if escape_result is not None:
                return escape_result
        else:
            (
                activeA_next, modeA_next, curA_next, segA_next, tgtB_for_A_next, _escape_okA, escape_result
            ) = _escape_one_tree(
                tree=treeA,
                other=treeB,
                active=activeA,
                mode=modeA,
                cur=curA,
                seg=segA,
                tgt_other=tgtB_for_A,
                needEsc=needEscA,
                valid_b=valid_b,
                banks=banks,
                joint_limits=joint_limits,
                cfg=cfg,
                checker=checker,
                edge_checker=edge_checker,
                projector=projector,
                iter_idx=it,
                _sample_ts_ids_from_banks_batch=_sample_ts_ids_from_banks_batch,
                _gather_selected_ts_tensors=_gather_selected_ts_tensors,
                _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
                _apply_overlap_discard_qrand_batch=_apply_overlap_discard_qrand_batch,
                _select_escape_q_hint=_select_escape_q_hint,
                _extend_one_step_from_parent=_extend_one_step_from_parent,
                nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
                generator=g,
                projector_cache=projector_cache,
                prealloc=prealloc,
                after_step_connect=lambda step_idx, step_ok, escape_step_idx: _try_new_node_connection_candidates(
                    tree_from=treeA,
                    other=treeB,
                    node_idx=step_idx,
                    new_mask=step_ok,
                    from_start_tree=True,
                    discard_overlap=use_overlap_discard,
                    iter_idx=it,
                    reached_reason="escape_reached",
                    bridge_reason="escape_direct_bridge",
                ),
            )
            activeA.copy_(activeA_next)
            modeA.copy_(modeA_next)
            curA.copy_(curA_next)
            segA.copy_(segA_next)
            tgtB_for_A.copy_(tgtB_for_A_next)
            if escape_result is not None:
                return escape_result

            (
                activeB_next, modeB_next, curB_next, segB_next, tgtA_for_B_next, _escape_okB, escape_result
            ) = _escape_one_tree(
                tree=treeB,
                other=treeA,
                active=activeB,
                mode=modeB,
                cur=curB,
                seg=segB,
                tgt_other=tgtA_for_B,
                needEsc=needEscB,
                valid_b=valid_b,
                banks=banks,
                joint_limits=joint_limits,
                cfg=cfg,
                checker=checker,
                edge_checker=edge_checker,
                projector=projector,
                iter_idx=it,
                _sample_ts_ids_from_banks_batch=_sample_ts_ids_from_banks_batch,
                _gather_selected_ts_tensors=_gather_selected_ts_tensors,
                _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
                _apply_overlap_discard_qrand_batch=_apply_overlap_discard_qrand_batch,
                _select_escape_q_hint=_select_escape_q_hint,
                _extend_one_step_from_parent=_extend_one_step_from_parent,
                nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
                generator=g,
                projector_cache=projector_cache,
                prealloc=prealloc,
                after_step_connect=lambda step_idx, step_ok, escape_step_idx: _try_new_node_connection_candidates(
                    tree_from=treeB,
                    other=treeA,
                    node_idx=step_idx,
                    new_mask=step_ok,
                    from_start_tree=False,
                    discard_overlap=use_overlap_discard,
                    iter_idx=it,
                    reached_reason="escape_reached",
                    bridge_reason="escape_direct_bridge",
                ),
            )
            activeB.copy_(activeB_next)
            modeB.copy_(modeB_next)
            curB.copy_(curB_next)
            segB.copy_(segB_next)
            tgtA_for_B.copy_(tgtA_for_B_next)
            if escape_result is not None:
                return escape_result

        stagnA.copy_(torch.where(activeA & (modeA == 0), stagnA, prealloc.zero_long(Kmax)))
        stagnB.copy_(torch.where(activeB & (modeB == 0), stagnB, prealloc.zero_long(Kmax)))
        edge_region_hitA.copy_(torch.where(activeA & (modeA == 0), edge_region_hitA, prealloc.zero_long(Kmax)))
        edge_region_hitB.copy_(torch.where(activeB & (modeB == 0), edge_region_hitB, prealloc.zero_long(Kmax)))
        _record_iter_summary(it)

    elapsed = time.time() - t0
    stats = PlanStats(
        iters=int(cfg.max_iters),
        nodes_A=int(treeA.n_nodes.max().item()),
        nodes_B=int(treeB.n_nodes.max().item()),
        ts_count=max(len(bk.spaces) for bk in banks),
        time_sec=elapsed,
        extra={
            "reason": "timeout_or_maxiters",
            "escape_fuse_trees": bool(getattr(cfg, "escape_fuse_trees", False)),
            "tree_storage_side_batch": False,
            "state_batch_shape": [2, B, Kmax],
            "connection_attempt_total": int(connection_attempt_total),
            "connection_attempt_success": int(connection_attempt_success),
            "connection_lazy_failures": int(connection_lazy_failures),
            "connection_region_skips": int(connection_region_skips),
            "connection_edge_region_skips": int(connection_edge_region_skips),
            "connection_segment_precheck_failures": int(connection_segment_precheck_failures),
            "connection_lazy_projection_failures": int(connection_lazy_projection_failures),
            "connection_lazy_edge_failures": int(connection_lazy_edge_failures),
            "connection_lazy_batch_calls": int(connection_lazy_batch_calls),
            "connection_lazy_batch_candidates": int(connection_lazy_batch_candidates),
            "connection_lazy_prealloc_overflows": int(connection_lazy_prealloc_overflows),
            "connection_lazy_prealloc_max_candidates": int(prealloc.lazy_max_paths),
            "connection_lazy_prealloc_max_path_points": int(prealloc.lazy_max_points_per_path),
            "connection_banned_branch_skips": int(connection_banned_branch_skips),
            "connection_lazy_edge_banned_nodes": int(connection_lazy_edge_banned_nodes),
            "connection_lazy_edge_single_banned_nodes": int(connection_lazy_edge_single_banned_nodes),
            "connection_lazy_edge_single_ban_fallbacks": int(connection_lazy_edge_single_ban_fallbacks),
            "connection_lazy_edge_subtree_ban_max_nodes": int(edge_subtree_ban_max_nodes),
            "connection_lazy_edge_connector_failures": int(connection_lazy_edge_connector_failures),
            "connection_banned_current_resets": int(connection_banned_current_resets),
            "connection_edge_region_retargets": int(connection_edge_region_retargets),
            "connection_edge_region_escapes": int(connection_edge_region_escapes),
            "failed_connection_regions_recorded": int(failed_connection_regions_recorded),
            "failed_edge_collision_regions_recorded": int(failed_edge_collision_regions_recorded),
            "failed_connection_region_radius": float(fail_region_radius),
            "failed_connection_region_max": int(fail_region_max),
            "failed_edge_region_radius": float(edge_fail_region_radius),
            "failed_edge_region_max": int(edge_fail_region_max),
            "connection_attempts_by_reason": dict(connection_attempts_by_reason),
            "connection_lazy_failures_by_reason": dict(connection_lazy_failures_by_reason),
            "batch_B": B,
            "block_K": Kmax,
            "winner_b": -1,
            "winner_k": -1,
            "tree_start": treeA,
            "tree_goal": treeB,
            "q_start_proj": q_start_proj,
            "q_goal_proj": q_goal_proj,
        },
        total_time_sec=elapsed,
    )
    return PlanResult(False, None, stats)

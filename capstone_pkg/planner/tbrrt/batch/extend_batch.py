from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjectorTorch as ManifoldProjector
from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.tangent_space import TangentSpace, project_vector_to_tangent
from capstone_pkg.planner.tbrrt.ts_bank import TSBank

from .connect_batch import (
    _build_projector_jacobian_flat_and_offsets,
    _append_projected_nodes_as_new_ts_two_trees,
    _filter_candidate_add_mask,
    _update_ts_bank_after_add,
    _update_ts_bank_after_collision,
)
from .prealloc import BatchConextPrealloc
from .tree_batch import TreeBatchGPU
from .nn_batch import nn_1_tree_all_candidates_cdist
from .projector_stack import ProjectorStackCache, bank_len as _bank_len, spaces_of as _spaces_of


def _bank_append(bank, ts: TangentSpace) -> None:
    if hasattr(bank, "add"):
        bank.add(ts)
    else:
        bank.append(ts)


def _make_tangent_space_from_svd(
    *,
    q: torch.Tensor,
    J: torch.Tensor,
    V: torch.Tensor,
    rank: int,
    ts_id: int,
    created_iter: int,
) -> TangentSpace:
    D = int(q.shape[0])
    r = int(rank)
    if r >= D:
        basis = torch.zeros((D, 0), device=q.device, dtype=q.dtype)
        projector = torch.zeros((D, D), device=q.device, dtype=q.dtype)
    else:
        basis = V[:, r:].contiguous()
        projector = (basis @ basis.transpose(0, 1)).contiguous()
    return TangentSpace(
        ts_id=int(ts_id),
        root=q.detach().clone().contiguous(),
        J=J.detach().clone().contiguous(),
        basis=basis.detach().clone().contiguous(),
        projector=projector.detach().clone().contiguous(),
        rank=r,
        created_iter=int(created_iter),
    )


def _empty_new_index_matrix(
    *,
    B: int,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.full((B, K), -1, device=device, dtype=torch.long)


@torch.no_grad()
def _build_projector_flat_and_offsets(
    ts_bank,
    device: torch.device,
    dtype: torch.dtype,
    cache: ProjectorStackCache | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      P_flat: (T_total, D, D) stacked projectors for all b
      offsets: (B,) offsets[b] where ts_id for batch b starts in P_flat
    """
    if cache is not None:
        return cache.get(ts_bank=ts_bank, device=device, dtype=dtype)

    counts = [_bank_len(bank) for bank in ts_bank]
    if not any(count > 0 for count in counts):
        raise RuntimeError("ts_bank is empty for all batches")
    counts_t = torch.tensor(counts, device=device, dtype=torch.long)

    offsets = torch.cumsum(
        torch.cat([torch.zeros((1,), device=device, dtype=torch.long), counts_t[:-1]], dim=0),
        dim=0,
    )
    proj_list = [
        torch.stack([ts.projector.to(device=device, dtype=dtype) for ts in _spaces_of(bank)], dim=0)
        for bank in ts_bank
        if _bank_len(bank) > 0
    ]
    P_flat = torch.cat(proj_list, dim=0)  # (T_total,D,D)
    return P_flat, offsets


@torch.no_grad()
def _build_tangent_spaces_fd_batch(
    *,
    q_roots: torch.Tensor,  # (N,D)
    projector: ManifoldProjector,
    svd_tol: float,
    ts_ids: torch.Tensor,   # (N,)
    created_iter: int,
) -> List[TangentSpace]:
    if q_roots.ndim != 2:
        raise ValueError("q_roots must be (N,D)")

    N, D = q_roots.shape
    if N == 0:
        return []

    q = q_roots.contiguous()
    h0, Jb = projector.residual_and_jacobian_if_available(q)
    if Jb is None:
        Jb = projector._jacobian_fd(q, h0)
    Jb = Jb.contiguous()  # (N,m,D)
    _, S, Vh = torch.linalg.svd(Jb, full_matrices=True)
    ranks = (S > float(svd_tol)).sum(dim=-1).to(torch.long)  # (N,)
    Vb = Vh.transpose(1, 2).contiguous()  # (N,D,D)

    q_list = q.unbind(0)
    J_list = Jb.unbind(0)
    V_list = Vb.unbind(0)
    rank_list = ranks.tolist()
    ts_id_list = ts_ids.to(dtype=torch.long).tolist()

    return [
        _make_tangent_space_from_svd(
            q=qi,
            J=Ji,
            V=Vi,
            rank=ri,
            ts_id=tsi,
            created_iter=created_iter,
        )
        for qi, Ji, Vi, ri, tsi in zip(q_list, J_list, V_list, rank_list, ts_id_list)
    ]


@torch.no_grad()
def _append_projected_nodes_as_new_ts(
    *,
    ts_bank,
    base_ts_id: torch.Tensor,   # (B,K)
    q_nodes: torch.Tensor,      # (B,K,D)
    was_proj: torch.Tensor,     # (B,K) bool
    projector: ManifoldProjector,
    svd_tol: float,
    iter_idx: int,
) -> torch.Tensor:
    out_ts_id = base_ts_id.clone()
    rows = torch.nonzero(was_proj, as_tuple=False)
    if rows.numel() == 0:
        return out_ts_id

    b_rows = rows[:, 0].to(torch.long)
    k_rows = rows[:, 1].to(torch.long)
    q_roots = q_nodes[b_rows, k_rows, :].contiguous()

    device = q_nodes.device
    B = len(ts_bank)
    base_ids = torch.tensor([_bank_len(bank) for bank in ts_bank], device=device, dtype=torch.long)

    order = torch.argsort(b_rows, stable=True)
    b_sorted = b_rows.index_select(0, order)
    cnt_per_b = torch.bincount(b_sorted, minlength=B)
    start_per_b = torch.cumsum(cnt_per_b, dim=0) - cnt_per_b
    local_sorted = torch.arange(order.numel(), device=device, dtype=torch.long) - start_per_b[b_sorted]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=device, dtype=torch.long)
    local_rank = local_sorted.index_select(0, inv)

    new_ids = base_ids.index_select(0, b_rows) + local_rank
    ts_new = _build_tangent_spaces_fd_batch(
        q_roots=q_roots,
        projector=projector,
        svd_tol=float(svd_tol),
        ts_ids=new_ids,
        created_iter=int(iter_idx),
    )

    ts_new_sorted = [ts_new[i] for i in order.tolist()]
    for ts, b in zip(ts_new_sorted, b_sorted.tolist()):
        _bank_append(ts_bank[int(b)], ts)

    out_ts_id[b_rows, k_rows] = new_ids
    return out_ts_id


@dataclass
class BatchExtendOut:
    """
    Batched extend output returning ALL K candidates per batch element.

    cand_ok: (B,K) - candidate is valid (has NN + point&edge free + proj success if needed)
    q_new:   (B,K,D)
    parent_idx: (B,K) - nearest node index in tree for each candidate
    ts_id:   (B,K) - tangent space id to assign to each candidate if added to tree
    dist_to_target: (B,K) - distance to its q_target after step/projection (debug/score)
    was_proj: (B,K) - whether this candidate went through manifold projection (and succeeded)
    """
    cand_ok: torch.Tensor
    q_new: torch.Tensor
    parent_idx: torch.Tensor
    ts_id: torch.Tensor
    dist_to_target: torch.Tensor
    was_proj: torch.Tensor


@torch.no_grad()
def _build_extend_candidates_from_parent(
    *,
    tree: TreeBatchGPU,
    banks,
    parent_idx: torch.Tensor,     # (B,K)
    q_targets: torch.Tensor,      # (B,K,D)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask: torch.Tensor,           # (B,K)
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
) -> BatchExtendOut:
    """Shared EXTEND core: build (B,K) candidates from explicit parent indices."""
    device = tree.device
    dtype = tree.dtype
    B, K, D = q_targets.shape
    BK = B * K
    EM = float(cfg.EM)

    tree_q, _ = tree.get_nodes()
    parent_idx_clamped = parent_idx.clamp_min(0)
    if prealloc is not None and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).contiguous()

    q_near = tree_q[b_idx, parent_idx_clamped, :]
    ts_id = tree.ts_id[b_idx, parent_idx_clamped]

    P_flat, offsets = _build_projector_flat_and_offsets(
        ts_bank=banks,
        device=device,
        dtype=dtype,
        cache=projector_cache,
    )
    global_ts = offsets.view(B, 1).expand(B, K) + ts_id
    P = P_flat.index_select(0, global_ts.reshape(BK))

    step_size = float(cfg.step_size)
    q_near_flat = q_near.reshape(BK, D)
    q_targets_flat = q_targets.reshape(BK, D)
    v = q_targets_flat - q_near_flat
    v_tan = project_vector_to_tangent(P, v)
    nrm = torch.linalg.norm(v_tan, dim=-1).clamp_min(1e-12)
    dir_tan = v_tan / nrm.unsqueeze(-1)
    q_step = (q_near_flat + step_size * dir_tan).contiguous()
    dist_to_target = torch.linalg.norm(q_targets_flat - q_step, dim=-1)

    q_new = q_step.clone()
    free_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    was_proj_flat = torch.zeros((BK,), device=device, dtype=torch.bool)

    active_flat = mask.reshape(BK)
    idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
    if idx.numel() > 0:
        q_step_a = q_step.index_select(0, idx)
        q_near_a = q_near_flat.index_select(0, idx)

        free_a = checker.get_collision_free_mask(q_step_a, margin=0.0)
        edge_res = edge_checker.check_edges_batch(
            q_near_a,
            q_step_a,
            step_q=float(cfg.edge_step_q),
            max_steps=int(cfg.edge_max_steps),
        )
        edge_free_a = edge_res.bool() if torch.is_tensor(edge_res) else (~edge_res.edge_in_collision.bool())
        free_a = free_a & edge_free_a

        res = projector.residual(q_step_a)
        res_norm = torch.linalg.norm(res, dim=-1)
        need_proj_a = res_norm > EM
        q_step2_a = q_step_a
        was_proj_a = torch.zeros((q_step_a.shape[0],), device=device, dtype=torch.bool)

        if bool(need_proj_a.any()):
            idx_local = torch.nonzero(need_proj_a, as_tuple=False).view(-1)
            pr = projector.project_batch(
                q_step_a.index_select(0, idx_local),
                h0_init=res.index_select(0, idx_local),
            )
            q_step2_a = q_step_a.clone()
            q_step2_a.index_copy_(0, idx_local, pr.q_proj)

            succ_local = pr.success_mask.bool()
            failed_local = idx_local[~succ_local]
            if failed_local.numel() > 0:
                free_a.index_fill_(0, failed_local, False)

            ok_local = idx_local[succ_local]
            if ok_local.numel() > 0:
                q_ok = q_step2_a.index_select(0, ok_local)
                free_ok = checker.get_collision_free_mask(q_ok, margin=0.0)
                near_ok = q_near_a.index_select(0, ok_local)
                edge2 = edge_checker.check_edges_batch(
                    near_ok,
                    q_ok,
                    step_q=float(cfg.edge_step_q),
                    max_steps=int(cfg.edge_max_steps),
                )
                edge2_free = edge2.bool() if torch.is_tensor(edge2) else (~edge2.edge_in_collision.bool())
                free_ok = free_ok & edge2_free
                free_a.index_copy_(0, ok_local, free_ok)
                still_ok = ok_local[free_ok]
                if still_ok.numel() > 0:
                    was_proj_a.index_fill_(0, still_ok, True)

        q_new.index_copy_(0, idx, q_step2_a)
        free_flat.index_copy_(0, idx, free_a)
        was_proj_flat.index_copy_(0, idx, was_proj_a)

    dist_to_target = torch.linalg.norm(q_targets_flat - q_new, dim=-1)
    cand_ok = free_flat.view(B, K) & mask

    ts_id = ts_id.clone()
    was_proj = was_proj_flat.view(B, K)
    was_proj_bk = was_proj & cand_ok
    if bool(was_proj_bk.any()):
        ts_id = _append_projected_nodes_as_new_ts(
            ts_bank=banks,
            base_ts_id=ts_id,
            q_nodes=q_new.view(B, K, D),
            was_proj=was_proj_bk,
            projector=projector,
            svd_tol=float(cfg.svd_tol),
            iter_idx=int(iter_idx),
        )
        if projector_cache is not None:
            projector_cache.invalidate()

    return BatchExtendOut(
        cand_ok=cand_ok,
        q_new=q_new.view(B, K, D),
        parent_idx=parent_idx,
        ts_id=ts_id,
        dist_to_target=dist_to_target.view(B, K),
        was_proj=was_proj,
    )


@torch.no_grad()
def _extend_one_step_from_parent(
    *,
    tree: TreeBatchGPU,
    banks: List[TSBank],
    parent_idx: torch.Tensor,     # (B,K)
    q_rand: torch.Tensor,         # (B,K,D)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask: torch.Tensor,           # (B,K)
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
    trace_phase: str | None = None,
    trace_escape_step: int | None = None,
) -> torch.Tensor:
    """One EXTEND step from explicitly provided parent indices."""
    device = tree.device
    dtype = tree.dtype
    B, K, D = q_rand.shape
    BK = B * K
    EM = float(cfg.EM)
    active_flat = mask.reshape(BK)
    idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return torch.full((B, K), -1, device=device, dtype=torch.long)

    tree_q, _ = tree.get_nodes()
    parent_idx_clamped = parent_idx.clamp_min(0)
    if prealloc is not None and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).contiguous()

    q_near = tree_q[b_idx, parent_idx_clamped, :]
    ts_id = tree.ts_id[b_idx, parent_idx_clamped]

    P_flat, J_flat, offsets = _build_projector_jacobian_flat_and_offsets(
        banks=banks,
        device=device,
        dtype=dtype,
        cache=projector_cache,
    )
    global_ts = offsets.view(B, 1).expand(B, K) + ts_id
    global_ts_flat = global_ts.reshape(BK)
    P = P_flat.index_select(0, global_ts_flat)
    J_cur = J_flat.index_select(0, global_ts_flat)

    step_size = float(cfg.step_size)
    q_near_flat = q_near.reshape(BK, D)
    q_rand_flat = q_rand.reshape(BK, D)
    v = q_rand_flat - q_near_flat
    v_tan = project_vector_to_tangent(P, v)
    nrm = torch.linalg.norm(v_tan, dim=-1).clamp_min(1e-12)
    dir_tan = v_tan / nrm.unsqueeze(-1)
    q_next = q_near_flat + step_size * dir_tan

    q_next_a = q_next.index_select(0, idx)
    q_near_a = q_near_flat.index_select(0, idx)
    J_cur_a = J_cur.index_select(0, idx)

    q_free_a = checker.get_collision_free_mask(q_next_a, margin=0.0)
    edge_res = edge_checker.check_edges_batch(
        q_near_a,
        q_next_a,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )
    edge_free_a = edge_res.bool() if torch.is_tensor(edge_res) else (~edge_res.edge_in_collision.bool())
    collision_blocked_a = (~q_free_a) | (~edge_free_a)
    free_a = q_free_a & edge_free_a

    res = projector.residual(q_next_a)
    res_norm = torch.linalg.norm(res, dim=-1)
    need_proj_a = res_norm > EM
    q_next2_a = q_next_a
    was_proj_a = torch.zeros((q_next_a.shape[0],), device=device, dtype=torch.bool)

    if bool(need_proj_a.any()):
        idx_local = torch.nonzero(need_proj_a, as_tuple=False).view(-1)
        pr = projector.project_batch(
            q_next_a.index_select(0, idx_local),
            h0_init=res.index_select(0, idx_local),
            J0_init=J_cur_a.index_select(0, idx_local),
        )
        q_next2_a = q_next_a.clone()
        q_next2_a.index_copy_(0, idx_local, pr.q_proj)

        succ_local = pr.success_mask.bool()
        failed_local = idx_local[~succ_local]
        if failed_local.numel() > 0:
            free_a.index_fill_(0, failed_local, False)

        ok_local = idx_local[succ_local]
        if ok_local.numel() > 0:
            q_ok = q_next2_a.index_select(0, ok_local)
            q_free_ok = checker.get_collision_free_mask(q_ok, margin=0.0)
            near_ok = q_near_a.index_select(0, ok_local)
            edge2 = edge_checker.check_edges_batch(
                near_ok,
                q_ok,
                step_q=float(cfg.edge_step_q),
                max_steps=int(cfg.edge_max_steps),
            )
            edge2_free = edge2.bool() if torch.is_tensor(edge2) else (~edge2.edge_in_collision.bool())
            free_ok = q_free_ok & edge2_free
            collision_ok = (~q_free_ok) | (~edge2_free)
            collision_blocked_a.index_copy_(0, ok_local, collision_ok)
            free_a.index_copy_(0, ok_local, free_ok)
            still_ok = ok_local[free_ok]
            if still_ok.numel() > 0:
                was_proj_a.index_fill_(0, still_ok, True)

    q_next2 = q_next.clone()
    q_next2.index_copy_(0, idx, q_next2_a)

    add_mask_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    add_mask_flat.index_copy_(0, idx, free_a)
    add_mask = add_mask_flat.view(B, K) & mask
    add_mask = _filter_candidate_add_mask(
        add_mask=add_mask,
        tree_q=tree_q,
        tree_n=tree.n_nodes,
        q_parent=q_near,
        q_target=q_rand,
        q_new=q_next2.view(B, K, D),
        cfg=cfg,
    )
    collision_blocked_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    collision_blocked_flat.index_copy_(0, idx, collision_blocked_a)
    collision_blocked = collision_blocked_flat.view(B, K) & mask & (~add_mask)

    new_ts_id = ts_id.clone()
    was_proj_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    was_proj_flat.index_copy_(0, idx, was_proj_a)
    was_proj = was_proj_flat.view(B, K) & add_mask

    if bool(was_proj.any()):
        q_next2_bk = q_next2.view(B, K, D)
        new_ts_id = _append_projected_nodes_as_new_ts(
            ts_bank=banks,
            base_ts_id=new_ts_id,
            q_nodes=q_next2_bk,
            was_proj=was_proj,
            projector=projector,
            svd_tol=float(cfg.svd_tol),
            iter_idx=int(iter_idx),
        )
        if projector_cache is not None:
            projector_cache.invalidate()

    new_idx = tree.add_nodes_k_per_batch(
        q_new=q_next2.view(B, K, D),
        parent_idx=parent_idx,
        ts_id=new_ts_id,
        mask=add_mask,
        is_proj_root=was_proj,
        trace_iter_idx=int(iter_idx),
        trace_phase=trace_phase,
        trace_escape_step=trace_escape_step,
    )
    advanced = (new_idx >= 0) & mask
    if bool(advanced.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=ts_id,
            assigned_ts_id=new_ts_id,
            q_new=q_next2.view(B, K, D),
            added_mask=advanced,
            was_proj=was_proj,
            cfg=cfg,
        )
    if bool(collision_blocked.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=ts_id,
            collision_mask=collision_blocked,
        )
    return new_idx


@torch.no_grad()
def extend_two_trees_one_step_from_parent(
    *,
    tree_a: TreeBatchGPU,
    tree_b: TreeBatchGPU,
    banks: List[TSBank],
    parent_idx_a: torch.Tensor,
    parent_idx_b: torch.Tensor,
    q_rand_a: torch.Tensor,
    q_rand_b: torch.Tensor,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
    trace_phase: str | None = None,
    trace_escape_step: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse A/B EXTEND-from-parent candidate computation into one virtual 2B batch."""
    device = tree_a.device
    dtype = tree_a.dtype
    B, K, D = q_rand_a.shape
    if q_rand_b.shape != (B, K, D):
        raise ValueError(f"q_rand_b shape {tuple(q_rand_b.shape)} != {(B, K, D)}")
    BK2 = 2 * B * K
    EM = float(cfg.EM)

    if not bool(mask_a.any() | mask_b.any()):
        return (
            _empty_new_index_matrix(B=B, K=K, device=device),
            _empty_new_index_matrix(B=B, K=K, device=device),
        )

    if prealloc is not None and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).contiguous()

    tree_q_a, _ = tree_a.get_nodes()
    tree_q_b, _ = tree_b.get_nodes()
    parent_a_clamped = parent_idx_a.clamp_min(0)
    parent_b_clamped = parent_idx_b.clamp_min(0)
    q_near_a = tree_q_a[b_idx, parent_a_clamped, :]
    q_near_b = tree_q_b[b_idx, parent_b_clamped, :]
    ts_id_a = tree_a.ts_id[b_idx, parent_a_clamped]
    ts_id_b = tree_b.ts_id[b_idx, parent_b_clamped]

    use_ab_prealloc = (
        prealloc is not None
        and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype)
    )
    if use_ab_prealloc:
        q_near_ab = prealloc.ab_q0[:, :K, :]
        q_rand_ab = prealloc.ab_q1[:, :K, :]
        ts_id_ab = prealloc.ab_long0[:, :K]
        mask_ab = prealloc.ab_bool0[:, :K]
        q_near_ab[:B].copy_(q_near_a)
        q_near_ab[B:2 * B].copy_(q_near_b)
        q_rand_ab[:B].copy_(q_rand_a)
        q_rand_ab[B:2 * B].copy_(q_rand_b)
        ts_id_ab[:B].copy_(ts_id_a)
        ts_id_ab[B:2 * B].copy_(ts_id_b)
        mask_ab[:B].copy_(mask_a)
        mask_ab[B:2 * B].copy_(mask_b)
    else:
        q_near_ab = torch.cat([q_near_a, q_near_b], dim=0)
        q_rand_ab = torch.cat([q_rand_a, q_rand_b], dim=0)
        ts_id_ab = torch.cat([ts_id_a, ts_id_b], dim=0)
        mask_ab = torch.cat([mask_a, mask_b], dim=0)
    banks_ab = banks + banks

    active_flat = mask_ab.reshape(BK2)
    idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return (
            _empty_new_index_matrix(B=B, K=K, device=device),
            _empty_new_index_matrix(B=B, K=K, device=device),
        )

    P_flat, J_flat, offsets = _build_projector_jacobian_flat_and_offsets(
        banks=banks_ab,
        device=device,
        dtype=dtype,
        cache=projector_cache,
    )
    global_ts = offsets.view(2 * B, 1).expand(2 * B, K) + ts_id_ab
    global_ts_flat = global_ts.reshape(BK2)
    P = P_flat.index_select(0, global_ts_flat)
    J_cur = J_flat.index_select(0, global_ts_flat)

    step_size = float(cfg.step_size)
    q_near_flat = q_near_ab.reshape(BK2, D)
    q_rand_flat = q_rand_ab.reshape(BK2, D)
    v = q_rand_flat - q_near_flat
    v_tan = project_vector_to_tangent(P, v)
    nrm = torch.linalg.norm(v_tan, dim=-1).clamp_min(1e-12)
    dir_tan = v_tan / nrm.unsqueeze(-1)
    q_next = q_near_flat + step_size * dir_tan

    q_next_active = q_next.index_select(0, idx)
    q_near_active = q_near_flat.index_select(0, idx)
    J_cur_active = J_cur.index_select(0, idx)

    q_free_active = checker.get_collision_free_mask(q_next_active, margin=0.0)
    edge_res = edge_checker.check_edges_batch(
        q_near_active,
        q_next_active,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )
    edge_free_active = edge_res.bool() if torch.is_tensor(edge_res) else (~edge_res.edge_in_collision.bool())
    collision_blocked_active = (~q_free_active) | (~edge_free_active)
    free_active = q_free_active & edge_free_active

    res = projector.residual(q_next_active)
    res_norm = torch.linalg.norm(res, dim=-1)
    need_proj_active = res_norm > EM
    q_next2_active = q_next_active
    was_proj_active = torch.zeros((q_next_active.shape[0],), device=device, dtype=torch.bool)

    if bool(need_proj_active.any()):
        idx_local = torch.nonzero(need_proj_active, as_tuple=False).view(-1)
        pr = projector.project_batch(
            q_next_active.index_select(0, idx_local),
            h0_init=res.index_select(0, idx_local),
            J0_init=J_cur_active.index_select(0, idx_local),
        )
        q_next2_active = q_next_active.clone()
        q_next2_active.index_copy_(0, idx_local, pr.q_proj)

        succ_local = pr.success_mask.bool()
        failed_local = idx_local[~succ_local]
        if failed_local.numel() > 0:
            free_active.index_fill_(0, failed_local, False)

        ok_local = idx_local[succ_local]
        if ok_local.numel() > 0:
            q_ok = q_next2_active.index_select(0, ok_local)
            q_free_ok = checker.get_collision_free_mask(q_ok, margin=0.0)
            near_ok = q_near_active.index_select(0, ok_local)
            edge2 = edge_checker.check_edges_batch(
                near_ok,
                q_ok,
                step_q=float(cfg.edge_step_q),
                max_steps=int(cfg.edge_max_steps),
            )
            edge2_free = edge2.bool() if torch.is_tensor(edge2) else (~edge2.edge_in_collision.bool())
            free_ok = q_free_ok & edge2_free
            collision_ok = (~q_free_ok) | (~edge2_free)
            collision_blocked_active.index_copy_(0, ok_local, collision_ok)
            free_active.index_copy_(0, ok_local, free_ok)
            still_ok = ok_local[free_ok]
            if still_ok.numel() > 0:
                was_proj_active.index_fill_(0, still_ok, True)

    q_next2 = q_next.clone()
    q_next2.index_copy_(0, idx, q_next2_active)
    q_next2_ab = q_next2.view(2 * B, K, D)
    q_next2_a = q_next2_ab[:B]
    q_next2_b = q_next2_ab[B:]

    if use_ab_prealloc:
        add_mask_ab = prealloc.ab_bool1[:, :K]
        add_mask_ab.zero_()
        add_mask_flat = add_mask_ab.reshape(BK2)
    else:
        add_mask_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
    add_mask_flat.index_copy_(0, idx, free_active)
    add_mask_ab = add_mask_flat.view(2 * B, K)
    add_mask_ab.logical_and_(mask_ab)
    add_mask_a = _filter_candidate_add_mask(
        add_mask=add_mask_ab[:B].clone(),
        tree_q=tree_q_a,
        tree_n=tree_a.n_nodes,
        q_parent=q_near_a,
        q_target=q_rand_a,
        q_new=q_next2_a,
        cfg=cfg,
    )
    add_mask_b = _filter_candidate_add_mask(
        add_mask=add_mask_ab[B:].clone(),
        tree_q=tree_q_b,
        tree_n=tree_b.n_nodes,
        q_parent=q_near_b,
        q_target=q_rand_b,
        q_new=q_next2_b,
        cfg=cfg,
    )

    if use_ab_prealloc:
        collision_blocked_ab = prealloc.ab_bool2[:, :K]
        collision_blocked_ab.zero_()
        collision_blocked_flat = collision_blocked_ab.reshape(BK2)
    else:
        collision_blocked_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
    collision_blocked_flat.index_copy_(0, idx, collision_blocked_active)
    collision_blocked_ab = collision_blocked_flat.view(2 * B, K)
    collision_blocked_ab.logical_and_(mask_ab)
    collision_blocked_a = collision_blocked_ab[:B] & (~add_mask_a)
    collision_blocked_b = collision_blocked_ab[B:] & (~add_mask_b)

    if use_ab_prealloc:
        was_proj_ab = prealloc.ab_bool3[:, :K]
        was_proj_ab.zero_()
        was_proj_flat = was_proj_ab.reshape(BK2)
    else:
        was_proj_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
    was_proj_flat.index_copy_(0, idx, was_proj_active)
    was_proj_ab = was_proj_flat.view(2 * B, K)
    was_proj_a = was_proj_ab[:B] & add_mask_a
    was_proj_b = was_proj_ab[B:] & add_mask_b

    new_ts_id_a = ts_id_a.clone()
    new_ts_id_b = ts_id_b.clone()
    if bool(was_proj_a.any() | was_proj_b.any()):
        new_ts_id_a, new_ts_id_b = _append_projected_nodes_as_new_ts_two_trees(
            banks=banks,
            base_ts_id_a=new_ts_id_a,
            base_ts_id_b=new_ts_id_b,
            q_nodes_a=q_next2_a,
            q_nodes_b=q_next2_b,
            was_proj_a=was_proj_a,
            was_proj_b=was_proj_b,
            projector=projector,
            svd_tol=float(cfg.svd_tol),
            iter_idx=int(iter_idx),
        )
        if projector_cache is not None:
            projector_cache.invalidate()

    new_idx_a = tree_a.add_nodes_k_per_batch(
        q_new=q_next2_a,
        parent_idx=parent_idx_a,
        ts_id=new_ts_id_a,
        mask=add_mask_a,
        is_proj_root=was_proj_a,
        trace_iter_idx=int(iter_idx),
        trace_phase=trace_phase,
        trace_escape_step=trace_escape_step,
    )
    new_idx_b = tree_b.add_nodes_k_per_batch(
        q_new=q_next2_b,
        parent_idx=parent_idx_b,
        ts_id=new_ts_id_b,
        mask=add_mask_b,
        is_proj_root=was_proj_b,
        trace_iter_idx=int(iter_idx),
        trace_phase=trace_phase,
        trace_escape_step=trace_escape_step,
    )

    advanced_a = (new_idx_a >= 0) & mask_a
    advanced_b = (new_idx_b >= 0) & mask_b
    if bool(advanced_a.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=ts_id_a,
            assigned_ts_id=new_ts_id_a,
            q_new=q_next2_a,
            added_mask=advanced_a,
            was_proj=was_proj_a,
            cfg=cfg,
        )
    if bool(advanced_b.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=ts_id_b,
            assigned_ts_id=new_ts_id_b,
            q_new=q_next2_b,
            added_mask=advanced_b,
            was_proj=was_proj_b,
            cfg=cfg,
        )
    if bool(collision_blocked_a.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=ts_id_a,
            collision_mask=collision_blocked_a,
        )
    if bool(collision_blocked_b.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=ts_id_b,
            collision_mask=collision_blocked_b,
        )

    return new_idx_a, new_idx_b


@torch.no_grad()
def extend_one_step_from_parent(
    *,
    tree: TreeBatchGPU,
    banks: List[TSBank],
    parent_idx: torch.Tensor,     # (B,K)
    q_rand: torch.Tensor,         # (B,K,D)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask: torch.Tensor,           # (B,K)
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
    trace_phase: str | None = None,
    trace_escape_step: int | None = None,
) -> torch.Tensor:
    return _extend_one_step_from_parent(
        tree=tree,
        banks=banks,
        parent_idx=parent_idx,
        q_rand=q_rand,
        cfg=cfg,
        checker=checker,
        edge_checker=edge_checker,
        projector=projector,
        iter_idx=iter_idx,
        mask=mask,
        projector_cache=projector_cache,
        prealloc=prealloc,
        trace_phase=trace_phase,
        trace_escape_step=trace_escape_step,
    )


@torch.no_grad()
def extend_extcon_block(
    *,
    tree: TreeBatchGPU,
    ts_bank,
    q_targets: torch.Tensor,          # (B,K,D)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
) -> BatchExtendOut:
    """Batched version of extend_extcon_once with K candidate targets per batch.

    Matches BASIC semantics for each candidate:
      - NN is over all tree nodes (not TS-restricted)
      - direction is projected into tangent space of the NN node
      - manifold projection is applied only if residual norm > EM
      - candidates that fail projection or collision checks are invalid

    Returns ALL K candidates for each batch element (no "best" selection here).
    """

    device = tree.device
    B, K, _ = q_targets.shape

    tree_q, tree_size = tree.get_nodes()

    # NN over all nodes for each candidate target
    nn_idx = nn_1_tree_all_candidates_cdist(
        tree_q,
        tree_size,
        q_targets,
        exclude_mask=getattr(tree, "blocked_node", getattr(tree, "banned_node", None)),
    )  # (B,K)
    valid_nn = nn_idx >= 0
    out = _build_extend_candidates_from_parent(
        tree=tree,
        banks=ts_bank,
        parent_idx=nn_idx,
        q_targets=q_targets,
        cfg=cfg,
        checker=checker,
        edge_checker=edge_checker,
        projector=projector,
        iter_idx=iter_idx,
        mask=(prealloc.one_bool(K) if prealloc is not None and prealloc.supports(B=B, K=K, D=tree.D, device=device, dtype=tree.dtype) else torch.ones((B, K), device=device, dtype=torch.bool)),
        projector_cache=projector_cache,
        prealloc=prealloc,
    )
    return BatchExtendOut(
        cand_ok=out.cand_ok & valid_nn,
        q_new=out.q_new,
        parent_idx=out.parent_idx,
        ts_id=out.ts_id,
        dist_to_target=out.dist_to_target,
        was_proj=out.was_proj,
    )

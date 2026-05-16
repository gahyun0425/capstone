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

from .nn_batch import nn_1_tree_all_candidates_cdist_with_dist
from .prealloc import BatchConextPrealloc
from .projector_stack import ProjectorStackCache, bank_len as _bank_len, spaces_of as _spaces_of
from .tree_batch import TreeBatchGPU


def _bank_append(bank, ts: TangentSpace) -> None:
    if hasattr(bank, "add"):
        bank.add(ts)
    else:
        bank.append(ts)


@torch.no_grad()
def _filter_candidate_add_mask(
    *,
    add_mask: torch.Tensor,      # (B,K)
    tree_q: torch.Tensor,        # (B,N,D)
    tree_n: torch.Tensor,        # (B,)
    q_parent: torch.Tensor,      # (B,K,D)
    q_target: torch.Tensor,      # (B,K,D)
    q_new: torch.Tensor,         # (B,K,D)
    cfg: TBRRTConfig,
) -> torch.Tensor:
    if not bool(add_mask.any()):
        return add_mask

    step_size = float(getattr(cfg, "step_size", 0.0))
    min_progress = float(getattr(cfg, "min_progress_ratio", 0.0)) * step_size
    min_sep_parent = float(getattr(cfg, "min_separation_parent_ratio", 0.0)) * step_size
    min_sep_tree = float(getattr(cfg, "min_separation_tree_ratio", 0.0)) * step_size

    if min_progress <= 0.0 and min_sep_parent <= 0.0 and min_sep_tree <= 0.0:
        return add_mask

    keep = add_mask.clone()

    if min_progress > 0.0:
        dist_parent = torch.linalg.norm(q_target - q_parent, dim=-1)
        dist_new = torch.linalg.norm(q_target - q_new, dim=-1)
        progress = dist_parent - dist_new
        keep = keep & (progress >= min_progress)

    if min_sep_parent > 0.0:
        sep_parent = torch.linalg.norm(q_new - q_parent, dim=-1)
        keep = keep & (sep_parent >= min_sep_parent)

    if min_sep_tree > 0.0 and bool(keep.any()):
        _, nn_dist = nn_1_tree_all_candidates_cdist_with_dist(
            tree_q=tree_q,
            tree_size=tree_n,
            q_targets=q_new,
        )
        keep = keep & (nn_dist >= min_sep_tree)

    return keep


@torch.no_grad()
def _build_projector_jacobian_flat_and_offsets(
    *,
    banks: List[TSBank],
    device: torch.device,
    dtype: torch.dtype,
    cache: ProjectorStackCache | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache is not None:
        return cache.get_with_jacobian(ts_bank=banks, device=device, dtype=dtype)

    counts = [_bank_len(bank) for bank in banks]
    if not any(count > 0 for count in counts):
        raise RuntimeError("ts_bank is empty for all batches")
    counts_t = torch.tensor(counts, device=device, dtype=torch.long)

    offsets = torch.cumsum(
        torch.cat([torch.zeros((1,), device=device, dtype=torch.long), counts_t[:-1]], dim=0),
        dim=0,
    )
    proj_list = [
        torch.stack([ts.projector.to(device=device, dtype=dtype) for ts in _spaces_of(bank)], dim=0)
        for bank in banks
        if _bank_len(bank) > 0
    ]
    jac_list = [
        torch.stack([ts.J.to(device=device, dtype=dtype) for ts in _spaces_of(bank)], dim=0)
        for bank in banks
        if _bank_len(bank) > 0
    ]
    P_flat = torch.cat(proj_list, dim=0)  # (T_total,D,D)
    J_flat = torch.cat(jac_list, dim=0)  # (T_total,m,D)
    return P_flat, J_flat, offsets


@torch.no_grad()
def _build_tangent_spaces_batch(
    *,
    q_roots: torch.Tensor,  # (N,D)
    projector: ManifoldProjector,
    svd_tol: float,
    ts_ids: torch.Tensor,   # (N,)
    created_iter: int,
    debug_label: str = "projected",
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

    out = [
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
    return out


@torch.no_grad()
def _append_projected_nodes_as_new_ts(
    *,
    banks: List[TSBank],
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
    B = len(banks)
    base_ids = torch.tensor([_bank_len(bank) for bank in banks], device=device, dtype=torch.long)

    order = torch.argsort(b_rows, stable=True)
    b_sorted = b_rows.index_select(0, order)
    cnt_per_b = torch.bincount(b_sorted, minlength=B)
    start_per_b = torch.cumsum(cnt_per_b, dim=0) - cnt_per_b
    local_sorted = torch.arange(order.numel(), device=device, dtype=torch.long) - start_per_b[b_sorted]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=device, dtype=torch.long)
    local_rank = local_sorted.index_select(0, inv)

    new_ids = base_ids.index_select(0, b_rows) + local_rank
    ts_new = _build_tangent_spaces_batch(
        q_roots=q_roots,
        projector=projector,
        svd_tol=float(svd_tol),
        ts_ids=new_ids,
        created_iter=int(iter_idx),
        debug_label="projected",
    )

    ts_new_sorted = [ts_new[i] for i in order.tolist()]
    for ts, b in zip(ts_new_sorted, b_sorted.tolist()):
        _bank_append(banks[int(b)], ts)

    out_ts_id[b_rows, k_rows] = new_ids
    return out_ts_id


@torch.no_grad()
def _append_projected_nodes_as_new_ts_two_trees(
    *,
    banks: List[TSBank],
    base_ts_id_a: torch.Tensor,   # (B,K)
    base_ts_id_b: torch.Tensor,   # (B,K)
    q_nodes_a: torch.Tensor,      # (B,K,D)
    q_nodes_b: torch.Tensor,      # (B,K,D)
    was_proj_a: torch.Tensor,     # (B,K) bool
    was_proj_b: torch.Tensor,     # (B,K) bool
    projector: ManifoldProjector,
    svd_tol: float,
    iter_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_a = base_ts_id_a.clone()
    out_b = base_ts_id_b.clone()

    rows_a = torch.nonzero(was_proj_a, as_tuple=False)
    rows_b = torch.nonzero(was_proj_b, as_tuple=False)
    if rows_a.numel() == 0 and rows_b.numel() == 0:
        return out_a, out_b

    device = q_nodes_a.device
    B, K, _D = q_nodes_a.shape
    parts_b = []
    parts_k = []
    parts_side = []
    parts_q = []
    if rows_a.numel() > 0:
        parts_b.append(rows_a[:, 0].to(torch.long))
        parts_k.append(rows_a[:, 1].to(torch.long))
        parts_side.append(torch.zeros((rows_a.shape[0],), device=device, dtype=torch.long))
        parts_q.append(q_nodes_a[rows_a[:, 0], rows_a[:, 1], :])
    if rows_b.numel() > 0:
        parts_b.append(rows_b[:, 0].to(torch.long))
        parts_k.append(rows_b[:, 1].to(torch.long))
        parts_side.append(torch.ones((rows_b.shape[0],), device=device, dtype=torch.long))
        parts_q.append(q_nodes_b[rows_b[:, 0], rows_b[:, 1], :])

    b_rows = torch.cat(parts_b, dim=0)
    k_rows = torch.cat(parts_k, dim=0)
    side = torch.cat(parts_side, dim=0)
    q_roots = torch.cat(parts_q, dim=0).contiguous()

    base_ids = torch.tensor([_bank_len(bank) for bank in banks], device=device, dtype=torch.long)
    sort_key = b_rows * (2 * K + 1) + side * K + k_rows
    order = torch.argsort(sort_key, stable=True)
    b_sorted = b_rows.index_select(0, order)
    cnt_per_b = torch.bincount(b_sorted, minlength=B)
    start_per_b = torch.cumsum(cnt_per_b, dim=0) - cnt_per_b
    local_sorted = torch.arange(order.numel(), device=device, dtype=torch.long) - start_per_b[b_sorted]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=device, dtype=torch.long)
    local_rank = local_sorted.index_select(0, inv)
    new_ids = base_ids.index_select(0, b_rows) + local_rank

    ts_new = _build_tangent_spaces_batch(
        q_roots=q_roots,
        projector=projector,
        svd_tol=float(svd_tol),
        ts_ids=new_ids,
        created_iter=int(iter_idx),
        debug_label="projected",
    )

    ts_new_sorted = [ts_new[i] for i in order.tolist()]
    for ts, b in zip(ts_new_sorted, b_sorted.tolist()):
        _bank_append(banks[int(b)], ts)

    side_a = side == 0
    side_b = ~side_a
    if bool(side_a.any()):
        out_a[b_rows[side_a], k_rows[side_a]] = new_ids[side_a]
    if bool(side_b.any()):
        out_b[b_rows[side_b], k_rows[side_b]] = new_ids[side_b]
    return out_a, out_b


@torch.no_grad()
def _update_ts_bank_after_add(
    *,
    banks: List[TSBank],
    used_ts_id: torch.Tensor,      # (B,K)
    assigned_ts_id: torch.Tensor,  # (B,K)
    q_new: torch.Tensor,           # (B,K,D)
    added_mask: torch.Tensor,      # (B,K)
    was_proj: torch.Tensor,        # (B,K)
    cfg: TBRRTConfig,
) -> None:
    if not bool(added_mask.any()):
        return

    dynamic_domain = bool(getattr(cfg, "dynamic_domain_enable", False))
    expand_frac = float(getattr(cfg, "ts_domain_expand_frac", 0.9))
    shrink_frac = float(getattr(cfg, "ts_domain_shrink_frac", 0.4))
    expand_ratio = float(getattr(cfg, "ts_domain_expand_ratio", 1.2))
    shrink_ratio = float(getattr(cfg, "ts_domain_shrink_ratio", 0.8))
    dom_min = float(getattr(cfg, "ts_domain_min", 0.05))
    dom_max = float(getattr(cfg, "ts_domain_max", 5.0))

    for b, bank in enumerate(banks):
        mask_b = added_mask[b]
        if not bool(mask_b.any()):
            continue

        assigned_b = assigned_ts_id[b, mask_b]
        bank.increment_counts_batch(assigned_b)

        if not dynamic_domain:
            continue

        bank.update_domains_batch(
            used_ts_id=used_ts_id[b, mask_b],
            q_added=q_new[b, mask_b, :],
            was_proj=was_proj[b, mask_b],
            expand_frac=expand_frac,
            shrink_frac=shrink_frac,
            expand_ratio=expand_ratio,
            shrink_ratio=shrink_ratio,
            dom_min=dom_min,
            dom_max=dom_max,
        )


@torch.no_grad()
def _update_ts_bank_after_collision(
    *,
    banks: List[TSBank],
    used_ts_id: torch.Tensor,      # (B,K)
    collision_mask: torch.Tensor,  # (B,K)
) -> None:
    if not bool(collision_mask.any()):
        return

    for b, bank in enumerate(banks):
        mask_b = collision_mask[b]
        if not bool(mask_b.any()):
            continue

        bank.increment_collision_counts_batch(used_ts_id[b, mask_b])


@dataclass
class ConnectStepOut:
    advanced: torch.Tensor
    new_idx: torch.Tensor
    pre_idx: torch.Tensor
    progress: torch.Tensor
    reached: torch.Tensor
    reached_idx: torch.Tensor
    target_idx_other: torch.Tensor


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


def _empty_connect_step_out(
    *,
    B: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype,
    cur_idx: torch.Tensor,
    target_idx_other: torch.Tensor,
) -> ConnectStepOut:
    return ConnectStepOut(
        advanced=torch.zeros((B, K), device=device, dtype=torch.bool),
        new_idx=torch.full((B, K), -1, device=device, dtype=torch.long),
        pre_idx=cur_idx.clone(),
        progress=torch.zeros((B, K), device=device, dtype=dtype),
        reached=torch.zeros((B, K), device=device, dtype=torch.bool),
        reached_idx=torch.full((B, K), -1, device=device, dtype=torch.long),
        target_idx_other=target_idx_other,
    )


def _record_connect_step_summary(
    *,
    iter_idx: int,
    tree: TreeBatchGPU,
    mask: torch.Tensor,
    advanced: torch.Tensor,
    point_collision: torch.Tensor,
    edge_collision: torch.Tensor,
    collision_blocked: torch.Tensor,
    projection_needed: torch.Tensor,
    projection_failed: torch.Tensor,
    was_proj: torch.Tensor,
    filter_rejected: torch.Tensor,
    reached: torch.Tensor,
    progress: torch.Tensor,
) -> None:
    rec = getattr(tree, "trace_recorder", None)
    if rec is None or not rec.wants("summary"):
        return
    progress_vals = progress[advanced]
    rec.record_connect_step(
        iter_idx=int(iter_idx),
        tree=str(tree.trace_tree_name),
        active=int(mask.sum().item()),
        advanced=int(advanced.sum().item()),
        trapped=int((mask & (~advanced)).sum().item()),
        point_collision=int(point_collision.sum().item()),
        edge_collision=int(edge_collision.sum().item()),
        collision_blocked=int(collision_blocked.sum().item()),
        projection_needed=int(projection_needed.sum().item()),
        projection_failed=int(projection_failed.sum().item()),
        projected=int((was_proj & advanced).sum().item()),
        filter_rejected=int(filter_rejected.sum().item()),
        reached=int(reached.sum().item()),
        mean_progress=(float(progress_vals.mean().item()) if progress_vals.numel() > 0 else 0.0),
        max_progress=(float(progress_vals.max().item()) if progress_vals.numel() > 0 else 0.0),
    )


@torch.no_grad()
def connect_one_step_with_state(
    *,
    tree: TreeBatchGPU,
    banks: List[TSBank],
    cur_idx: torch.Tensor,           # (B,K)
    q_target: torch.Tensor,          # (B,K,D)
    target_idx_other: torch.Tensor,  # (B,K)
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask: torch.Tensor,              # (B,K)
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
) -> ConnectStepOut:
    """One CONNECT step while preserving per-block state."""
    device = tree.device
    dtype = tree.dtype
    B, K, D = q_target.shape
    BK = B * K
    EM = float(cfg.EM)
    advanced = torch.zeros((B, K), device=device, dtype=torch.bool)
    new_idx = torch.full((B, K), -1, device=device, dtype=torch.long)
    progress = torch.zeros((B, K), device=device, dtype=dtype)
    reached = torch.zeros((B, K), device=device, dtype=torch.bool)
    reached_idx = torch.full((B, K), -1, device=device, dtype=torch.long)
    pre_idx = cur_idx.clone()

    active_flat = mask.reshape(BK)
    idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return ConnectStepOut(advanced, new_idx, pre_idx, progress, reached, reached_idx, target_idx_other)

    tree_q, _ = tree.get_nodes()
    cur_idx_clamped = cur_idx.clamp_min(0)
    if prealloc is not None and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).contiguous()
    cur_q = tree_q[b_idx, cur_idx_clamped, :]
    cur_ts_id = tree.ts_id[b_idx, cur_idx_clamped]

    P_flat, J_flat, offsets = _build_projector_jacobian_flat_and_offsets(
        banks=banks,
        device=device,
        dtype=dtype,
        cache=projector_cache,
    )
    global_ts = offsets.view(B, 1).expand(B, K) + cur_ts_id
    global_ts_flat = global_ts.reshape(BK)
    P = P_flat.index_select(0, global_ts_flat)
    J_cur = J_flat.index_select(0, global_ts_flat)

    step_size = float(cfg.step_size)
    cur_q_flat = cur_q.reshape(BK, D)
    q_tgt_flat = q_target.reshape(BK, D)
    v = q_tgt_flat - cur_q_flat
    v_tan = project_vector_to_tangent(P, v)
    nrm = torch.linalg.norm(v_tan, dim=-1).clamp_min(1e-12)
    dir_tan = v_tan / nrm.unsqueeze(-1)
    q_next = cur_q_flat + step_size * dir_tan

    q_next_a = q_next.index_select(0, idx)
    cur_q_a = cur_q_flat.index_select(0, idx)
    J_cur_a = J_cur.index_select(0, idx)

    q_free_a = checker.get_collision_free_mask(q_next_a, margin=0.0)
    edge_res = edge_checker.check_edges_batch(
        cur_q_a,
        q_next_a,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )
    edge_free_a = edge_res.bool() if torch.is_tensor(edge_res) else (~edge_res.edge_in_collision.bool())
    point_collision_a = ~q_free_a
    edge_collision_a = ~edge_free_a
    collision_blocked_a = (~q_free_a) | (~edge_free_a)
    free_a = q_free_a & edge_free_a

    res = projector.residual(q_next_a)
    res_norm = torch.linalg.norm(res, dim=-1)
    need_proj_a = res_norm > EM
    q_next2_a = q_next_a
    was_proj_a = torch.zeros((q_next_a.shape[0],), device=device, dtype=torch.bool)
    projection_failed_a = torch.zeros((q_next_a.shape[0],), device=device, dtype=torch.bool)

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
            projection_failed_a.index_fill_(0, failed_local, True)

        ok_local = idx_local[succ_local]
        if ok_local.numel() > 0:
            q_ok = q_next2_a.index_select(0, ok_local)
            q_free_ok = checker.get_collision_free_mask(q_ok, margin=0.0)
            cur_ok = cur_q_a.index_select(0, ok_local)
            edge2 = edge_checker.check_edges_batch(
                cur_ok,
                q_ok,
                step_q=float(cfg.edge_step_q),
                max_steps=int(cfg.edge_max_steps),
            )
            edge2_free = edge2.bool() if torch.is_tensor(edge2) else (~edge2.edge_in_collision.bool())
            free_ok = q_free_ok & edge2_free
            collision_ok = (~q_free_ok) | (~edge2_free)
            point_collision_a.index_copy_(0, ok_local, ~q_free_ok)
            edge_collision_a.index_copy_(0, ok_local, ~edge2_free)
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
    add_mask_pre_filter = add_mask.clone()
    add_mask = _filter_candidate_add_mask(
        add_mask=add_mask,
        tree_q=tree_q,
        tree_n=tree.n_nodes,
        q_parent=cur_q,
        q_target=q_target,
        q_new=q_next2.view(B, K, D),
        cfg=cfg,
    )
    filter_rejected = add_mask_pre_filter & (~add_mask)
    collision_blocked_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    collision_blocked_flat.index_copy_(0, idx, collision_blocked_a)
    collision_blocked = collision_blocked_flat.view(B, K) & mask & (~add_mask)

    new_ts_id = cur_ts_id.clone()
    was_proj_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
    was_proj_flat.index_copy_(0, idx, was_proj_a)
    was_proj = was_proj_flat.view(B, K) & add_mask

    if bool(was_proj.any()):
        q_next2_bk = q_next2.view(B, K, D)
        new_ts_id = _append_projected_nodes_as_new_ts(
            banks=banks,
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
        parent_idx=cur_idx,
        ts_id=new_ts_id,
        mask=add_mask,
        is_proj_root=was_proj,
        trace_iter_idx=int(iter_idx),
        trace_phase="connect",
    )
    advanced = (new_idx >= 0) & mask
    if bool(advanced.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=cur_ts_id,
            assigned_ts_id=new_ts_id,
            q_new=q_next2.view(B, K, D),
            added_mask=advanced,
            was_proj=was_proj,
            cfg=cfg,
        )
    if bool(collision_blocked.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=cur_ts_id,
            collision_mask=collision_blocked,
        )

    if bool(advanced.any()):
        tree_q, _ = tree.get_nodes()
        cur_idx2 = torch.where(advanced, new_idx, cur_idx)
        cur_q2 = tree_q[b_idx, cur_idx2.clamp_min(0), :]
        dist_prev = torch.linalg.norm(q_target - cur_q, dim=-1)
        dist = torch.linalg.norm(q_target - cur_q2, dim=-1)
        progress = torch.where(advanced, dist_prev - dist, progress)
        just_reached = advanced & (dist <= float(cfg.goal_threshold))
        reached = just_reached
        reached_idx = torch.where(just_reached, cur_idx2, reached_idx)

    rec = getattr(tree, "trace_recorder", None)
    if rec is not None and rec.wants("summary"):
        point_collision_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
        edge_collision_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
        projection_needed_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
        projection_failed_flat = torch.zeros((BK,), device=device, dtype=torch.bool)
        point_collision_flat.index_copy_(0, idx, point_collision_a)
        edge_collision_flat.index_copy_(0, idx, edge_collision_a)
        projection_needed_flat.index_copy_(0, idx, need_proj_a)
        projection_failed_flat.index_copy_(0, idx, projection_failed_a)
        point_collision = point_collision_flat.view(B, K) & mask
        edge_collision = edge_collision_flat.view(B, K) & mask
        projection_needed = projection_needed_flat.view(B, K) & mask
        projection_failed = projection_failed_flat.view(B, K) & mask
        progress_vals = progress[advanced]
        rec.record_connect_step(
            iter_idx=int(iter_idx),
            tree=str(tree.trace_tree_name),
            active=int(mask.sum().item()),
            advanced=int(advanced.sum().item()),
            trapped=int((mask & (~advanced)).sum().item()),
            point_collision=int(point_collision.sum().item()),
            edge_collision=int(edge_collision.sum().item()),
            collision_blocked=int(collision_blocked.sum().item()),
            projection_needed=int(projection_needed.sum().item()),
            projection_failed=int(projection_failed.sum().item()),
            projected=int((was_proj & advanced).sum().item()),
            filter_rejected=int(filter_rejected.sum().item()),
            reached=int(reached.sum().item()),
            mean_progress=(float(progress_vals.mean().item()) if progress_vals.numel() > 0 else 0.0),
            max_progress=(float(progress_vals.max().item()) if progress_vals.numel() > 0 else 0.0),
        )

    return ConnectStepOut(advanced, new_idx, pre_idx, progress, reached, reached_idx, target_idx_other)


@torch.no_grad()
def connect_two_trees_one_step_with_state(
    *,
    tree_a: TreeBatchGPU,
    tree_b: TreeBatchGPU,
    banks: List[TSBank],
    cur_idx_a: torch.Tensor,
    cur_idx_b: torch.Tensor,
    q_target_a: torch.Tensor,
    q_target_b: torch.Tensor,
    target_idx_other_a: torch.Tensor,
    target_idx_other_b: torch.Tensor,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    projector_cache: ProjectorStackCache | None = None,
    prealloc: BatchConextPrealloc | None = None,
) -> Tuple[ConnectStepOut, ConnectStepOut]:
    """Fuse start-tree and goal-tree CONNECT candidate computation into one GPU batch.

    The two tree containers are still committed separately; this function only fuses
    q_next, collision, edge, projection, and residual work over a virtual 2B batch.
    """
    device = tree_a.device
    dtype = tree_a.dtype
    B, K, D = q_target_a.shape
    if q_target_b.shape != (B, K, D):
        raise ValueError(f"q_target_b shape {tuple(q_target_b.shape)} != {(B, K, D)}")
    BK2 = 2 * B * K
    EM = float(cfg.EM)

    if not bool(mask_a.any() | mask_b.any()):
        return (
            _empty_connect_step_out(
                B=B,
                K=K,
                device=device,
                dtype=dtype,
                cur_idx=cur_idx_a,
                target_idx_other=target_idx_other_a,
            ),
            _empty_connect_step_out(
                B=B,
                K=K,
                device=device,
                dtype=dtype,
                cur_idx=cur_idx_b,
                target_idx_other=target_idx_other_b,
            ),
        )

    if prealloc is not None and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype):
        b_idx = prealloc.b_idx(K)
    else:
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K).contiguous()

    tree_q_a, _ = tree_a.get_nodes()
    tree_q_b, _ = tree_b.get_nodes()
    cur_idx_a_clamped = cur_idx_a.clamp_min(0)
    cur_idx_b_clamped = cur_idx_b.clamp_min(0)
    cur_q_a = tree_q_a[b_idx, cur_idx_a_clamped, :]
    cur_q_b = tree_q_b[b_idx, cur_idx_b_clamped, :]
    cur_ts_id_a = tree_a.ts_id[b_idx, cur_idx_a_clamped]
    cur_ts_id_b = tree_b.ts_id[b_idx, cur_idx_b_clamped]

    use_ab_prealloc = (
        prealloc is not None
        and prealloc.supports(B=B, K=K, D=D, device=device, dtype=dtype)
    )
    if use_ab_prealloc:
        cur_q_ab = prealloc.ab_q0[:, :K, :]
        q_target_ab = prealloc.ab_q1[:, :K, :]
        cur_ts_id_ab = prealloc.ab_long0[:, :K]
        mask_ab = prealloc.ab_bool0[:, :K]
        cur_q_ab[:B].copy_(cur_q_a)
        cur_q_ab[B:2 * B].copy_(cur_q_b)
        q_target_ab[:B].copy_(q_target_a)
        q_target_ab[B:2 * B].copy_(q_target_b)
        cur_ts_id_ab[:B].copy_(cur_ts_id_a)
        cur_ts_id_ab[B:2 * B].copy_(cur_ts_id_b)
        mask_ab[:B].copy_(mask_a)
        mask_ab[B:2 * B].copy_(mask_b)
    else:
        cur_q_ab = torch.cat([cur_q_a, cur_q_b], dim=0)
        q_target_ab = torch.cat([q_target_a, q_target_b], dim=0)
        cur_ts_id_ab = torch.cat([cur_ts_id_a, cur_ts_id_b], dim=0)
        mask_ab = torch.cat([mask_a, mask_b], dim=0)
    banks_ab = banks + banks

    active_flat = mask_ab.reshape(BK2)
    idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return (
            _empty_connect_step_out(
                B=B,
                K=K,
                device=device,
                dtype=dtype,
                cur_idx=cur_idx_a,
                target_idx_other=target_idx_other_a,
            ),
            _empty_connect_step_out(
                B=B,
                K=K,
                device=device,
                dtype=dtype,
                cur_idx=cur_idx_b,
                target_idx_other=target_idx_other_b,
            ),
        )

    P_flat, J_flat, offsets = _build_projector_jacobian_flat_and_offsets(
        banks=banks_ab,
        device=device,
        dtype=dtype,
        cache=projector_cache,
    )
    global_ts = offsets.view(2 * B, 1).expand(2 * B, K) + cur_ts_id_ab
    global_ts_flat = global_ts.reshape(BK2)
    P = P_flat.index_select(0, global_ts_flat)
    J_cur = J_flat.index_select(0, global_ts_flat)

    step_size = float(cfg.step_size)
    cur_q_flat = cur_q_ab.reshape(BK2, D)
    q_tgt_flat = q_target_ab.reshape(BK2, D)
    v = q_tgt_flat - cur_q_flat
    v_tan = project_vector_to_tangent(P, v)
    nrm = torch.linalg.norm(v_tan, dim=-1).clamp_min(1e-12)
    dir_tan = v_tan / nrm.unsqueeze(-1)
    q_next = cur_q_flat + step_size * dir_tan

    q_next_active = q_next.index_select(0, idx)
    cur_q_active = cur_q_flat.index_select(0, idx)
    J_cur_active = J_cur.index_select(0, idx)

    q_free_active = checker.get_collision_free_mask(q_next_active, margin=0.0)
    edge_res = edge_checker.check_edges_batch(
        cur_q_active,
        q_next_active,
        step_q=float(cfg.edge_step_q),
        max_steps=int(cfg.edge_max_steps),
    )
    edge_free_active = edge_res.bool() if torch.is_tensor(edge_res) else (~edge_res.edge_in_collision.bool())
    point_collision_active = ~q_free_active
    edge_collision_active = ~edge_free_active
    collision_blocked_active = (~q_free_active) | (~edge_free_active)
    free_active = q_free_active & edge_free_active

    res = projector.residual(q_next_active)
    res_norm = torch.linalg.norm(res, dim=-1)
    need_proj_active = res_norm > EM
    q_next2_active = q_next_active
    was_proj_active = torch.zeros((q_next_active.shape[0],), device=device, dtype=torch.bool)
    projection_failed_active = torch.zeros((q_next_active.shape[0],), device=device, dtype=torch.bool)

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
            projection_failed_active.index_fill_(0, failed_local, True)

        ok_local = idx_local[succ_local]
        if ok_local.numel() > 0:
            q_ok = q_next2_active.index_select(0, ok_local)
            q_free_ok = checker.get_collision_free_mask(q_ok, margin=0.0)
            cur_ok = cur_q_active.index_select(0, ok_local)
            edge2 = edge_checker.check_edges_batch(
                cur_ok,
                q_ok,
                step_q=float(cfg.edge_step_q),
                max_steps=int(cfg.edge_max_steps),
            )
            edge2_free = edge2.bool() if torch.is_tensor(edge2) else (~edge2.edge_in_collision.bool())
            free_ok = q_free_ok & edge2_free
            collision_ok = (~q_free_ok) | (~edge2_free)
            point_collision_active.index_copy_(0, ok_local, ~q_free_ok)
            edge_collision_active.index_copy_(0, ok_local, ~edge2_free)
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
    add_mask_a_pre = add_mask_ab[:B].clone()
    add_mask_b_pre = add_mask_ab[B:].clone()
    add_mask_a = _filter_candidate_add_mask(
        add_mask=add_mask_a_pre,
        tree_q=tree_q_a,
        tree_n=tree_a.n_nodes,
        q_parent=cur_q_a,
        q_target=q_target_a,
        q_new=q_next2_a,
        cfg=cfg,
    )
    add_mask_b = _filter_candidate_add_mask(
        add_mask=add_mask_b_pre,
        tree_q=tree_q_b,
        tree_n=tree_b.n_nodes,
        q_parent=cur_q_b,
        q_target=q_target_b,
        q_new=q_next2_b,
        cfg=cfg,
    )
    filter_rejected_a = add_mask_a_pre & (~add_mask_a)
    filter_rejected_b = add_mask_b_pre & (~add_mask_b)

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

    new_ts_id_a = cur_ts_id_a.clone()
    new_ts_id_b = cur_ts_id_b.clone()
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
        parent_idx=cur_idx_a,
        ts_id=new_ts_id_a,
        mask=add_mask_a,
        is_proj_root=was_proj_a,
        trace_iter_idx=int(iter_idx),
        trace_phase="connect",
    )
    new_idx_b = tree_b.add_nodes_k_per_batch(
        q_new=q_next2_b,
        parent_idx=cur_idx_b,
        ts_id=new_ts_id_b,
        mask=add_mask_b,
        is_proj_root=was_proj_b,
        trace_iter_idx=int(iter_idx),
        trace_phase="connect",
    )

    advanced_a = (new_idx_a >= 0) & mask_a
    advanced_b = (new_idx_b >= 0) & mask_b
    if bool(advanced_a.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=cur_ts_id_a,
            assigned_ts_id=new_ts_id_a,
            q_new=q_next2_a,
            added_mask=advanced_a,
            was_proj=was_proj_a,
            cfg=cfg,
        )
    if bool(advanced_b.any()):
        _update_ts_bank_after_add(
            banks=banks,
            used_ts_id=cur_ts_id_b,
            assigned_ts_id=new_ts_id_b,
            q_new=q_next2_b,
            added_mask=advanced_b,
            was_proj=was_proj_b,
            cfg=cfg,
        )
    if bool(collision_blocked_a.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=cur_ts_id_a,
            collision_mask=collision_blocked_a,
        )
    if bool(collision_blocked_b.any()):
        _update_ts_bank_after_collision(
            banks=banks,
            used_ts_id=cur_ts_id_b,
            collision_mask=collision_blocked_b,
        )

    progress_a = torch.zeros((B, K), device=device, dtype=dtype)
    progress_b = torch.zeros((B, K), device=device, dtype=dtype)
    reached_a = torch.zeros((B, K), device=device, dtype=torch.bool)
    reached_b = torch.zeros((B, K), device=device, dtype=torch.bool)
    reached_idx_a = torch.full((B, K), -1, device=device, dtype=torch.long)
    reached_idx_b = torch.full((B, K), -1, device=device, dtype=torch.long)

    if bool(advanced_a.any()):
        tree_q_a, _ = tree_a.get_nodes()
        cur_idx2_a = torch.where(advanced_a, new_idx_a, cur_idx_a)
        cur_q2_a = tree_q_a[b_idx, cur_idx2_a.clamp_min(0), :]
        dist_prev_a = torch.linalg.norm(q_target_a - cur_q_a, dim=-1)
        dist_a = torch.linalg.norm(q_target_a - cur_q2_a, dim=-1)
        progress_a = torch.where(advanced_a, dist_prev_a - dist_a, progress_a)
        reached_a = advanced_a & (dist_a <= float(cfg.goal_threshold))
        reached_idx_a = torch.where(reached_a, cur_idx2_a, reached_idx_a)

    if bool(advanced_b.any()):
        tree_q_b, _ = tree_b.get_nodes()
        cur_idx2_b = torch.where(advanced_b, new_idx_b, cur_idx_b)
        cur_q2_b = tree_q_b[b_idx, cur_idx2_b.clamp_min(0), :]
        dist_prev_b = torch.linalg.norm(q_target_b - cur_q_b, dim=-1)
        dist_b = torch.linalg.norm(q_target_b - cur_q2_b, dim=-1)
        progress_b = torch.where(advanced_b, dist_prev_b - dist_b, progress_b)
        reached_b = advanced_b & (dist_b <= float(cfg.goal_threshold))
        reached_idx_b = torch.where(reached_b, cur_idx2_b, reached_idx_b)

    rec_a = getattr(tree_a, "trace_recorder", None)
    rec_b = getattr(tree_b, "trace_recorder", None)
    wants_summary = (
        (rec_a is not None and rec_a.wants("summary"))
        or (rec_b is not None and rec_b.wants("summary"))
    )
    if wants_summary:
        point_collision_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
        edge_collision_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
        projection_needed_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
        projection_failed_flat = torch.zeros((BK2,), device=device, dtype=torch.bool)
        point_collision_flat.index_copy_(0, idx, point_collision_active)
        edge_collision_flat.index_copy_(0, idx, edge_collision_active)
        projection_needed_flat.index_copy_(0, idx, need_proj_active)
        projection_failed_flat.index_copy_(0, idx, projection_failed_active)
        point_collision_ab = point_collision_flat.view(2 * B, K) & mask_ab
        edge_collision_ab = edge_collision_flat.view(2 * B, K) & mask_ab
        projection_needed_ab = projection_needed_flat.view(2 * B, K) & mask_ab
        projection_failed_ab = projection_failed_flat.view(2 * B, K) & mask_ab

        _record_connect_step_summary(
            iter_idx=int(iter_idx),
            tree=tree_a,
            mask=mask_a,
            advanced=advanced_a,
            point_collision=point_collision_ab[:B],
            edge_collision=edge_collision_ab[:B],
            collision_blocked=collision_blocked_a,
            projection_needed=projection_needed_ab[:B],
            projection_failed=projection_failed_ab[:B],
            was_proj=was_proj_a,
            filter_rejected=filter_rejected_a,
            reached=reached_a,
            progress=progress_a,
        )
        _record_connect_step_summary(
            iter_idx=int(iter_idx),
            tree=tree_b,
            mask=mask_b,
            advanced=advanced_b,
            point_collision=point_collision_ab[B:],
            edge_collision=edge_collision_ab[B:],
            collision_blocked=collision_blocked_b,
            projection_needed=projection_needed_ab[B:],
            projection_failed=projection_failed_ab[B:],
            was_proj=was_proj_b,
            filter_rejected=filter_rejected_b,
            reached=reached_b,
            progress=progress_b,
        )

    return (
        ConnectStepOut(advanced_a, new_idx_a, cur_idx_a.clone(), progress_a, reached_a, reached_idx_a, target_idx_other_a),
        ConnectStepOut(advanced_b, new_idx_b, cur_idx_b.clone(), progress_b, reached_b, reached_idx_b, target_idx_other_b),
    )


__all__ = [
    "ConnectStepOut",
    "connect_one_step_with_state",
    "connect_two_trees_one_step_with_state",
    "_filter_candidate_add_mask",
]

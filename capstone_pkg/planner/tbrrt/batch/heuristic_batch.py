from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from capstone_pkg.planner.tbrrt.ts_bank import TSBank
from capstone_pkg.utils.joint_limit import JointLimitsTorch

from .nn_batch import nn_1_tree_all_candidates_cdist
from .prealloc import BatchConextPrealloc
from .tree_batch import TreeBatchGPU


@torch.no_grad()
def _sample_ts_ids_from_banks_batch(
    *,
    banks: List[TSBank],
    active_rows: torch.Tensor,  # (B,) bool
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = active_rows.device
    B = len(banks)
    weights = [bank.get_sampling_weights(device=device, dtype=torch.float32) for bank in banks]
    w_pad = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True)  # (B,Smax)
    w_sum = w_pad.sum(dim=1, keepdim=True)
    bad = w_sum.squeeze(1) <= 0
    w_pad[:, 0] = torch.where(bad, torch.ones_like(w_pad[:, 0]), w_pad[:, 0])
    w_sum = w_pad.sum(dim=1, keepdim=True)
    w_pad = w_pad / w_sum.clamp_min(1e-12)

    sampled = torch.multinomial(w_pad, num_samples=1, replacement=True, generator=generator).squeeze(1).to(torch.long)
    minus_one = torch.full((B,), -1, device=device, dtype=torch.long)
    return torch.where(active_rows, sampled, minus_one)


@torch.no_grad()
def _sample_in_selected_ts_ball_batch(
    *,
    roots: torch.Tensor,      # (B,D)
    basis: torch.Tensor,      # (B,D,Kmax)
    dim: torch.Tensor,        # (B,)
    domain: torch.Tensor,     # (B,)
    n_per_batch: torch.Tensor,  # (B,)
    joint_limits: JointLimitsTorch,
    q_target: Optional[torch.Tensor] = None,  # (B,D)
    enable_halfspace: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D = roots.shape
    n_max = int(n_per_batch.max().item()) if B > 0 else 0
    if n_max <= 0:
        empty_q = torch.zeros((B, 0, D), device=roots.device, dtype=roots.dtype)
        empty_m = torch.zeros((B, 0), device=roots.device, dtype=torch.bool)
        return empty_q, empty_m

    valid_sample = torch.arange(n_max, device=roots.device).view(1, n_max) < n_per_batch.view(B, 1)
    kmax = int(basis.shape[2])

    if kmax <= 0:
        q = roots.unsqueeze(1).expand(B, n_max, D).contiguous()
        q = joint_limits.clamp(q.view(B * n_max, D)).view(B, n_max, D)
        return q, valid_sample

    dtype = roots.dtype
    v = torch.randn((B, n_max, kmax), device=roots.device, dtype=dtype, generator=generator)
    k_mask = torch.arange(kmax, device=roots.device).view(1, 1, kmax) < dim.view(B, 1, 1)
    v = v * k_mask
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12)

    if enable_halfspace and (q_target is not None):
        d_amb = (q_target - roots).to(device=roots.device, dtype=dtype)
        d_tan = torch.einsum("bd,bdk->bk", d_amb, basis)
        d_tan = d_tan * (torch.arange(kmax, device=roots.device).view(1, kmax) < dim.view(B, 1))
        d_tan = d_tan / torch.linalg.norm(d_tan, dim=-1, keepdim=True).clamp_min(1e-12)
        flip = (v * d_tan.unsqueeze(1)).sum(dim=-1) < 0.0
        v = torch.where(flip.unsqueeze(-1), -v, v)

    u = torch.rand((B, n_max, 1), device=roots.device, dtype=dtype, generator=generator)
    kf = dim.to(dtype).clamp_min(1.0).view(B, 1, 1)
    r = (u ** (1.0 / kf)) * domain.view(B, 1, 1)
    a = v * r
    q = roots.unsqueeze(1) + torch.einsum("bnk,bdk->bnd", a, basis)

    is0 = dim == 0
    q = torch.where(is0.view(B, 1, 1), roots.unsqueeze(1), q)

    q = joint_limits.clamp(q.view(B * n_max, D)).view(B, n_max, D)
    return q, valid_sample


@torch.no_grad()
def _apply_overlap_discard_qrand_batch(
    *,
    tree: TreeBatchGPU,
    q_rand: torch.Tensor,      # (B,K,D)
    mask: torch.Tensor,        # (B,K)
    roots: torch.Tensor,       # (B,D)
    basis: torch.Tensor,       # (B,D,Kmax)
    dim: torch.Tensor,         # (B,)
    domain: torch.Tensor,      # (B,)
    joint_limits: JointLimitsTorch,
    q_target: Optional[torch.Tensor] = None,  # (B,D)
    enable_halfspace: bool = True,
    max_tries: int = 20,
    generator: Optional[torch.Generator] = None,
    prealloc: Optional[BatchConextPrealloc] = None,
) -> torch.Tensor:
    if max_tries <= 0 or (not bool(mask.any())):
        return q_rand

    tree_q_all, tree_n_all = tree.get_nodes()
    out = q_rand.clone()
    B, K, _ = out.shape
    device = out.device
    if prealloc is not None and prealloc.supports(B=B, K=K, D=tree.D, device=device, dtype=out.dtype):
        slot_ids = prealloc.slot_ids(K)
        row_ids = prealloc.row_ids(K)
    else:
        slot_ids = torch.arange(K, device=device).view(1, K).expand(B, K)
        row_ids = torch.arange(B, device=device).view(B, 1).expand(B, K)

    for attempt in range(max_tries):
        nn_idx = nn_1_tree_all_candidates_cdist(
            tree_q_all,
            tree_n_all,
            out,
            exclude_mask=getattr(tree, "blocked_node", getattr(tree, "banned_node", None)),
        )
        nn_safe = nn_idx.clamp_min(0)
        invalid = mask & (nn_idx < 0)
        invalid = invalid | (mask & (
            tree.is_proj_root[row_ids, nn_safe]
            | tree.is_parent_of_proj_root[row_ids, nn_safe]
        ))
        if not bool(invalid.any()):
            break
        if attempt + 1 >= max_tries:
            break

        n_replace = invalid.sum(dim=1).to(torch.long)
        q_new, _ = _sample_in_selected_ts_ball_batch(
            roots=roots,
            basis=basis,
            dim=dim,
            domain=domain,
            n_per_batch=n_replace,
            joint_limits=joint_limits,
            q_target=q_target,
            enable_halfspace=enable_halfspace,
            generator=generator,
        )
        max_replace = int(q_new.shape[1])
        if max_replace <= 0:
            break

        large = torch.full((B, K), K + 1, device=device, dtype=torch.long)
        sort_key = torch.where(invalid, slot_ids, large)
        slot_order = torch.argsort(sort_key, dim=1)
        slot_sel = slot_order[:, :max_replace]
        rank = torch.arange(max_replace, device=device).view(1, max_replace)
        repl_valid = rank < n_replace.view(B, 1)
        repl_rows = torch.nonzero(repl_valid, as_tuple=False)
        if repl_rows.numel() == 0:
            break

        b_rep = repl_rows[:, 0]
        r_rep = repl_rows[:, 1]
        dst_cols = slot_sel[b_rep, r_rep]
        out[b_rep, dst_cols, :] = q_new[b_rep, r_rep, :]

    return out


@torch.no_grad()
def _select_escape_q_hint(
    *,
    other: TreeBatchGPU,
    need_esc: torch.Tensor,   # (B,K)
    tgt_other: torch.Tensor,  # (B,K)
    device: torch.device,
    dtype: torch.dtype,
    prealloc: Optional[BatchConextPrealloc] = None,
) -> torch.Tensor:
    root_hint = other.q[:, 0, :].to(device=device, dtype=dtype)
    target_valid = need_esc & (tgt_other >= 0)
    B, K = target_valid.shape
    tree_q_all, _ = other.get_nodes()
    if prealloc is not None and prealloc.supports(B=B, K=K, D=other.D, device=device, dtype=dtype):
        b_idx = prealloc.b_ids
        slot_rank = prealloc.slot_ids(K)
    else:
        b_idx = torch.arange(B, device=device)
        slot_rank = torch.arange(K, device=device).view(1, K).expand(B, K)
    large = torch.full((B, K), K + 1, device=device, dtype=torch.long)
    banned_nodes = getattr(other, "blocked_node", getattr(other, "banned_node", None))
    if banned_nodes is not None:
        safe_tgt = tgt_other.clamp(min=0, max=max(0, int(banned_nodes.shape[1]) - 1))
        row_ids = torch.arange(B, device=device).view(B, 1).expand(B, K)
        target_valid = target_valid & (~banned_nodes[row_ids, safe_tgt])
    first_slot = torch.argmin(torch.where(target_valid, slot_rank, large), dim=1)
    chosen_target = tgt_other.gather(1, first_slot.unsqueeze(1)).squeeze(1).clamp_min(0)
    target_hint = tree_q_all[b_idx, chosen_target, :].to(device=device, dtype=dtype)
    has_target = target_valid.any(dim=1, keepdim=True)
    return torch.where(has_target, target_hint, root_hint)


@torch.no_grad()
def _select_connect_target_idx_batch(
    *,
    other: TreeBatchGPU,
    q_query: torch.Tensor,     # (B,K,D)
    active_mask: torch.Tensor, # (B,K)
    discard_overlap: bool,
    exclude_idx: Optional[torch.Tensor] = None,  # (B,K), -1 allowed
    prealloc: Optional[BatchConextPrealloc] = None,
) -> torch.Tensor:
    tree_q_all, tree_n_all = other.get_nodes()
    B, N_full, _ = tree_q_all.shape
    _, K, _ = q_query.shape
    device = q_query.device
    dtype = q_query.dtype

    if prealloc is not None and prealloc.supports(B=B, K=K, D=other.D, device=device, dtype=dtype):
        minus_one_out = prealloc.minus_one_long(K).clone()
    else:
        minus_one_out = torch.full((B, K), -1, device=device, dtype=torch.long)

    if not bool(active_mask.any()):
        return minus_one_out

    max_n = int(tree_n_all.max().item()) if tree_n_all.numel() > 0 else 0
    max_n = max(0, min(max_n, int(N_full)))
    if max_n <= 0:
        return minus_one_out

    active_cols = active_mask.any(dim=0)
    active_col_idx = torch.nonzero(active_cols, as_tuple=False).view(-1)
    if active_col_idx.numel() == 0:
        return minus_one_out

    q_query_eff = q_query.index_select(1, active_col_idx)
    active_eff = active_mask.index_select(1, active_col_idx)
    K_eff = int(active_col_idx.numel())
    tree_q_eff = tree_q_all[:, :max_n, :]
    N = max_n

    dist = torch.cdist(q_query_eff, tree_q_eff)  # (B,K_eff,N_live)
    n_idx = torch.arange(N, device=device).view(1, 1, N)
    valid_nodes = n_idx < tree_n_all.view(B, 1, 1)
    banned_nodes = getattr(other, "blocked_node", getattr(other, "banned_node", None))
    if banned_nodes is not None:
        valid_nodes = valid_nodes & (~banned_nodes[:, :N].bool().view(B, 1, N))
    dist = dist.masked_fill(~valid_nodes, float("inf"))

    orig_idx = torch.argmin(dist, dim=-1)
    orig_dist = dist.gather(-1, orig_idx.unsqueeze(-1)).squeeze(-1)
    orig_idx = torch.where(torch.isfinite(orig_dist), orig_idx, torch.full_like(orig_idx, -1))
    dist_work = dist
    if exclude_idx is not None:
        if exclude_idx.shape != (B, K):
            raise ValueError(f"exclude_idx shape {tuple(exclude_idx.shape)} != {(B, K)}")
        exclude_eff = exclude_idx.index_select(1, active_col_idx)
        excluded = (exclude_eff >= 0).view(B, K_eff, 1) & (n_idx == exclude_eff.clamp_min(0).view(B, K_eff, 1))
        dist_work = dist.masked_fill(excluded, float("inf"))

    base_idx = torch.argmin(dist_work, dim=-1)
    base_dist = dist_work.gather(-1, base_idx.unsqueeze(-1)).squeeze(-1)
    base_idx = torch.where(torch.isfinite(base_dist), base_idx, orig_idx)
    sel_idx = base_idx

    if discard_overlap:
        overlap_nodes = (other.is_proj_root[:, :N] | other.is_parent_of_proj_root[:, :N]).view(B, 1, N)
        dist_non_overlap = dist_work.masked_fill(overlap_nodes, float("inf"))
        non_overlap_idx = torch.argmin(dist_non_overlap, dim=-1)
        non_overlap_dist = dist_non_overlap.gather(-1, non_overlap_idx.unsqueeze(-1)).squeeze(-1)
        has_non_overlap = torch.isfinite(non_overlap_dist)
        sel_idx = torch.where(has_non_overlap, non_overlap_idx, base_idx)

    empty = tree_n_all <= 0
    sel_idx = sel_idx.masked_fill(empty.view(B, 1), -1)

    out_eff = torch.where(active_eff, sel_idx, torch.full_like(sel_idx, -1))
    minus_one_out.index_copy_(1, active_col_idx, out_eff)
    return minus_one_out


__all__ = [
    "_sample_ts_ids_from_banks_batch",
    "_sample_in_selected_ts_ball_batch",
    "_apply_overlap_discard_qrand_batch",
    "_select_escape_q_hint",
    "_select_connect_target_idx_batch",
]

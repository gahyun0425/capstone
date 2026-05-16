from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import torch

from capstone_pkg.collision_check.collision import SelfCollisionChecker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjectorTorch as ManifoldProjector
from capstone_pkg.planner.tbrrt.config import TBRRTConfig
from capstone_pkg.planner.tbrrt.ts_bank import TSBank
from capstone_pkg.utils.joint_limit import JointLimitsTorch

from .prealloc import BatchConextPrealloc
from .projector_stack import ProjectorStackCache
from .tree_batch import TreeBatchGPU


class _BatchEscapeTreeView:
    def __init__(
        self,
        *,
        D: int,
        q: torch.Tensor,
        n_nodes: torch.Tensor,
        is_proj_root: torch.Tensor,
        is_parent_of_proj_root: torch.Tensor,
        banned_node: torch.Tensor,
    ) -> None:
        self.D = D
        self.q = q
        self.n_nodes = n_nodes
        self.is_proj_root = is_proj_root
        self.is_parent_of_proj_root = is_parent_of_proj_root
        self.banned_node = banned_node
        self.blocked_node = banned_node

    def get_nodes(self):
        return self.q, self.n_nodes


def _pad_dim1(t: torch.Tensor, cap: int) -> torch.Tensor:
    if int(t.shape[1]) == cap:
        return t
    out_shape = list(t.shape)
    out_shape[1] = cap
    out = t.new_zeros(tuple(out_shape))
    out[:, : int(t.shape[1]), ...] = t
    return out


def _record_spawn_per_slot(
    *,
    tree: TreeBatchGPU,
    offset: int,
    B: int,
    iter_idx: int,
    need_esc_ab: torch.Tensor,
    active_ab: torch.Tensor,
    source_b: torch.Tensor,
    assigned_by_source: torch.Tensor,
    source_ts: torch.Tensor,
    dim_src: torch.Tensor,
    domain_src: torch.Tensor,
    escape_spawn_blocks: int,
    cfg: TBRRTConfig,
) -> None:
    rec = getattr(tree, "trace_recorder", None)
    if rec is None or not rec.wants("summary"):
        return
    tree_name = str(getattr(tree, "trace_tree_name", ""))
    for src_i in range(int(source_b.numel())):
        row = int(source_b[src_i].item())
        if row < offset or row >= offset + B:
            continue
        b = row - offset
        rec.record_escape_spawn_batch(
            iter_idx=int(iter_idx),
            tree=tree_name,
            batch_idx=int(b),
            need_slots=1,
            candidate_slots=int((need_esc_ab[row] | (~active_ab[row])).sum().item()),
            assigned_slots=int(assigned_by_source[src_i].item()),
            ts_id=int(source_ts[src_i].item()),
            ts_dim=int(dim_src[src_i].item()),
            ts_domain=float(domain_src[src_i].item()),
            n_draw=escape_spawn_blocks,
            enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
            enable_overlap_discard=bool(getattr(cfg, "enable_overlap_discard", True)),
        )


def _record_escape_result(
    *,
    tree: TreeBatchGPU,
    esc_mask: torch.Tensor,
    ok: torch.Tensor,
    fail: torch.Tensor,
    iter_idx: int,
    n_escape_steps: int,
) -> None:
    rec = getattr(tree, "trace_recorder", None)
    if rec is None or not rec.wants("summary"):
        return
    rec.record_escape_result(
        iter_idx=int(iter_idx),
        tree=str(getattr(tree, "trace_tree_name", "")),
        spawned=int(esc_mask.sum().item()),
        succeeded=int(ok.sum().item()),
        failed=int(fail.sum().item()),
        extend_steps=int(n_escape_steps),
    )


def _pack_ts_banks_for_rows(
    banks: List[TSBank],
    *,
    device: torch.device,
    dtype: torch.dtype,
    prealloc: Optional[BatchConextPrealloc],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-row TSBank objects into padded tensors for batched slot sampling."""
    rows = len(banks)
    smax = max(1, max((len(bank.spaces) for bank in banks), default=1))
    if prealloc is not None and prealloc.device == device and prealloc.dtype == dtype and prealloc.D > 0:
        prealloc.ensure_ts_scratch(rows=rows, capacity=smax)
        if (
            prealloc.ts_roots is not None
            and prealloc.ts_basis is not None
            and prealloc.ts_dim is not None
            and prealloc.ts_domain is not None
            and prealloc.ts_weight is not None
            and prealloc.ts_valid is not None
        ):
            roots = prealloc.ts_roots[:rows, :smax, :]
            basis = prealloc.ts_basis[:rows, :smax, :, :]
            dim = prealloc.ts_dim[:rows, :smax]
            domain = prealloc.ts_domain[:rows, :smax]
            weight = prealloc.ts_weight[:rows, :smax]
            valid = prealloc.ts_valid[:rows, :smax]
        else:
            roots = torch.empty((rows, smax, prealloc.D), device=device, dtype=dtype)
            basis = torch.empty((rows, smax, prealloc.D, prealloc.D), device=device, dtype=dtype)
            dim = torch.empty((rows, smax), device=device, dtype=torch.long)
            domain = torch.empty((rows, smax), device=device, dtype=dtype)
            weight = torch.empty((rows, smax), device=device, dtype=torch.float32)
            valid = torch.empty((rows, smax), device=device, dtype=torch.bool)
    else:
        D = int(banks[0].spaces[0].root.numel()) if rows > 0 and len(banks[0].spaces) > 0 else 0
        roots = torch.empty((rows, smax, D), device=device, dtype=dtype)
        basis = torch.empty((rows, smax, D, D), device=device, dtype=dtype)
        dim = torch.empty((rows, smax), device=device, dtype=torch.long)
        domain = torch.empty((rows, smax), device=device, dtype=dtype)
        weight = torch.empty((rows, smax), device=device, dtype=torch.float32)
        valid = torch.empty((rows, smax), device=device, dtype=torch.bool)

    roots.zero_()
    basis.zero_()
    dim.zero_()
    domain.fill_(1.0)
    weight.zero_()
    valid.zero_()
    counts = torch.zeros((rows,), device=device, dtype=torch.long)

    for row, bank in enumerate(banks):
        roots_i, basis_i, dim_i, domain_i, weight_i = bank.pack_tensors()
        n_i = int(roots_i.shape[0])
        k_i = int(basis_i.shape[2])
        roots[row, :n_i, :].copy_(roots_i.to(device=device, dtype=dtype))
        basis[row, :n_i, :, :k_i].copy_(basis_i.to(device=device, dtype=dtype))
        dim[row, :n_i].copy_(dim_i.to(device=device))
        domain[row, :n_i].copy_(domain_i.to(device=device, dtype=dtype))
        weight[row, :n_i].copy_(weight_i.to(device=device, dtype=torch.float32))
        valid[row, :n_i] = True
        counts[row] = n_i

    weight_sum = weight.sum(dim=1, keepdim=True)
    bad = weight_sum.squeeze(1) <= 0.0
    if bool(bad.any()):
        weight[bad, 0] = 1.0
        valid[bad, 0] = True
        counts[bad] = 1
        weight_sum = weight.sum(dim=1, keepdim=True)
    weight.div_(weight_sum.clamp_min(1e-12))
    return roots, basis, dim, domain, weight, valid, counts


def _sample_ts_ids_for_escape_slot_matrix(
    *,
    weight: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    need_esc: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Return a TS id for every ESCAPE slot using batched GPU sampling."""
    rows, K = need_esc.shape
    S = int(weight.shape[1])
    if S <= 0:
        return torch.full((rows, K), -1, device=need_esc.device, dtype=torch.long)

    max_need = int(need_esc.sum(dim=1).max().item())
    if max_need <= 0:
        return torch.full((rows, K), -1, device=need_esc.device, dtype=torch.long)

    # Weighted sampling without replacement via Gumbel top-k.  We only compute
    # the first max_need entries instead of sorting all TS ids for every row.
    k_top = min(max_need, S)
    u = torch.rand(weight.shape, device=weight.device, dtype=torch.float32, generator=generator).clamp_(1e-12, 1.0 - 1e-12)
    gumbel = -torch.log(-torch.log(u))
    scores = torch.where(valid, torch.log(weight.clamp_min(1e-12)) + gumbel, torch.full_like(weight, -float("inf")))
    distinct = torch.topk(scores, k=k_top, dim=1).indices

    # If one row needs more escape TS choices than it has valid TS entries,
    # fill the remaining slots with weighted sampling with replacement.
    replacement = torch.multinomial(weight, num_samples=max_need, replacement=True, generator=generator)

    source_rank = torch.cumsum(need_esc.to(torch.long), dim=1) - 1
    rank_s = source_rank.clamp(min=0, max=k_top - 1)
    rank_k = source_rank.clamp(min=0, max=max_need - 1)
    distinct_choice = distinct.gather(1, rank_s)
    replacement_choice = replacement.gather(1, rank_k)
    use_distinct = source_rank < counts.view(rows, 1)
    ts = torch.where(use_distinct, distinct_choice, replacement_choice)
    return torch.where(need_esc, ts, torch.full_like(ts, -1))


@torch.no_grad()
def _apply_overlap_discard_qrand_slots(
    *,
    tree: Any,
    q_rand: torch.Tensor,
    mask: torch.Tensor,
    slot_source: torch.Tensor,
    roots: torch.Tensor,
    basis: torch.Tensor,
    dim: torch.Tensor,
    domain: torch.Tensor,
    joint_limits: JointLimitsTorch,
    q_target: Optional[torch.Tensor],
    enable_halfspace: bool,
    max_tries: int,
    generator: Optional[torch.Generator],
    nn_1_tree_all_candidates_cdist,
    _sample_in_selected_ts_ball_batch,
) -> torch.Tensor:
    if max_tries <= 0 or (not bool(mask.any())):
        return q_rand

    tree_q_all, tree_n_all = tree.get_nodes()
    out = q_rand.clone()
    B, K, _ = out.shape
    device = out.device
    row_ids = torch.arange(B, device=device).view(B, 1).expand(B, K)

    for attempt in range(max_tries):
        nn_idx = nn_1_tree_all_candidates_cdist(
            tree_q_all,
            tree_n_all,
            out,
            exclude_mask=getattr(tree, "blocked_node", getattr(tree, "banned_node", None)),
        )
        nn_safe = nn_idx.clamp_min(0)
        invalid = mask & (slot_source >= 0) & (nn_idx < 0)
        invalid = invalid | (mask & (slot_source >= 0) & (
            tree.is_proj_root[row_ids, nn_safe]
            | tree.is_parent_of_proj_root[row_ids, nn_safe]
        ))
        if not bool(invalid.any()):
            break
        if attempt + 1 >= max_tries:
            break

        invalid_rows = torch.nonzero(invalid, as_tuple=False)
        if invalid_rows.numel() == 0:
            break
        src = slot_source[invalid_rows[:, 0], invalid_rows[:, 1]].to(torch.long)
        q_new, _ = _sample_in_selected_ts_ball_batch(
            roots=roots.index_select(0, src),
            basis=basis.index_select(0, src),
            dim=dim.index_select(0, src),
            domain=domain.index_select(0, src),
            n_per_batch=torch.ones((src.numel(),), device=device, dtype=torch.long),
            joint_limits=joint_limits,
            q_target=(q_target.index_select(0, src) if q_target is not None else None),
            enable_halfspace=enable_halfspace,
            generator=generator,
        )
        if q_new.numel() == 0:
            break
        out[invalid_rows[:, 0], invalid_rows[:, 1], :] = q_new[:, 0, :]

    return out


@torch.no_grad()
def _build_per_slot_ts_escape_spawn(
    *,
    tree: Any,
    other_q_all: torch.Tensor,
    other_n_nodes: torch.Tensor,
    other_blocked: Optional[torch.Tensor],
    active: torch.Tensor,
    need_esc: torch.Tensor,
    tgt_other: torch.Tensor,
    valid_b: torch.Tensor,
    banks: List[TSBank],
    joint_limits: JointLimitsTorch,
    cfg: TBRRTConfig,
    _gather_selected_ts_tensors,
    _sample_in_selected_ts_ball_batch,
    nn_1_tree_all_candidates_cdist,
    generator: Optional[torch.Generator] = None,
    prealloc: Optional[BatchConextPrealloc] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = active.device
    dtype = tree.q.dtype
    rows, Kmax = active.shape
    D = int(tree.D)
    escape_spawn_blocks = max(1, int(getattr(cfg, "escape_spawn_blocks", 3)))

    if prealloc is not None and prealloc.supports(B=prealloc.B, K=Kmax, D=D, device=device, dtype=dtype):
        if rows == prealloc.B:
            esc_mask = prealloc.escape_mask[:, :Kmax]
            esc_qrand = prealloc.escape_qrand[:, :Kmax, :]
            esc_parent = prealloc.escape_parent[:, :Kmax]
        elif rows == 2 * prealloc.B:
            esc_mask = prealloc.escape_mask_ab[:, :Kmax]
            esc_qrand = prealloc.escape_qrand_ab[:, :Kmax, :]
            esc_parent = prealloc.escape_parent_ab[:, :Kmax]
        else:
            esc_mask = torch.zeros((rows, Kmax), device=device, dtype=torch.bool)
            esc_qrand = torch.zeros((rows, Kmax, D), device=device, dtype=dtype)
            esc_parent = torch.zeros((rows, Kmax), device=device, dtype=torch.long)
    else:
        esc_mask = torch.zeros((rows, Kmax), device=device, dtype=torch.bool)
        esc_qrand = torch.zeros((rows, Kmax, D), device=device, dtype=dtype)
        esc_parent = torch.zeros((rows, Kmax), device=device, dtype=torch.long)

    esc_mask.zero_()
    esc_qrand.zero_()
    esc_parent.zero_()
    slot_source = torch.full((rows, Kmax), -1, device=device, dtype=torch.long)

    active_rows = valid_b & need_esc.any(dim=1)
    if not bool(active_rows.any()):
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_float = torch.empty((0,), device=device, dtype=dtype)
        return esc_mask, esc_qrand, esc_parent, slot_source, empty_long, empty_long, empty_long, empty_long, empty_long, empty_float

    roots_all, basis_all, dim_all, domain_all, weight_all, valid_ts, ts_counts = _pack_ts_banks_for_rows(
        banks,
        device=device,
        dtype=dtype,
        prealloc=prealloc,
    )
    source_ts_by_slot = _sample_ts_ids_for_escape_slot_matrix(
        weight=weight_all,
        valid=valid_ts,
        counts=ts_counts,
        need_esc=need_esc & active_rows.view(rows, 1),
        generator=generator,
    )

    src_linear = torch.nonzero((source_ts_by_slot >= 0).reshape(-1), as_tuple=False).view(-1)
    if src_linear.numel() == 0:
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_float = torch.empty((0,), device=device, dtype=dtype)
        return esc_mask, esc_qrand, esc_parent, slot_source, empty_long, empty_long, empty_long, empty_long, empty_long, empty_float

    source_b = torch.div(src_linear, Kmax, rounding_mode="floor")
    source_slot = src_linear.remainder(Kmax)
    source_ts = source_ts_by_slot.reshape(-1).index_select(0, src_linear)
    n_sources = int(src_linear.numel())

    roots_src = roots_all[source_b, source_ts, :]
    basis_src = basis_all[source_b, source_ts, :, :]
    dim_src = dim_all[source_b, source_ts]
    domain_src = domain_all[source_b, source_ts]

    target_idx = tgt_other[source_b, source_slot]
    target_safe = target_idx.clamp(min=0, max=max(0, int(other_q_all.shape[1]) - 1))
    target_valid = (target_idx >= 0) & (target_idx < other_n_nodes.index_select(0, source_b))
    if other_blocked is not None:
        target_valid = target_valid & (~other_blocked[source_b, target_safe])
    root_hint = other_q_all[source_b, 0, :].to(device=device, dtype=dtype)
    target_hint = other_q_all[source_b, target_safe, :].to(device=device, dtype=dtype)
    q_hint_src = torch.where(target_valid.view(n_sources, 1), target_hint, root_hint)

    q_samples, _ = _sample_in_selected_ts_ball_batch(
        roots=roots_src,
        basis=basis_src,
        dim=dim_src,
        domain=domain_src,
        n_per_batch=torch.full((n_sources,), escape_spawn_blocks, device=device, dtype=torch.long),
        joint_limits=joint_limits,
        q_target=q_hint_src,
        enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
        generator=generator,
    )

    cand_slot_mask = need_esc | (~active)
    num_need = need_esc.sum(dim=1).to(torch.long)
    cand_count = cand_slot_mask.sum(dim=1).to(torch.long)
    max_assign = torch.minimum(cand_count, num_need * escape_spawn_blocks)
    max_assign = torch.where(active_rows, max_assign, torch.zeros_like(max_assign))

    if prealloc is not None and rows == prealloc.B and prealloc.Kmax >= Kmax:
        slot_ids = prealloc.slot_ids(Kmax)
    elif prealloc is not None and rows == 2 * prealloc.B and prealloc.Kmax >= Kmax:
        slot_ids = prealloc.slot_ids_ab(Kmax)
    else:
        slot_ids = torch.arange(Kmax, device=device, dtype=torch.long).view(1, Kmax).expand(rows, Kmax)

    prio = torch.full((rows, Kmax), 2, device=device, dtype=torch.long)
    prio = torch.where(need_esc, torch.zeros_like(prio), prio)
    prio = torch.where((~active) & (~need_esc), torch.ones_like(prio), prio)
    sort_key = torch.where(cand_slot_mask, prio * (Kmax + 1) + slot_ids, torch.full_like(prio, 3 * (Kmax + 1)))
    slot_order = torch.argsort(sort_key, dim=1)

    source_rank_by_slot = torch.cumsum(need_esc.to(torch.long), dim=1) - 1
    source_rank = source_rank_by_slot[source_b, source_slot]
    sample_rank = torch.arange(escape_spawn_blocks, device=device, dtype=torch.long).view(1, escape_spawn_blocks)
    extra_pos = num_need.index_select(0, source_b).view(n_sources, 1) + source_rank.view(n_sources, 1) * (escape_spawn_blocks - 1) + sample_rank - 1
    assign_pos = torch.where(sample_rank == 0, source_rank.view(n_sources, 1), extra_pos)
    assign_valid = assign_pos < max_assign.index_select(0, source_b).view(n_sources, 1)

    dst_slot = slot_order[source_b.view(n_sources, 1), assign_pos.clamp(min=0, max=Kmax - 1)]
    dst_row = source_b.view(n_sources, 1).expand(n_sources, escape_spawn_blocks)
    src_id = torch.arange(n_sources, device=device, dtype=torch.long).view(n_sources, 1).expand(n_sources, escape_spawn_blocks)

    flat_valid = assign_valid.reshape(-1)
    valid_idx = torch.nonzero(flat_valid, as_tuple=False).view(-1)
    flat_row = dst_row.reshape(-1).index_select(0, valid_idx)
    flat_slot = dst_slot.reshape(-1).index_select(0, valid_idx)
    flat_src = src_id.reshape(-1).index_select(0, valid_idx)
    flat_q = q_samples.reshape(n_sources * escape_spawn_blocks, D).index_select(0, valid_idx)

    esc_mask[flat_row, flat_slot] = True
    esc_qrand[flat_row, flat_slot, :] = flat_q
    slot_source[flat_row, flat_slot] = flat_src
    assigned_by_source = assign_valid.sum(dim=1).to(torch.long)

    if bool(getattr(cfg, "enable_overlap_discard", True)):
        esc_qrand = _apply_overlap_discard_qrand_slots(
            tree=tree,
            q_rand=esc_qrand,
            mask=esc_mask,
            slot_source=slot_source,
            roots=roots_src,
            basis=basis_src,
            dim=dim_src,
            domain=domain_src,
            joint_limits=joint_limits,
            q_target=q_hint_src,
            enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
            max_tries=int(getattr(cfg, "discard_overlap_max_tries", 20)),
            generator=generator,
            nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
            _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
        )

    tree_q_all, tree_n_all = tree.get_nodes()
    nn_idx = nn_1_tree_all_candidates_cdist(
        tree_q_all,
        tree_n_all,
        esc_qrand,
        exclude_mask=getattr(tree, "blocked_node", getattr(tree, "banned_node", None)),
    )
    esc_parent = torch.where(esc_mask, nn_idx, esc_parent)

    return (
        esc_mask,
        esc_qrand,
        esc_parent,
        slot_source,
        source_b,
        source_slot,
        source_ts,
        assigned_by_source,
        dim_src,
        domain_src,
    )


@torch.no_grad()
def _escape_one_tree(
    *,
    tree: TreeBatchGPU,
    other: TreeBatchGPU,
    active: torch.Tensor,
    mode: torch.Tensor,
    cur: torch.Tensor,
    seg: torch.Tensor,
    tgt_other: torch.Tensor,
    needEsc: torch.Tensor,
    valid_b: torch.Tensor,
    banks: List[TSBank],
    joint_limits: JointLimitsTorch,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    _sample_ts_ids_from_banks_batch,
    _gather_selected_ts_tensors,
    _sample_in_selected_ts_ball_batch,
    _apply_overlap_discard_qrand_batch,
    _select_escape_q_hint,
    _extend_one_step_from_parent,
    nn_1_tree_all_candidates_cdist,
    generator: Optional[torch.Generator] = None,
    projector_cache: ProjectorStackCache | None = None,
    counter_prefix: str = "",
    prealloc: Optional[BatchConextPrealloc] = None,
    after_step_connect: Optional[Callable[..., Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
    if not bool(needEsc.any()):
        return active, mode, cur, seg, tgt_other, torch.zeros_like(active), None

    device = tree.device
    dtype = tree.dtype
    B, Kmax = active.shape
    D = tree.D
    escape_spawn_blocks = max(1, int(getattr(cfg, "escape_spawn_blocks", 3)))
    use_prealloc = (
        prealloc is not None
        and prealloc.supports(B=B, K=Kmax, D=D, device=device, dtype=dtype)
    )
    if use_prealloc:
        esc_mask = prealloc.escape_mask
        esc_qrand = prealloc.escape_qrand
        esc_parent = prealloc.escape_parent
        esc_mask.zero_()
        esc_qrand.zero_()
        esc_parent.zero_()
    else:
        esc_mask = torch.zeros((B, Kmax), device=device, dtype=torch.bool)
        esc_qrand = torch.zeros((B, Kmax, D), device=device, dtype=dtype)
        esc_parent = torch.zeros((B, Kmax), device=device, dtype=torch.long)

    num_need = needEsc.sum(dim=1).to(torch.long)
    esc_rows = valid_b & (num_need > 0)
    if bool(esc_rows.any()):
        other_q_all, other_n_nodes = other.get_nodes()
        (
            esc_mask,
            esc_qrand,
            esc_parent,
            _slot_source,
            source_b,
            _source_slot,
            source_ts,
            assigned_by_source,
            dim_src,
            domain_src,
        ) = _build_per_slot_ts_escape_spawn(
            tree=tree,
            other_q_all=other_q_all,
            other_n_nodes=other_n_nodes,
            other_blocked=getattr(other, "blocked_node", getattr(other, "banned_node", None)),
            active=active,
            need_esc=needEsc,
            tgt_other=tgt_other,
            valid_b=valid_b,
            banks=banks,
            joint_limits=joint_limits,
            cfg=cfg,
            _gather_selected_ts_tensors=_gather_selected_ts_tensors,
            _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
            nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
            generator=generator,
            prealloc=prealloc,
        )
        rec = getattr(tree, "trace_recorder", None)
        if rec is not None and rec.wants("summary"):
            tree_name = str(getattr(tree, "trace_tree_name", ""))
            for src_i in range(int(source_b.numel())):
                b = int(source_b[src_i].item())
                rec.record_escape_spawn_batch(
                    iter_idx=int(iter_idx),
                    tree=tree_name,
                    batch_idx=b,
                    need_slots=1,
                    candidate_slots=int((needEsc[b] | (~active[b])).sum().item()),
                    assigned_slots=int(assigned_by_source[src_i].item()),
                    ts_id=int(source_ts[src_i].item()),
                    ts_dim=int(dim_src[src_i].item()),
                    ts_domain=float(domain_src[src_i].item()),
                    n_draw=escape_spawn_blocks,
                    enable_halfspace=bool(getattr(cfg, "enable_halfspace", True)),
                    enable_overlap_discard=bool(getattr(cfg, "enable_overlap_discard", True)),
                )

        active = active | esc_mask
        tgt_other = torch.where(esc_mask, (prealloc.minus_one_long(Kmax) if use_prealloc else torch.full_like(tgt_other, -1)), tgt_other)
        seg = torch.where(esc_mask, (prealloc.zero_long(Kmax) if use_prealloc else torch.zeros_like(seg)), seg)
    n_escape_steps = max(1, int(getattr(cfg, "escape_extend_steps", 1)))
    if use_prealloc:
        new_idx = prealloc.escape_new_idx
        alive = prealloc.escape_alive
        parent = prealloc.escape_parent_work
        new_idx.fill_(-1)
        alive.copy_(esc_mask)
        parent.copy_(esc_parent)
    else:
        new_idx = torch.full((B, Kmax), -1, device=device, dtype=torch.long)
        alive = esc_mask.clone()
        parent = esc_parent.clone()

    connection_result: Optional[Any] = None
    for escape_step_idx in range(n_escape_steps):
        step_idx = _extend_one_step_from_parent(
            tree=tree,
            banks=banks,
            parent_idx=parent,
            q_rand=esc_qrand,
            cfg=cfg,
            checker=checker,
            edge_checker=edge_checker,
            projector=projector,
            iter_idx=iter_idx,
            mask=alive,
            projector_cache=projector_cache,
            prealloc=prealloc,
            trace_phase="escape_extend",
            trace_escape_step=int(escape_step_idx),
        )
        step_ok = (step_idx >= 0) & alive
        if after_step_connect is not None and bool(step_ok.any()):
            connection_result = after_step_connect(
                step_idx=step_idx,
                step_ok=step_ok,
                escape_step_idx=int(escape_step_idx),
            )
        new_idx = torch.where(step_ok, step_idx, new_idx)
        parent = torch.where(step_ok, step_idx, parent)
        alive = step_ok
        if connection_result is not None:
            break
        if not bool(alive.any()):
            break

    ok = (new_idx >= 0) & esc_mask
    escape_ok = ok.clone()
    cur = torch.where(ok, new_idx, cur)
    mode = torch.where(ok, (prealloc.zero_long(Kmax) if use_prealloc else torch.zeros_like(mode)), mode)

    fail = esc_mask & (~ok)
    mode = torch.where(fail, (prealloc.one_long(Kmax) if use_prealloc else torch.ones_like(mode)), mode)
    rec = getattr(tree, "trace_recorder", None)
    if rec is not None and rec.wants("summary"):
        rec.record_escape_result(
            iter_idx=int(iter_idx),
            tree=str(getattr(tree, "trace_tree_name", "")),
            spawned=int(esc_mask.sum().item()),
            succeeded=int(ok.sum().item()),
            failed=int(fail.sum().item()),
            extend_steps=int(n_escape_steps),
        )
    return active, mode, cur, seg, tgt_other, escape_ok, connection_result


@torch.no_grad()
def _escape_two_trees(
    *,
    tree_a: TreeBatchGPU,
    tree_b: TreeBatchGPU,
    active_a: torch.Tensor,
    active_b: torch.Tensor,
    mode_a: torch.Tensor,
    mode_b: torch.Tensor,
    cur_a: torch.Tensor,
    cur_b: torch.Tensor,
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
    tgt_other_a: torch.Tensor,
    tgt_other_b: torch.Tensor,
    need_esc_a: torch.Tensor,
    need_esc_b: torch.Tensor,
    valid_b: torch.Tensor,
    banks: List[TSBank],
    joint_limits: JointLimitsTorch,
    cfg: TBRRTConfig,
    checker: SelfCollisionChecker,
    edge_checker: EdgeCollisionChecker,
    projector: ManifoldProjector,
    iter_idx: int,
    _sample_ts_ids_from_banks_batch,
    _gather_selected_ts_tensors,
    _sample_in_selected_ts_ball_batch,
    _apply_overlap_discard_qrand_batch,
    _select_escape_q_hint,
    _extend_two_trees_one_step_from_parent,
    nn_1_tree_all_candidates_cdist,
    generator: Optional[torch.Generator] = None,
    projector_cache: ProjectorStackCache | None = None,
    prealloc: Optional[BatchConextPrealloc] = None,
    after_step_connect_a: Optional[Callable[..., Any]] = None,
    after_step_connect_b: Optional[Callable[..., Any]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[Any],
]:
    if not bool(need_esc_a.any() | need_esc_b.any()):
        return (
            active_a, mode_a, cur_a, seg_a, tgt_other_a, torch.zeros_like(active_a),
            active_b, mode_b, cur_b, seg_b, tgt_other_b, torch.zeros_like(active_b),
            None,
        )

    device = tree_a.device
    dtype = tree_a.dtype
    B, Kmax = active_a.shape
    D = tree_a.D
    escape_spawn_blocks = max(1, int(getattr(cfg, "escape_spawn_blocks", 3)))
    use_prealloc = (
        prealloc is not None
        and prealloc.supports(B=B, K=Kmax, D=D, device=device, dtype=dtype)
    )

    if use_prealloc:
        active_ab = prealloc.ab_bool0[:, :Kmax]
        need_esc_ab = prealloc.ab_bool1[:, :Kmax]
        tgt_other_ab = prealloc.ab_long0[:, :Kmax]
        active_ab[:B].copy_(active_a)
        active_ab[B:2 * B].copy_(active_b)
        need_esc_ab[:B].copy_(need_esc_a)
        need_esc_ab[B:2 * B].copy_(need_esc_b)
        tgt_other_ab[:B].copy_(tgt_other_a)
        tgt_other_ab[B:2 * B].copy_(tgt_other_b)
    else:
        active_ab = torch.cat([active_a, active_b], dim=0)
        need_esc_ab = torch.cat([need_esc_a, need_esc_b], dim=0)
        tgt_other_ab = torch.cat([tgt_other_a, tgt_other_b], dim=0)
    valid_ab = valid_b.repeat(2)
    banks_ab = banks + banks

    if use_prealloc:
        esc_mask_ab = prealloc.escape_mask_ab[:, :Kmax]
        esc_qrand_ab = prealloc.escape_qrand_ab[:, :Kmax, :]
        esc_parent_ab = prealloc.escape_parent_ab[:, :Kmax]
        esc_mask_ab.zero_()
        esc_qrand_ab.zero_()
        esc_parent_ab.zero_()
    else:
        esc_mask_ab = torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool)
        esc_qrand_ab = torch.zeros((2 * B, Kmax, D), device=device, dtype=dtype)
        esc_parent_ab = torch.zeros((2 * B, Kmax), device=device, dtype=torch.long)

    num_need = need_esc_ab.sum(dim=1).to(torch.long)
    esc_rows = valid_ab & (num_need > 0)
    if bool(esc_rows.any()):
        tree_cap = max(int(tree_a.q.shape[1]), int(tree_b.q.shape[1]))
        tree_a_blocked = getattr(tree_a, "blocked_node", tree_a.banned_node)
        tree_b_blocked = getattr(tree_b, "blocked_node", tree_b.banned_node)

        if use_prealloc:
            prealloc.ensure_tree_scratch(tree_cap)
            tree_q_ab = prealloc.tree_q_ab[:, :tree_cap, :]
            tree_n_ab = prealloc.tree_n_ab
            is_proj_root_ab = prealloc.tree_is_proj_root_ab[:, :tree_cap]
            is_parent_of_proj_root_ab = prealloc.tree_is_parent_of_proj_root_ab[:, :tree_cap]
            banned_node_ab = prealloc.tree_banned_node_ab[:, :tree_cap]
            other_q_ab = prealloc.tree_other_q_ab[:, :tree_cap, :]
            other_n_ab = prealloc.tree_other_n_ab
            other_blocked_ab = prealloc.tree_other_banned_node_ab[:, :tree_cap]

            tree_q_ab.zero_()
            is_proj_root_ab.zero_()
            is_parent_of_proj_root_ab.zero_()
            banned_node_ab.zero_()
            other_q_ab.zero_()
            other_blocked_ab.zero_()

            tree_q_ab[:B, : tree_a.q.shape[1], :].copy_(tree_a.q)
            tree_q_ab[B:2 * B, : tree_b.q.shape[1], :].copy_(tree_b.q)
            tree_n_ab[:B].copy_(tree_a.n_nodes)
            tree_n_ab[B:2 * B].copy_(tree_b.n_nodes)
            is_proj_root_ab[:B, : tree_a.is_proj_root.shape[1]].copy_(tree_a.is_proj_root)
            is_proj_root_ab[B:2 * B, : tree_b.is_proj_root.shape[1]].copy_(tree_b.is_proj_root)
            is_parent_of_proj_root_ab[:B, : tree_a.is_parent_of_proj_root.shape[1]].copy_(tree_a.is_parent_of_proj_root)
            is_parent_of_proj_root_ab[B:2 * B, : tree_b.is_parent_of_proj_root.shape[1]].copy_(tree_b.is_parent_of_proj_root)
            banned_node_ab[:B, : tree_a_blocked.shape[1]].copy_(tree_a_blocked)
            banned_node_ab[B:2 * B, : tree_b_blocked.shape[1]].copy_(tree_b_blocked)

            other_q_ab[:B, : tree_b.q.shape[1], :].copy_(tree_b.q)
            other_q_ab[B:2 * B, : tree_a.q.shape[1], :].copy_(tree_a.q)
            other_n_ab[:B].copy_(tree_b.n_nodes)
            other_n_ab[B:2 * B].copy_(tree_a.n_nodes)
            other_blocked_ab[:B, : tree_b_blocked.shape[1]].copy_(tree_b_blocked)
            other_blocked_ab[B:2 * B, : tree_a_blocked.shape[1]].copy_(tree_a_blocked)
        else:
            tree_q_ab = torch.cat([_pad_dim1(tree_a.q, tree_cap), _pad_dim1(tree_b.q, tree_cap)], dim=0)
            tree_n_ab = torch.cat([tree_a.n_nodes, tree_b.n_nodes], dim=0)
            is_proj_root_ab = torch.cat(
                [_pad_dim1(tree_a.is_proj_root, tree_cap), _pad_dim1(tree_b.is_proj_root, tree_cap)],
                dim=0,
            )
            is_parent_of_proj_root_ab = torch.cat(
                [
                    _pad_dim1(tree_a.is_parent_of_proj_root, tree_cap),
                    _pad_dim1(tree_b.is_parent_of_proj_root, tree_cap),
                ],
                dim=0,
            )
            banned_node_ab = torch.cat([_pad_dim1(tree_a_blocked, tree_cap), _pad_dim1(tree_b_blocked, tree_cap)], dim=0)
            other_q_ab = torch.cat([_pad_dim1(tree_b.q, tree_cap), _pad_dim1(tree_a.q, tree_cap)], dim=0)
            other_n_ab = torch.cat([tree_b.n_nodes, tree_a.n_nodes], dim=0)
            other_blocked_ab = torch.cat([_pad_dim1(tree_b_blocked, tree_cap), _pad_dim1(tree_a_blocked, tree_cap)], dim=0)

        (
            esc_mask_ab,
            esc_qrand_ab,
            esc_parent_ab,
            _slot_source,
            source_b,
            _source_slot,
            source_ts,
            assigned_by_source,
            dim_src,
            domain_src,
        ) = _build_per_slot_ts_escape_spawn(
            tree=_BatchEscapeTreeView(
                D=tree_a.D,
                q=tree_q_ab,
                n_nodes=tree_n_ab,
                is_proj_root=is_proj_root_ab,
                is_parent_of_proj_root=is_parent_of_proj_root_ab,
                banned_node=banned_node_ab,
            ),
            other_q_all=other_q_ab,
            other_n_nodes=other_n_ab,
            other_blocked=other_blocked_ab,
            active=active_ab,
            need_esc=need_esc_ab,
            tgt_other=tgt_other_ab,
            valid_b=valid_ab,
            banks=banks_ab,
            joint_limits=joint_limits,
            cfg=cfg,
            _gather_selected_ts_tensors=_gather_selected_ts_tensors,
            _sample_in_selected_ts_ball_batch=_sample_in_selected_ts_ball_batch,
            nn_1_tree_all_candidates_cdist=nn_1_tree_all_candidates_cdist,
            generator=generator,
            prealloc=prealloc,
        )

        _record_spawn_per_slot(
            tree=tree_a,
            offset=0,
            B=B,
            iter_idx=int(iter_idx),
            need_esc_ab=need_esc_ab,
            active_ab=active_ab,
            source_b=source_b,
            assigned_by_source=assigned_by_source,
            source_ts=source_ts,
            dim_src=dim_src,
            domain_src=domain_src,
            escape_spawn_blocks=escape_spawn_blocks,
            cfg=cfg,
        )
        _record_spawn_per_slot(
            tree=tree_b,
            offset=B,
            B=B,
            iter_idx=int(iter_idx),
            need_esc_ab=need_esc_ab,
            active_ab=active_ab,
            source_b=source_b,
            assigned_by_source=assigned_by_source,
            source_ts=source_ts,
            dim_src=dim_src,
            domain_src=domain_src,
            escape_spawn_blocks=escape_spawn_blocks,
            cfg=cfg,
        )
    esc_mask_a = esc_mask_ab[:B]
    esc_mask_b = esc_mask_ab[B:]
    esc_qrand_a = esc_qrand_ab[:B]
    esc_qrand_b = esc_qrand_ab[B:]
    esc_parent_a = esc_parent_ab[:B]
    esc_parent_b = esc_parent_ab[B:]

    minus_one = prealloc.minus_one_long(Kmax) if use_prealloc else torch.full_like(tgt_other_a, -1)
    zero_long = prealloc.zero_long(Kmax) if use_prealloc else torch.zeros_like(seg_a)
    one_long = prealloc.one_long(Kmax) if use_prealloc else torch.ones_like(mode_a)

    active_a = active_a | esc_mask_a
    active_b = active_b | esc_mask_b
    tgt_other_a = torch.where(esc_mask_a, minus_one, tgt_other_a)
    tgt_other_b = torch.where(esc_mask_b, minus_one, tgt_other_b)
    seg_a = torch.where(esc_mask_a, zero_long, seg_a)
    seg_b = torch.where(esc_mask_b, zero_long, seg_b)

    n_escape_steps = max(1, int(getattr(cfg, "escape_extend_steps", 1)))
    if use_prealloc:
        new_idx_ab = prealloc.escape_new_idx_ab[:, :Kmax]
        alive_ab = prealloc.escape_alive_ab[:, :Kmax]
        parent_ab = prealloc.escape_parent_work_ab[:, :Kmax]
        new_idx_ab.fill_(-1)
        alive_ab.copy_(esc_mask_ab)
        parent_ab.copy_(esc_parent_ab)
        new_idx_a = new_idx_ab[:B]
        new_idx_b = new_idx_ab[B:]
        alive_a = alive_ab[:B]
        alive_b = alive_ab[B:]
        parent_a = parent_ab[:B]
        parent_b = parent_ab[B:]
    else:
        new_idx_a = torch.full((B, Kmax), -1, device=device, dtype=torch.long)
        new_idx_b = torch.full((B, Kmax), -1, device=device, dtype=torch.long)
        alive_a = esc_mask_a.clone()
        alive_b = esc_mask_b.clone()
        parent_a = esc_parent_a.clone()
        parent_b = esc_parent_b.clone()

    connection_result: Optional[Any] = None
    for escape_step_idx in range(n_escape_steps):
        step_idx_a, step_idx_b = _extend_two_trees_one_step_from_parent(
            tree_a=tree_a,
            tree_b=tree_b,
            banks=banks,
            parent_idx_a=parent_a,
            parent_idx_b=parent_b,
            q_rand_a=esc_qrand_a,
            q_rand_b=esc_qrand_b,
            cfg=cfg,
            checker=checker,
            edge_checker=edge_checker,
            projector=projector,
            iter_idx=iter_idx,
            mask_a=alive_a,
            mask_b=alive_b,
            projector_cache=projector_cache,
            prealloc=prealloc,
            trace_phase="escape_extend",
            trace_escape_step=int(escape_step_idx),
        )
        step_ok_a = (step_idx_a >= 0) & alive_a
        step_ok_b = (step_idx_b >= 0) & alive_b

        if after_step_connect_a is not None and bool(step_ok_a.any()):
            connection_result = after_step_connect_a(
                step_idx=step_idx_a,
                step_ok=step_ok_a,
                escape_step_idx=int(escape_step_idx),
            )
        if connection_result is None and after_step_connect_b is not None and bool(step_ok_b.any()):
            connection_result = after_step_connect_b(
                step_idx=step_idx_b,
                step_ok=step_ok_b,
                escape_step_idx=int(escape_step_idx),
            )

        new_idx_a = torch.where(step_ok_a, step_idx_a, new_idx_a)
        new_idx_b = torch.where(step_ok_b, step_idx_b, new_idx_b)
        parent_a = torch.where(step_ok_a, step_idx_a, parent_a)
        parent_b = torch.where(step_ok_b, step_idx_b, parent_b)
        alive_a = step_ok_a
        alive_b = step_ok_b
        if connection_result is not None:
            break
        if not bool(alive_a.any() | alive_b.any()):
            break

    ok_a = (new_idx_a >= 0) & esc_mask_a
    ok_b = (new_idx_b >= 0) & esc_mask_b
    escape_ok_a = ok_a.clone()
    escape_ok_b = ok_b.clone()
    cur_a = torch.where(ok_a, new_idx_a, cur_a)
    cur_b = torch.where(ok_b, new_idx_b, cur_b)
    mode_a = torch.where(ok_a, zero_long, mode_a)
    mode_b = torch.where(ok_b, zero_long, mode_b)

    fail_a = esc_mask_a & (~ok_a)
    fail_b = esc_mask_b & (~ok_b)
    mode_a = torch.where(fail_a, one_long, mode_a)
    mode_b = torch.where(fail_b, one_long, mode_b)

    _record_escape_result(
        tree=tree_a,
        esc_mask=esc_mask_a,
        ok=ok_a,
        fail=fail_a,
        iter_idx=int(iter_idx),
        n_escape_steps=int(n_escape_steps),
    )
    _record_escape_result(
        tree=tree_b,
        esc_mask=esc_mask_b,
        ok=ok_b,
        fail=fail_b,
        iter_idx=int(iter_idx),
        n_escape_steps=int(n_escape_steps),
    )

    return (
        active_a, mode_a, cur_a, seg_a, tgt_other_a, escape_ok_a,
        active_b, mode_b, cur_b, seg_b, tgt_other_b, escape_ok_b,
        connection_result,
    )


__all__ = ["_escape_one_tree", "_escape_two_trees"]

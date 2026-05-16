from __future__ import annotations

import torch


def _slice_tree_to_live_nodes(tree_q: torch.Tensor, tree_size: torch.Tensor) -> tuple[torch.Tensor, int]:
    max_n = int(tree_size.max().item()) if tree_size.numel() > 0 else 0
    max_n = max(0, min(max_n, int(tree_q.shape[1])))
    return tree_q[:, :max_n, :], max_n


@torch.no_grad()
def nn_1_tree_all(tree_q: torch.Tensor, tree_size: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Nearest neighbor for each batch element over all its nodes.

    tree_q: (B,cap,D)
    tree_size: (B,) long
    q: (B,D)
    returns idx: (B,) long
    """
    device = tree_q.device
    B, cap, D = tree_q.shape
    assert q.shape == (B, D)
    tree_q, cap = _slice_tree_to_live_nodes(tree_q, tree_size)
    if cap <= 0:
        return torch.full((B,), -1, device=device, dtype=torch.long)

    # dist: (B,1,cap) -> (B,cap)
    dist = torch.cdist(q.view(B, 1, D), tree_q).squeeze(1)
    n_idx = torch.arange(cap, device=device).view(1, cap)
    valid = n_idx < tree_size.view(B, 1)
    dist = dist.masked_fill(~valid, float("inf"))
    idx_out = torch.argmin(dist, dim=1)

    empty = tree_size <= 0
    return idx_out.masked_fill(empty, -1)


@torch.no_grad()
def nn_1_tree_all_candidates(tree_q: torch.Tensor, tree_size: torch.Tensor, q_targets: torch.Tensor) -> torch.Tensor:
    """Nearest neighbor for each (b,k) candidate.

    tree_q: (B,cap,D)
    tree_size:(B,)
    q_targets:(B,K,D)
    returns idx:(B,K)
    """
    device = tree_q.device
    B, cap, D = tree_q.shape
    B2, K, D2 = q_targets.shape
    assert B2 == B and D2 == D

    return nn_1_tree_all_candidates_cdist(tree_q, tree_size, q_targets)

@torch.no_grad()
def nn_1_tree_all_candidates_cdist(
    tree_q: torch.Tensor,      # (B,N,D)
    tree_size: torch.Tensor,   # (B,)
    q_targets: torch.Tensor,   # (B,K,D)
    exclude_mask: torch.Tensor | None = None,  # (B,N) bool, True nodes are excluded
) -> torch.Tensor:
    """
    Returns nn_idx: (B,K) with -1 if tree_size[b]==0
    Uses torch.cdist batched: dist(B,K,N)
    """
    B, N, D = tree_q.shape
    _, K, _ = q_targets.shape
    device = tree_q.device
    tree_q, N = _slice_tree_to_live_nodes(tree_q, tree_size)
    if N <= 0:
        return torch.full((B, K), -1, device=device, dtype=torch.long)

    # dist: (B,K,N)
    dist = torch.cdist(q_targets, tree_q)  # batched cdist

    # mask invalid nodes beyond tree_size -> +inf
    n_idx = torch.arange(N, device=device).view(1, 1, N)  # (1,1,N)
    valid_mask = n_idx < tree_size.view(B, 1, 1)          # (B,1,N)
    if exclude_mask is not None:
        exclude_live = exclude_mask[:, :N].bool().view(B, 1, N)
        valid_mask = valid_mask & (~exclude_live)
    dist = dist.masked_fill(~valid_mask, float("inf"))

    nn_idx = torch.argmin(dist, dim=-1)  # (B,K)
    nn_dist = dist.gather(-1, nn_idx.unsqueeze(-1)).squeeze(-1)
    nn_idx = torch.where(torch.isfinite(nn_dist), nn_idx, torch.full_like(nn_idx, -1))

    # if no nodes at all for a batch, set -1
    empty = tree_size <= 0
    return nn_idx.masked_fill(empty.view(B, 1), -1)


@torch.no_grad()
def nn_1_tree_all_candidates_cdist_with_dist(
    tree_q: torch.Tensor,      # (B,N,D)
    tree_size: torch.Tensor,   # (B,)
    q_targets: torch.Tensor,   # (B,K,D)
    exclude_mask: torch.Tensor | None = None,  # (B,N) bool, True nodes are excluded
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      nn_idx:  (B,K) nearest-node indices, -1 if tree_size[b] == 0
      nn_dist: (B,K) nearest distances, +inf if tree_size[b] == 0
    """
    B, N, D = tree_q.shape
    _, K, _ = q_targets.shape
    device = tree_q.device
    tree_q, N = _slice_tree_to_live_nodes(tree_q, tree_size)
    if N <= 0:
        return (
            torch.full((B, K), -1, device=device, dtype=torch.long),
            torch.full((B, K), float("inf"), device=device, dtype=q_targets.dtype),
        )

    dist = torch.cdist(q_targets, tree_q)  # (B,K,N)
    n_idx = torch.arange(N, device=device).view(1, 1, N)
    valid_mask = n_idx < tree_size.view(B, 1, 1)
    if exclude_mask is not None:
        exclude_live = exclude_mask[:, :N].bool().view(B, 1, N)
        valid_mask = valid_mask & (~exclude_live)
    dist = dist.masked_fill(~valid_mask, float("inf"))

    nn_dist, nn_idx = torch.min(dist, dim=-1)

    empty = tree_size <= 0
    nn_idx = nn_idx.masked_fill(empty.view(B, 1), -1)
    nn_dist = nn_dist.masked_fill(empty.view(B, 1), float("inf"))
    nn_idx = torch.where(torch.isfinite(nn_dist), nn_idx, torch.full_like(nn_idx, -1))
    return nn_idx, nn_dist

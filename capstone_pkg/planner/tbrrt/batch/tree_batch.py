from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

@dataclass
class TreeBatchGPU:
    """A batched tree container.

    Stores B independent trees of max `capacity` nodes each.

    - q:      (B, capacity, D)
    - parent: (B, capacity)
    - ts_id:  (B, capacity)
    - n_nodes:(B,)

    add_nodes_one_per_batch adds at most one node per batch element.
    """

    device: torch.device
    dtype: torch.dtype
    B: int
    D: int
    capacity: int = 32768
    trace_recorder: Optional[Any] = None
    trace_tree_name: str = ""

    def __post_init__(self) -> None:
        self.q = torch.zeros((self.B, self.capacity, self.D), device=self.device, dtype=self.dtype)
        self.parent = torch.full((self.B, self.capacity), -1, device=self.device, dtype=torch.long)
        self.ts_id = torch.zeros((self.B, self.capacity), device=self.device, dtype=torch.long)
        self.n_nodes = torch.zeros((self.B,), device=self.device, dtype=torch.long)
        self.is_proj_root = torch.zeros((self.B, self.capacity), device=self.device, dtype=torch.bool)
        self.is_parent_of_proj_root = torch.zeros((self.B, self.capacity), device=self.device, dtype=torch.bool)
        self.banned_node = torch.zeros((self.B, self.capacity), device=self.device, dtype=torch.bool)
        self.blocked_node = torch.zeros((self.B, self.capacity), device=self.device, dtype=torch.bool)

    def __len__(self) -> int:
        # total nodes across batches
        return int(self.n_nodes.sum().item())

    @torch.no_grad()
    def init_roots(self, q_root: torch.Tensor, ts_root: torch.Tensor) -> None:
        """Initialize root node at index 0 for each batch.

        q_root: (B,D)
        ts_root:(B,) long
        """
        assert q_root.shape == (self.B, self.D)
        assert ts_root.shape == (self.B,)
        self.q[:, 0, :] = q_root
        self.parent[:, 0] = -1
        self.ts_id[:, 0] = ts_root
        self.n_nodes[:] = 1
        self.is_proj_root.zero_()
        self.is_parent_of_proj_root.zero_()
        self.banned_node.zero_()
        self.blocked_node.zero_()
        if self.trace_recorder is not None and self.trace_tree_name:
            self.trace_recorder.record_nodes_batch(
                tree=self.trace_tree_name,
                batch_idx=torch.arange(self.B, device=self.device, dtype=torch.long),
                node_idx=torch.zeros((self.B,), device=self.device, dtype=torch.long),
                parent_idx=torch.full((self.B,), -1, device=self.device, dtype=torch.long),
                ts_id=ts_root,
                q=q_root,
                is_proj_root=torch.zeros((self.B,), device=self.device, dtype=torch.bool),
                iter_idx=-1,
                phase="root",
                slot_idx=torch.zeros((self.B,), device=self.device, dtype=torch.long),
            )

    @torch.no_grad()
    def get_nodes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q, n_nodes)."""
        return self.q, self.n_nodes

    @torch.no_grad()
    def add_nodes_one_per_batch(
        self,
        *,
        q_new: torch.Tensor,       # (B,D)
        parent_idx: torch.Tensor,  # (B,)
        ts_id: torch.Tensor,       # (B,)
        mask: torch.Tensor,        # (B,) bool
        is_proj_root: torch.Tensor | None = None,  # (B,) bool
        trace_iter_idx: int | None = None,
        trace_phase: str | None = None,
        trace_slot_idx: torch.Tensor | None = None,
        trace_escape_step: int | None = None,
    ) -> torch.Tensor:
        """Add one node per batch where mask is True.

        Returns new_idx (B,) long with -1 where not added.
        """
        assert q_new.shape == (self.B, self.D)
        assert parent_idx.shape == (self.B,)
        assert ts_id.shape == (self.B,)
        assert mask.shape == (self.B,)
        if is_proj_root is None:
            is_proj_root = torch.zeros((self.B,), device=self.device, dtype=torch.bool)
        else:
            assert is_proj_root.shape == (self.B,)

        new_idx = torch.full((self.B,), -1, device=self.device, dtype=torch.long)
        b = torch.nonzero(mask, as_tuple=False).view(-1)
        if b.numel() == 0:
            return new_idx

        idx = self.n_nodes.clone()  # insertion indices
        # capacity check (fail-fast)
        if (idx.index_select(0, b) >= self.capacity).any():
            # mark those as not added
            ok = mask & (idx < self.capacity)
        else:
            ok = mask

        safe_parent_idx = parent_idx.clamp(min=0, max=max(0, self.capacity - 1))
        parent_valid = parent_idx >= 0
        parent_blocked = (
            self.banned_node[torch.arange(self.B, device=self.device), safe_parent_idx]
            | self.blocked_node[torch.arange(self.B, device=self.device), safe_parent_idx]
        ) & parent_valid
        ok = ok & parent_valid & (~parent_blocked)

        b = torch.nonzero(ok, as_tuple=False).view(-1)
        if b.numel() > 0:
            ii = idx.index_select(0, b)
            self.q[b, ii, :] = q_new.index_select(0, b)
            self.parent[b, ii] = parent_idx.index_select(0, b)
            self.ts_id[b, ii] = ts_id.index_select(0, b)
            self.banned_node[b, ii] = False
            self.blocked_node[b, ii] = False
            proj_b = is_proj_root.index_select(0, b).bool()
            self.is_proj_root[b, ii] = proj_b
            parent_b = parent_idx.index_select(0, b)
            parent_ok = proj_b & (parent_b >= 0)
            if parent_ok.any():
                self.is_parent_of_proj_root[b[parent_ok], parent_b[parent_ok]] = True
            self.n_nodes[b] = ii + 1
            new_idx[b] = ii
            if self.trace_recorder is not None and self.trace_tree_name:
                if trace_slot_idx is None:
                    slot_idx = torch.zeros((b.numel(),), device=self.device, dtype=torch.long)
                else:
                    slot_idx = trace_slot_idx.index_select(0, b)
                self.trace_recorder.record_nodes_batch(
                    tree=self.trace_tree_name,
                    batch_idx=b,
                    node_idx=ii,
                    parent_idx=self.parent[b, ii],
                    ts_id=self.ts_id[b, ii],
                    q=self.q[b, ii],
                    is_proj_root=self.is_proj_root[b, ii],
                    iter_idx=trace_iter_idx,
                    phase=trace_phase,
                    slot_idx=slot_idx,
                    escape_step=trace_escape_step,
                )

        return new_idx

    @torch.no_grad()
    def backtrack_path(self, b: int, idx: int) -> torch.Tensor:
        """Return path root->idx for a single batch element."""
        if idx < 0:
            raise ValueError("idx must be >=0")
        n = int(self.n_nodes[b].item())
        if idx >= n:
            raise ValueError("idx out of range")
        # collect indices
        cur = idx
        out = []
        while cur >= 0:
            out.append(cur)
            cur = int(self.parent[b, cur].item())
        out.reverse()
        return self.q[b, torch.tensor(out, device=self.device), :].detach().clone()

    @torch.no_grad()
    def add_nodes_k_per_batch(
        self,
        *,
        q_new: torch.Tensor,        # (B,K,D)
        parent_idx: torch.Tensor,   # (B,K) long
        ts_id: torch.Tensor,        # (B,K) long
        mask: torch.Tensor,         # (B,K) bool
        is_proj_root: torch.Tensor | None = None,  # (B,K) bool
        trace_iter_idx: int | None = None,
        trace_phase: str | None = None,
        trace_escape_step: int | None = None,
    ) -> torch.Tensor:
        """
        Add up to K nodes per batch element.
        Returns:
        new_idx: (B,K) long, -1 where not added
        """
        device = self.device
        B, K, D = q_new.shape
        assert B == self.B and D == self.D
        assert parent_idx.shape == (B, K)
        assert ts_id.shape == (B, K)
        assert mask.shape == (B, K)
        if is_proj_root is None:
            is_proj_root = torch.zeros((B, K), device=device, dtype=torch.bool)
        else:
            assert is_proj_root.shape == (B, K)

        b_idx_full = torch.arange(B, device=device).view(B, 1).expand(B, K)
        safe_parent_idx = parent_idx.clamp(min=0, max=max(0, self.capacity - 1))
        parent_valid = parent_idx >= 0
        parent_blocked = (self.banned_node[b_idx_full, safe_parent_idx] | self.blocked_node[b_idx_full, safe_parent_idx]) & parent_valid
        mask = mask & parent_valid & (~parent_blocked)

        add_counts = mask.sum(dim=1).to(torch.long)  # (B,)
        new_cap_needed = int((self.n_nodes + add_counts).max().item())
        if new_cap_needed > self.capacity:
            self._grow(new_cap_needed)

        new_idx = torch.full((B, K), -1, device=device, dtype=torch.long)

        base = self.n_nodes.clone()  # (B,)
        self.n_nodes = self.n_nodes + add_counts

        rows = torch.nonzero(mask, as_tuple=False)
        if rows.numel() == 0:
            return new_idx

        b_sel = rows[:, 0].to(torch.long)
        k_sel = rows[:, 1].to(torch.long)
        local_rank = (torch.cumsum(mask.to(torch.long), dim=1) - 1)[b_sel, k_sel]
        dst_idx = base[b_sel] + local_rank

        new_idx[b_sel, k_sel] = dst_idx
        self.q[b_sel, dst_idx, :] = q_new[b_sel, k_sel, :]
        self.parent[b_sel, dst_idx] = parent_idx[b_sel, k_sel]
        self.ts_id[b_sel, dst_idx] = ts_id[b_sel, k_sel]
        self.banned_node[b_sel, dst_idx] = False
        self.blocked_node[b_sel, dst_idx] = False

        proj_sel = is_proj_root[b_sel, k_sel].bool()
        self.is_proj_root[b_sel, dst_idx] = proj_sel
        parent_sel = parent_idx[b_sel, k_sel]
        parent_ok = proj_sel & (parent_sel >= 0)
        if bool(parent_ok.any()):
            self.is_parent_of_proj_root[b_sel[parent_ok], parent_sel[parent_ok]] = True

        if self.trace_recorder is not None and self.trace_tree_name:
            trace_batch_idx = b_sel
            trace_node_idx = dst_idx
            self.trace_recorder.record_nodes_batch(
                tree=self.trace_tree_name,
                batch_idx=trace_batch_idx,
                node_idx=trace_node_idx,
                parent_idx=self.parent[b_sel, dst_idx],
                ts_id=self.ts_id[b_sel, dst_idx],
                q=self.q[b_sel, dst_idx, :],
                is_proj_root=self.is_proj_root[b_sel, dst_idx],
                iter_idx=trace_iter_idx,
                phase=trace_phase,
                slot_idx=k_sel,
                escape_step=trace_escape_step,
            )

        return new_idx
    
    def _grow(self, new_capacity: int):
        """
        Increase internal buffers to at least new_capacity.
        Keeps existing data. Doubles capacity strategy for amortized efficiency.
        """
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity:
            return

        # choose new cap (at least double)
        new_cap = max(new_capacity, int(self.capacity * 2))

        device = self.device
        dtype = self.dtype
        B = self.B
        D = self.D

        # allocate new buffers
        q_new = torch.empty((B, new_cap, D), device=device, dtype=dtype)
        parent_new = torch.empty((B, new_cap), device=device, dtype=torch.long)
        tsid_new = torch.empty((B, new_cap), device=device, dtype=torch.long)
        proj_root_new = torch.zeros((B, new_cap), device=device, dtype=torch.bool)
        parent_proj_new = torch.zeros((B, new_cap), device=device, dtype=torch.bool)
        banned_node_new = torch.zeros((B, new_cap), device=device, dtype=torch.bool)
        blocked_node_new = torch.zeros((B, new_cap), device=device, dtype=torch.bool)

        # copy old
        q_new[:, : self.capacity, :] = self.q
        parent_new[:, : self.capacity] = self.parent
        tsid_new[:, : self.capacity] = self.ts_id
        proj_root_new[:, : self.capacity] = self.is_proj_root
        parent_proj_new[:, : self.capacity] = self.is_parent_of_proj_root
        banned_node_new[:, : self.capacity] = self.banned_node
        blocked_node_new[:, : self.capacity] = self.blocked_node

        # swap
        self.q = q_new
        self.parent = parent_new
        self.ts_id = tsid_new
        self.is_proj_root = proj_root_new
        self.is_parent_of_proj_root = parent_proj_new
        self.banned_node = banned_node_new
        self.blocked_node = blocked_node_new
        self.capacity = new_cap

    @torch.no_grad()
    def ban_subtree(self, b: int, root_idx: int) -> int:
        """Mark root_idx and its existing descendants as banned for batch b."""
        b_i = int(b)
        root = int(root_idx)
        if b_i < 0 or b_i >= self.B:
            return 0
        n = int(self.n_nodes[b_i].item())
        if root <= 0 or root >= n:
            return 0

        before = self.banned_node[b_i, :n].clone()
        banned = before.clone()
        banned[root] = True
        parent_b = self.parent[b_i, :n]
        for idx in range(root + 1, n):
            p = int(parent_b[idx].item())
            if p >= 0 and bool(banned[p].item()):
                banned[idx] = True
        self.banned_node[b_i, :n] = banned
        self.blocked_node[b_i, :n] = banned | self.blocked_node[b_i, :n]
        return int((banned & (~before)).sum().item())

    @torch.no_grad()
    def ban_node(self, b: int, node_idx: int) -> int:
        """Mark exactly one existing node as banned."""
        b_i = int(b)
        node = int(node_idx)
        if b_i < 0 or b_i >= self.B:
            return 0
        n = int(self.n_nodes[b_i].item())
        if node <= 0 or node >= n:
            return 0
        if bool(self.banned_node[b_i, node].item()):
            return 0
        self.banned_node[b_i, node] = True
        self.blocked_node[b_i, node] = True
        return 1

    @torch.no_grad()
    def ban_subtree_limited(self, b: int, root_idx: int, max_new_nodes: int) -> tuple[int, bool]:
        """Ban a subtree unless that would newly ban more than max_new_nodes.

        Returns (new_banned_count, subtree_banned). If the subtree is too large,
        only root_idx is banned and subtree_banned is False.
        """
        b_i = int(b)
        root = int(root_idx)
        if b_i < 0 or b_i >= self.B:
            return 0, False
        n = int(self.n_nodes[b_i].item())
        if root <= 0 or root >= n:
            return 0, False

        before_banned = self.banned_node[b_i, :n].clone()
        before_blocked = self.blocked_node[b_i, :n].clone()
        banned = before_banned.clone()
        blocked = before_blocked.clone()
        banned[root] = True
        blocked[root] = True
        parent_b = self.parent[b_i, :n]
        for idx in range(root + 1, n):
            p = int(parent_b[idx].item())
            if p >= 0 and bool(banned[p].item()):
                banned[idx] = True
            if p >= 0 and bool(blocked[p].item()):
                blocked[idx] = True

        new_count = int((banned & (~before_banned)).sum().item())
        if max_new_nodes > 0 and new_count > int(max_new_nodes):
            new_single = 0 if bool(self.banned_node[b_i, root].item()) else 1
            self.banned_node[b_i, root] = True
            self.blocked_node[b_i, :n] = blocked
            return new_single, False

        self.banned_node[b_i, :n] = banned
        self.blocked_node[b_i, :n] = blocked
        return new_count, True

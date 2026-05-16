from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BatchConextPrealloc:
    B: int
    Kmax: int
    D: int
    device: torch.device
    dtype: torch.dtype

    b_ids: torch.Tensor
    b_idx_bk: torch.Tensor
    b_idx_abk: torch.Tensor
    row_ids_bk: torch.Tensor
    slot_ids_bk: torch.Tensor
    slot_ids_abk: torch.Tensor

    zero_long_b: torch.Tensor
    one_long_b: torch.Tensor
    minus_one_long_b: torch.Tensor
    zero_bool_b: torch.Tensor
    one_bool_b: torch.Tensor

    zero_long_bk: torch.Tensor
    one_long_bk: torch.Tensor
    minus_one_long_bk: torch.Tensor
    zero_bool_bk: torch.Tensor
    one_bool_bk: torch.Tensor

    escape_mask: torch.Tensor
    escape_qrand: torch.Tensor
    escape_parent: torch.Tensor
    escape_new_idx: torch.Tensor
    escape_alive: torch.Tensor
    escape_parent_work: torch.Tensor

    connect_diff: torch.Tensor
    connect_dist: torch.Tensor
    connect_node_curvature: torch.Tensor
    connect_target_valid: torch.Tensor
    connect_reached: torch.Tensor
    connect_close: torch.Tensor
    connect_aux_bool: torch.Tensor

    ab_q0: torch.Tensor
    ab_q1: torch.Tensor
    ab_long0: torch.Tensor
    ab_long1: torch.Tensor
    ab_long2: torch.Tensor
    ab_bool0: torch.Tensor
    ab_bool1: torch.Tensor
    ab_bool2: torch.Tensor
    ab_bool3: torch.Tensor
    ab_vec0: torch.Tensor

    escape_mask_ab: torch.Tensor
    escape_qrand_ab: torch.Tensor
    escape_parent_ab: torch.Tensor
    escape_new_idx_ab: torch.Tensor
    escape_alive_ab: torch.Tensor
    escape_parent_work_ab: torch.Tensor

    state_active: torch.Tensor
    state_cur: torch.Tensor
    state_mode: torch.Tensor
    state_seg: torch.Tensor
    state_target: torch.Tensor
    state_stagn: torch.Tensor
    state_ban_target: torch.Tensor
    state_ban_cd: torch.Tensor

    tree_scratch_capacity: int = 0
    tree_q_ab: Optional[torch.Tensor] = None
    tree_n_ab: Optional[torch.Tensor] = None
    tree_is_proj_root_ab: Optional[torch.Tensor] = None
    tree_is_parent_of_proj_root_ab: Optional[torch.Tensor] = None
    tree_banned_node_ab: Optional[torch.Tensor] = None
    tree_other_q_ab: Optional[torch.Tensor] = None
    tree_other_n_ab: Optional[torch.Tensor] = None
    tree_other_banned_node_ab: Optional[torch.Tensor] = None

    ts_scratch_rows: int = 0
    ts_scratch_capacity: int = 0
    ts_roots: Optional[torch.Tensor] = None
    ts_basis: Optional[torch.Tensor] = None
    ts_dim: Optional[torch.Tensor] = None
    ts_domain: Optional[torch.Tensor] = None
    ts_weight: Optional[torch.Tensor] = None
    ts_valid: Optional[torch.Tensor] = None

    lazy_max_paths: int = 0
    lazy_max_points_per_path: int = 0
    lazy_flat_capacity: int = 0
    lazy_edge_capacity: int = 0
    lazy_pair_q: Optional[torch.Tensor] = None
    lazy_flat_q: Optional[torch.Tensor] = None
    lazy_edge_q0: Optional[torch.Tensor] = None
    lazy_edge_q1: Optional[torch.Tensor] = None
    lazy_edge_owner: Optional[torch.Tensor] = None
    lazy_edge_path_idx: Optional[torch.Tensor] = None
    lazy_edge_ok: Optional[torch.Tensor] = None
    lazy_pair_ok: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        *,
        B: int,
        Kmax: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BatchConextPrealloc":
        b_ids = torch.arange(B, device=device, dtype=torch.long)
        b_idx_bk = b_ids.view(B, 1).expand(B, Kmax)
        b_idx_abk = torch.arange(2 * B, device=device, dtype=torch.long).view(2 * B, 1).expand(2 * B, Kmax)
        row_ids_bk = b_idx_bk
        slot_ids_bk = torch.arange(Kmax, device=device, dtype=torch.long).view(1, Kmax).expand(B, Kmax)
        slot_ids_abk = torch.arange(Kmax, device=device, dtype=torch.long).view(1, Kmax).expand(2 * B, Kmax)

        zero_long_b = torch.zeros((B,), device=device, dtype=torch.long)
        one_long_b = torch.ones((B,), device=device, dtype=torch.long)
        minus_one_long_b = torch.full((B,), -1, device=device, dtype=torch.long)
        zero_bool_b = torch.zeros((B,), device=device, dtype=torch.bool)
        one_bool_b = torch.ones((B,), device=device, dtype=torch.bool)

        zero_long_bk = torch.zeros((B, Kmax), device=device, dtype=torch.long)
        one_long_bk = torch.ones((B, Kmax), device=device, dtype=torch.long)
        minus_one_long_bk = torch.full((B, Kmax), -1, device=device, dtype=torch.long)
        zero_bool_bk = torch.zeros((B, Kmax), device=device, dtype=torch.bool)
        one_bool_bk = torch.ones((B, Kmax), device=device, dtype=torch.bool)

        return cls(
            B=B,
            Kmax=Kmax,
            D=D,
            device=device,
            dtype=dtype,
            b_ids=b_ids,
            b_idx_bk=b_idx_bk,
            b_idx_abk=b_idx_abk,
            row_ids_bk=row_ids_bk,
            slot_ids_bk=slot_ids_bk,
            slot_ids_abk=slot_ids_abk,
            zero_long_b=zero_long_b,
            one_long_b=one_long_b,
            minus_one_long_b=minus_one_long_b,
            zero_bool_b=zero_bool_b,
            one_bool_b=one_bool_b,
            zero_long_bk=zero_long_bk,
            one_long_bk=one_long_bk,
            minus_one_long_bk=minus_one_long_bk,
            zero_bool_bk=zero_bool_bk,
            one_bool_bk=one_bool_bk,
            escape_mask=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            escape_qrand=torch.zeros((B, Kmax, D), device=device, dtype=dtype),
            escape_parent=torch.zeros((B, Kmax), device=device, dtype=torch.long),
            escape_new_idx=torch.full((B, Kmax), -1, device=device, dtype=torch.long),
            escape_alive=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            escape_parent_work=torch.zeros((B, Kmax), device=device, dtype=torch.long),
            connect_diff=torch.zeros((B, Kmax, D), device=device, dtype=dtype),
            connect_dist=torch.zeros((B, Kmax), device=device, dtype=dtype),
            connect_node_curvature=torch.full((B, Kmax), float("inf"), device=device, dtype=dtype),
            connect_target_valid=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            connect_reached=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            connect_close=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            connect_aux_bool=torch.zeros((B, Kmax), device=device, dtype=torch.bool),
            ab_q0=torch.zeros((2 * B, Kmax, D), device=device, dtype=dtype),
            ab_q1=torch.zeros((2 * B, Kmax, D), device=device, dtype=dtype),
            ab_long0=torch.zeros((2 * B, Kmax), device=device, dtype=torch.long),
            ab_long1=torch.zeros((2 * B, Kmax), device=device, dtype=torch.long),
            ab_long2=torch.zeros((2 * B, Kmax), device=device, dtype=torch.long),
            ab_bool0=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            ab_bool1=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            ab_bool2=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            ab_bool3=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            ab_vec0=torch.zeros((2 * B, D), device=device, dtype=dtype),
            escape_mask_ab=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            escape_qrand_ab=torch.zeros((2 * B, Kmax, D), device=device, dtype=dtype),
            escape_parent_ab=torch.zeros((2 * B, Kmax), device=device, dtype=torch.long),
            escape_new_idx_ab=torch.full((2 * B, Kmax), -1, device=device, dtype=torch.long),
            escape_alive_ab=torch.zeros((2 * B, Kmax), device=device, dtype=torch.bool),
            escape_parent_work_ab=torch.zeros((2 * B, Kmax), device=device, dtype=torch.long),
            state_active=torch.zeros((2, B, Kmax), device=device, dtype=torch.bool),
            state_cur=torch.zeros((2, B, Kmax), device=device, dtype=torch.long),
            state_mode=torch.zeros((2, B, Kmax), device=device, dtype=torch.long),
            state_seg=torch.zeros((2, B, Kmax), device=device, dtype=torch.long),
            state_target=torch.full((2, B, Kmax), -1, device=device, dtype=torch.long),
            state_stagn=torch.zeros((2, B, Kmax), device=device, dtype=torch.long),
            state_ban_target=torch.full((2, B), -1, device=device, dtype=torch.long),
            state_ban_cd=torch.zeros((2, B), device=device, dtype=torch.long),
        )

    def supports(
        self,
        *,
        B: int,
        K: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        return (
            self.B == B
            and self.Kmax >= K
            and self.D == D
            and self.device == device
            and self.dtype == dtype
        )

    def b_idx(self, K: int) -> torch.Tensor:
        return self.b_idx_bk[:, :K]

    def b_idx_ab(self, K: int) -> torch.Tensor:
        return self.b_idx_abk[:, :K]

    def row_ids(self, K: int) -> torch.Tensor:
        return self.row_ids_bk[:, :K]

    def slot_ids(self, K: int) -> torch.Tensor:
        return self.slot_ids_bk[:, :K]

    def slot_ids_ab(self, K: int) -> torch.Tensor:
        return self.slot_ids_abk[:, :K]

    def zero_long(self, K: int) -> torch.Tensor:
        return self.zero_long_bk[:, :K]

    def one_long(self, K: int) -> torch.Tensor:
        return self.one_long_bk[:, :K]

    def minus_one_long(self, K: int) -> torch.Tensor:
        return self.minus_one_long_bk[:, :K]

    def zero_bool(self, K: int) -> torch.Tensor:
        return self.zero_bool_bk[:, :K]

    def one_bool(self, K: int) -> torch.Tensor:
        return self.one_bool_bk[:, :K]

    def ensure_tree_scratch(self, capacity: int) -> None:
        cap = int(capacity)
        if (
            self.tree_q_ab is not None
            and self.tree_n_ab is not None
            and self.tree_is_proj_root_ab is not None
            and self.tree_is_parent_of_proj_root_ab is not None
            and self.tree_banned_node_ab is not None
            and self.tree_other_q_ab is not None
            and self.tree_other_n_ab is not None
            and self.tree_other_banned_node_ab is not None
            and self.tree_scratch_capacity >= cap
        ):
            return

        self.tree_scratch_capacity = cap
        self.tree_q_ab = torch.zeros((2 * self.B, cap, self.D), device=self.device, dtype=self.dtype)
        self.tree_n_ab = torch.zeros((2 * self.B,), device=self.device, dtype=torch.long)
        self.tree_is_proj_root_ab = torch.zeros((2 * self.B, cap), device=self.device, dtype=torch.bool)
        self.tree_is_parent_of_proj_root_ab = torch.zeros((2 * self.B, cap), device=self.device, dtype=torch.bool)
        self.tree_banned_node_ab = torch.zeros((2 * self.B, cap), device=self.device, dtype=torch.bool)
        self.tree_other_q_ab = torch.zeros((2 * self.B, cap, self.D), device=self.device, dtype=self.dtype)
        self.tree_other_n_ab = torch.zeros((2 * self.B,), device=self.device, dtype=torch.long)
        self.tree_other_banned_node_ab = torch.zeros((2 * self.B, cap), device=self.device, dtype=torch.bool)

    def ensure_ts_scratch(self, *, rows: int, capacity: int) -> None:
        n_rows = max(0, int(rows))
        cap = max(0, int(capacity))
        if n_rows <= 0 or cap <= 0:
            return
        if (
            self.ts_roots is not None
            and self.ts_basis is not None
            and self.ts_dim is not None
            and self.ts_domain is not None
            and self.ts_weight is not None
            and self.ts_valid is not None
            and self.ts_scratch_rows >= n_rows
            and self.ts_scratch_capacity >= cap
        ):
            return

        self.ts_scratch_rows = n_rows
        self.ts_scratch_capacity = cap
        self.ts_roots = torch.empty((n_rows, cap, self.D), device=self.device, dtype=self.dtype)
        self.ts_basis = torch.empty((n_rows, cap, self.D, self.D), device=self.device, dtype=self.dtype)
        self.ts_dim = torch.empty((n_rows, cap), device=self.device, dtype=torch.long)
        self.ts_domain = torch.empty((n_rows, cap), device=self.device, dtype=self.dtype)
        self.ts_weight = torch.empty((n_rows, cap), device=self.device, dtype=torch.float32)
        self.ts_valid = torch.empty((n_rows, cap), device=self.device, dtype=torch.bool)

    def ensure_lazy_scratch(self, *, max_paths: int, max_points_per_path: int) -> None:
        paths = max(0, int(max_paths))
        points_per_path = max(0, int(max_points_per_path))
        flat_capacity = paths * points_per_path
        edge_capacity = paths * max(0, points_per_path - 1)
        if paths <= 0 or points_per_path <= 0:
            return
        if (
            self.lazy_pair_q is not None
            and self.lazy_flat_q is not None
            and self.lazy_edge_q0 is not None
            and self.lazy_edge_q1 is not None
            and self.lazy_edge_owner is not None
            and self.lazy_edge_path_idx is not None
            and self.lazy_edge_ok is not None
            and self.lazy_pair_ok is not None
            and self.lazy_max_paths >= paths
            and self.lazy_flat_capacity >= flat_capacity
            and self.lazy_edge_capacity >= edge_capacity
        ):
            return

        self.lazy_max_paths = paths
        self.lazy_max_points_per_path = points_per_path
        self.lazy_flat_capacity = flat_capacity
        self.lazy_edge_capacity = edge_capacity
        self.lazy_pair_q = torch.empty((2 * paths, self.D), device=self.device, dtype=self.dtype)
        self.lazy_flat_q = torch.empty((flat_capacity, self.D), device=self.device, dtype=self.dtype)
        self.lazy_edge_q0 = torch.empty((edge_capacity, self.D), device=self.device, dtype=self.dtype)
        self.lazy_edge_q1 = torch.empty((edge_capacity, self.D), device=self.device, dtype=self.dtype)
        self.lazy_edge_owner = torch.empty((edge_capacity,), device=self.device, dtype=torch.long)
        self.lazy_edge_path_idx = torch.empty((edge_capacity,), device=self.device, dtype=torch.long)
        self.lazy_edge_ok = torch.empty((paths,), device=self.device, dtype=torch.bool)
        self.lazy_pair_ok = torch.empty((paths,), device=self.device, dtype=torch.bool)

    def lazy_scratch_fits(
        self,
        *,
        n_paths: int,
        total_points: int,
        total_edges: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        return (
            self.lazy_pair_q is not None
            and self.lazy_flat_q is not None
            and self.lazy_edge_q0 is not None
            and self.lazy_edge_q1 is not None
            and self.lazy_edge_owner is not None
            and self.lazy_edge_path_idx is not None
            and self.lazy_edge_ok is not None
            and self.lazy_pair_ok is not None
            and self.device == device
            and self.dtype == dtype
            and self.lazy_max_paths >= int(n_paths)
            and self.lazy_flat_capacity >= int(total_points)
            and self.lazy_edge_capacity >= int(total_edges)
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .tangent_space import TangentSpace
from .stats import get_stats


@dataclass
class TSBank:
    """Stores all tangent spaces and provides sampling utilities."""

    spaces: List[TangentSpace]
    ts_radius: float

    # --- TS selection heuristics (paper Sec. 3.7) ---
    # Larger values increase the effect.
    bias_volume: float = 1.0        # w ∝ r^k (legacy behavior)
    bias_curvature: float = 1.0     # prefer lower "curvature" (proxy) TS
    bias_nodecount: float = 1.0     # prefer less explored TS (fewer nodes)
    bias_collision: float = 0.0     # prefer TS with fewer collision-blocked steps
    curv_eps: float = 1e-3

    # runtime stats
    _node_counts: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long), init=False, repr=False)
    _dims: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long), init=False, repr=False)
    _curv_cache: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.float32), init=False, repr=False)
    _collision_counts: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long), init=False, repr=False)
    _domains: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.float32), init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_runtime_arrays()

    def __len__(self) -> int:
        return len(self.spaces)

    def add(self, ts: TangentSpace) -> int:
        self._ensure_runtime_arrays()
        self.spaces.append(ts)
        dev = ts.root.device
        self._node_counts = torch.cat(
            [self._node_counts.to(device=dev), torch.zeros((1,), device=dev, dtype=torch.long)],
            dim=0,
        )
        self._dims = torch.cat(
            [
                self._dims.to(device=dev),
                torch.tensor([int(ts.dim)], device=dev, dtype=torch.long),
            ],
            dim=0,
        )
        self._curv_cache = torch.cat(
            [
                self._curv_cache.to(device=dev),
                torch.tensor([float(self._estimate_curvature(ts))], device=dev, dtype=torch.float32),
            ],
            dim=0,
        )
        self._collision_counts = torch.cat(
            [self._collision_counts.to(device=dev), torch.zeros((1,), device=dev, dtype=torch.long)],
            dim=0,
        )
        self._domains = torch.cat(
            [
                self._domains.to(device=dev),
                torch.tensor([float(self.ts_radius)], device=dev, dtype=torch.float32),
            ],
            dim=0,
        )
        return ts.ts_id

    def get(self, ts_id: int) -> TangentSpace:
        return self.spaces[int(ts_id)]

    def increment_count(self, ts_id: int, inc: int = 1) -> None:
        """Increment node count associated with a tangent space."""
        i = int(ts_id)
        if i < 0:
            return
        self._ensure_runtime_arrays()
        self._node_counts[i] += int(inc)

    @torch.no_grad()
    def increment_counts_batch(self, ts_ids: torch.Tensor) -> None:
        self._ensure_runtime_arrays()
        if ts_ids.numel() == 0 or len(self.spaces) == 0:
            return
        ids = ts_ids.to(device=self._node_counts.device, dtype=torch.long).view(-1)
        valid = (ids >= 0) & (ids < len(self.spaces))
        if not bool(valid.any()):
            return
        counts = torch.bincount(ids[valid], minlength=len(self.spaces))
        self._node_counts[: counts.shape[0]] += counts.to(device=self._node_counts.device, dtype=self._node_counts.dtype)

    def increment_collision_count(self, ts_id: int, inc: int = 1) -> None:
        """Increment collision-blocked step count associated with a tangent space."""
        i = int(ts_id)
        if i < 0:
            return
        self._ensure_runtime_arrays()
        self._collision_counts[i] += int(inc)

    @torch.no_grad()
    def increment_collision_counts_batch(self, ts_ids: torch.Tensor) -> None:
        self._ensure_runtime_arrays()
        if ts_ids.numel() == 0 or len(self.spaces) == 0:
            return
        ids = ts_ids.to(device=self._collision_counts.device, dtype=torch.long).view(-1)
        valid = (ids >= 0) & (ids < len(self.spaces))
        if not bool(valid.any()):
            return
        counts = torch.bincount(ids[valid], minlength=len(self.spaces))
        self._collision_counts[: counts.shape[0]] += counts.to(
            device=self._collision_counts.device,
            dtype=self._collision_counts.dtype,
        )

    def get_domain(self, ts_id: int) -> float:
        i = int(ts_id)
        if i < 0:
            return float(self.ts_radius)
        self._ensure_runtime_arrays()
        return float(self._domains[i].item())

    def get_curvature(self, ts_id: int) -> float:
        i = int(ts_id)
        if i < 0:
            return float("inf")
        self._ensure_runtime_arrays()
        return float(self._curv_cache[i].item())

    def get_curvatures_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self._ensure_runtime_arrays()
        return self._curv_cache.to(device=device, dtype=dtype)

    def get_dims_tensor(self, *, device: torch.device) -> torch.Tensor:
        self._ensure_runtime_arrays()
        return self._dims.to(device=device, dtype=torch.long)

    def set_domain(self, ts_id: int, domain: float) -> None:
        i = int(ts_id)
        if i < 0:
            return
        self._ensure_runtime_arrays()
        self._domains[i] = float(domain)

    @torch.no_grad()
    def update_domains_batch(
        self,
        *,
        used_ts_id: torch.Tensor,
        q_added: torch.Tensor,
        was_proj: torch.Tensor,
        expand_frac: float,
        shrink_frac: float,
        expand_ratio: float,
        shrink_ratio: float,
        dom_min: float,
        dom_max: float,
    ) -> None:
        self._ensure_runtime_arrays()
        if used_ts_id.numel() == 0 or len(self.spaces) == 0:
            return

        ids = used_ts_id.view(-1).to(dtype=torch.long)
        proj = was_proj.view(-1).to(dtype=torch.bool)
        valid = (ids >= 0) & (ids < len(self.spaces))
        if not bool(valid.any()):
            return

        ids = ids[valid]
        proj = proj[valid]
        q_valid = q_added.view(-1, q_added.shape[-1])[valid]

        unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        root_dtype = q_valid.dtype
        root_device = q_valid.device
        roots = torch.stack(
            [self.spaces[int(ts_id)].root.to(device=root_device, dtype=root_dtype) for ts_id in unique_ids.tolist()],
            dim=0,
        )
        dist = torch.linalg.norm(q_valid - roots.index_select(0, inverse.to(device=roots.device)), dim=-1)

        order = torch.argsort(ids, stable=True)
        ids_sorted = ids[order]
        dist_sorted = dist[order]
        proj_sorted = proj[order]

        unique_ids_sorted, counts = torch.unique_consecutive(ids_sorted, return_counts=True)
        cursor = 0
        for ts_id_t, count_t in zip(unique_ids_sorted.tolist(), counts.tolist()):
            ts_id = int(ts_id_t)
            count = int(count_t)
            dom = float(self._domains[ts_id].item())
            dist_group = dist_sorted[cursor : cursor + count]
            proj_group = proj_sorted[cursor : cursor + count]
            for projected_t, dist_t in zip(proj_group.tolist(), dist_group.tolist()):
                projected = bool(projected_t)
                dist_val = float(dist_t)
                if (not projected) and (dist_val > expand_frac * dom):
                    dom = dom * expand_ratio
                elif projected and (dist_val < shrink_frac * dom):
                    dom = dom * shrink_ratio
                dom = max(dom_min, min(dom_max, dom))
            self._domains[ts_id] = dom
            cursor += count

    def _estimate_curvature(self, ts: TangentSpace) -> float:
        """Heuristic curvature proxy.

        The paper suggests preferring low-curvature tangent spaces. We don't have
        direct curvature, so we use a conservative proxy derived from Jacobian
        conditioning: higher condition numbers often correlate with locally "hard"
        regions.

        Returns a positive scalar where smaller is preferred.
        """
        try:
            S = torch.linalg.svdvals(ts.J)
            if S.numel() == 0:
                return 1.0
            r = max(1, int(ts.rank))
            s_max = float(S[0].item())
            s_min = float(S[min(r - 1, S.numel() - 1)].item())
            cond = s_max / max(s_min, float(self.curv_eps))
            return float(torch.log1p(torch.tensor(cond)).item())
        except Exception:
            return 1.0

    @torch.no_grad()
    def _ensure_runtime_arrays(self) -> None:
        if int(self._node_counts.numel()) == len(self.spaces):
            return
        self._rebuild_runtime_arrays()

    def _rebuild_runtime_arrays(self) -> None:
        n = len(self.spaces)
        if n == 0:
            self._node_counts = torch.zeros((0,), dtype=torch.long)
            self._dims = torch.zeros((0,), dtype=torch.long)
            self._curv_cache = torch.zeros((0,), dtype=torch.float32)
            self._collision_counts = torch.zeros((0,), dtype=torch.long)
            self._domains = torch.zeros((0,), dtype=torch.float32)
            return

        dev = self.spaces[0].root.device
        old_node = self._node_counts.to(device=dev, dtype=torch.long)
        old_dims = self._dims.to(device=dev, dtype=torch.long)
        old_curv = self._curv_cache.to(device=dev, dtype=torch.float32)
        old_collision = self._collision_counts.to(device=dev, dtype=torch.long)
        old_domains = self._domains.to(device=dev, dtype=torch.float32)

        self._node_counts = torch.zeros((n,), device=dev, dtype=torch.long)
        self._dims = torch.empty((n,), device=dev, dtype=torch.long)
        self._curv_cache = torch.empty((n,), device=dev, dtype=torch.float32)
        self._collision_counts = torch.zeros((n,), device=dev, dtype=torch.long)
        self._domains = torch.full((n,), float(self.ts_radius), device=dev, dtype=torch.float32)

        if old_node.numel() > 0:
            keep = min(n, int(old_node.numel()))
            self._node_counts[:keep] = old_node[:keep]
        if old_collision.numel() > 0:
            keep = min(n, int(old_collision.numel()))
            self._collision_counts[:keep] = old_collision[:keep]
        if old_domains.numel() > 0:
            keep = min(n, int(old_domains.numel()))
            self._domains[:keep] = old_domains[:keep]

        keep_dims = min(n, int(old_dims.numel()))
        if keep_dims > 0:
            self._dims[:keep_dims] = old_dims[:keep_dims]
        for j in range(keep_dims, n):
            self._dims[j] = int(self.spaces[j].dim)

        keep_curv = min(n, int(old_curv.numel()))
        if keep_curv > 0:
            self._curv_cache[:keep_curv] = old_curv[:keep_curv]
        for j in range(keep_curv, n):
            self._curv_cache[j] = float(self._estimate_curvature(self.spaces[j]))

    @torch.no_grad()
    def get_sampling_weights(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return TS selection probabilities without packing roots/bases."""
        if len(self.spaces) == 0:
            raise ValueError("TSBank has no tangent spaces")

        self._ensure_runtime_arrays()

        dev = self.spaces[0].root.device if device is None else device
        ks = self._dims.to(device=dev, dtype=dtype).clamp_min(1.0)
        counts = self._node_counts.to(device=dev, dtype=dtype)
        curv = self._curv_cache.to(device=dev, dtype=dtype).clamp_min(float(self.curv_eps))
        collisions = self._collision_counts.to(device=dev, dtype=dtype)
        domains = self._domains[: len(self.spaces)].to(device=dev, dtype=dtype).clamp_min(1e-12)

        logw = torch.zeros((len(self.spaces),), device=dev, dtype=dtype)
        if float(self.bias_volume) != 0.0:
            logw = logw + float(self.bias_volume) * (ks * torch.log(domains))
        if float(self.bias_curvature) != 0.0:
            logw = logw + float(self.bias_curvature) * (-torch.log(curv))
        if float(self.bias_nodecount) != 0.0:
            logw = logw + float(self.bias_nodecount) * (-torch.log(counts + 1.0))
        if float(self.bias_collision) != 0.0:
            logw = logw + float(self.bias_collision) * (-torch.log(collisions + 1.0))
        return torch.softmax(logw, dim=0)

    @torch.no_grad()
    def pack_tensors(self):
        """Pack tangent space data into padded tensors for vectorized sampling.

        Returns:
            roots:  (S, D)
            basis:  (S, D, Kmax)  # padded on K
            dim:    (S,)          # k per TS
            domain: (S,)
            w:      (S,)          # sampling probability over TS
        """
        if len(self.spaces) == 0:
            raise ValueError("TSBank has no tangent spaces")

        dev = self.spaces[0].root.device
        dtype = self.spaces[0].root.dtype
        S = len(self.spaces)
        D = int(self.spaces[0].root.numel())
        self._ensure_runtime_arrays()
        dims = self._dims[:S].to(device=dev, dtype=torch.long)
        domains = self._domains[:S].to(device=dev, dtype=torch.float32)
        Kmax = int(dims.max().item()) if S > 0 else 0

        roots = torch.empty((S, D), device=dev, dtype=dtype)
        basis = torch.zeros((S, D, Kmax), device=dev, dtype=dtype)
        dim = dims.clone()
        domain = domains.clone()

        for i, ts in enumerate(self.spaces):
            roots[i] = ts.root
            k = int(dim[i].item())
            if k > 0:
                basis[i, :, :k] = ts.basis

        w = self.get_sampling_weights(device=dev, dtype=torch.float32)
        return roots, basis, dim, domain, w

    @torch.no_grad()
    def sample_ts_id(self, *, generator: Optional[torch.Generator] = None) -> int:
        """Sample a tangent space id.

        Paper-style heuristics (Sec. 3.7): choose which tangent space to sample from
        with a bias function. We combine four factors:

          (1) volume term:        r^{k_i}                (legacy)
          (2) curvature proxy:    1 / (curv_i + eps)     (prefer low curvature)
          (3) exploration term:   1 / (count_i + 1)      (prefer less explored)
          (4) collision term:     1 / (coll_i + 1)       (prefer fewer collision blocks)

        Bias strengths are controlled by bias_* fields.
        """
        if len(self.spaces) == 1:
            st = get_stats()
            st.add_hist("ts_selected", 0, 1)
            return 0

        dev = self.spaces[0].root.device
        w = self.get_sampling_weights(device=dev, dtype=torch.float32)
        idx = torch.multinomial(w, num_samples=1, replacement=True, generator=generator)
        out_id = int(idx.item())
        st = get_stats()
        st.add_hist("ts_selected", out_id, 1)
        return out_id

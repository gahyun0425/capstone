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
    curv_eps: float = 1e-3

    # runtime stats
    _node_counts: List[int] = field(default_factory=list, init=False, repr=False)
    _curv_cache: List[float] = field(default_factory=list, init=False, repr=False)
    _domains: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # If constructed with pre-filled spaces, initialize per-TS domains.
        if len(self._domains) != len(self.spaces):
            self._domains = [float(self.ts_radius) for _ in self.spaces]

    def __len__(self) -> int:
        return len(self.spaces)

    def add(self, ts: TangentSpace) -> int:
        self.spaces.append(ts)
        # keep stats arrays aligned
        self._node_counts.append(0)
        self._curv_cache.append(float(self._estimate_curvature(ts)))
        self._domains.append(float(self.ts_radius))
        return ts.ts_id

    def get(self, ts_id: int) -> TangentSpace:
        return self.spaces[int(ts_id)]

    def increment_count(self, ts_id: int, inc: int = 1) -> None:
        """Increment node count associated with a tangent space."""
        i = int(ts_id)
        if i < 0:
            return
        # allow lazy growth if TSBank constructed with pre-filled spaces
        while len(self._node_counts) < len(self.spaces):
            j = len(self._node_counts)
            self._node_counts.append(0)
            self._curv_cache.append(float(self._estimate_curvature(self.spaces[j])))
            self._domains.append(float(self.ts_radius))
        self._node_counts[i] += int(inc)

    def get_domain(self, ts_id: int) -> float:
        i = int(ts_id)
        if i < 0:
            return float(self.ts_radius)
        while len(self._domains) < len(self.spaces):
            self._domains.append(float(self.ts_radius))
        return float(self._domains[i])

    def set_domain(self, ts_id: int, domain: float) -> None:
        i = int(ts_id)
        if i < 0:
            return
        while len(self._domains) < len(self.spaces):
            self._domains.append(float(self.ts_radius))
        self._domains[i] = float(domain)

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
        Kmax = max(int(ts.dim) for ts in self.spaces)

        roots = torch.empty((S, D), device=dev, dtype=dtype)
        basis = torch.zeros((S, D, Kmax), device=dev, dtype=dtype)
        dim = torch.empty((S,), device=dev, dtype=torch.long)
        domain = torch.empty((S,), device=dev, dtype=torch.float32)

        for i, ts in enumerate(self.spaces):
            roots[i] = ts.root
            dim[i] = int(ts.dim)
            domain[i] = float(self.get_domain(ts.ts_id))
            k = int(ts.dim)
            if k > 0:
                basis[i, :, :k] = ts.basis

        # Build weights with the same logic as sample_ts_id.
        if len(self._node_counts) != len(self.spaces):
            self._node_counts = [0 for _ in self.spaces]
            self._curv_cache = [float(self._estimate_curvature(ts)) for ts in self.spaces]

        ks = dim.to(torch.float32).clamp_min(1.0)
        counts = torch.tensor(self._node_counts, device=dev, dtype=torch.float32)
        curv = torch.tensor(self._curv_cache, device=dev, dtype=torch.float32).clamp_min(float(self.curv_eps))
        domains = domain.clamp_min(1e-12)

        logw = torch.zeros((S,), device=dev, dtype=torch.float32)
        if float(self.bias_volume) != 0.0:
            logw = logw + float(self.bias_volume) * (ks * torch.log(domains))
        if float(self.bias_curvature) != 0.0:
            logw = logw + float(self.bias_curvature) * (-torch.log(curv))
        if float(self.bias_nodecount) != 0.0:
            logw = logw + float(self.bias_nodecount) * (-torch.log(counts + 1.0))

        w = torch.softmax(logw, dim=0)
        return roots, basis, dim, domain, w

    @torch.no_grad()
    def sample_ts_id(self, *, generator: Optional[torch.Generator] = None) -> int:
        """Sample a tangent space id.

        Paper-style heuristics (Sec. 3.7): choose which tangent space to sample from
        with a bias function. We combine three factors:

          (1) volume term:        r^{k_i}                (legacy)
          (2) curvature proxy:    1 / (curv_i + eps)     (prefer low curvature)
          (3) exploration term:   1 / (count_i + 1)      (prefer less explored)

        Bias strengths are controlled by bias_* fields.
        """
        if len(self.spaces) == 1:
            st = get_stats()
            st.add_hist("ts_selected", 0, 1)
            return 0

        # lazy-init stats if TSBank was constructed with pre-filled spaces
        if len(self._node_counts) != len(self.spaces):
            self._node_counts = [0 for _ in self.spaces]
            self._curv_cache = [float(self._estimate_curvature(ts)) for ts in self.spaces]

        dev = self.spaces[0].root.device
        dtype = torch.float32
        ks = torch.tensor([max(1, ts.dim) for ts in self.spaces], device=dev, dtype=dtype)
        counts = torch.tensor(self._node_counts, device=dev, dtype=dtype)
        curv = torch.tensor(self._curv_cache, device=dev, dtype=dtype).clamp_min(float(self.curv_eps))
        domains = torch.tensor([float(self.get_domain(ts.ts_id)) for ts in self.spaces], device=dev, dtype=dtype).clamp_min(1e-12)

        # log weights to avoid overflow
        logw = torch.zeros((len(self.spaces),), device=dev, dtype=dtype)
        if float(self.bias_volume) != 0.0:
            logw = logw + float(self.bias_volume) * (
                ks * torch.log(domains)
            )
        if float(self.bias_curvature) != 0.0:
            logw = logw + float(self.bias_curvature) * (-torch.log(curv))
        if float(self.bias_nodecount) != 0.0:
            logw = logw + float(self.bias_nodecount) * (-torch.log(counts + 1.0))

        w = torch.softmax(logw, dim=0)
        idx = torch.multinomial(w, num_samples=1, replacement=True, generator=generator)
        out_id = int(idx.item())
        st = get_stats()
        st.add_hist("ts_selected", out_id, 1)
        return out_id

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from capstone_pkg.constraint_projection.projection import ManifoldProjector, ManifoldProjectorTorch


@dataclass
class TangentSpace:
    """A single tangent space in the tangent bundle."""

    ts_id: int
    root: torch.Tensor                 # (D,)
    J: torch.Tensor                    # (m,D)
    basis: torch.Tensor                # (D,k) nullspace basis
    projector: torch.Tensor            # (D,D) = basis @ basis.T
    rank: int
    created_iter: int

    @property
    def dim(self) -> int:
        return int(self.basis.shape[1])


@torch.no_grad()
def build_tangent_space_fd(
    *,
    q_root: torch.Tensor,              # (D,)
    projector: Union[ManifoldProjector, ManifoldProjectorTorch],
    svd_tol: float = 1e-6,
    ts_id: int = 0,
    created_iter: int = 0,
) -> TangentSpace:
    """Build tangent space at q_root using finite-difference Jacobian.

    We reuse the same FD Jacobian method as the manifold projector to keep
    definitions consistent.

    Returns a TangentSpace containing:
      - J (m,D)
      - nullspace basis B (D,k)
      - orthogonal projector P = B B^T (D,D)
    """

    if q_root.ndim != 1:
        raise ValueError("q_root must be (D,)")

    q = q_root.view(1, -1)
    h0, Jb = projector.residual_and_jacobian_if_available(q)
    if Jb is None:
        # Fall back to the projector's FD Jacobian only when no analytic backend exists.
        Jb = projector._jacobian_fd(q, h0)
    J = Jb.squeeze(0).contiguous()     # (m,D)

    # SVD for nullspace
    # J = U S Vh
    U, S, Vh = torch.linalg.svd(J, full_matrices=True)

    # rank estimation
    r = int((S > float(svd_tol)).sum().item())
    D = int(J.shape[1])

    # Nullspace basis: columns of V corresponding to zero singular values.
    # Vh: (D,D) => V = Vh.T
    V = Vh.transpose(0, 1)
    if r >= D:
        # No nullspace (should not happen for typical constrained problems)
        basis = torch.zeros((D, 0), device=J.device, dtype=J.dtype)
        P = torch.zeros((D, D), device=J.device, dtype=J.dtype)
    else:
        basis = V[:, r:].contiguous()  # (D, D-r)
        P = (basis @ basis.transpose(0, 1)).contiguous()  # (D,D)

    return TangentSpace(
        ts_id=int(ts_id),
        root=q_root.detach().clone().contiguous(),
        J=J.detach().clone().contiguous(),
        basis=basis.detach().clone().contiguous(),
        projector=P.detach().clone().contiguous(),
        rank=r,
        created_iter=int(created_iter),
    )


@torch.no_grad()
def project_vector_to_tangent(ts, v: torch.Tensor) -> torch.Tensor:
    """Project vector(s) v onto the tangent space.

    Supports:
      - ts: TangentSpace-like object with attribute `projector` (D,D) or (N,D,D)
      - ts: projector tensor directly (D,D) or (N,D,D)
    v:
      - (D,) or (N,D)
    """
    P = ts.projector if hasattr(ts, "projector") else ts
    if not torch.is_tensor(P):
        raise TypeError(f"projector must be a torch.Tensor, got {type(P)}")

    if v.ndim == 1:
        v2 = v.unsqueeze(0)
        squeeze = True
    else:
        v2 = v
        squeeze = False

    if P.ndim == 2:
        out = (P.unsqueeze(0) @ v2.unsqueeze(-1)).squeeze(-1)
    elif P.ndim == 3:
        if P.shape[0] != v2.shape[0]:
            raise ValueError(f"P batch {P.shape[0]} != v batch {v2.shape[0]}")
        out = torch.bmm(P, v2.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"projector must be (D,D) or (N,D,D), got {tuple(P.shape)}")

    return out.squeeze(0) if squeeze else out


@torch.no_grad()
def tangent_coords(ts: TangentSpace, q: torch.Tensor) -> torch.Tensor:
    """Compute tangent coordinates (k,) for q relative to ts.root by least squares.

    This is used mainly for debugging / radius checks.
    """
    if ts.dim == 0:
        return torch.zeros((0,), device=q.device, dtype=q.dtype)
    dq = (q - ts.root)
    # Solve basis * a ≈ dq
    a, *_ = torch.linalg.lstsq(ts.basis, dq.view(-1, 1))
    return a.view(-1)


@torch.no_grad()
def from_tangent_coords(ts: TangentSpace, a: torch.Tensor) -> torch.Tensor:
    """Map tangent coordinates back to configuration space."""
    if ts.dim == 0:
        return ts.root.clone()
    return (ts.root + ts.basis @ a)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


def spaces_of(bank):
    return bank.spaces if hasattr(bank, "spaces") else bank


def bank_len(bank) -> int:
    return len(spaces_of(bank))


@dataclass
class ProjectorStackCache:
    counts: Optional[Tuple[int, ...]] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    P_flat: Optional[torch.Tensor] = None
    J_flat: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None

    @torch.no_grad()
    def get(
        self,
        *,
        ts_bank: Sequence,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = tuple(bank_len(bank) for bank in ts_bank)
        if self._is_usable(counts=counts, device=device, dtype=dtype, require_jacobian=False):
            return self.P_flat, self.offsets  # type: ignore[return-value]

        P_flat, _J_flat, offsets = self._rebuild(
            ts_bank=ts_bank,
            counts=counts,
            device=device,
            dtype=dtype,
            include_jacobian=False,
        )
        return P_flat, offsets

    @torch.no_grad()
    def get_with_jacobian(
        self,
        *,
        ts_bank: Sequence,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = tuple(bank_len(bank) for bank in ts_bank)
        if self._is_usable(counts=counts, device=device, dtype=dtype, require_jacobian=True):
            return self.P_flat, self.J_flat, self.offsets  # type: ignore[return-value]

        return self._rebuild(
            ts_bank=ts_bank,
            counts=counts,
            device=device,
            dtype=dtype,
            include_jacobian=True,
        )

    @torch.no_grad()
    def _rebuild(
        self,
        *,
        ts_bank: Sequence,
        counts: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        include_jacobian: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if not any(count > 0 for count in counts):
            raise RuntimeError("ts_bank is empty for all batches")
        counts_t = torch.tensor(counts, device=device, dtype=torch.long)

        offsets = torch.cumsum(
            torch.cat([torch.zeros((1,), device=device, dtype=torch.long), counts_t[:-1]], dim=0),
            dim=0,
        )
        proj_list = [
            torch.stack([ts.projector.to(device=device, dtype=dtype) for ts in spaces_of(bank)], dim=0)
            for bank, count in zip(ts_bank, counts)
            if count > 0
        ]
        P_flat = torch.cat(proj_list, dim=0)
        J_flat = None
        if include_jacobian:
            jac_list = [
                torch.stack([ts.J.to(device=device, dtype=dtype) for ts in spaces_of(bank)], dim=0)
                for bank, count in zip(ts_bank, counts)
                if count > 0
            ]
            J_flat = torch.cat(jac_list, dim=0)

        self.counts = counts
        self.device = device
        self.dtype = dtype
        self.P_flat = P_flat
        self.J_flat = J_flat
        self.offsets = offsets
        return P_flat, J_flat, offsets

    def invalidate(self) -> None:
        self.counts = None
        self.device = None
        self.dtype = None
        self.P_flat = None
        self.J_flat = None
        self.offsets = None

    def _is_usable(
        self,
        *,
        counts: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        require_jacobian: bool,
    ) -> bool:
        return (
            self.P_flat is not None
            and self.offsets is not None
            and ((not require_jacobian) or (self.J_flat is not None))
            and self.counts == counts
            and self.device == device
            and self.dtype == dtype
        )

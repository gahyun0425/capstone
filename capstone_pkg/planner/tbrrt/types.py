from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Node:
    q: torch.Tensor            # (D,)
    parent: int                # parent index (-1 for root)
    ts_id: int                 # tangent space id


@dataclass
class PlanStats:
    iters: int
    nodes_A: int
    nodes_B: int
    ts_count: int
    time_sec: float
    extra: Dict[str, Any]


@dataclass
class PlanResult:
    success: bool
    path: Optional[List[List[float]]]
    stats: PlanStats
    conn_idx_A: Optional[int] = None
    conn_idx_B: Optional[int] = None

from __future__ import annotations

import time
import torch


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def now(device: torch.device) -> float:
    sync_if_cuda(device)
    return time.perf_counter()


def fmt_s(x: float) -> str:
    return f"{x:.6f} s"

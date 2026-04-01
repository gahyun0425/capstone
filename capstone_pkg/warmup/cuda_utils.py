from __future__ import annotations

import torch


def log_cuda_info(prefix: str = "[CUDA]") -> None:
    print(f"{prefix} torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        print(f"{prefix} current_device = {dev} ({name})")
        try:
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            print(f"{prefix} memory_allocated = {alloc:.1f} MB")
            print(f"{prefix} memory_reserved  = {reserv:.1f} MB")
        except Exception:
            pass


def cuda_context_warmup(device: torch.device, n: int = 2) -> None:
    """첫 CUDA 호출 튐 제거"""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    for _ in range(max(1, int(n))):
        _ = torch.zeros(1, device=device)
    torch.cuda.synchronize()


def set_torch_perf_flags() -> None:
    """init 초반에 한번만 호출하는 게 보통 좋음"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

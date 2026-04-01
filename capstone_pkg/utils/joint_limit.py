# waypoint_only.py
# "시작 관절각(q_start)과 목표 관절각(q_goal) 사이에서"
# 랜덤 waypoint(중간 관절각 후보)들을 여러 개 샘플링하는 유틸 코드

from __future__ import annotations

from dataclasses import dataclass

import torch
import yaml


@dataclass(frozen=True)
class JointLimitsTorch:
    lower: torch.Tensor  # (D,)
    upper: torch.Tensor  # (D,)

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.min(torch.max(q, self.lower), self.upper)

# 역할: YAML 파일에서 joint limit(lower/upper)을 읽어 PyTorch 텐서로 만들고 JointLimitsTorch로 반환.
def load_joint_limits_torch(joint_limits_yml: str, device: torch.device, dtype: torch.dtype) -> JointLimitsTorch:
    with open(joint_limits_yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jl = data.get("joint_limits", data)
    lower = torch.tensor(jl["lower"], device=device, dtype=dtype)
    upper = torch.tensor(jl["upper"], device=device, dtype=dtype)

    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError(f"joint limits shape error: lower={lower.shape}, upper={upper.shape}")

    return JointLimitsTorch(lower=lower, upper=upper)
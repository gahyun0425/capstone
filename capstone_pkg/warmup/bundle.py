from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from curobo.wrap.model.robot_world import RobotWorld

from capstone_pkg.kinematics.curobo_ik import FastBimanualIK


@dataclass
class WarmupBundle:
    device: torch.device
    dtype: torch.dtype

    ik_solver: FastBimanualIK
    self_collision_checker: Any  # get_self_collision_checker() 반환

    # 배치 collision(trajectory) 용 RobotWorld
    robot_world: RobotWorld
    robot_cfg_dict: Dict[str, Any]

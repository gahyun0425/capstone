from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.types.robot import RobotConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from capstone_pkg.collision_check.collision_link import (
    ensure_collision_fields,
    add_connected_link_collision_ignores,
)

from .timing import sync_if_cuda


def build_robot_world_for_self_collision(
    robot_yml: str,
    device: torch.device,
) -> Tuple[RobotWorld, Dict[str, Any]]:
    """trajectory/edge batch collision용 RobotWorld 생성"""
    tensor_args = TensorDeviceType(device=device)
    cfg = load_yaml(robot_yml)
    robot_cfg_dict = cfg["robot_cfg"]

    ensure_collision_fields(robot_cfg_dict)
    add_connected_link_collision_ignores(robot_cfg_dict, only_collision_links=True)

    robot_cfg_for_curobo = dict(robot_cfg_dict)
    robot_cfg_for_curobo.pop("cspace", None)

    try:
        robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo, tensor_args)
    except TypeError:
        robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo)

    rw_cfg = RobotWorldConfig.load_from_config(
        robot_config=robot_cfg,
        world_model=None,
        tensor_args=tensor_args,
        collision_activation_distance=0.0,
        self_collision_activation_distance=0.0,
    )
    return RobotWorld(rw_cfg), robot_cfg_dict


@torch.no_grad()
def robotworld_warmup(robot_world: RobotWorld, device: torch.device, iters: int = 2) -> None:
    """RobotWorld(배치 collision용) 내부 커널/FK/거리계산 워밍업"""
    if iters <= 0:
        return

    dof = len(robot_world.kinematics.joint_names)

    for _ in range(int(iters)):
        q = torch.zeros((1, dof), device=device, dtype=torch.float32)
        st = robot_world.get_kinematics(q)

        x_sph = None
        for name in ["robot_spheres", "link_spheres", "spheres"]:
            if hasattr(st, name):
                cand = getattr(st, name)
                if cand is not None:
                    x_sph = cand
                    break

        if x_sph is None:
            fk_out = robot_world.kinematics.forward(q)
            x_sph = fk_out[-1]

        if x_sph.dim() == 3:
            x_sph = x_sph.unsqueeze(1)

        _ = robot_world.get_self_collision_distance(x_sph)

    sync_if_cuda(device)

from __future__ import annotations

import torch
from curobo.types.math import Pose

from capstone_pkg.kinematics.curobo_ik import FastBimanualIK
from .timing import sync_if_cuda


def ik_solver_warmup_reachable(
    ik: FastBimanualIK,
    device: torch.device,
    iters: int = 3,
    batch_size: int = 100,
) -> None:
    """
    reachable warmup:
      - sample_configs(batch) -> fk -> goal Pose -> solve_batch
    """
    if iters <= 0:
        return

    for _ in range(int(iters)):
        ql = ik.left_solver.sample_configs(batch_size)
        kin_l = ik.left_solver.fk(ql)
        goal_l = Pose(kin_l.ee_position, kin_l.ee_quaternion)

        qr = ik.right_solver.sample_configs(batch_size)
        kin_r = ik.right_solver.fk(qr)
        goal_r = Pose(kin_r.ee_position, kin_r.ee_quaternion)

        # cuRobo IK는 enable_grad 맥락을 요구하는 경우가 있어 유지
        with torch.enable_grad():
            _ = ik.left_solver.solve_batch(goal_l)
            _ = ik.right_solver.solve_batch(goal_r)

    sync_if_cuda(device)

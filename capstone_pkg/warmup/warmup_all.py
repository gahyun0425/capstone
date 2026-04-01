#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional
import torch

from capstone_pkg.kinematics.curobo_ik import FastBimanualIK
from capstone_pkg.collision_check.collision import get_self_collision_checker

from .bundle import WarmupBundle
from .timing import now, fmt_s, sync_if_cuda
from .cuda_utils import log_cuda_info, cuda_context_warmup, set_torch_perf_flags
from .ik_warmup import ik_solver_warmup_reachable
from .robotworld_utils import build_robot_world_for_self_collision, robotworld_warmup


def init_all_and_warmup(
    *,
    robot_yml: str,
    left_ee: str,
    right_ee: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    ik_warmup_iters: int = 3,
    cuda_ctx_warmup_iters: int = 2,
    robotworld_warmup_iters: int = 2,
    selfcol_warmup_iters: int = 1, 
    log_prefix: str = "[INIT]",
    world_yml: Optional[str] = None,
) -> WarmupBundle:
    """
    모든 초기화 + 웜업을 좌표 입력 전에 끝내기 위한 함수
    """
    cpu = (device.type == "cpu")

    # torch perf flags
    set_torch_perf_flags()

    log_cuda_info(f"{log_prefix}[CUDA]")

    t0 = now(device)

    # 1) CUDA context warmup
    t_ctx0 = now(device)
    cuda_context_warmup(device=device, n=cuda_ctx_warmup_iters)
    t_ctx1 = now(device)

    # 2) IK solver init
    t_ik_init0 = now(device)
    ik = FastBimanualIK(
        robot_yml,
        left_ee=left_ee,
        right_ee=right_ee,
        cpu=cpu,
        world_yml=world_yml,  # FastBimanualIK가 받으면 사용, 아니면 무시될 수 있음(구현에 따라)
    )
    t_ik_init1 = now(device)

    # 3) IK warmup
    t_ik_w0 = now(device)
    ik_solver_warmup_reachable(ik, device=device, iters=ik_warmup_iters, batch_size=100)
    t_ik_w1 = now(device)

    # 4) Self collision checker init
    t_sc_init0 = now(device)
    sc_checker = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
    t_sc_init1 = now(device)

    # 5) Self collision warmup
    t_sc_w0 = now(device)
    try:
        # cspace 14개 기준으로 0 (로봇에 따라 다르면 여기서 예외 날 수 있음)
        _ = sc_checker.check_single([0.0] * 14)
    except Exception:
        pass
    sync_if_cuda(device)
    t_sc_w1 = now(device)

    # 6) RobotWorld for batch collision (trajectory용)
    t_rw0 = now(device)
    robot_world, robot_cfg_dict = build_robot_world_for_self_collision(robot_yml, device=device)
    t_rw1 = now(device)

    # 7) RobotWorld warmup
    t_rw_w0 = now(device)
    robotworld_warmup(robot_world, device=device, iters=robotworld_warmup_iters)
    t_rw_w1 = now(device)

    t1 = now(device)

    print("\n================ INIT/WARMUP BREAKDOWN ================")
    print(f"[TIME] cuda_context_warmup (n={cuda_ctx_warmup_iters})     : {fmt_s(t_ctx1 - t_ctx0)}")
    print(f"[TIME] IK init                                             : {fmt_s(t_ik_init1 - t_ik_init0)}")
    print(f"[TIME] IK warmup reachable (n={ik_warmup_iters})           : {fmt_s(t_ik_w1 - t_ik_w0)}")
    print(f"[TIME] SelfCollisionChecker init                           : {fmt_s(t_sc_init1 - t_sc_init0)}")
    print(f"[TIME] SelfCollisionChecker warmup                         : {fmt_s(t_sc_w1 - t_sc_w0)}")
    print(f"[TIME] RobotWorld init (batch collision)                   : {fmt_s(t_rw1 - t_rw0)}")
    print(f"[TIME] RobotWorld warmup                                   : {fmt_s(t_rw_w1 - t_rw_w0)}")
    print(f"[TIME] TOTAL init+warmup                                   : {fmt_s(t1 - t0)}")
    print("=======================================================\n")

    return WarmupBundle(
        device=device,
        dtype=dtype,
        ik_solver=ik,
        self_collision_checker=sc_checker,
        robot_world=robot_world,
        robot_cfg_dict=robot_cfg_dict,
    )


# (선택) 이 파일을 단독 실행할 때만 테스트용으로 돌리고 싶으면 사용
def _main_smoke() -> None:
    from capstone_pkg.utils.config import ROBOT_YAML_STR, LEFT_GRIPPER, RIGHT_GRIPPER

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = init_all_and_warmup(
        robot_yml=ROBOT_YAML_STR,
        left_ee=LEFT_GRIPPER,
        right_ee=RIGHT_GRIPPER,
        device=dev,
    )


if __name__ == "__main__":
    _main_smoke()

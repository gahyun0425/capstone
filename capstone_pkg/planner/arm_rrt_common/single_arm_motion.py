from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np


@dataclass
class SingleArmMotionPlan:
    arm: str
    cspace_joint_names: list[str]
    active_joint_names: list[str]
    q_start_cspace: list[float]
    q_goal_cspace: list[float]
    raw_path: list[list[float]]
    spline_path: list[list[float]]


def normalize_arm_name(arm: str) -> str:
    raw = str(arm).strip().lower()
    aliases = {
        "l": "left",
        "left": "left",
        "left_arm": "left",
        "left-arm": "left",
        "r": "right",
        "right": "right",
        "right_arm": "right",
        "right-arm": "right",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        raise ValueError("arm must be one of: left, right")
    return normalized


def xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> list[float]:
    if len(quat_xyzw) != 4:
        raise ValueError("quat must contain 4 values in xyzw order")
    x, y, z, w = [float(v) for v in quat_xyzw]
    return [w, x, y, z]


def normalize_single_arm_planner_backend(planner_backend: str) -> str:
    raw = str(planner_backend).strip().lower()
    aliases = {
        "tbrrt": "tbrrt_batch_conext",
        "tb-rrt": "tbrrt_batch_conext",
        "batch_conext": "tbrrt_batch_conext",
        "batch-conext": "tbrrt_batch_conext",
        "tbrrt_batch_conext": "tbrrt_batch_conext",
        "tbrrt-batch-conext": "tbrrt_batch_conext",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        raise ValueError("planner_backend must be tbrrt_batch_conext")
    return normalized


def _build_ik_seed_batch(
    q_start: Sequence[float],
    *,
    batch_size: int,
    noise_std: float,
    random_seed: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[list[float]]:
    q0 = np.asarray(q_start, dtype=np.float64)
    if q0.ndim != 1:
        raise ValueError(f"q_start must be 1-D, got shape={q0.shape}")

    out: list[list[float]] = [q0.tolist()]
    if batch_size <= 1 or noise_std <= 0.0:
        return out

    rng = np.random.default_rng(int(random_seed))
    for _ in range(int(batch_size) - 1):
        q_seed = q0 + rng.normal(loc=0.0, scale=float(noise_std), size=q0.shape)
        q_seed = np.clip(q_seed, lower, upper)
        out.append(q_seed.tolist())
    return out


def _dedupe_q_candidates(
    candidates: Sequence[Sequence[float]],
    *,
    atol: float,
) -> list[list[float]]:
    if atol <= 0.0:
        return [[float(v) for v in q] for q in candidates]

    unique: list[np.ndarray] = []
    out: list[list[float]] = []
    for q in candidates:
        q_np = np.asarray(q, dtype=np.float64)
        if any(float(np.linalg.norm(q_np - u)) <= float(atol) for u in unique):
            continue
        unique.append(q_np)
        out.append([float(v) for v in q_np.tolist()])
    return out


def plan_single_arm_motion(
    *,
    robot_yml: str,
    arm: str,
    target_xyz: Sequence[float],
    target_quat_xyzw: Sequence[float],
    world_yml: str | None,
    cpu: bool,
    joint_state_topic: str,
    joint_state_wait_s: float,
    use_current_joint_state_start: bool,
    q_start_cspace: Sequence[float] | None = None,
    step: float,
    max_iters: int,
    goal_bias: float,
    connect_threshold: float,
    planner_backend: str = "tbrrt_batch_conext",
    joint_limit_yml: str | None = None,
    ik_batch: int = 100,
    ik_seed_noise_std: float = 0.25,
    ik_seed_random_seed: int = 0,
    ik_goal_dedupe_tol: float = 1.0e-3,
    tbrrt_cfg=None,
    tbrrt_block_k: int = 32,
    spline_dt: float = 0.01,
) -> SingleArmMotionPlan:
    from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
    from capstone_pkg.planner.arm_rrt_common.path_publisher import read_joint_positions_once
    from capstone_pkg.utils.joint_limit import load_joint_limits_torch
    from capstone_pkg.utils.config import JOINT_LIMIT, LEFT_JOINTS, RIGHT_JOINTS
    import torch

    normalized_arm = normalize_arm_name(arm)
    normalized_backend = normalize_single_arm_planner_backend(planner_backend)
    ik = get_single_arm_ik(
        robot_yml,
        arm=normalized_arm,
        cpu=cpu,
        world_yml=world_yml,
    )
    resolved_q_start_cspace = [0.0 for _ in ik.cspace_joint_names]
    if q_start_cspace is not None:
        if len(q_start_cspace) != len(ik.cspace_joint_names):
            raise ValueError(
                f"q_start_cspace length {len(q_start_cspace)} "
                f"!= cspace dof {len(ik.cspace_joint_names)}"
            )
        resolved_q_start_cspace = [float(v) for v in q_start_cspace]
    elif use_current_joint_state_start:
        resolved_q_start_cspace = read_joint_positions_once(
            ik.cspace_joint_names,
            topic=joint_state_topic,
            wait_s=joint_state_wait_s,
        )

    jl = load_joint_limits_torch(str(joint_limit_yml or JOINT_LIMIT), device=torch.device("cpu"), dtype=torch.float32)
    ik_seed_batch = _build_ik_seed_batch(
        resolved_q_start_cspace,
        batch_size=max(1, int(ik_batch)),
        noise_std=float(ik_seed_noise_std),
        random_seed=int(ik_seed_random_seed),
        lower=jl.lower.detach().cpu().numpy(),
        upper=jl.upper.detach().cpu().numpy(),
    )
    target_xyz_list = [float(v) for v in target_xyz]
    target_quat_wxyz = xyzw_to_wxyz(target_quat_xyzw)
    ik_outs = ik.solve_batch(
        [list(target_xyz_list) for _ in range(len(ik_seed_batch))],
        [list(target_quat_wxyz) for _ in range(len(ik_seed_batch))],
        q_start_cspace=resolved_q_start_cspace,
        q_seed_cspace_batch=ik_seed_batch,
    )

    cand_q = [
        list(out.q_cspace)
        for out in ik_outs
        if out.success and out.q_cspace is not None
    ]
    cand_q = _dedupe_q_candidates(
        cand_q,
        atol=float(ik_goal_dedupe_tol),
    )
    if not cand_q:
        raise RuntimeError("IK failed or target pose is in collision")

    q_start_np = np.asarray(resolved_q_start_cspace, dtype=np.float64)
    q_goal = min(
        cand_q,
        key=lambda q: float(np.linalg.norm(np.asarray(q, dtype=np.float64) - q_start_np)),
    )

    active_joint_names = list(LEFT_JOINTS if normalized_arm == "left" else RIGHT_JOINTS)
    from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
        plan_single_arm_tbrrt_batch_conext,
    )
    from capstone_pkg.planner.tbrrt.config import TBRRTConfig

    cfg = tbrrt_cfg
    if cfg is None:
        cfg = TBRRTConfig(
            step_size=float(step),
            goal_threshold=float(connect_threshold),
            goal_bias=float(goal_bias),
            max_iters=int(max_iters),
            topp_output_dt=max(1.0e-3, float(spline_dt)),
        )
    else:
        cfg = replace(
            cfg,
            topp_output_dt=max(1.0e-3, float(spline_dt)),
        )

    out = plan_single_arm_tbrrt_batch_conext(
        robot_yml=robot_yml,
        arm=normalized_arm,
        q_start=resolved_q_start_cspace,
        q_goals=[q_goal],
        world_yml=world_yml,
        cpu=cpu,
        cfg=cfg,
        joint_limit_yml=str(joint_limit_yml or JOINT_LIMIT),
        block_k=int(tbrrt_block_k),
    )
    if not out.success or not out.path:
        raise RuntimeError(f"TB-RRT batch_conext failed: {out.stats.extra}")

    raw_path = [[float(v) for v in q] for q in out.path]
    spline_path = [[float(v) for v in q] for q in out.path]

    return SingleArmMotionPlan(
        arm=normalized_arm,
        cspace_joint_names=list(ik.cspace_joint_names),
        active_joint_names=active_joint_names,
        q_start_cspace=[float(v) for v in resolved_q_start_cspace],
        q_goal_cspace=[float(v) for v in q_goal],
        raw_path=[[float(v) for v in q] for q in raw_path],
        spline_path=[[float(v) for v in q] for q in spline_path],
    )


def build_active_joint_path(plan: SingleArmMotionPlan) -> tuple[list[str], list[list[float]]]:
    name_to_idx = {name: idx for idx, name in enumerate(plan.cspace_joint_names)}
    joint_names = [name for name in plan.active_joint_names if name in name_to_idx]
    if not joint_names:
        raise RuntimeError("No active joints found in cspace joint names")

    active_path = [
        [float(q[name_to_idx[name]]) for name in joint_names]
        for q in plan.spline_path
    ]
    return joint_names, active_path


def _wrapped_joint_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def _resolve_arrival_wait_s(
    *,
    path_len: int,
    dt: float,
    configured_wait_s: float,
) -> float:
    if float(configured_wait_s) >= 0.0:
        return float(configured_wait_s)
    traj_duration_s = max(0.0, float(max(0, path_len - 1)) * float(dt))
    return max(2.0, traj_duration_s + 2.0)


def _nearest_waypoint_index(
    current_positions: Sequence[float],
    path: Sequence[Sequence[float]],
) -> int:
    best_idx = 0
    best_err = float("inf")
    for idx, waypoint in enumerate(path):
        err = max(
            _wrapped_joint_delta(float(current_positions[j]), float(waypoint[j]))
            for j in range(min(len(current_positions), len(waypoint)))
        )
        if err < best_err:
            best_err = err
            best_idx = idx
    return int(best_idx)


def _build_retry_path(
    *,
    current_positions: Sequence[float],
    original_path: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not original_path:
        raise ValueError("original_path is empty")
    goal = [float(v) for v in original_path[-1]]
    current = [float(v) for v in current_positions]
    if len(original_path) == 1:
        return [current, goal]

    nearest_idx = _nearest_waypoint_index(current, original_path)
    if nearest_idx >= len(original_path) - 1:
        return [current, goal]

    retry_path = [current]
    retry_path.extend(
        [[float(v) for v in waypoint] for waypoint in original_path[nearest_idx + 1 :]]
    )
    if len(retry_path) == 1:
        retry_path.append(goal)
    return retry_path


def execute_single_arm_motion(plan: SingleArmMotionPlan, args) -> None:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        publish_joint_path,
        publish_joint_trajectory,
        read_joint_positions_once,
        send_joint_trajectory_action,
        wait_for_joint_positions,
    )

    joint_names, active_path = build_active_joint_path(plan)
    goal_positions = [float(v) for v in active_path[-1]]
    max_attempts = max(1, int(getattr(args, "arrival_max_retries", 1)))

    def _send_command(cmd_path: Sequence[Sequence[float]]) -> None:
        if args.publish_mode == "real":
            topic = args.real_left_topic if plan.arm == "left" else args.real_right_topic
            action_name = args.real_left_action if plan.arm == "left" else args.real_right_action

            if args.real_use_action:
                try:
                    send_joint_trajectory_action(
                        cmd_path,
                        joint_names,
                        action_name=action_name,
                        dt=float(args.publish_dt),
                        wait_server_s=float(args.action_wait_server_s),
                        wait_result_s=float(args.action_wait_result_s),
                    )
                    return
                except RuntimeError:
                    if not args.real_action_fallback_to_topic:
                        raise

            publish_joint_trajectory(
                cmd_path,
                joint_names,
                topic=topic,
                dt=float(args.publish_dt),
                wait_subscriber_s=float(args.publish_wait_subscriber_s),
                require_subscriber=bool(args.publish_require_subscriber),
                retry_until_subscriber=bool(args.publish_retry_until_subscriber),
                publish_repeat=int(args.publish_repeat),
                publish_period_s=float(args.publish_period_s),
                wait_ack_s=float(args.publish_wait_ack_s),
                keep_alive_s=float(args.publish_keep_alive_s),
                reliability=str(getattr(args, "publish_reliability", "best_effort")),
                durability=(
                    "transient_local"
                    if bool(getattr(args, "publish_transient_local", False))
                    else str(getattr(args, "publish_durability", "volatile"))
                ),
                qos_depth=int(getattr(args, "publish_qos_depth", 1)),
                start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
            )
            return

        publish_joint_path(
            cmd_path,
            joint_names,
            topic=args.publish_topic,
            dt=float(args.publish_dt),
            wait_subscriber_s=float(args.publish_wait_subscriber_s),
        )

    last_err = float("inf")
    for attempt_idx in range(max_attempts):
        if attempt_idx == 0:
            cmd_path = active_path
        else:
            current_positions = read_joint_positions_once(
                joint_names,
                topic=str(args.joint_state_topic),
                wait_s=float(args.joint_state_wait_s),
            )
            cmd_path = _build_retry_path(
                current_positions=current_positions,
                original_path=active_path,
            )
            print(
                f"[ARRIVAL] retry {attempt_idx + 1}/{max_attempts}: "
                "re-publishing remaining path toward goal."
            )

        try:
            _send_command(cmd_path)

            arrived, _current_positions, max_abs_err = wait_for_joint_positions(
                joint_names,
                goal_positions,
                topic=str(args.joint_state_topic),
                wait_s=_resolve_arrival_wait_s(
                    path_len=len(cmd_path),
                    dt=float(args.publish_dt),
                    configured_wait_s=float(getattr(args, "arrival_wait_s", -1.0)),
                ),
                tolerance=float(getattr(args, "arrival_joint_tolerance", 0.05)),
                poll_period_s=float(getattr(args, "arrival_poll_s", 0.05)),
            )
        except RuntimeError as exc:
            if attempt_idx + 1 >= max_attempts:
                raise
            print(
                f"[ARRIVAL] attempt {attempt_idx + 1}/{max_attempts} failed: {exc} "
                "-> re-publishing."
            )
            continue
        if arrived:
            print(
                f"[ARRIVAL] confirmed: max_abs_err={max_abs_err:.6f} "
                f"(tol={float(getattr(args, 'arrival_joint_tolerance', 0.05)):.6f})"
            )
            return

        last_err = max_abs_err
        print(
            f"[ARRIVAL] not reached after attempt {attempt_idx + 1}/{max_attempts}: "
            f"max_abs_err={max_abs_err:.6f}"
        )

    raise RuntimeError(
        "Failed to confirm stage arrival after "
        f"{max_attempts} attempt(s); last max abs joint error={last_err:.6f}"
    )

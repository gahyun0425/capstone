from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import torch

from capstone_pkg.utils.config import (
    ROBOT_YAML,
    JOINT_LIMIT,
    ROBOT_XML,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    CSPACE_JOINT_NAMES_14,
    LEFT_JOINTS,
    RIGHT_JOINTS,
)
from capstone_pkg.planner.start_goal import get_start_and_goal_from_topic_and_ik
from capstone_pkg.planner.bidir_rrt.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_trajectory_group,
    send_joint_trajectory_action_group,
)
from capstone_pkg.utils.joint_limit import load_joint_limits_torch
from capstone_pkg.constraint_projection.constraint import RigidConstraint
from capstone_pkg.constraint_projection.projection import ManifoldProjector
from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.planner.tbrrt import TBRRTConfig
from capstone_pkg.planner.tbrrt.basic_tbrrt.basic_tbrrt import plan_tbrrt_extcon
from capstone_pkg.utils.jointstate_publisher import publish_q_path_as_jointstate_keep_gripper_closed


def build_tbrrt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", type=str, default=ROBOT_YAML)
    ap.add_argument("--world_yml", type=str, default=None, help="world collision yaml; None이면 world collision 비활성화")
    ap.add_argument("--joint_limit_yml", type=str, default=JOINT_LIMIT)
    ap.add_argument("--jointstate_topic", type=str, default="/joint_states")
    ap.add_argument("--cmd_topic", type=str, default="/joint_states_cmd")
    ap.add_argument("--publish_mode", choices=("joint_state", "real"), default="joint_state")
    ap.add_argument("--mujoco_xml", type=str, default=ROBOT_XML)
    ap.add_argument("--left_ee", type=str, default=LEFT_EE_FRAME)
    ap.add_argument("--right_ee", type=str, default=RIGHT_EE_FRAME)
    ap.add_argument("--goal_topk", type=int, default=16)
    ap.add_argument("--select_goal", type=str, default="first_free", choices=["first_free", "min_penetration"])
    ap.add_argument("--fail_if_start_in_collision", action="store_true")
    ap.add_argument("--target_left_xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--target_left_rpy_deg", type=float, nargs=3, required=True, metavar=("R", "P", "Y"))
    ap.add_argument(
        "--planar_xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep x-y planar motion: current left z and roll/pitch are preserved, yaw is allowed",
    )
    ap.add_argument("--rigid_orientation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lock_z", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--step_size", type=float, default=0.1)
    ap.add_argument("--goal_threshold", type=float, default=0.10)
    ap.add_argument("--EM", type=float, default=0.02)
    ap.add_argument("--E_conn", type=float, default=0.025)
    ap.add_argument("--ts_radius", type=float, default=1.1)
    ap.add_argument("--p_uniform", type=float, default=0.25)
    ap.add_argument("--goal_bias", type=float, default=0.15)
    ap.add_argument("--connect_max_steps", type=int, default=10)
    ap.add_argument("--escape_extend_steps", type=int, default=5)
    ap.add_argument("--max_iters", type=int, default=500000)
    ap.add_argument("--time_limit_sec", type=float, default=900.0)
    ap.add_argument("--proj_iters", type=int, default=60)
    ap.add_argument("--proj_tol", type=float, default=1e-3)
    ap.add_argument("--proj_fd_eps", type=float, default=1e-3)
    ap.add_argument("--edge_step_q", type=float, default=0.03)
    ap.add_argument("--edge_max_steps", type=int, default=128)
    ap.add_argument("--svd_tol", type=float, default=1e-6)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--start_delay_s", type=float, default=0.2)
    ap.add_argument("--hold_last_s", type=float, default=0.0)
    ap.add_argument("--close_value", type=float, default=1.0)
    ap.add_argument("--real_left_topic", default="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory")
    ap.add_argument("--real_right_topic", default="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory")
    ap.add_argument("--real_left_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_left/joint_trajectory")
    ap.add_argument("--real_right_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_right/joint_trajectory")
    ap.add_argument("--real_use_action", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--real_action_fallback_to_topic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--real_left_action", default="/leader/joint_trajectory_command_broadcaster_left/follow_joint_trajectory")
    ap.add_argument("--real_right_action", default="/leader/joint_trajectory_command_broadcaster_right/follow_joint_trajectory")
    ap.add_argument("--real_left_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_left/follow_joint_trajectory")
    ap.add_argument("--real_right_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_right/follow_joint_trajectory")
    ap.add_argument("--action_wait_server_s", type=float, default=2.0)
    ap.add_argument("--action_wait_result_s", type=float, default=-1.0)
    ap.add_argument("--publish_wait_subscriber_s", type=float, default=5.0)
    ap.add_argument("--publish_require_subscriber", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--publish_retry_until_subscriber", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--publish_repeat", type=int, default=2)
    ap.add_argument("--publish_period_s", type=float, default=0.05)
    ap.add_argument("--publish_wait_ack_s", type=float, default=1.0)
    ap.add_argument("--publish_keep_alive_s", type=float, default=0.5)
    ap.add_argument("--publish_transient_local", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--attach_object", action="store_true", default=False)
    return ap


def main_tbrrt(argv: Sequence[str] | None = None) -> int:
    args = build_tbrrt_parser().parse_args(list(argv) if argv is not None else None)
    t0 = time.time()
    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float32

    q_start, q_goals, best_pen, t_pose_done = get_start_and_goal_from_topic_and_ik(
        robot_yml=str(args.robot_yml),
        jointstate_topic=str(args.jointstate_topic),
        joint_names=list(CSPACE_JOINT_NAMES_14),
        target_left_xyz=list(args.target_left_xyz),
        target_left_rpy_deg=list(args.target_left_rpy_deg),
        left_ee=str(args.left_ee),
        right_ee=str(args.right_ee),
        select=args.select_goal,
        device_str=("cuda" if device.type == "cuda" else "cpu"),
        world_yml=(None if args.world_yml in (None, "", "none", "None") else str(args.world_yml)),
        fail_if_start_in_collision=bool(args.fail_if_start_in_collision),
        topk=int(args.goal_topk),
        planar_xy=bool(args.planar_xy),
    )
    print(f"[tbrrt] got start + {len(q_goals)} goals. best_pen={best_pen:.6f} t_pose_done={t_pose_done:.3f}s")

    _world_yml = None if args.world_yml in (None, "", "none", "None") else str(args.world_yml)
    checker = get_self_collision_checker(str(args.robot_yml), cpu=(device.type == "cpu"), world_yml=_world_yml)

    if bool(args.attach_object):
        if not hasattr(checker, "_build_q_active_from_cspace"):
            raise RuntimeError("SelfCollisionChecker has no _build_q_active_from_cspace()")
        q_start_c = torch.tensor(q_start, device=device, dtype=torch.float32).view(1, -1)
        q_model = checker._build_q_active_from_cspace(q_start_c)[0]
        checker.attach_mujoco_object_to_robot(
            mujoco_xml_path=str(args.mujoco_xml),
            q_model_order=q_model,
            link_name=str(args.left_ee),
            name_prefix="att_",
            disable_in_world=False,
        )
        print("[tbrrt] attached mujoco object to robot")

    jl = load_joint_limits_torch(str(args.joint_limit_yml), device=device, dtype=dtype)
    q_ref = torch.tensor(q_start, device=device, dtype=dtype)
    lock_z = bool(args.lock_z or args.planar_xy)
    if bool(args.planar_xy):
        print("[tbrrt] planar_xy enabled: enforcing left z lock and yaw-only absolute rotation")
    if bool(args.planar_xy) and bool(args.rigid_orientation):
        print("[tbrrt][WARN] rigid_orientation is also enabled, so yaw will be locked too.")
    constraint = RigidConstraint(
        robot_yml=str(args.robot_yml),
        left_ee=str(args.left_ee),
        right_ee=str(args.right_ee),
        q_ref=q_ref,
        device=device,
        dtype=dtype,
        mode="se3",
        rigid_orientation=bool(args.rigid_orientation),
        lock_z=lock_z,
        planar_xy=bool(args.planar_xy),
    )
    projector = ManifoldProjector(constraint=constraint, limits=jl, max_iters=int(args.proj_iters), tol=float(args.proj_tol), fd_eps=float(args.proj_fd_eps))

    cfg = TBRRTConfig(
        step_size=float(args.step_size),
        goal_threshold=float(args.goal_threshold),
        EM=float(args.EM),
        E_conn=float(args.E_conn),
        ts_radius=float(args.ts_radius),
        p_uniform=float(args.p_uniform),
        goal_bias=float(args.goal_bias),
        connect_max_steps=int(args.connect_max_steps),
        escape_extend_steps=max(1, int(args.escape_extend_steps)),
        max_iters=int(args.max_iters),
        time_limit_sec=float(args.time_limit_sec),
        edge_step_q=float(args.edge_step_q),
        edge_max_steps=int(args.edge_max_steps),
        svd_tol=float(args.svd_tol),
        seed=(None if int(args.seed) < 0 else int(args.seed)),
    )

    out = plan_tbrrt_extcon(q_start=q_start, q_goals=q_goals, cfg=cfg, checker=checker, projector=projector, joint_limits=jl, device=device)
    if not out.success:
        print(f"[tbrrt] planning failed: {out.stats.extra}")
        print(f"[tbrrt] total wall time = {time.time() - t0:.3f}s")
        return 1

    print("\n[tbrrt] SUCCESS")
    print(f"  iters={out.stats.iters} nodesA={out.stats.nodes_A} nodesB={out.stats.nodes_B} ts={out.stats.ts_count} time={out.stats.time_sec:.3f}s")
    if out.path is not None:
        print(f"  path_len={len(out.path)}")

    traj_dt = 1.0 / max(1e-6, float(args.hz))
    path_rows = [[float(v) for v in row] for row in out.path]
    left_dof = len(LEFT_JOINTS)
    left_path = [row[:left_dof] for row in path_rows]
    right_path = [row[left_dof:] for row in path_rows]

    def _send_real_commands(
        *,
        topic_commands: list[JointTrajectoryCommand],
        action_commands: list[JointTrajectoryCommand],
        dt: float,
        start_time_delay_s: float,
        label: str,
    ) -> str:
        if args.real_use_action:
            try:
                action_targets = ", ".join(cmd.endpoint for cmd in action_commands)
                print(
                    f"[{label}] Sending FollowJointTrajectory -> {action_targets} "
                    f"(dt={dt:.3f}s)"
                )
                send_joint_trajectory_action_group(
                    action_commands,
                    dt=dt,
                    wait_server_s=float(args.action_wait_server_s),
                    wait_result_s=float(args.action_wait_result_s),
                    start_time_delay_s=float(start_time_delay_s),
                )
                return "action"
            except RuntimeError as exc:
                print(f"[ACTION] {exc}")
                if not args.real_action_fallback_to_topic:
                    raise
                topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
                print(f"[ACTION] Falling back to JointTrajectory topics -> {topic_targets}")

        topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
        print(f"[{label}] Publishing JointTrajectory -> {topic_targets} (dt={dt:.3f}s)")
        publish_joint_trajectory_group(
            topic_commands,
            dt=dt,
            wait_subscriber_s=float(args.publish_wait_subscriber_s),
            require_subscriber=bool(args.publish_require_subscriber),
            retry_until_subscriber=bool(args.publish_retry_until_subscriber),
            publish_repeat=int(args.publish_repeat),
            publish_period_s=float(args.publish_period_s),
            wait_ack_s=float(args.publish_wait_ack_s),
            keep_alive_s=float(args.publish_keep_alive_s),
            transient_local=bool(args.publish_transient_local),
            start_time_delay_s=float(start_time_delay_s),
        )
        return "topic"

    if args.publish_mode == "real":
        gripper_dt = min(0.1, traj_dt) if traj_dt > 0.0 else 0.1
        gripper_topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_gripper_topic),
                joint_names=["gripper_l_joint1"],
                path=[[float(args.close_value)]],
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_gripper_topic),
                joint_names=["gripper_r_joint1"],
                path=[[float(args.close_value)]],
            ),
        ]
        gripper_action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_gripper_action),
                joint_names=["gripper_l_joint1"],
                path=[[float(args.close_value)]],
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_gripper_action),
                joint_names=["gripper_r_joint1"],
                path=[[float(args.close_value)]],
            ),
        ]
        _send_real_commands(
            topic_commands=gripper_topic_commands,
            action_commands=gripper_action_commands,
            dt=gripper_dt,
            start_time_delay_s=0.0,
            label="GRIPPER",
        )

        arm_topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_topic),
                joint_names=list(LEFT_JOINTS),
                path=left_path,
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_topic),
                joint_names=list(RIGHT_JOINTS),
                path=right_path,
            ),
        ]
        arm_action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_action),
                joint_names=list(LEFT_JOINTS),
                path=left_path,
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_action),
                joint_names=list(RIGHT_JOINTS),
                path=right_path,
            ),
        ]
        _send_real_commands(
            topic_commands=arm_topic_commands,
            action_commands=arm_action_commands,
            dt=traj_dt,
            start_time_delay_s=float(args.start_delay_s),
            label="REAL",
        )
    else:
        q_path = torch.tensor(out.path, device=device, dtype=dtype)
        publish_q_path_as_jointstate_keep_gripper_closed(
            q_path=q_path,
            robot_yml=str(args.robot_yml),
            mujoco_xml=str(args.mujoco_xml),
            close_value=float(args.close_value),
            cmd_topic=str(args.cmd_topic),
            hz=float(args.hz),
            repeat=int(args.repeat),
            start_delay_s=float(args.start_delay_s),
            hold_last_s=float(args.hold_last_s),
            node_name="tbrrt_path_publisher",
        )

    print("  q_start(path) =", out.path[0])
    print("  q_goal(path)  =", out.path[-1])
    print(f"[tbrrt] total wall time = {time.time() - t0:.3f}s")
    return 0


# Backward-compatible entrypoint expected by capstone_pkg.main
run_tbrrt = main_tbrrt

from __future__ import annotations

import argparse
import json
from typing import Sequence

from capstone_pkg.kinematics.force_curobo_ik import ForceCuroboIK
from capstone_pkg.planner.arm_rrt_common.dual_arm_runner import (
    _normalized_world_yml,
    _publish_real_group,
    _read_target_pose,
    _validate_dual_path,
    main_dual_arm,
)
from capstone_pkg.planner.arm_rrt_common.path_publisher import (
    publish_joint_path,
    read_joint_positions_once,
)
from capstone_pkg.planner.arm_rrt_common.plot import (
    publish_joint_path_plot,
    show_joint_path_plot_matplotlib,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
    build_single_arm_tbrrt_config,
)
from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
    plan_single_arm_tbrrt_batch_conext,
)
from capstone_pkg.utils.config import (
    CSPACE_JOINT_NAMES_14,
    JOINT_LIMIT,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    ROBOT_URDF,
    ROBOT_YAML,
    WORLD_YAML,
)
from capstone_pkg.utils.world_collision_bridge import (
    DEFAULT_WORLD_COLLISION_TOPIC,
    publish_world_collision_yaml,
)

_DEFAULT_STORED_TRAJECTORY_JSON = (
    "/home/gaga/capstone_ws/src/capstone_pkg/data/arm_cart_picking_trajectory.json"
)


def _strip_option_with_nargs(
    src: Sequence[str],
    option: str,
    nargs: int,
) -> list[str]:
    out: list[str] = []
    skip = 0

    for token in src:
        if skip > 0:
            skip -= 1
            continue
        if token == option:
            skip = int(nargs)
            continue
        if token.startswith(f"{option}="):
            continue
        out.append(token)

    return out


def _build_fixed_arm_cart_picking_args(argv: Sequence[str] | None) -> list[str]:
    src = list(argv) if argv is not None else []
    args = _strip_option_with_nargs(src, "--world_yml", 1)
    args = _strip_option_with_nargs(args, "--planner_mode", 1)
    args = _strip_option_with_nargs(args, "--plot_path", 0)
    args = _strip_option_with_nargs(args, "--no-plot_path", 0)
    args = _strip_option_with_nargs(args, "--left_xyz", 3)
    args = _strip_option_with_nargs(args, "--left_quat_xyzw", 4)
    args = _strip_option_with_nargs(args, "--right_xyz", 3)
    args = _strip_option_with_nargs(args, "--right_quat_xyzw", 4)
    has_stored_json = any(
        token == "--stored_trajectory_json" or token.startswith("--stored_trajectory_json=")
        for token in args
    )

    stored_json_args = []
    if not has_stored_json:
        stored_json_args = ["--stored_trajectory_json", _DEFAULT_STORED_TRAJECTORY_JSON]

    return [
        "--world_yml",
        "none",
        "--planner_mode",
        "spline_only",
        "--no-plot_path",
        "--left_xyz",
        "0.4",
        "0.2",
        "1.0",
        "--left_quat_xyzw",
        "0.5",
        "0.5",
        "0.5",
        "0.5",
        "--right_xyz",
        "0.4",
        "-0.2",
        "1.0",
        "--right_quat_xyzw",
        "0.5",
        "-0.5",
        "0.5",
        "-0.5",
        *stored_json_args,
        *args,
    ]


def build_arm_cart_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--urdf_path", default=ROBOT_URDF)
    ap.add_argument("--joint_limit_yml", default=JOINT_LIMIT)
    ap.add_argument(
        "--world_yml",
        default=WORLD_YAML,
        help="world collision yaml; 'none' or '' disables world collision",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--save", default="", help="optional path to save result json")
    ap.add_argument(
        "--publish_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish planned path commands",
    )
    ap.add_argument(
        "--publish_mode",
        choices=("joint_state", "real"),
        default="joint_state",
        help="publish backend: joint_state or real(JointTrajectory)",
    )
    ap.add_argument(
        "--publish_topic",
        default="/joint_states_cmd",
        help="target topic for JointState command stream",
    )
    ap.add_argument(
        "--joint_state_topic",
        default="/joint_states",
        help="topic used to read the current robot joint state",
    )
    ap.add_argument(
        "--joint_state_wait_s",
        type=float,
        default=5.0,
        help="wait time [s] for the current robot joint state",
    )
    ap.add_argument(
        "--use_current_joint_state_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read current /joint_states and plan from that start state",
    )
    ap.add_argument(
        "--real_left_topic",
        default="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
        help="target topic for real left arm JointTrajectory",
    )
    ap.add_argument(
        "--real_right_topic",
        default="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
        help="target topic for real right arm JointTrajectory",
    )
    ap.add_argument(
        "--real_use_action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prefer FollowJointTrajectory action over raw JointTrajectory topic in real mode",
    )
    ap.add_argument(
        "--real_action_fallback_to_topic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fall back to raw JointTrajectory topic if FollowJointTrajectory is unavailable",
    )
    ap.add_argument(
        "--real_left_action",
        default="/leader/joint_trajectory_command_broadcaster_left/follow_joint_trajectory",
        help="FollowJointTrajectory action name for the real left arm",
    )
    ap.add_argument(
        "--real_right_action",
        default="/leader/joint_trajectory_command_broadcaster_right/follow_joint_trajectory",
        help="FollowJointTrajectory action name for the real right arm",
    )
    ap.add_argument(
        "--action_wait_server_s",
        type=float,
        default=2.0,
        help="wait time [s] for FollowJointTrajectory action server",
    )
    ap.add_argument(
        "--action_wait_result_s",
        type=float,
        default=-1.0,
        help="wait time [s] for FollowJointTrajectory result, -1 waits until execution completes",
    )
    ap.add_argument(
        "--publish_dt",
        type=float,
        default=0.01,
        help="trajectory sampling/publish period [s] between waypoints",
    )
    ap.add_argument(
        "--start_delay_s",
        type=float,
        default=0.2,
        help="delay [s] before shared JointTrajectory execution starts",
    )
    ap.add_argument(
        "--publish_wait_subscriber_s",
        type=float,
        default=5.0,
        help="wait time [s] for subscriber discovery before publishing, -1 waits forever",
    )
    ap.add_argument(
        "--publish_require_subscriber",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail if no matching JointTrajectory subscriber is found before publishing",
    )
    ap.add_argument(
        "--publish_retry_until_subscriber",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep re-publishing JointTrajectory until a subscriber match appears",
    )
    ap.add_argument(
        "--publish_repeat",
        type=int,
        default=8,
        help="number of times to publish the same JointTrajectory command",
    )
    ap.add_argument(
        "--publish_period_s",
        type=float,
        default=0.03,
        help="period [s] between repeated JointTrajectory publishes",
    )
    ap.add_argument(
        "--publish_wait_ack_s",
        type=float,
        default=0.0,
        help="wait time [s] for DDS acknowledgements after each JointTrajectory publish",
    )
    ap.add_argument(
        "--publish_keep_alive_s",
        type=float,
        default=1.0,
        help="keep publisher alive [s] after JointTrajectory publish",
    )
    ap.add_argument(
        "--publish_reliability",
        choices=("reliable", "best_effort"),
        default="best_effort",
        help="QoS reliability used for real JointTrajectory publishers",
    )
    ap.add_argument(
        "--publish_durability",
        choices=("volatile", "transient_local"),
        default="volatile",
        help="QoS durability used for real JointTrajectory publishers",
    )
    ap.add_argument(
        "--publish_qos_depth",
        type=int,
        default=1,
        help="QoS depth used for real JointTrajectory publishers",
    )
    ap.add_argument(
        "--arrival_wait_s",
        type=float,
        default=-1.0,
        help="wait time [s] to confirm real robot arrival, -1 uses trajectory duration + margin",
    )
    ap.add_argument(
        "--arrival_joint_tolerance",
        type=float,
        default=0.05,
        help="max abs joint error tolerance [rad] used to confirm arrival",
    )
    ap.add_argument(
        "--arrival_poll_s",
        type=float,
        default=0.05,
        help="poll period [s] while checking real robot arrival",
    )
    ap.add_argument(
        "--arrival_max_retries",
        type=int,
        default=-1,
        help="max re-publish attempts for real robot arrival confirmation, <=0 retries forever",
    )
    ap.add_argument(
        "--publish_transient_local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="deprecated compatibility flag; use --publish_durability",
    )
    ap.add_argument(
        "--plot_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish planned path as RViz2 MarkerArray plot",
    )
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show planned path in a matplotlib window",
    )
    ap.add_argument(
        "--plot_topic",
        default="/arm_rrt/joint_path_plot",
        help="topic for RViz2 MarkerArray plot",
    )
    ap.add_argument("--plot_frame", default="map", help="frame_id for RViz2 plot markers")
    ap.add_argument("--plot_x_step", type=float, default=0.05, help="x-axis spacing between waypoints [m]")
    ap.add_argument("--plot_y_scale", type=float, default=1.0, help="scale for joint value axis")
    ap.add_argument("--plot_z_sep", type=float, default=0.25, help="z separation between joints [m]")
    ap.add_argument("--plot_lifetime", type=float, default=0.0, help="marker lifetime [s], 0 means keep forever")
    ap.add_argument("--plot_keep_alive", type=float, default=5.0, help="keep plot publisher alive after publish [s], -1 means forever")
    ap.add_argument("--left_xyz", nargs=3, type=float, help="left target xyz in meters")
    ap.add_argument("--left_quat_xyzw", nargs=4, type=float, help="left target quaternion in xyzw order")
    ap.add_argument("--right_xyz", nargs=3, type=float, help="right target xyz in meters")
    ap.add_argument("--right_quat_xyzw", nargs=4, type=float, help="right target quaternion in xyzw order")
    ap.add_argument(
        "--force_ik_num_trials",
        type=int,
        default=24,
        help="number of dual-arm IK trials with random seed perturbations",
    )
    ap.add_argument(
        "--force_ik_seed_noise_std",
        type=float,
        default=0.25,
        help="stddev for IK seed perturbation",
    )
    ap.add_argument(
        "--force_ik_random_seed",
        type=int,
        default=0,
        help="random seed for dual-arm IK candidate search",
    )
    ap.add_argument(
        "--force_ik_num_seeds",
        type=int,
        default=20,
        help="cuRobo IK internal seed count per arm",
    )
    ap.add_argument(
        "--forward_direction_base",
        nargs=3,
        type=float,
        default=(1.0, 0.0, 0.0),
        help="base-frame direction used by force-based IK scoring",
    )
    ap.add_argument("--max_iters", type=int, default=100000)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--goal_bias", type=float, default=0.30)
    ap.add_argument("--connect_threshold", type=float, default=0.20)
    ap.add_argument("--tbrrt_block_k", type=int, default=32)
    ap.add_argument("--tbrrt_time_limit_sec", type=float, default=60.0)
    ap.add_argument("--tbrrt_seed", type=int, default=-1)
    ap.add_argument(
        "--validate_combined_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate the synchronized dual-arm path in collision after merging",
    )
    ap.add_argument(
        "--publish_world_collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish selected world collision yaml to MuJoCo simulation",
    )
    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC, help="MuJoCo world collision cuboid topic")
    ap.add_argument("--world_collision_wait_subscriber_s", type=float, default=1.0, help="wait for MuJoCo subscriber before publishing world collision")
    ap.add_argument("--world_collision_keep_alive_s", type=float, default=0.5, help="keep world collision publisher alive after publishing")
    return ap


def _project_full_path_to_active(
    path: Sequence[Sequence[float]],
    *,
    full_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> list[list[float]]:
    name_to_idx = {name: idx for idx, name in enumerate(full_joint_names)}
    active_idx = [name_to_idx[name] for name in active_joint_names]
    return [[float(q[idx]) for idx in active_idx] for q in path]


def _synchronize_single_arm_paths(
    q_start_cspace: Sequence[float],
    left_path_full: Sequence[Sequence[float]],
    right_path_full: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not left_path_full or not right_path_full:
        raise ValueError("left_path_full and right_path_full must be non-empty")

    name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}
    left_idx = [name_to_idx[name] for name in LEFT_JOINTS]
    right_idx = [name_to_idx[name] for name in RIGHT_JOINTS]
    q_start = [float(v) for v in q_start_cspace]

    out: list[list[float]] = []
    n_steps = max(len(left_path_full), len(right_path_full))
    for step_idx in range(n_steps):
        left_q = left_path_full[min(step_idx, len(left_path_full) - 1)]
        right_q = right_path_full[min(step_idx, len(right_path_full) - 1)]
        q = list(q_start)
        for idx in left_idx:
            q[idx] = float(left_q[idx])
        for idx in right_idx:
            q[idx] = float(right_q[idx])
        out.append(q)
    return out


def _publish_world_collision(args: argparse.Namespace, world_yml: str | None) -> None:
    if not bool(args.publish_world_collision):
        return
    if world_yml is None:
        print("[WORLD] no world_yml; skip MuJoCo world collision publish.")
        return
    try:
        count = publish_world_collision_yaml(
            world_yml,
            topic=str(args.world_collision_topic),
            wait_subscriber_s=float(args.world_collision_wait_subscriber_s),
            keep_alive_s=float(args.world_collision_keep_alive_s),
            node_name="arm_cart_world_collision_publisher",
        )
        print(
            f"[WORLD] published {count} collision cuboid(s) "
            f"to MuJoCo topic {args.world_collision_topic}"
        )
    except Exception as exc:
        print(f"[WORLD][WARN] failed to publish collision cuboids to MuJoCo: {exc}")


def _plan_single_arm_from_dual_goal(
    *,
    arm: str,
    robot_yml: str,
    joint_limit_yml: str,
    q_start_cspace: Sequence[float],
    q_goal_dual: Sequence[float],
    world_yml: str | None,
    cpu: bool,
    args: argparse.Namespace,
):
    out = plan_single_arm_tbrrt_batch_conext(
        robot_yml=robot_yml,
        arm=arm,
        q_start=q_start_cspace,
        q_goals=[[float(v) for v in q_goal_dual]],
        world_yml=world_yml,
        cpu=cpu,
        cfg=build_single_arm_tbrrt_config(args),
        joint_limit_yml=joint_limit_yml,
        block_k=int(args.tbrrt_block_k),
    )
    if not out.success or not out.path:
        raise RuntimeError(f"{arm} single-arm TBRRT failed: {out.stats.extra}")
    return out


def main_arm_cart(argv: Sequence[str] | None = None) -> int:
    args = build_arm_cart_parser().parse_args(list(argv) if argv is not None else None)
    world_yml = _normalized_world_yml(args.world_yml)

    _publish_world_collision(args, world_yml)

    q_start_cspace = [0.0 for _ in CSPACE_JOINT_NAMES_14]
    if args.use_current_joint_state_start:
        try:
            q_start_cspace = read_joint_positions_once(
                CSPACE_JOINT_NAMES_14,
                topic=str(args.joint_state_topic),
                wait_s=float(args.joint_state_wait_s),
            )
            print(f"[JOINTS] Using current start state from {args.joint_state_topic}")
        except RuntimeError as exc:
            print(f"[JOINTS] {exc}")
            if args.publish_mode == "real":
                print(
                    "[JOINTS] Real mode requires a valid current joint state. "
                    "Use --joint_state_topic to fix the topic or "
                    "--no-use_current_joint_state_start to override."
                )
                return 1
            print("[JOINTS] Falling back to zero start state.")

    left_xyz, left_quat_wxyz = _read_target_pose(
        args.left_xyz,
        args.left_quat_xyzw,
        label="Left",
        default_xyz="0.4 0.2 1.0",
        default_quat_xyzw="0.5 0.5 0.5 0.5",
    )
    right_xyz, right_quat_wxyz = _read_target_pose(
        args.right_xyz,
        args.right_quat_xyzw,
        label="Right",
        default_xyz="0.4 -0.2 1.0",
        default_quat_xyzw="0.5 -0.5 0.5 -0.5",
    )

    print("\n[1/4] Solving dual-arm IK with ForceCuroboIK...")
    solver = ForceCuroboIK(
        robot_yml=str(args.robot_yml),
        urdf_path=str(args.urdf_path),
        world_yml=world_yml,
        cpu=bool(args.cpu),
        num_seeds=int(args.force_ik_num_seeds),
    )
    ik_out = solver.solve_max_forward_force(
        left_xyz=left_xyz,
        left_quat_wxyz=left_quat_wxyz,
        right_xyz=right_xyz,
        right_quat_wxyz=right_quat_wxyz,
        q_start_cspace=q_start_cspace,
        forward_direction_base=[float(v) for v in args.forward_direction_base],
        num_trials=int(args.force_ik_num_trials),
        seed_noise_std=float(args.force_ik_seed_noise_std),
        random_seed=int(args.force_ik_random_seed),
    )
    if not ik_out.success or ik_out.q_cspace is None:
        print("[IK] Failed to find a collision-free dual-arm IK solution.")
        return 1

    q_goal_dual = [float(v) for v in ik_out.q_cspace]
    print(
        f"[IK] success. score={ik_out.score:.6f} "
        f"left_force={ik_out.left_force_capacity:.6f} "
        f"right_force={ik_out.right_force_capacity:.6f} "
        f"valid={ik_out.valid_candidates}/{ik_out.tried_candidates}"
    )

    print("\n[2/4] Planning each arm with single-arm TBRRT...")
    try:
        left_out = _plan_single_arm_from_dual_goal(
            arm="left",
            robot_yml=str(args.robot_yml),
            joint_limit_yml=str(args.joint_limit_yml),
            q_start_cspace=q_start_cspace,
            q_goal_dual=q_goal_dual,
            world_yml=world_yml,
            cpu=bool(args.cpu),
            args=args,
        )
        print(
            f"[LEFT] path_len={len(left_out.path)} "
            f"iters={left_out.stats.iters} time={left_out.stats.time_sec:.3f}s"
        )
        right_out = _plan_single_arm_from_dual_goal(
            arm="right",
            robot_yml=str(args.robot_yml),
            joint_limit_yml=str(args.joint_limit_yml),
            q_start_cspace=q_start_cspace,
            q_goal_dual=q_goal_dual,
            world_yml=world_yml,
            cpu=bool(args.cpu),
            args=args,
        )
        print(
            f"[RIGHT] path_len={len(right_out.path)} "
            f"iters={right_out.stats.iters} time={right_out.stats.time_sec:.3f}s"
        )
    except RuntimeError as exc:
        print(f"[TBRRT] {exc}")
        return 1

    left_path_full = [[float(v) for v in row] for row in left_out.path]
    right_path_full = [[float(v) for v in row] for row in right_out.path]
    combined_path = _synchronize_single_arm_paths(
        q_start_cspace=q_start_cspace,
        left_path_full=left_path_full,
        right_path_full=right_path_full,
    )
    left_path = _project_full_path_to_active(
        combined_path,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=LEFT_JOINTS,
    )
    right_path = _project_full_path_to_active(
        combined_path,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=RIGHT_JOINTS,
    )
    print(f"[SYNC] merged single-arm paths -> combined_len={len(combined_path)}")

    if args.validate_combined_path:
        print("[3/4] Validating merged dual-arm path for collision...")
        try:
            _validate_dual_path(
                combined_path,
                robot_yml=str(args.robot_yml),
                cpu=bool(args.cpu),
                world_yml=world_yml,
            )
        except RuntimeError as exc:
            print(f"[SYNC] {exc}")
            print(
                "[SYNC] The two single-arm plans are individually valid, "
                "but their synchronized merge is not collision-free."
            )
            return 1
        print("[SYNC] Collision check passed.")

    save_payload = {
        "mode": "arm_cart_single_tbrrt",
        "q_start_cspace": [float(v) for v in q_start_cspace],
        "q_goal_cspace": q_goal_dual,
        "cspace_joint_names": list(CSPACE_JOINT_NAMES_14),
        "left_joint_names": list(LEFT_JOINTS),
        "right_joint_names": list(RIGHT_JOINTS),
        "left_path_full": left_path_full,
        "right_path_full": right_path_full,
        "left_path": left_path,
        "right_path": right_path,
        "combined_path": combined_path,
        "publish_dt": float(args.publish_dt),
        "ik": {
            "score": float(ik_out.score),
            "left_force_capacity": float(ik_out.left_force_capacity),
            "right_force_capacity": float(ik_out.right_force_capacity),
            "tried_candidates": int(ik_out.tried_candidates),
            "valid_candidates": int(ik_out.valid_candidates),
        },
        "left_tbrrt": {
            "iters": int(left_out.stats.iters),
            "time_sec": float(left_out.stats.time_sec),
            "path_len": int(len(left_path_full)),
        },
        "right_tbrrt": {
            "iters": int(right_out.stats.iters),
            "time_sec": float(right_out.stats.time_sec),
            "path_len": int(len(right_path_full)),
        },
    }
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2)
        print(f"[SAVE] Saved: {args.save}")

    if args.publish_path:
        print("[4/4] Publishing synchronized dual-arm trajectory...")
        try:
            if args.publish_mode == "real":
                _publish_real_group(args, left_path, right_path)
            else:
                print(
                    f"[PUBLISH] Publishing JointState path -> {args.publish_topic} "
                    f"(dt={float(args.publish_dt):.3f}s)"
                )
                publish_joint_path(
                    combined_path,
                    list(CSPACE_JOINT_NAMES_14),
                    topic=str(args.publish_topic),
                    dt=float(args.publish_dt),
                    wait_subscriber_s=float(args.publish_wait_subscriber_s),
                )
        except RuntimeError as exc:
            print(f"[PUBLISH] {exc}")
            return 1
        print("[PUBLISH] done.")

    if args.plot_path:
        print(f"[PLOT] Publishing joint plot -> {args.plot_topic} (frame={args.plot_frame})")
        publish_joint_path_plot(
            combined_path,
            list(CSPACE_JOINT_NAMES_14),
            topic=str(args.plot_topic),
            frame_id=str(args.plot_frame),
            x_step=float(args.plot_x_step),
            y_scale=float(args.plot_y_scale),
            z_separation=float(args.plot_z_sep),
            marker_lifetime_s=float(args.plot_lifetime),
            keep_alive_s=float(args.plot_keep_alive),
        )
        print("[PLOT] done.")

    if args.plot:
        print("[PLOT] Showing matplotlib joint path window...")
        try:
            show_joint_path_plot_matplotlib(
                combined_path,
                list(CSPACE_JOINT_NAMES_14),
                x_step=float(args.plot_x_step),
                y_scale=float(args.plot_y_scale),
                z_separation=float(args.plot_z_sep),
                title="ARM_CART Single TBRRT Joint Path",
            )
        except Exception as exc:
            print(f"[PLOT] matplotlib plot failed: {exc}")
            return 1

    return 0


def main_arm_cart_picking(argv: Sequence[str] | None = None) -> int:
    return main_dual_arm(_build_fixed_arm_cart_picking_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_cart_picking(argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from typing import Sequence

from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.kinematics.force_curobo_ik import ForceCuroboIK
from capstone_pkg.planner.bidir_rrt.birrt import plan_birrt_jointspace
from capstone_pkg.planner.bidir_rrt.input_utils import read_vec, xyzw_to_wxyz
from capstone_pkg.planner.bidir_rrt.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_path,
    publish_joint_trajectory_group,
    read_joint_positions_once,
    send_joint_trajectory_action_group,
)
from capstone_pkg.planner.bidir_rrt.plot import publish_joint_path_plot
from capstone_pkg.planner.bidir_rrt.spline import spline_interpolate_path
from capstone_pkg.utils.config import (
    CSPACE_JOINT_NAMES_14,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    ROBOT_URDF,
    ROBOT_YAML,
    WORLD_YAML,
)


def build_dual_birrt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--urdf_path", default=ROBOT_URDF)
    ap.add_argument(
        "--world_yml",
        default=WORLD_YAML,
        help="world collision yaml; 'none' or '' disables world collision",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--max_iters", type=int, default=100000)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--goal_bias", type=float, default=0.30)
    ap.add_argument("--connect_threshold", type=float, default=0.20)
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
        default=2.0,
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
        default=2,
        help="number of times to publish the same JointTrajectory command",
    )
    ap.add_argument(
        "--publish_period_s",
        type=float,
        default=0.05,
        help="period [s] between repeated JointTrajectory publishes",
    )
    ap.add_argument(
        "--publish_wait_ack_s",
        type=float,
        default=1.0,
        help="wait time [s] for DDS acknowledgements after each JointTrajectory publish",
    )
    ap.add_argument(
        "--publish_keep_alive_s",
        type=float,
        default=0.5,
        help="keep publisher alive [s] after JointTrajectory publish",
    )
    ap.add_argument(
        "--publish_transient_local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use TRANSIENT_LOCAL durability for JointTrajectory publisher",
    )
    ap.add_argument(
        "--plot_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish planned path as RViz2 MarkerArray plot",
    )
    ap.add_argument(
        "--plot_topic",
        default="/bidir_rrt/joint_path_plot",
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
    ap.add_argument(
        "--validate_combined_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate the synchronized dual-arm path in collision after merging",
    )
    return ap


def _normalized_world_yml(raw_world_yml: str | None) -> str | None:
    if raw_world_yml in (None, "", "none", "None"):
        return None
    return str(raw_world_yml)


def _read_target_pose(
    xyz: Sequence[float] | None,
    quat_xyzw: Sequence[float] | None,
    *,
    label: str,
    default_xyz: str,
    default_quat_xyzw: str,
) -> tuple[list[float], list[float]]:
    if xyz is None:
        xyz_out = read_vec(f"{label} target xyz (m)", 3, default_xyz)
    else:
        xyz_out = [float(v) for v in xyz]

    if quat_xyzw is None:
        quat_xyzw_out = read_vec(f"{label} target quat (xyzw)", 4, default_quat_xyzw)
    else:
        quat_xyzw_out = [float(v) for v in quat_xyzw]

    return xyz_out, xyzw_to_wxyz(quat_xyzw_out)


def _project_full_path_to_active(
    path: Sequence[Sequence[float]],
    *,
    full_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> list[list[float]]:
    name_to_idx = {name: idx for idx, name in enumerate(full_joint_names)}
    active_idx = [name_to_idx[name] for name in active_joint_names]
    return [[float(q[idx]) for idx in active_idx] for q in path]


def _arm_goal_from_dual_goal(
    q_start_cspace: Sequence[float],
    q_goal_cspace: Sequence[float],
    *,
    active_joint_names: Sequence[str],
    cspace_joint_names: Sequence[str],
) -> list[float]:
    name_to_goal = {name: float(val) for name, val in zip(cspace_joint_names, q_goal_cspace)}
    q_goal_arm = [float(v) for v in q_start_cspace]
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    for joint_name in active_joint_names:
        q_goal_arm[name_to_idx[joint_name]] = name_to_goal[joint_name]
    return q_goal_arm


def _pad_path(path: Sequence[Sequence[float]], length: int) -> list[list[float]]:
    if not path:
        raise ValueError("path is empty")
    out = [[float(v) for v in q] for q in path]
    while len(out) < length:
        out.append(list(out[-1]))
    return out


def _merge_arm_paths(
    left_path: Sequence[Sequence[float]],
    right_path: Sequence[Sequence[float]],
    *,
    cspace_joint_names: Sequence[str],
) -> list[list[float]]:
    max_len = max(len(left_path), len(right_path))
    left_sync = _pad_path(left_path, max_len)
    right_sync = _pad_path(right_path, max_len)

    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    left_idx = [name_to_idx[name] for name in LEFT_JOINTS]
    right_idx = [name_to_idx[name] for name in RIGHT_JOINTS]
    q_path: list[list[float]] = []

    for left_q, right_q in zip(left_sync, right_sync):
        q = [0.0] * len(cspace_joint_names)
        for idx, value in zip(left_idx, left_q):
            q[idx] = float(value)
        for idx, value in zip(right_idx, right_q):
            q[idx] = float(value)
        q_path.append(q)

    return q_path


def _validate_dual_path(
    q_path: Sequence[Sequence[float]],
    *,
    robot_yml: str,
    cpu: bool,
    world_yml: str | None,
) -> None:
    sc = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
    ec = EdgeCollisionChecker(robot_yml, cpu=cpu, world_yml=world_yml)

    for idx, q in enumerate(q_path):
        in_collision, _, _ = sc.check_single([float(v) for v in q])
        if in_collision:
            raise RuntimeError(f"Synchronized dual-arm path is in collision at waypoint {idx}.")

    for idx in range(len(q_path) - 1):
        edge = ec.check_edge(
            [float(v) for v in q_path[idx]],
            [float(v) for v in q_path[idx + 1]],
            return_first_hit=False,
        )
        if edge.edge_in_collision:
            raise RuntimeError(
                f"Synchronized dual-arm path is in collision on edge {idx} -> {idx + 1}."
            )


def _publish_real_group(args: argparse.Namespace, left_path: Sequence[Sequence[float]], right_path: Sequence[Sequence[float]]) -> None:
    topic_commands = [
        JointTrajectoryCommand(
            endpoint=str(args.real_left_topic),
            joint_names=list(LEFT_JOINTS),
            path=left_path,
            label="left",
        ),
        JointTrajectoryCommand(
            endpoint=str(args.real_right_topic),
            joint_names=list(RIGHT_JOINTS),
            path=right_path,
            label="right",
        ),
    ]
    action_commands = [
        JointTrajectoryCommand(
            endpoint=str(args.real_left_action),
            joint_names=list(LEFT_JOINTS),
            path=left_path,
            label="left",
        ),
        JointTrajectoryCommand(
            endpoint=str(args.real_right_action),
            joint_names=list(RIGHT_JOINTS),
            path=right_path,
            label="right",
        ),
    ]

    if args.real_use_action:
        try:
            action_targets = ", ".join(cmd.endpoint for cmd in action_commands)
            print(
                f"[PUBLISH] Sending FollowJointTrajectory -> {action_targets} "
                f"(dt={float(args.publish_dt):.3f}s)"
            )
            send_joint_trajectory_action_group(
                action_commands,
                dt=float(args.publish_dt),
                wait_server_s=float(args.action_wait_server_s),
                wait_result_s=float(args.action_wait_result_s),
                start_time_delay_s=float(args.start_delay_s),
            )
            return
        except RuntimeError as exc:
            print(f"[ACTION] {exc}")
            if not args.real_action_fallback_to_topic:
                raise
            topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
            print(f"[ACTION] Falling back to JointTrajectory topics -> {topic_targets}")

    topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
    print(f"[PUBLISH] Publishing JointTrajectory -> {topic_targets} (dt={float(args.publish_dt):.3f}s)")
    publish_joint_trajectory_group(
        topic_commands,
        dt=float(args.publish_dt),
        wait_subscriber_s=float(args.publish_wait_subscriber_s),
        require_subscriber=bool(args.publish_require_subscriber),
        retry_until_subscriber=bool(args.publish_retry_until_subscriber),
        publish_repeat=int(args.publish_repeat),
        publish_period_s=float(args.publish_period_s),
        wait_ack_s=float(args.publish_wait_ack_s),
        keep_alive_s=float(args.publish_keep_alive_s),
        transient_local=bool(args.publish_transient_local),
        start_time_delay_s=float(args.start_delay_s),
    )


def main_dual_birrt(argv: Sequence[str] | None = None) -> int:
    args = build_dual_birrt_parser().parse_args(list(argv) if argv is not None else None)
    world_yml = _normalized_world_yml(args.world_yml)

    left_xyz, left_quat_wxyz = _read_target_pose(
        args.left_xyz,
        args.left_quat_xyzw,
        label="Left",
        default_xyz="0.4 0.2 1.2",
        default_quat_xyzw="0.5 0.5 0.5 0.5",
    )
    right_xyz, right_quat_wxyz = _read_target_pose(
        args.right_xyz,
        args.right_quat_xyzw,
        label="Right",
        default_xyz="0.4 -0.2 1.2",
        default_quat_xyzw="0.5 -0.5 0.5 -0.5",
    )

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

    q_goal_left = _arm_goal_from_dual_goal(
        q_start_cspace,
        q_goal_dual,
        active_joint_names=LEFT_JOINTS,
        cspace_joint_names=CSPACE_JOINT_NAMES_14,
    )
    q_goal_right = _arm_goal_from_dual_goal(
        q_start_cspace,
        q_goal_dual,
        active_joint_names=RIGHT_JOINTS,
        cspace_joint_names=CSPACE_JOINT_NAMES_14,
    )

    print("\n[2/4] Running left-arm BiRRT...")
    ok_left, left_raw_path_full = plan_birrt_jointspace(
        robot_yml=str(args.robot_yml),
        q_start=q_start_cspace,
        q_goal=q_goal_left,
        active_joint_names=list(LEFT_JOINTS),
        cspace_joint_names=list(CSPACE_JOINT_NAMES_14),
        cpu=bool(args.cpu),
        step=float(args.step),
        max_iters=int(args.max_iters),
        goal_bias=float(args.goal_bias),
        connect_threshold=float(args.connect_threshold),
        world_yml=world_yml,
    )
    if not ok_left:
        print("[BiRRT] Failed to find a left-arm path.")
        return 1

    print("[3/4] Running right-arm BiRRT...")
    ok_right, right_raw_path_full = plan_birrt_jointspace(
        robot_yml=str(args.robot_yml),
        q_start=q_start_cspace,
        q_goal=q_goal_right,
        active_joint_names=list(RIGHT_JOINTS),
        cspace_joint_names=list(CSPACE_JOINT_NAMES_14),
        cpu=bool(args.cpu),
        step=float(args.step),
        max_iters=int(args.max_iters),
        goal_bias=float(args.goal_bias),
        connect_threshold=float(args.connect_threshold),
        world_yml=world_yml,
    )
    if not ok_right:
        print("[BiRRT] Failed to find a right-arm path.")
        return 1

    left_raw_path = _project_full_path_to_active(
        left_raw_path_full,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=LEFT_JOINTS,
    )
    right_raw_path = _project_full_path_to_active(
        right_raw_path_full,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=RIGHT_JOINTS,
    )

    left_spline_path = spline_interpolate_path(left_raw_path, dt=float(args.publish_dt))
    right_spline_path = spline_interpolate_path(right_raw_path, dt=float(args.publish_dt))
    q_path = _merge_arm_paths(
        left_spline_path,
        right_spline_path,
        cspace_joint_names=CSPACE_JOINT_NAMES_14,
    )
    left_path = _project_full_path_to_active(
        q_path,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=LEFT_JOINTS,
    )
    right_path = _project_full_path_to_active(
        q_path,
        full_joint_names=CSPACE_JOINT_NAMES_14,
        active_joint_names=RIGHT_JOINTS,
    )

    print(
        "[SYNC] left_raw_len={} right_raw_len={} -> left_spline_len={} right_spline_len={} -> combined_len={}".format(
            len(left_raw_path),
            len(right_raw_path),
            len(left_spline_path),
            len(right_spline_path),
            len(q_path),
        )
    )

    if args.validate_combined_path:
        print("[SYNC] Validating synchronized dual-arm path for collision...")
        try:
            _validate_dual_path(
                q_path,
                robot_yml=str(args.robot_yml),
                cpu=bool(args.cpu),
                world_yml=world_yml,
            )
        except RuntimeError as exc:
            print(f"[SYNC] {exc}")
            return 1
        print("[SYNC] Collision check passed.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "dual_birrt",
                    "q_start_cspace": q_start_cspace,
                    "q_goal_cspace": q_goal_dual,
                    "cspace_joint_names": list(CSPACE_JOINT_NAMES_14),
                    "left_joint_names": list(LEFT_JOINTS),
                    "right_joint_names": list(RIGHT_JOINTS),
                    "left_path": left_path,
                    "right_path": right_path,
                    "combined_path": q_path,
                    "publish_dt": float(args.publish_dt),
                    "ik": {
                        "score": float(ik_out.score),
                        "left_force_capacity": float(ik_out.left_force_capacity),
                        "right_force_capacity": float(ik_out.right_force_capacity),
                        "tried_candidates": int(ik_out.tried_candidates),
                        "valid_candidates": int(ik_out.valid_candidates),
                    },
                },
                f,
                indent=2,
            )
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
                    q_path,
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
            q_path,
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

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_dual_birrt(argv)


if __name__ == "__main__":
    raise SystemExit(main())

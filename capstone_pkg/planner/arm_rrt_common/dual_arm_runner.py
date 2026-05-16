from __future__ import annotations

import argparse
import json
from typing import Sequence

from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.kinematics.force_curobo_ik import ForceCuroboIK
from capstone_pkg.planner.arm_rrt_common.input_utils import read_vec, xyzw_to_wxyz
from capstone_pkg.planner.arm_rrt_common.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_path,
    publish_joint_trajectory_group,
    read_joint_positions_once,
    send_joint_trajectory_action_group,
    wait_for_joint_positions,
)
from capstone_pkg.planner.arm_rrt_common.plot import (
    publish_joint_path_plot,
    show_joint_path_plot_matplotlib,
)
from capstone_pkg.planner.arm_rrt_common.spline import spline_interpolate_path
from capstone_pkg.utils.config import (
    CSPACE_JOINT_NAMES_14,
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


def build_dual_arm_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--urdf_path", default=ROBOT_URDF)
    ap.add_argument(
        "--world_yml",
        default=WORLD_YAML,
        help="world collision yaml; 'none' or '' disables world collision",
    )
    ap.add_argument(
        "--planner_mode",
        choices=("spline_only",),
        default="spline_only",
        help="move directly from q_start to the dual-arm IK goal with spline interpolation",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--save", default="", help="optional path to save result json")
    ap.add_argument(
        "--use_stored_trajectory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="load a previously saved dual-arm trajectory json and execute it directly",
    )
    ap.add_argument(
        "--stored_trajectory_json",
        default="",
        help="path to a saved dual-arm trajectory json used when --use_stored_trajectory is enabled",
    )
    ap.add_argument(
        "--stored_path_start_tol",
        type=float,
        default=0.25,
        help="max abs joint error tolerance between current state and stored trajectory start",
    )
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
    ap.add_argument(
        "--validate_combined_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate the synchronized dual-arm path in collision after merging",
    )
    ap.add_argument("--publish_world_collision", action=argparse.BooleanOptionalAction, default=True, help="publish selected world collision yaml to MuJoCo simulation")
    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC, help="MuJoCo world collision cuboid topic")
    ap.add_argument("--world_collision_wait_subscriber_s", type=float, default=1.0, help="wait for MuJoCo subscriber before publishing world collision")
    ap.add_argument("--world_collision_keep_alive_s", type=float, default=0.5, help="keep world collision publisher alive after publishing")
    return ap


def _normalized_world_yml(raw_world_yml: str | None) -> str | None:
    if raw_world_yml in (None, "", "none", "None"):
        return None
    return str(raw_world_yml)


def _compute_max_abs_joint_error(a: Sequence[float], b: Sequence[float]) -> float:
    dof = min(len(a), len(b))
    if dof <= 0:
        raise ValueError("joint vectors are empty")
    return max(abs(float(a[i]) - float(b[i])) for i in range(dof))


def _load_stored_dual_trajectory(
    path: str,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("stored trajectory json must be a dict")

    q_path_raw = raw.get("combined_path")
    if not isinstance(q_path_raw, list) or not q_path_raw:
        raise ValueError("stored trajectory json must contain non-empty combined_path")

    q_path = [[float(v) for v in row] for row in q_path_raw]
    if len(q_path[0]) != len(CSPACE_JOINT_NAMES_14):
        raise ValueError(
            "stored combined_path dimension mismatch: "
            f"{len(q_path[0])} != {len(CSPACE_JOINT_NAMES_14)}"
        )

    left_path_raw = raw.get("left_path")
    right_path_raw = raw.get("right_path")
    if isinstance(left_path_raw, list) and left_path_raw:
        left_path = [[float(v) for v in row] for row in left_path_raw]
    else:
        left_path = _project_full_path_to_active(
            q_path,
            full_joint_names=CSPACE_JOINT_NAMES_14,
            active_joint_names=LEFT_JOINTS,
        )

    if isinstance(right_path_raw, list) and right_path_raw:
        right_path = [[float(v) for v in row] for row in right_path_raw]
    else:
        right_path = _project_full_path_to_active(
            q_path,
            full_joint_names=CSPACE_JOINT_NAMES_14,
            active_joint_names=RIGHT_JOINTS,
        )

    return q_path, left_path, right_path, raw


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


def _wrapped_joint_delta(a: float, b: float) -> float:
    import math

    return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def _nearest_waypoint_index(
    current_positions: Sequence[float],
    path: Sequence[Sequence[float]],
) -> int:
    if not path:
        raise ValueError("path is empty")

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


def _retry_attempt_limit(max_retries: int) -> int | None:
    retries = int(max_retries)
    if retries <= 0:
        return None
    return max(1, retries)


def _format_attempt(attempt_idx: int, max_attempts: int | None) -> str:
    if max_attempts is None:
        return f"{attempt_idx + 1}/inf"
    return f"{attempt_idx + 1}/{max_attempts}"


def _publish_real_group(
    args: argparse.Namespace,
    left_path: Sequence[Sequence[float]],
    right_path: Sequence[Sequence[float]],
) -> None:
    left_goal = [float(v) for v in left_path[-1]]
    right_goal = [float(v) for v in right_path[-1]]
    max_attempts = _retry_attempt_limit(int(getattr(args, "arrival_max_retries", -1)))
    attempt_idx = 0
    last_failure = ""

    while max_attempts is None or attempt_idx < max_attempts:
        cmd_left_path = [[float(v) for v in q] for q in left_path]
        cmd_right_path = [[float(v) for v in q] for q in right_path]
        if attempt_idx > 0:
            try:
                left_current = read_joint_positions_once(
                    LEFT_JOINTS,
                    topic=str(args.joint_state_topic),
                    wait_s=float(args.joint_state_wait_s),
                )
                right_current = read_joint_positions_once(
                    RIGHT_JOINTS,
                    topic=str(args.joint_state_topic),
                    wait_s=float(args.joint_state_wait_s),
                )
                left_nearest_idx = _nearest_waypoint_index(left_current, left_path)
                right_nearest_idx = _nearest_waypoint_index(right_current, right_path)
                cmd_left_path = _build_retry_path(
                    current_positions=left_current,
                    original_path=left_path,
                )
                cmd_right_path = _build_retry_path(
                    current_positions=right_current,
                    original_path=right_path,
                )
                left_desc = (
                    "no progress"
                    if left_nearest_idx == 0
                    else f"waypoint {left_nearest_idx + 1}/{len(left_path)}"
                )
                right_desc = (
                    "no progress"
                    if right_nearest_idx == 0
                    else f"waypoint {right_nearest_idx + 1}/{len(right_path)}"
                )
                print(
                    f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for dual-arm real publish: "
                    f"left at {left_desc}, right at {right_desc}."
                )
            except RuntimeError as exc:
                print(
                    f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for dual-arm real publish: "
                    f"failed to read current joints ({exc}); re-publishing full path."
                )

        topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_topic),
                joint_names=list(LEFT_JOINTS),
                path=cmd_left_path,
                label="left",
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_topic),
                joint_names=list(RIGHT_JOINTS),
                path=cmd_right_path,
                label="right",
            ),
        ]
        action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_action),
                joint_names=list(LEFT_JOINTS),
                path=cmd_left_path,
                label="left",
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_action),
                joint_names=list(RIGHT_JOINTS),
                path=cmd_right_path,
                label="right",
            ),
        ]

        try:
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
                except RuntimeError as exc:
                    print(f"[ACTION] {exc}")
                    if not args.real_action_fallback_to_topic:
                        raise
                    topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
                    print(f"[ACTION] Falling back to JointTrajectory topics -> {topic_targets}")
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
                        reliability=str(getattr(args, "publish_reliability", "best_effort")),
                        durability=(
                            "transient_local"
                            if bool(getattr(args, "publish_transient_local", False))
                            else str(getattr(args, "publish_durability", "volatile"))
                        ),
                        qos_depth=int(getattr(args, "publish_qos_depth", 1)),
                        start_time_delay_s=float(args.start_delay_s),
                    )
            else:
                topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
                print(
                    f"[PUBLISH] Publishing JointTrajectory -> {topic_targets} "
                    f"(dt={float(args.publish_dt):.3f}s)"
                )
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
                    reliability=str(getattr(args, "publish_reliability", "best_effort")),
                    durability=(
                        "transient_local"
                        if bool(getattr(args, "publish_transient_local", False))
                        else str(getattr(args, "publish_durability", "volatile"))
                    ),
                    qos_depth=int(getattr(args, "publish_qos_depth", 1)),
                    start_time_delay_s=float(args.start_delay_s),
                )
        except RuntimeError as exc:
            attempt_idx += 1
            if max_attempts is not None and attempt_idx >= max_attempts:
                raise RuntimeError(
                    f"Failed to publish dual-arm real command after {attempt_idx} attempt(s): {exc}"
                ) from exc
            print(
                f"[ARRIVAL] publish attempt {_format_attempt(attempt_idx - 1, max_attempts)} "
                f"failed for dual-arm real publish: {exc}. Retrying."
            )
            continue

        wait_s = _resolve_arrival_wait_s(
            path_len=max(len(cmd_left_path), len(cmd_right_path)),
            dt=float(args.publish_dt),
            configured_wait_s=float(getattr(args, "arrival_wait_s", -1.0)),
        )
        failures: list[str] = []
        for arm_name, joint_names, goal in (
            ("left", LEFT_JOINTS, left_goal),
            ("right", RIGHT_JOINTS, right_goal),
        ):
            arrived, _current_positions, max_abs_err = wait_for_joint_positions(
                joint_names,
                goal,
                topic=str(args.joint_state_topic),
                wait_s=wait_s,
                tolerance=float(getattr(args, "arrival_joint_tolerance", 0.05)),
                poll_period_s=float(getattr(args, "arrival_poll_s", 0.05)),
            )
            if not arrived:
                failures.append(f"{arm_name}: max_abs_err={max_abs_err:.6f}")

        if not failures:
            if attempt_idx > 0:
                print("[ARRIVAL] dual-arm real publish confirmed after retry.")
            return

        last_failure = "; ".join(failures)
        attempt_idx += 1
        print(
            f"[ARRIVAL] dual-arm real publish not confirmed after attempt "
            f"{_format_attempt(attempt_idx - 1, max_attempts)}: {last_failure}. "
            "Re-publishing toward the remaining path."
        )

    raise RuntimeError(
        "Failed to confirm dual-arm real publish arrival after "
        f"{'infinite retry loop interruption' if max_attempts is None else f'{max_attempts} attempt(s)'}; "
        f"last failure={last_failure or 'unknown'}"
    )


def main_dual_arm(argv: Sequence[str] | None = None) -> int:
    args = build_dual_arm_parser().parse_args(list(argv) if argv is not None else None)
    world_yml = _normalized_world_yml(args.world_yml)

    if bool(args.publish_world_collision):
        if world_yml is None:
            print("[WORLD] no world_yml; skip MuJoCo world collision publish.")
        else:
            try:
                count = publish_world_collision_yaml(
                    world_yml,
                    topic=str(args.world_collision_topic),
                    wait_subscriber_s=float(args.world_collision_wait_subscriber_s),
                    keep_alive_s=float(args.world_collision_keep_alive_s),
                    node_name="dual_arm_world_collision_publisher",
                )
                print(
                    f"[WORLD] published {count} collision cuboid(s) "
                    f"to MuJoCo topic {args.world_collision_topic}"
                )
            except Exception as exc:
                print(f"[WORLD][WARN] failed to publish collision cuboids to MuJoCo: {exc}")

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

    save_payload: dict | None = None
    if bool(args.use_stored_trajectory):
        stored_json = str(args.stored_trajectory_json).strip()
        if not stored_json:
            raise ValueError("--stored_trajectory_json is required when --use_stored_trajectory is enabled")

        q_path, left_path, right_path, loaded_payload = _load_stored_dual_trajectory(stored_json)
        q_goal_dual = [float(v) for v in q_path[-1]]
        max_abs_start_err = _compute_max_abs_joint_error(q_path[0], q_start_cspace)
        loaded_publish_dt = float(loaded_payload.get("publish_dt", args.publish_dt))
        args.publish_dt = loaded_publish_dt
        print("\n[1/4] USING STORED TRAJECTORY")
        print(f"  file={stored_json}")
        print(f"  path_len={len(q_path)}")
        print(f"  publish_dt={loaded_publish_dt:.6f}s")
        print(f"  stored_path_start_max_abs_err={max_abs_start_err:.6f}")
        if max_abs_start_err > float(args.stored_path_start_tol):
            raise RuntimeError(
                "stored trajectory start is too far from current state: "
                f"{max_abs_start_err:.6f} > {float(args.stored_path_start_tol):.6f}"
            )
        save_payload = {
            "mode": "stored_trajectory",
            "planner_mode": str(loaded_payload.get("planner_mode", "stored_trajectory")),
            "source_json": stored_json,
            "q_start_cspace": [float(v) for v in q_start_cspace],
            "q_goal_cspace": q_goal_dual,
            "cspace_joint_names": list(CSPACE_JOINT_NAMES_14),
            "left_joint_names": list(LEFT_JOINTS),
            "right_joint_names": list(RIGHT_JOINTS),
            "left_path": left_path,
            "right_path": right_path,
            "combined_path": q_path,
            "publish_dt": float(loaded_publish_dt),
        }
    else:
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

        print("\n[2/4] planner_mode=spline_only: generating a direct dual-arm spline path...")
        q_path = spline_interpolate_path(
            [q_start_cspace, q_goal_dual],
            dt=float(args.publish_dt),
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
        print(f"[SYNC] direct_spline start_to_goal -> combined_len={len(q_path)}")
        save_payload = {
            "mode": "dual_arm",
            "planner_mode": str(args.planner_mode),
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
        }

    if args.validate_combined_path:
        print("[3/4] Validating synchronized dual-arm path for collision...")
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

    if args.plot:
        print("[PLOT] Showing matplotlib joint path window...")
        try:
            show_joint_path_plot_matplotlib(
                q_path,
                list(CSPACE_JOINT_NAMES_14),
                x_step=float(args.plot_x_step),
                y_scale=float(args.plot_y_scale),
                z_separation=float(args.plot_z_sep),
                title="Dual Arm Joint Path",
            )
        except Exception as exc:
            print(f"[PLOT] matplotlib plot failed: {exc}")
            return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_dual_arm(argv)


if __name__ == "__main__":
    raise SystemExit(main())

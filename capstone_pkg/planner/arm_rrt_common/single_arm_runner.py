from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _merge_world_with_user_object(
    base_world_yml: str | None,
    *,
    center_xyz: Sequence[float],
    dims_xyz: Sequence[float],
    quat_wxyz: Sequence[float] | None = None,
    object_name: str = "grasp_object",
) -> str:
    import yaml

    quat = list(quat_wxyz or [1.0, 0.0, 0.0, 0.0])
    raw: dict[str, Any] = {}
    if base_world_yml not in (None, "", "none", "None"):
        with open(str(base_world_yml), "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            raw = loaded

    if "cuboid" not in raw or not isinstance(raw["cuboid"], dict):
        raw["cuboid"] = {}

    raw["cuboid"][object_name] = {
        "dims": [float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])],
        "pose": [
            float(center_xyz[0]),
            float(center_xyz[1]),
            float(center_xyz[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ],
    }

    tmp = tempfile.NamedTemporaryFile(
        prefix="capstone_world_with_object_",
        suffix=".yaml",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    return str(tmp_path)


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} > ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _publish_world_collision_for_mujoco(args: argparse.Namespace, world_yml: str | None) -> None:
    if not getattr(args, "publish_world_collision", True):
        return
    if world_yml in (None, "", "none", "None"):
        print("[WORLD] no world_yml; skip MuJoCo world collision publish.")
        return

    from capstone_pkg.utils.world_collision_bridge import publish_world_collision_yaml

    try:
        count = publish_world_collision_yaml(
            str(world_yml),
            topic=str(args.world_collision_topic),
            wait_subscriber_s=float(args.world_collision_wait_subscriber_s),
            keep_alive_s=float(args.world_collision_keep_alive_s),
            node_name="single_arm_world_collision_publisher",
        )
        print(
            f"[WORLD] published {count} collision cuboid(s) "
            f"to MuJoCo topic {args.world_collision_topic}"
        )
    except Exception as exc:
        print(f"[WORLD][WARN] failed to publish collision cuboids to MuJoCo: {exc}")


def build_single_arm_parser(
    *,
    default_world_yml: str | None,
    collision_models: Mapping[str, str] | None = None,
    default_collision_model: str | None = None,
) -> argparse.ArgumentParser:
    from capstone_pkg.utils.config import JOINT_LIMIT, ROBOT_YAML
    from capstone_pkg.utils.world_collision_bridge import DEFAULT_WORLD_COLLISION_TOPIC

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--joint_limit_yml", default=JOINT_LIMIT)
    ap.add_argument(
        "--world_yml",
        default=default_world_yml,
        help="world collision yaml; 'none'이면 world collision 비활성화",
    )
    if collision_models:
        ap.add_argument(
            "--collision_model",
            choices=tuple(collision_models.keys()),
            default=default_collision_model,
            help="predefined world collision model selection",
        )
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument(
        "--planner_backend",
        choices=("tbrrt_batch_conext",),
        default="tbrrt_batch_conext",
        help="single-arm planner backend",
    )
    ap.add_argument("--max_iters", type=int, default=100000)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--goal_bias", type=float, default=0.30)
    ap.add_argument("--connect_threshold", type=float, default=0.20)
    ap.add_argument("--tbrrt_block_k", type=int, default=32)
    ap.add_argument("--tbrrt_time_limit_sec", type=float, default=60.0)
    ap.add_argument("--tbrrt_seed", type=int, default=-1)
    ap.add_argument(
        "--ik_batch",
        type=int,
        default=100,
        help="number of single-arm CuRobo seed trials to evaluate per target pose",
    )
    ap.add_argument(
        "--ik_seed_noise_std",
        type=float,
        default=0.25,
        help="Gaussian std [rad] for perturbing q_start into single-arm CuRobo IK seeds",
    )
    ap.add_argument(
        "--ik_seed",
        type=int,
        default=0,
        help="random seed used for single-arm CuRobo IK seed perturbations",
    )
    ap.add_argument(
        "--ik_goal_dedupe_tol",
        type=float,
        default=1.0e-3,
        help="merge single-arm IK goals whose joint-space distance is within this tolerance",
    )
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
        "--real_left_gripper_topic",
        default="/leader/joint_trajectory_command_broadcaster_gripper_left/joint_trajectory",
        help="target topic for real left gripper JointTrajectory",
    )
    ap.add_argument(
        "--real_right_gripper_topic",
        default="/leader/joint_trajectory_command_broadcaster_gripper_right/joint_trajectory",
        help="target topic for real right gripper JointTrajectory",
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
        "--real_left_gripper_action",
        default="/leader/joint_trajectory_command_broadcaster_gripper_left/follow_joint_trajectory",
        help="FollowJointTrajectory action name for the real left gripper",
    )
    ap.add_argument(
        "--real_right_gripper_action",
        default="/leader/joint_trajectory_command_broadcaster_gripper_right/follow_joint_trajectory",
        help="FollowJointTrajectory action name for the real right gripper",
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
        "--arrival_joint_tolerance",
        type=float,
        default=0.05,
        help="max abs joint error tolerance [rad] used to confirm stage arrival",
    )
    ap.add_argument(
        "--arrival_wait_s",
        type=float,
        default=-1.0,
        help="wait time [s] for stage arrival confirmation after command publish, -1 uses trajectory duration + margin",
    )
    ap.add_argument(
        "--arrival_poll_s",
        type=float,
        default=0.05,
        help="poll period [s] for arrival confirmation against /joint_states",
    )
    ap.add_argument(
        "--arrival_max_retries",
        type=int,
        default=3,
        help="max command publish attempts per stage when arrival confirmation fails",
    )
    ap.add_argument("--publish_dt", type=float, default=0.01, help="publish period [s] between waypoints")
    ap.add_argument(
        "--start_delay_s",
        type=float,
        default=0.2,
        help="delay [s] before JointTrajectory execution starts on the robot",
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
    ap.add_argument("--publish_repeat", type=int, default=8, help="number of times to publish the same JointTrajectory command")
    ap.add_argument("--publish_period_s", type=float, default=0.03, help="period [s] between repeated JointTrajectory publishes")
    ap.add_argument("--publish_wait_ack_s", type=float, default=0.0, help="wait time [s] for DDS acknowledgements after each JointTrajectory publish")
    ap.add_argument("--publish_keep_alive_s", type=float, default=1.0, help="keep publisher alive [s] after JointTrajectory publish")
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
    ap.add_argument("--publish_transient_local", action=argparse.BooleanOptionalAction, default=False, help="deprecated compatibility flag; use --publish_durability")
    ap.add_argument("--gripper_delay_s", type=float, default=3.0, help="delay [s] after trajectory execution before gripper command")
    ap.add_argument("--gripper_target", type=float, default=1.0, help="target command value for selected gripper joint")
    ap.add_argument("--plot_path", action=argparse.BooleanOptionalAction, default=True, help="publish planned path as RViz2 MarkerArray plot")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="show planned path in a matplotlib window")
    ap.add_argument("--plot_topic", default="/arm_rrt/joint_path_plot", help="topic for RViz2 MarkerArray plot")
    ap.add_argument("--plot_frame", default="map", help="frame_id for RViz2 plot markers")
    ap.add_argument("--plot_x_step", type=float, default=0.05, help="x-axis spacing between waypoints [m]")
    ap.add_argument("--plot_y_scale", type=float, default=1.0, help="scale for joint value axis")
    ap.add_argument("--plot_z_sep", type=float, default=0.25, help="z separation between joints [m]")
    ap.add_argument("--plot_lifetime", type=float, default=0.0, help="marker lifetime [s], 0 means keep forever")
    ap.add_argument("--plot_keep_alive", type=float, default=5.0, help="keep plot publisher alive after publish [s], -1 means forever")
    ap.add_argument("--publish_world_collision", action=argparse.BooleanOptionalAction, default=True, help="publish selected world collision yaml to MuJoCo simulation")
    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC, help="MuJoCo world collision cuboid topic")
    ap.add_argument("--world_collision_wait_subscriber_s", type=float, default=1.0, help="wait for MuJoCo subscriber before publishing world collision")
    ap.add_argument("--world_collision_keep_alive_s", type=float, default=0.5, help="keep world collision publisher alive after publishing")
    return ap


def build_single_arm_tbrrt_config(args: argparse.Namespace):
    from capstone_pkg.planner.tbrrt import TBRRTConfig

    raw_seed = int(getattr(args, "tbrrt_seed", -1))
    seed = None if raw_seed < 0 else raw_seed
    publish_dt = max(1.0e-3, float(getattr(args, "publish_dt", 0.01)))
    return TBRRTConfig(
        step_size=float(args.step),
        goal_threshold=float(args.connect_threshold),
        goal_bias=float(args.goal_bias),
        max_iters=int(args.max_iters),
        time_limit_sec=float(getattr(args, "tbrrt_time_limit_sec", 60.0)),
        topp_output_dt=publish_dt,
        seed=seed,
    )


def _resolve_world_yml(
    args: argparse.Namespace,
    *,
    collision_models: Mapping[str, str] | None,
    default_world_yml: str | None,
) -> str | None:
    raw_world_yml = getattr(args, "world_yml", None)
    if raw_world_yml is None:
        pass
    elif raw_world_yml in ("", "none", "None"):
        return None
    else:
        return str(raw_world_yml)

    collision_model = getattr(args, "collision_model", None)
    if collision_model and collision_models:
        return str(collision_models[str(collision_model)])

    if default_world_yml in (None, "", "none", "None"):
        return None
    return str(default_world_yml)


def main_single_arm(
    argv: Sequence[str] | None = None,
    *,
    planner_name: str,
    collision_models: Mapping[str, str] | None = None,
    default_collision_model: str | None = None,
    default_world_yml: str | None = None,
) -> int:
    from capstone_pkg.collision_check.collision import get_self_collision_checker
    from capstone_pkg.kinematics.curobo_ik import SingleArmIK
    from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
        normalize_single_arm_planner_backend,
    )
    from capstone_pkg.planner.arm_rrt_common.input_utils import read_vec, xyzw_to_wxyz
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        publish_joint_path,
        publish_joint_trajectory,
        read_joint_positions_once,
        send_joint_trajectory_action,
    )
    from capstone_pkg.planner.arm_rrt_common.plot import (
        publish_joint_path_plot,
        show_joint_path_plot_matplotlib,
    )
    from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
        plan_single_arm_tbrrt_batch_conext,
    )
    from capstone_pkg.utils.config import (
        LEFT_EE_FRAME,
        LEFT_JOINTS,
        RIGHT_EE_FRAME,
        RIGHT_JOINTS,
    )
    import torch

    args = build_single_arm_parser(
        default_world_yml=default_world_yml,
        collision_models=collision_models,
        default_collision_model=default_collision_model,
    ).parse_args(list(argv) if argv is not None else None)

    resolved_world_yml = _resolve_world_yml(
        args,
        collision_models=collision_models,
        default_world_yml=default_world_yml,
    )
    if collision_models and getattr(args, "collision_model", None):
        print(
            f"[{planner_name}] collision_model={args.collision_model} "
            f"-> {resolved_world_yml}"
        )
    elif resolved_world_yml is None:
        print(f"[{planner_name}] world collision disabled")

    arm = input("Plan which arm? (left/right): ").strip().lower()
    if arm not in ("left", "right"):
        print("[ERROR] arm must be 'left' or 'right'")
        return 2

    xyz = read_vec("Target xyz (m)", 3, "0.4 0.2 1.65")
    input_x = float(xyz[0])
    xyz[0] = input_x
    print(f"[TARGET] applying x offset: input x={input_x:.3f} m -> target x={xyz[0]:.3f} m")
    q_xyzw = read_vec("Target quat (xyzw)", 4, "0 0 0.7071 0.7071")
    quat_wxyz = xyzw_to_wxyz(q_xyzw)

    use_object_collision = _prompt_yes_no("Add grasp object as collision cuboid?", default=True)
    object_center_xyz = None
    object_dims_xyz = None
    object_world_yml = resolved_world_yml
    if use_object_collision:
        object_center_xyz = read_vec("Object center xyz (m)", 3, "0.45 0.20 1.55")
        input_obj_x = float(object_center_xyz[0])
        object_center_xyz[0] = input_obj_x + 0.05
        object_dims_xyz = read_vec("Object size xyz (m)", 3, "0.05 0.05 0.18")
        object_world_yml = _merge_world_with_user_object(
            object_world_yml,
            center_xyz=object_center_xyz,
            dims_xyz=object_dims_xyz,
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            object_name="grasp_object",
        )
        print(f"[WORLD] added user object collision model -> {object_world_yml}")

    _publish_world_collision_for_mujoco(args, object_world_yml)

    q_start_cspace = None

    print("\n[1/2] Solving IK with cuRobo...")
    ik = SingleArmIK(args.robot_yml, arm=arm, cpu=args.cpu)
    q_start_cspace = [0.0 for _ in ik.cspace_joint_names]
    if args.use_current_joint_state_start:
        try:
            q_start_cspace = read_joint_positions_once(
                ik.cspace_joint_names,
                topic=args.joint_state_topic,
                wait_s=args.joint_state_wait_s,
            )
            print(f"[JOINTS] Using current start state from {args.joint_state_topic}")
        except RuntimeError as exc:
            print(f"[JOINTS] {exc}")
            if args.publish_mode == "real":
                print("[JOINTS] Real mode requires a valid current joint state. Use --joint_state_topic to fix the topic or --no-use_current_joint_state_start to override.")
                return 1
            print("[JOINTS] Falling back to zero start state.")

    ik_out = ik.solve(xyz, quat_wxyz, q_start_cspace=q_start_cspace)
    if not ik_out.success or ik_out.q_cspace is None:
        print("[IK] Failed or in collision.")
        return 1
    q_goal = ik_out.q_cspace
    print("[IK] success.\n")

    active = LEFT_JOINTS if arm == "left" else RIGHT_JOINTS
    planner_backend = normalize_single_arm_planner_backend(args.planner_backend)

    print("[2/2] Running TB-RRT batch_conext (single-arm)...")
    out = plan_single_arm_tbrrt_batch_conext(
        robot_yml=args.robot_yml,
        arm=arm,
        q_start=q_start_cspace,
        q_goals=[q_goal],
        world_yml=object_world_yml,
        cpu=bool(args.cpu),
        cfg=build_single_arm_tbrrt_config(args),
        joint_limit_yml=str(args.joint_limit_yml),
        block_k=int(args.tbrrt_block_k),
    )
    if not out.success or not out.path:
        print(f"[TBRRT] Failed to find a path: {out.stats.extra}")
        return 1

    raw_path = [[float(v) for v in q] for q in out.path]
    spline_path = [[float(v) for v in q] for q in out.path]
    print(
        f"[TBRRT] Success! path_len={len(spline_path)} "
        f"iters={out.stats.iters} time={out.stats.time_sec:.3f}s"
    )
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "path": spline_path, "cspace_joint_names": ik.cspace_joint_names}, f, indent=2)
        print(f"[PLANNER] Saved: {args.save}")

    pub_path = None
    pub_names = None
    if args.publish_path or args.plot_path or args.plot:
        name_to_idx = {n: i for i, n in enumerate(ik.cspace_joint_names)}
        pub_names = [n for n in active if n in name_to_idx]
        if not pub_names:
            print("[PATH] no active joints found in cspace_joint_names; skip publish/plot.")
            return 1

        pub_path = [[float(q[name_to_idx[n]]) for n in pub_names] for q in spline_path]

    if args.publish_path:
        if args.publish_mode == "real":
            real_topic = args.real_left_topic if arm == "left" else args.real_right_topic
            real_action = args.real_left_action if arm == "left" else args.real_right_action

            def _send_real_command(
                cmd_path,
                cmd_names,
                *,
                topic: str,
                action_name: str,
                dt: float,
                label: str,
            ) -> str:
                if args.real_use_action:
                    try:
                        print(f"[{label}] Sending FollowJointTrajectory -> {action_name} (dt={dt:.3f}s)")
                        send_joint_trajectory_action(
                            cmd_path,
                            cmd_names,
                            action_name=action_name,
                            dt=dt,
                            wait_server_s=args.action_wait_server_s,
                            wait_result_s=args.action_wait_result_s,
                        )
                        return "action"
                    except RuntimeError as exc:
                        print(f"[ACTION] {exc}")
                        if not args.real_action_fallback_to_topic:
                            raise
                        print(f"[ACTION] Falling back to JointTrajectory topic -> {topic}")

                publish_joint_trajectory(
                    cmd_path,
                    cmd_names,
                    topic=topic,
                    dt=dt,
                    wait_subscriber_s=args.publish_wait_subscriber_s,
                    require_subscriber=args.publish_require_subscriber,
                    retry_until_subscriber=args.publish_retry_until_subscriber,
                    publish_repeat=args.publish_repeat,
                    publish_period_s=args.publish_period_s,
                    wait_ack_s=args.publish_wait_ack_s,
                    keep_alive_s=args.publish_keep_alive_s,
                    reliability=str(getattr(args, "publish_reliability", "best_effort")),
                    durability=(
                        "transient_local"
                        if bool(getattr(args, "publish_transient_local", False))
                        else str(getattr(args, "publish_durability", "volatile"))
                    ),
                    qos_depth=int(getattr(args, "publish_qos_depth", 1)),
                    start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
                )
                return "topic"

            try:
                used_transport = _send_real_command(
                    pub_path,
                    pub_names,
                    topic=real_topic,
                    action_name=real_action,
                    dt=args.publish_dt,
                    label="3/4",
                )
            except RuntimeError as exc:
                print(f"[PUBLISH] {exc}")
                return 1

            if used_transport == "topic":
                traj_duration_s = max(0.0, float(len(pub_path) - 1) * float(args.publish_dt))
                if traj_duration_s > 0.0:
                    print(f"[GRIPPER] Waiting for trajectory execution: {traj_duration_s:.2f}s")
                    time.sleep(traj_duration_s)

            if args.gripper_delay_s > 0.0:
                print(f"[GRIPPER] Waiting additional delay: {args.gripper_delay_s:.2f}s")
                time.sleep(float(args.gripper_delay_s))

            gripper_topic = args.real_left_gripper_topic if arm == "left" else args.real_right_gripper_topic
            gripper_action = args.real_left_gripper_action if arm == "left" else args.real_right_gripper_action
            gripper_joint = "gripper_l_joint1" if arm == "left" else "gripper_r_joint1"
            print(
                f"[GRIPPER] Command -> {gripper_topic} "
                f"({gripper_joint}={args.gripper_target:.3f})"
            )
            try:
                _send_real_command(
                    [[float(args.gripper_target)]],
                    [gripper_joint],
                    topic=gripper_topic,
                    action_name=gripper_action,
                    dt=0.1,
                    label="GRIPPER",
                )
            except RuntimeError as exc:
                print(f"[GRIPPER] {exc}")
                return 1
            print("[GRIPPER] done.")
            if use_object_collision and object_center_xyz is not None and object_dims_xyz is not None:
                try:
                    checker = get_self_collision_checker(args.robot_yml, cpu=args.cpu, world_yml=object_world_yml)
                    q_end_cspace = spline_path[-1]
                    q_model = checker._build_q_active_from_cspace(
                        torch.tensor(q_end_cspace, device=checker.tensor_args.device, dtype=torch.float32).view(1, -1)
                    )[0]
                    checker.attach_box_object_to_robot(
                        center_xyz=object_center_xyz,
                        dims_xyz=object_dims_xyz,
                        q_model_order=q_model,
                        link_name=(LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME),
                        object_name="grasp_object",
                        disable_in_world=True,
                    )
                    print("[ATTACH] grasp object attached to end-effector collision model.")
                except Exception as exc:
                    print(f"[ATTACH] failed: {exc}")
        else:
            print(f"[3/4] Publishing JointState path -> {args.publish_topic} (dt={args.publish_dt:.3f}s)")
            publish_joint_path(
                pub_path,
                pub_names,
                topic=args.publish_topic,
                dt=args.publish_dt,
            )
        if args.publish_mode != "real" and use_object_collision and object_center_xyz is not None and object_dims_xyz is not None:
            try:
                checker = get_self_collision_checker(args.robot_yml, cpu=args.cpu, world_yml=object_world_yml)
                q_end_cspace = spline_path[-1]
                q_model = checker._build_q_active_from_cspace(
                    torch.tensor(q_end_cspace, device=checker.tensor_args.device, dtype=torch.float32).view(1, -1)
                )[0]
                checker.attach_box_object_to_robot(
                    center_xyz=object_center_xyz,
                    dims_xyz=object_dims_xyz,
                    q_model_order=q_model,
                    link_name=(LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME),
                    object_name="grasp_object",
                    disable_in_world=True,
                )
                print("[ATTACH] grasp object attached to end-effector collision model.")
            except Exception as exc:
                print(f"[ATTACH] failed: {exc}")
        print("[PUBLISH] done.")

    if args.plot_path:
        print(f"[4/4] Publishing joint plot -> {args.plot_topic} (frame={args.plot_frame})")
        publish_joint_path_plot(
            pub_path,
            pub_names,
            topic=args.plot_topic,
            frame_id=args.plot_frame,
            x_step=args.plot_x_step,
            y_scale=args.plot_y_scale,
            z_separation=args.plot_z_sep,
            marker_lifetime_s=args.plot_lifetime,
            keep_alive_s=args.plot_keep_alive,
        )
        print("[PLOT] done.")

    if args.plot:
        print("[PLOT] Showing matplotlib joint path window...")
        try:
            show_joint_path_plot_matplotlib(
                pub_path,
                pub_names,
                x_step=float(args.plot_x_step),
                y_scale=float(args.plot_y_scale),
                z_separation=float(args.plot_z_sep),
                title=f"{planner_name} Joint Path",
            )
        except Exception as exc:
            print(f"[PLOT] matplotlib plot failed: {exc}")
            return 1

    return 0

from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, List, Sequence



def _merge_world_with_user_object(base_world_yml: str, *, center_xyz: List[float], dims_xyz: List[float], quat_wxyz: List[float] | None = None, object_name: str = "grasp_object") -> str:
    import yaml

    quat = quat_wxyz or [1.0, 0.0, 0.0, 0.0]
    with open(base_world_yml, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if raw is None:
        raw = {}

    if "cuboid" not in raw or not isinstance(raw["cuboid"], dict):
        raw["cuboid"] = {}

    raw["cuboid"][object_name] = {
        "dims": [float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])],
        "pose": [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
    }

    tmp = tempfile.NamedTemporaryFile(prefix="capstone_world_with_object_", suffix=".yaml", delete=False)
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

_PLANNER_ALIASES = {
    "birrt": "birrt",
    "bidir_rrt": "birrt",
    "bidir-rrt": "birrt",
    "dual_birrt": "dual_birrt",
    "dual_bidir_rrt": "dual_birrt",
    "dual-birrt": "dual_birrt",
    "dual-bidir-rrt": "dual_birrt",
    "tbrrt": "tbrrt",
    "basic_tbrrt": "tbrrt",
    "basic-tbrrt": "tbrrt",
}


def _planner_usage() -> str:
    return (
        "Usage:\n"
        "  ros2 run capstone_pkg main -- planner birrt [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner dual_birrt [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner birrt [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner dual_birrt [planner args...]\n"
        "\n"
        "Available planners:\n"
        "  birrt\n"
        "  dual_birrt\n"
        "  tbrrt\n"
    )


def _normalize_argv(argv: Sequence[str] | None) -> List[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _resolve_planner(argv: Sequence[str] | None) -> tuple[str | None, List[str]]:
    args = _normalize_argv(argv)
    if not args:
        return None, []

    if args[0] in ("-h", "--help"):
        return "help", []

    if args[0] == "planner":
        if len(args) < 2:
            raise ValueError("planner name is required after 'planner'")
        planner = _PLANNER_ALIASES.get(args[1].lower())
        return planner, args[2:]

    if args[0] in ("-p", "--planner"):
        if len(args) < 2:
            raise ValueError("planner name is required after '--planner'")
        planner = _PLANNER_ALIASES.get(args[1].lower())
        return planner, args[2:]

    planner = _PLANNER_ALIASES.get(args[0].lower())
    return planner, args[1:]


def _build_birrt_parser() -> argparse.ArgumentParser:
    from capstone_pkg.utils.config import ROBOT_YAML

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--world_yml", default=None, help="world collision yaml; 생략하면 tbrrt는 world collision 비활성화")
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--max_iters", type=int, default=100000)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--goal_bias", type=float, default=0.30)
    ap.add_argument("--connect_threshold", type=float, default=0.20)
    ap.add_argument("--save", default="", help="optional path to save result json")
    ap.add_argument("--publish_path", action=argparse.BooleanOptionalAction, default=True, help="publish planned path commands")
    ap.add_argument("--publish_mode", choices=("joint_state", "real"), default="joint_state", help="publish backend: joint_state or real(JointTrajectory)")
    ap.add_argument("--publish_topic", default="/joint_states_cmd", help="target topic for JointState command stream")
    ap.add_argument("--joint_state_topic", default="/joint_states", help="topic used to read the current robot joint state")
    ap.add_argument("--joint_state_wait_s", type=float, default=2.0, help="wait time [s] for the current robot joint state")
    ap.add_argument("--use_current_joint_state_start", action=argparse.BooleanOptionalAction, default=True, help="read current /joint_states and plan from that start state")
    ap.add_argument("--real_left_topic", default="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory", help="target topic for real left arm JointTrajectory")
    ap.add_argument("--real_right_topic", default="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory", help="target topic for real right arm JointTrajectory")
    ap.add_argument("--real_left_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_left/joint_trajectory", help="target topic for real left gripper JointTrajectory")
    ap.add_argument("--real_right_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_right/joint_trajectory", help="target topic for real right gripper JointTrajectory")
    ap.add_argument("--real_use_action", action=argparse.BooleanOptionalAction, default=False, help="prefer FollowJointTrajectory action over raw JointTrajectory topic in real mode")
    ap.add_argument("--real_action_fallback_to_topic", action=argparse.BooleanOptionalAction, default=True, help="fall back to raw JointTrajectory topic if FollowJointTrajectory is unavailable")
    ap.add_argument("--real_left_action", default="/leader/joint_trajectory_command_broadcaster_left/follow_joint_trajectory", help="FollowJointTrajectory action name for the real left arm")
    ap.add_argument("--real_right_action", default="/leader/joint_trajectory_command_broadcaster_right/follow_joint_trajectory", help="FollowJointTrajectory action name for the real right arm")
    ap.add_argument("--real_left_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_left/follow_joint_trajectory", help="FollowJointTrajectory action name for the real left gripper")
    ap.add_argument("--real_right_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_right/follow_joint_trajectory", help="FollowJointTrajectory action name for the real right gripper")
    ap.add_argument("--action_wait_server_s", type=float, default=2.0, help="wait time [s] for FollowJointTrajectory action server")
    ap.add_argument("--action_wait_result_s", type=float, default=-1.0, help="wait time [s] for FollowJointTrajectory result, -1 waits until execution completes")
    ap.add_argument("--publish_dt", type=float, default=0.01, help="publish period [s] between waypoints")
    ap.add_argument("--publish_wait_subscriber_s", type=float, default=5.0, help="wait time [s] for subscriber discovery before publishing, -1 waits forever")
    ap.add_argument("--publish_require_subscriber", action=argparse.BooleanOptionalAction, default=True, help="fail if no matching JointTrajectory subscriber is found before publishing")
    ap.add_argument("--publish_retry_until_subscriber", action=argparse.BooleanOptionalAction, default=True, help="keep re-publishing JointTrajectory until a subscriber match appears")
    ap.add_argument("--publish_repeat", type=int, default=2, help="number of times to publish the same JointTrajectory command")
    ap.add_argument("--publish_period_s", type=float, default=0.05, help="period [s] between repeated JointTrajectory publishes")
    ap.add_argument("--publish_wait_ack_s", type=float, default=1.0, help="wait time [s] for DDS acknowledgements after each JointTrajectory publish")
    ap.add_argument("--publish_keep_alive_s", type=float, default=0.5, help="keep publisher alive [s] after JointTrajectory publish")
    ap.add_argument("--publish_transient_local", action=argparse.BooleanOptionalAction, default=False, help="use TRANSIENT_LOCAL durability for JointTrajectory publisher")
    ap.add_argument("--gripper_delay_s", type=float, default=3.0, help="delay [s] after trajectory execution before gripper command")
    ap.add_argument("--gripper_target", type=float, default=0.8, help="target command value for selected gripper joint")
    ap.add_argument("--plot_path", action=argparse.BooleanOptionalAction, default=True, help="publish planned path as RViz2 MarkerArray plot")
    ap.add_argument("--plot_topic", default="/bidir_rrt/joint_path_plot", help="topic for RViz2 MarkerArray plot")
    ap.add_argument("--plot_frame", default="map", help="frame_id for RViz2 plot markers")
    ap.add_argument("--plot_x_step", type=float, default=0.05, help="x-axis spacing between waypoints [m]")
    ap.add_argument("--plot_y_scale", type=float, default=1.0, help="scale for joint value axis")
    ap.add_argument("--plot_z_sep", type=float, default=0.25, help="z separation between joints [m]")
    ap.add_argument("--plot_lifetime", type=float, default=0.0, help="marker lifetime [s], 0 means keep forever")
    ap.add_argument("--plot_keep_alive", type=float, default=5.0, help="keep plot publisher alive after publish [s], -1 means forever")
    return ap


def main_birrt(argv: Sequence[str] | None = None) -> int:
    ap = _build_birrt_parser()
    args = ap.parse_args(_normalize_argv(argv))

    from capstone_pkg.kinematics.curobo_ik import SingleArmIK
    from capstone_pkg.planner.bidir_rrt.birrt import plan_birrt_jointspace
    from capstone_pkg.planner.bidir_rrt.input_utils import read_vec, xyzw_to_wxyz
    from capstone_pkg.planner.bidir_rrt.path_publisher import (
        publish_joint_path,
        publish_joint_trajectory,
        read_joint_positions_once,
        send_joint_trajectory_action,
    )
    from capstone_pkg.planner.bidir_rrt.plot import publish_joint_path_plot
    from capstone_pkg.planner.bidir_rrt.spline import spline_interpolate_path
    from capstone_pkg.collision_check.collision import get_self_collision_checker
    import torch
    from capstone_pkg.utils.config import LEFT_JOINTS, RIGHT_JOINTS, LEFT_EE_FRAME, RIGHT_EE_FRAME, WORLD_YAML

    arm = input("Plan which arm? (left/right): ").strip().lower()
    if arm not in ("left", "right"):
        print("[ERROR] arm must be 'left' or 'right'")
        return 2

    xyz = read_vec("Target xyz (m)", 3, "0.4 0.2 1.65")
    input_x = float(xyz[0])
    xyz[0] = input_x - 0.1
    print(f"[TARGET] applying x offset: input x={input_x:.3f} m -> target x={xyz[0]:.3f} m")
    q_xyzw = read_vec("Target quat (xyzw)", 4, "0 0 0.7071 0.7071")
    quat_wxyz = xyzw_to_wxyz(q_xyzw)

    use_object_collision = _prompt_yes_no("Add grasp object as collision cuboid?", default=True)
    object_center_xyz: List[float] | None = None
    object_dims_xyz: List[float] | None = None
    object_world_yml = args.world_yml or WORLD_YAML
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

    q_start_cspace: List[float] | None = None

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

    print("[2/2] Running BiRRT (joint-space)...")
    ok, path = plan_birrt_jointspace(
        robot_yml=args.robot_yml,
        q_start=q_start_cspace,
        q_goal=q_goal,
        active_joint_names=active,
        cspace_joint_names=ik.cspace_joint_names,
        cpu=args.cpu,
        step=args.step,
        max_iters=args.max_iters,
        goal_bias=args.goal_bias,
        connect_threshold=args.connect_threshold,
        world_yml=object_world_yml,
    )

    if not ok:
        print("[BiRRT] Failed to find a path.")
        return 1

    raw_path = path
    spline_path = spline_interpolate_path(raw_path, dt=0.01)
    print(f"[BiRRT] Success! raw_path_len={len(raw_path)} -> spline_path_len={len(spline_path)} (dt=0.01)")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "path": spline_path, "cspace_joint_names": ik.cspace_joint_names}, f, indent=2)
        print(f"[BiRRT] Saved: {args.save}")

    pub_path = None
    pub_names = None
    if args.publish_path or args.plot_path:
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
                    transient_local=args.publish_transient_local,
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

    return 0



def main_tbrrt(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.tbrrt_runner import main_tbrrt as _main_tbrrt
    return _main_tbrrt(_normalize_argv(argv))


def main_dual_birrt(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.bidir_rrt.dual_birrt import main_dual_birrt as _main_dual_birrt
    return _main_dual_birrt(_normalize_argv(argv))


def _build_tbrrt_parser() -> argparse.ArgumentParser:
    from capstone_pkg.utils.config import ROBOT_YAML

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--world_yml", default=None, help="world collision yaml; 생략하면 tbrrt는 world collision 비활성화")
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--target_left_xyz", nargs=3, type=float, required=True)
    ap.add_argument("--target_left_rpy_deg", nargs=3, type=float, required=True)
    ap.add_argument("--save", default="", help="optional path to save result json")
    ap.add_argument("--publish_path", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--publish_mode", choices=("joint_state", "real"), default="joint_state")
    ap.add_argument("--publish_topic", default="/joint_states_cmd")
    ap.add_argument("--joint_state_topic", default="/joint_states")
    return ap


def main_tbrrt(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.tbrrt_runner import main_tbrrt as _main_tbrrt
    return _main_tbrrt(_normalize_argv(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        planner, planner_args = _resolve_planner(argv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(_planner_usage(), file=sys.stderr)
        return 2

    if planner == "help":
        print(_planner_usage())
        return 0

    if planner == "birrt":
        return main_birrt(planner_args)
    if planner == "dual_birrt":
        return main_dual_birrt(planner_args)
    if planner == "tbrrt":
        return main_tbrrt(planner_args)

    print("[ERROR] planner must be one of: birrt, dual_birrt, tbrrt", file=sys.stderr)
    print(_planner_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

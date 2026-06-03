from __future__ import annotations

import argparse
import json
import math
import threading
import time
from typing import Sequence

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from capstone_pkg.kinematics.curobo_ik import (
    get_single_arm_ik,
    warmup_single_arm_ik_reachable,
)
from capstone_pkg.kinematics.force_curobo_ik import ForceCuroboIK
from capstone_pkg.planner.arm_rrt_common.dual_arm_runner import (
    _validate_dual_path,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
    SingleArmMotionPlan,
    build_active_joint_path,
    normalize_arm_name,
    xyzw_to_wxyz,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
    _publish_world_collision_for_mujoco,
    _resolve_world_yml,
    build_single_arm_parser,
    build_single_arm_tbrrt_config,
)
from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
    plan_single_arm_tbrrt_batch_conext,
)
from capstone_pkg.utils.config import (
    CSPACE_JOINT_NAMES_14,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    ROBOT_URDF,
)

_DEFAULT_LEFT_XYZ = (0.50, 0.20, 1.00)
_DEFAULT_LEFT_QUAT_XYZW = (0.5, 0.5, 0.5, 0.5)
_DEFAULT_RIGHT_XYZ = (0.50, -0.20, 1.00)
_DEFAULT_RIGHT_QUAT_XYZW = (0.5, -0.5, 0.5, -0.5)
_ZERO_GOAL_TOL = 1.0e-4


def _command_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _joint_state_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _duration_from_seconds(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((float(seconds) - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _real_traj_qos(args: argparse.Namespace) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.RELIABLE
        if str(getattr(args, "publish_reliability", "best_effort")).strip().lower()
        == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    durability = (
        DurabilityPolicy.TRANSIENT_LOCAL
        if str(getattr(args, "publish_durability", "volatile")).strip().lower()
        == "transient_local"
        or bool(getattr(args, "publish_transient_local", False))
        else DurabilityPolicy.VOLATILE
    )
    return QoSProfile(
        reliability=reliability,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(getattr(args, "publish_qos_depth", 1))),
    )


def _build_joint_trajectory(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    dt: float,
) -> JointTrajectory:
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    msg = JointTrajectory()
    msg.joint_names = [str(name) for name in joint_names]
    first_point_offset_s = float(dt) if len(path) == 1 else 0.0
    for idx, q in enumerate(path):
        if len(q) != len(joint_names):
            raise ValueError(
                f"path[{idx}] length {len(q)} != len(joint_names) {len(joint_names)}"
            )
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.time_from_start = _duration_from_seconds(
            first_point_offset_s + float(idx) * float(dt)
        )
        msg.points.append(point)
    return msg


def _arm_joint_names(arm: str) -> list[str]:
    return list(LEFT_JOINTS if arm == "left" else RIGHT_JOINTS)


def _all_joints_zero(
    q_cspace: Sequence[float],
    *,
    tol: float = _ZERO_GOAL_TOL,
) -> bool:
    if not q_cspace:
        return False
    return max(abs(float(v)) for v in q_cspace) <= float(tol)


def _project_full_path_to_active(
    path: Sequence[Sequence[float]],
    *,
    active_joint_names: Sequence[str],
) -> list[list[float]]:
    name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}
    active_idx = [name_to_idx[name] for name in active_joint_names]
    return [[float(q[idx]) for idx in active_idx] for q in path]


def _wrapped_joint_delta(a: float, b: float) -> float:
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


def _pad_path(path: Sequence[Sequence[float]], total: int) -> list[list[float]]:
    if not path:
        raise ValueError("path is empty")
    out = [[float(v) for v in row] for row in path]
    while len(out) < int(total):
        out.append(list(out[-1]))
    return out


def _combine_active_joint_paths(
    *,
    left_path: Sequence[Sequence[float]],
    right_path: Sequence[Sequence[float]],
) -> list[list[float]]:
    total = max(len(left_path), len(right_path))
    left_sync = _pad_path(left_path, total)
    right_sync = _pad_path(right_path, total)
    name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}

    full_path: list[list[float]] = []
    for left_q, right_q in zip(left_sync, right_sync):
        waypoint = [0.0 for _ in CSPACE_JOINT_NAMES_14]
        for joint_name, joint_value in zip(LEFT_JOINTS, left_q):
            waypoint[name_to_idx[joint_name]] = float(joint_value)
        for joint_name, joint_value in zip(RIGHT_JOINTS, right_q):
            waypoint[name_to_idx[joint_name]] = float(joint_value)
        full_path.append(waypoint)
    return full_path


def _retry_attempt_limit(max_retries: int) -> int | None:
    retries = int(max_retries)
    if retries <= 0:
        return None
    return max(1, retries)


def _format_attempt(attempt_idx: int, max_attempts: int | None) -> str:
    if max_attempts is None:
        return f"{attempt_idx + 1}/inf"
    return f"{attempt_idx + 1}/{max_attempts}"


def _synchronize_single_arm_paths(
    *,
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


def _format_pose_arg(xyz: Sequence[float], quat_xyzw: Sequence[float]) -> str:
    xyz_s = " ".join(f"{float(v):.3f}" for v in xyz)
    quat_s = " ".join(f"{float(v):.3f}" for v in quat_xyzw)
    return f"xyz=[{xyz_s}], quat_xyzw=[{quat_s}]"


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


def build_parser() -> argparse.ArgumentParser:
    ap = build_single_arm_parser(
        default_world_yml=None,
        collision_models=None,
        default_collision_model=None,
    )
    ap.description = "ARM_INIT sequence worker"
    ap.set_defaults(arrival_max_retries=-1)
    ap.add_argument("--urdf_path", default=ROBOT_URDF)
    ap.add_argument("--arm_init_start_topic", default="/arm_init_start")
    ap.add_argument("--arm_init_finish_topic", default="/arm_init_finish")
    ap.add_argument("--finish_publish_repeat", type=int, default=1)
    ap.add_argument("--finish_publish_period_s", type=float, default=0.05)
    ap.add_argument("--left_xyz", nargs=3, type=float, default=_DEFAULT_LEFT_XYZ)
    ap.add_argument(
        "--left_quat_xyzw",
        nargs=4,
        type=float,
        default=_DEFAULT_LEFT_QUAT_XYZW,
    )
    ap.add_argument("--right_xyz", nargs=3, type=float, default=_DEFAULT_RIGHT_XYZ)
    ap.add_argument(
        "--right_quat_xyzw",
        nargs=4,
        type=float,
        default=_DEFAULT_RIGHT_QUAT_XYZW,
    )
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
        help="stddev for dual-arm IK seed perturbation",
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
    ap.add_argument(
        "--startup_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pre-initialize and warm up IK/collision objects before processing start topics",
    )
    ap.add_argument(
        "--startup_warmup_iters",
        type=int,
        default=1,
        help="number of reachable IK warmup iterations to run per arm/world at startup",
    )
    ap.add_argument(
        "--startup_warmup_batch_size",
        type=int,
        default=None,
        help="batch size used for startup IK warmup solves; defaults to --ik_batch",
    )
    ap.add_argument(
        "--startup_force_ik_trials",
        type=int,
        default=1,
        help="number of fixed-target ForceCuroboIK trials to run at startup warmup",
    )
    return ap


class ArmInitNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_init")
        self._args = args
        self._request_cv = threading.Condition()
        self._pending_request_arm: str | None = None
        self._request_active = False
        self._joint_state_cv = threading.Condition()
        self._joint_state_by_name: dict[str, float] = {}
        self._force_ik_solver: ForceCuroboIK | None = None
        self._resolved_world_yml = _resolve_world_yml(
            args,
            collision_models=None,
            default_world_yml=None,
        )
        self._run_startup_warmup()

        self._finish_pub = self.create_publisher(
            Bool,
            str(args.arm_init_finish_topic),
            _command_qos(),
        )
        self._start_sub = self.create_subscription(
            String,
            str(args.arm_init_start_topic),
            self._start_callback,
            _command_qos(),
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            str(args.joint_state_topic),
            self._joint_state_callback,
            _joint_state_qos(),
        )
        self._joint_state_cmd_pub = self.create_publisher(
            JointState,
            str(args.publish_topic),
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )
        self._traj_publishers: dict[str, object] = {}
        self._arm_action_clients: dict[str, ActionClient] = {}
        if str(args.publish_mode) == "real":
            traj_qos = _real_traj_qos(args)
            self._traj_publishers = {
                str(args.real_left_topic): self.create_publisher(
                    JointTrajectory,
                    str(args.real_left_topic),
                    traj_qos,
                ),
                str(args.real_right_topic): self.create_publisher(
                    JointTrajectory,
                    str(args.real_right_topic),
                    traj_qos,
                ),
            }
            self._arm_action_clients = {
                str(args.real_left_action): ActionClient(
                    self,
                    FollowJointTrajectory,
                    str(args.real_left_action),
                ),
                str(args.real_right_action): ActionClient(
                    self,
                    FollowJointTrajectory,
                    str(args.real_right_action),
                ),
            }

        world_label = self._resolved_world_yml if self._resolved_world_yml is not None else "none"
        self.get_logger().info(
            "Listening on "
            f"{args.arm_init_start_topic}, publishing finish on {args.arm_init_finish_topic}, "
            f"world_yml={world_label}, "
            f"left_target={_format_pose_arg(args.left_xyz, args.left_quat_xyzw)}, "
            f"right_target={_format_pose_arg(args.right_xyz, args.right_quat_xyzw)}"
        )
        self._request_worker = threading.Thread(
            target=self._request_worker_loop,
            daemon=True,
        )
        self._request_worker.start()

    def _start_callback(self, msg: String) -> None:
        try:
            arm = normalize_arm_name(msg.data)
        except ValueError:
            self.get_logger().warn(
                f"Ignoring arm_init_start with invalid arm={msg.data!r}"
            )
            return

        with self._request_cv:
            if self._request_active or self._pending_request_arm is not None:
                state = "active" if self._request_active else "pending"
                self.get_logger().warning(
                    f"Ignoring arm_init_start arm={arm}: request already {state}"
                )
                return
            self._pending_request_arm = arm
            self._request_cv.notify()

        self.get_logger().info(
            f"Accepted arm_init_start arm={arm}"
        )

    def _request_worker_loop(self) -> None:
        while rclpy.ok():
            with self._request_cv:
                while self._pending_request_arm is None and rclpy.ok():
                    self._request_cv.wait(timeout=0.2)
                if not rclpy.ok():
                    return
                arm = self._pending_request_arm
                self._pending_request_arm = None
                self._request_active = True

            self.get_logger().info(
                f"Processing arm_init_start arm={arm}"
            )
            try:
                self._process_request(arm)
            finally:
                with self._request_cv:
                    self._request_active = False
                    self._request_cv.notify_all()

    def _joint_state_callback(self, msg: JointState) -> None:
        updates = {}
        for name, position in zip(list(msg.name), list(msg.position)):
            if isinstance(name, str):
                updates[str(name)] = float(position)
        if not updates:
            return
        with self._joint_state_cv:
            self._joint_state_by_name.update(updates)
            self._joint_state_cv.notify_all()

    def _target_pose_args(self) -> tuple[list[float], list[float], list[float], list[float]]:
        left_xyz = [float(v) for v in self._args.left_xyz]
        right_xyz = [float(v) for v in self._args.right_xyz]
        left_quat_wxyz = xyzw_to_wxyz([float(v) for v in self._args.left_quat_xyzw])
        right_quat_wxyz = xyzw_to_wxyz([float(v) for v in self._args.right_quat_xyzw])
        return left_xyz, left_quat_wxyz, right_xyz, right_quat_wxyz

    def _get_force_ik_solver(self) -> ForceCuroboIK:
        if self._force_ik_solver is None:
            self._force_ik_solver = ForceCuroboIK(
                robot_yml=str(self._args.robot_yml),
                urdf_path=str(self._args.urdf_path),
                world_yml=self._resolved_world_yml,
                cpu=bool(self._args.cpu),
                num_seeds=int(self._args.force_ik_num_seeds),
            )
        return self._force_ik_solver

    def _run_startup_warmup(self) -> None:
        if not bool(getattr(self._args, "startup_warmup", True)):
            self.get_logger().info("Startup warmup disabled")
            return

        warmup_iters = max(0, int(getattr(self._args, "startup_warmup_iters", 1)))
        configured_batch_size = getattr(self._args, "startup_warmup_batch_size", None)
        if configured_batch_size is None:
            configured_batch_size = getattr(self._args, "ik_batch", 100)
        warmup_batch_size = max(1, int(configured_batch_size))
        force_ik_trials = max(0, int(getattr(self._args, "startup_force_ik_trials", 1)))
        if warmup_iters <= 0 and force_ik_trials <= 0:
            self.get_logger().info(
                "Startup warmup skipped because startup_warmup_iters and startup_force_ik_trials are <= 0"
            )
            return

        from capstone_pkg.collision_check.collision import get_self_collision_checker

        world_label = self._resolved_world_yml if self._resolved_world_yml is not None else "none"
        t0 = time.monotonic()
        self.get_logger().info(
            "Starting ARM_INIT warmup "
            f"(iters={warmup_iters}, batch_size={warmup_batch_size}, "
            f"force_ik_trials={force_ik_trials}, world={world_label})"
        )

        checker = get_self_collision_checker(
            str(self._args.robot_yml),
            cpu=bool(self._args.cpu),
            world_yml=self._resolved_world_yml,
        )
        try:
            _ = checker.check_single([0.0 for _ in CSPACE_JOINT_NAMES_14])
        except Exception:
            pass

        if warmup_iters > 0:
            for arm_index, arm in enumerate(("left", "right")):
                ik = get_single_arm_ik(
                    str(self._args.robot_yml),
                    arm=arm,
                    cpu=bool(self._args.cpu),
                    world_yml=self._resolved_world_yml,
                )
                warmup_single_arm_ik_reachable(
                    ik,
                    iters=warmup_iters,
                    batch_size=warmup_batch_size,
                    noise_std=float(getattr(self._args, "ik_seed_noise_std", 0.25)),
                    random_seed=int(getattr(self._args, "ik_seed", 0)) + arm_index,
                )
                self.get_logger().info(f"[warmup] ready arm={arm} world={world_label}")

        solver = self._get_force_ik_solver()
        if force_ik_trials > 0:
            left_xyz, left_quat_wxyz, right_xyz, right_quat_wxyz = self._target_pose_args()
            try:
                ik_out = solver.solve_max_forward_force(
                    left_xyz=left_xyz,
                    left_quat_wxyz=left_quat_wxyz,
                    right_xyz=right_xyz,
                    right_quat_wxyz=right_quat_wxyz,
                    q_start_cspace=[0.0 for _ in CSPACE_JOINT_NAMES_14],
                    forward_direction_base=[
                        float(v) for v in self._args.forward_direction_base
                    ],
                    num_trials=force_ik_trials,
                    seed_noise_std=float(self._args.force_ik_seed_noise_std),
                    random_seed=int(self._args.force_ik_random_seed),
                )
                status = "success" if ik_out.success else "no valid target IK"
                self.get_logger().info(f"[warmup] ForceCuroboIK target solve: {status}")
            except Exception as exc:
                self.get_logger().warn(
                    f"[warmup] ForceCuroboIK target solve failed during warmup: {exc}"
                )

        self.get_logger().info(
            f"ARM_INIT warmup completed in {time.monotonic() - t0:.2f}s"
        )

    def _wait_for_joint_sample(
        self,
        joint_names: Sequence[str],
        *,
        wait_s: float,
    ) -> list[float]:
        deadline = time.monotonic() + max(0.0, float(wait_s))
        with self._joint_state_cv:
            while rclpy.ok():
                missing = [name for name in joint_names if name not in self._joint_state_by_name]
                if not missing:
                    return [float(self._joint_state_by_name[name]) for name in joint_names]

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    known = list(self._joint_state_by_name.keys())
                    if not known:
                        raise RuntimeError(
                            f"No JointState received on {self._args.joint_state_topic} "
                            f"within {float(wait_s):.2f}s"
                        )
                    raise RuntimeError(
                        f"Timed out waiting for joints on {self._args.joint_state_topic}; "
                        f"missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}"
                    )
                self._joint_state_cv.wait(timeout=min(0.1, remaining))

        raise RuntimeError("rclpy shutdown while waiting for JointState")

    def _publish_finish(self) -> None:
        repeat = max(1, int(self._args.finish_publish_repeat))
        period_s = max(0.0, float(self._args.finish_publish_period_s))
        msg = Bool()
        msg.data = True
        for _ in range(repeat):
            self._finish_pub.publish(msg)
            if period_s > 0.0:
                time.sleep(period_s)

    def _trajectory_stamp(self, delay_s: float):
        if float(delay_s) > 0.0:
            return (
                self.get_clock().now() + RclpyDuration(seconds=float(delay_s))
            ).to_msg()
        return self.get_clock().now().to_msg()

    def _get_traj_publisher(self, topic: str):
        pub = self._traj_publishers.get(str(topic))
        if pub is None:
            raise RuntimeError(
                f"No JointTrajectory publisher configured for topic: {topic}"
            )
        return pub

    def _get_arm_action_client(self, action_name: str) -> ActionClient:
        client = self._arm_action_clients.get(str(action_name))
        if client is None:
            raise RuntimeError(
                f"No FollowJointTrajectory action client configured for: {action_name}"
            )
        return client

    def _wait_for_action_server_with_timeout(
        self,
        client: ActionClient,
        action_name: str,
        *,
        wait_server_s: float,
    ) -> None:
        deadline = time.monotonic() + max(0.1, float(wait_server_s))
        while rclpy.ok():
            if client.wait_for_server(timeout_sec=0.2):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"FollowJointTrajectory action server not available: {action_name}"
                )
        raise RuntimeError(
            f"rclpy shutdown while waiting for FollowJointTrajectory action server: {action_name}"
        )

    def _send_follow_joint_trajectory_group(
        self,
        commands: Sequence[
            tuple[ActionClient, str, Sequence[str], Sequence[Sequence[float]]]
        ],
        *,
        dt: float,
        wait_server_s: float,
        wait_result_s: float,
        start_time_delay_s: float,
    ) -> None:
        if not commands:
            raise ValueError("commands is empty")

        normalized = [
            (client, str(action_name), list(joint_names), path)
            for client, action_name, joint_names, path in commands
        ]
        for client, action_name, _joint_names, _path in normalized:
            self._wait_for_action_server_with_timeout(
                client,
                action_name,
                wait_server_s=float(wait_server_s),
            )

        common_stamp = self._trajectory_stamp(float(start_time_delay_s))
        send_events: list[threading.Event] = []
        send_holders: list[dict[str, object]] = []
        send_futures = []

        for client, _action_name, joint_names, path in normalized:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = _build_joint_trajectory(path, joint_names, dt=float(dt))
            goal.trajectory.header.stamp = common_stamp

            done = threading.Event()
            holder: dict[str, object] = {}
            future = client.send_goal_async(goal)
            future.add_done_callback(
                lambda fut, holder=holder, done=done: (
                    holder.setdefault("future", fut),
                    done.set(),
                )
            )
            send_events.append(done)
            send_holders.append(holder)
            send_futures.append(future)

        send_deadline = time.monotonic() + max(1.0, float(wait_server_s) + 1.0)
        while rclpy.ok():
            if all(done.is_set() for done in send_events):
                break
            if time.monotonic() >= send_deadline:
                pending = [
                    normalized[idx][1]
                    for idx, done in enumerate(send_events)
                    if not done.is_set()
                ]
                raise RuntimeError(
                    "Timed out sending FollowJointTrajectory goal: "
                    + ", ".join(pending)
                )
            time.sleep(0.05)

        if not all(done.is_set() for done in send_events):
            pending = [
                normalized[idx][1]
                for idx, done in enumerate(send_events)
                if not done.is_set()
            ]
            raise RuntimeError(
                "rclpy shutdown while sending FollowJointTrajectory goal: "
                + ", ".join(pending)
            )

        goal_handles = []
        rejected = []
        failures = []
        for idx, future in enumerate(send_futures):
            done_future = send_holders[idx].get("future", future)
            exc = done_future.exception()
            if exc is not None:
                failures.append(f"{normalized[idx][1]}: {exc}")
                goal_handles.append(None)
                continue
            goal_handle = done_future.result()
            if goal_handle is None or not goal_handle.accepted:
                rejected.append(normalized[idx][1])
            goal_handles.append(goal_handle)

        if failures:
            raise RuntimeError(
                "Failed to send FollowJointTrajectory goal: " + "; ".join(failures)
            )
        if rejected:
            raise RuntimeError(
                "FollowJointTrajectory goal rejected: " + ", ".join(rejected)
            )

        result_events: list[threading.Event] = []
        result_holders: list[dict[str, object]] = []
        result_futures = []
        for goal_handle in goal_handles:
            done = threading.Event()
            holder: dict[str, object] = {}
            future = goal_handle.get_result_async()
            future.add_done_callback(
                lambda fut, holder=holder, done=done: (
                    holder.setdefault("future", fut),
                    done.set(),
                )
            )
            result_events.append(done)
            result_holders.append(holder)
            result_futures.append(future)

        result_deadline = (
            None
            if float(wait_result_s) < 0.0
            else time.monotonic() + float(wait_result_s)
        )
        while rclpy.ok():
            if all(done.is_set() for done in result_events):
                break
            if result_deadline is not None and time.monotonic() >= result_deadline:
                pending = [
                    normalized[idx][1]
                    for idx, done in enumerate(result_events)
                    if not done.is_set()
                ]
                raise RuntimeError(
                    "Timed out waiting for FollowJointTrajectory result: "
                    + ", ".join(pending)
                )
            time.sleep(0.05)

        if not all(done.is_set() for done in result_events):
            pending = [
                normalized[idx][1]
                for idx, done in enumerate(result_events)
                if not done.is_set()
            ]
            raise RuntimeError(
                "rclpy shutdown while waiting for FollowJointTrajectory result: "
                + ", ".join(pending)
            )

        result_failures = []
        for idx, future in enumerate(result_futures):
            done_future = result_holders[idx].get("future", future)
            exc = done_future.exception()
            if exc is not None:
                result_failures.append(f"{normalized[idx][1]}: {exc}")
                continue

            wrapped_result = done_future.result()
            result = wrapped_result.result if wrapped_result is not None else None
            if result is None:
                result_failures.append(
                    f"{normalized[idx][1]}: FollowJointTrajectory returned no result"
                )
                continue
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                detail = result.error_string.strip()
                suffix = f": {detail}" if detail else ""
                result_failures.append(
                    f"{normalized[idx][1]}: error_code {result.error_code}{suffix}"
                )

        if result_failures:
            raise RuntimeError(
                "FollowJointTrajectory failed on " + "; ".join(result_failures)
            )

    def _publish_joint_trajectory_group(
        self,
        commands: Sequence[tuple[str, Sequence[str], Sequence[Sequence[float]]]],
        *,
        dt: float,
        start_time_delay_s: float,
    ) -> None:
        if not commands:
            raise ValueError("commands is empty")

        normalized = [
            (str(topic), list(joint_names), path)
            for topic, joint_names, path in commands
        ]
        publishers = [
            self._get_traj_publisher(topic)
            for topic, _names, _path in normalized
        ]
        messages = [
            _build_joint_trajectory(path, joint_names, dt=float(dt))
            for _topic, joint_names, path in normalized
        ]

        retry_until_subscriber = bool(
            getattr(self._args, "publish_retry_until_subscriber", True)
        )
        initial_wait_s = (
            0.0
            if retry_until_subscriber
            and float(self._args.publish_wait_subscriber_s) < 0.0
            else float(self._args.publish_wait_subscriber_s)
        )
        wait_deadline = (
            None
            if initial_wait_s < 0.0
            else time.monotonic() + max(0.0, initial_wait_s)
        )
        unresolved = list(range(len(normalized)))
        next_log_t = 0.0

        while rclpy.ok() and unresolved:
            unresolved = [
                idx
                for idx in unresolved
                if publishers[idx].get_subscription_count() == 0
            ]
            if not unresolved:
                break

            now = time.monotonic()
            if wait_deadline is not None and now >= wait_deadline:
                break
            if next_log_t == 0.0 or now >= next_log_t:
                waiting = ", ".join(normalized[idx][0] for idx in unresolved)
                self.get_logger().info(
                    f"Waiting for at least 1 matching subscription(s) on: {waiting}"
                )
                next_log_t = now + 1.0
            time.sleep(0.05)

        if unresolved and retry_until_subscriber:
            waiting = ", ".join(normalized[idx][0] for idx in unresolved)
            self.get_logger().warning(
                f"No matching subscribers on {waiting}; re-publishing until all appear."
            )
            next_log_t = 0.0
            while rclpy.ok() and unresolved:
                now = time.monotonic()
                if next_log_t == 0.0 or now >= next_log_t:
                    still_waiting = ", ".join(normalized[idx][0] for idx in unresolved)
                    self.get_logger().info(
                        f"Still waiting for at least 1 matching subscription(s) on: {still_waiting}"
                    )
                    next_log_t = now + 1.0

                stamp = self._trajectory_stamp(float(start_time_delay_s))
                for idx in unresolved:
                    messages[idx].header.stamp = stamp
                    publishers[idx].publish(messages[idx])
                time.sleep(max(0.05, float(self._args.publish_period_s)))

                unresolved = [
                    idx
                    for idx in unresolved
                    if publishers[idx].get_subscription_count() == 0
                ]

            if not unresolved:
                matched = ", ".join(topic for topic, _names, _path in normalized)
                self.get_logger().info(f"Matched subscriptions on {matched}.")

        if unresolved:
            waiting = ", ".join(normalized[idx][0] for idx in unresolved)
            message = (
                f"No subscribers detected on {waiting} after waiting "
                f"{max(0.0, float(self._args.publish_wait_subscriber_s)):.2f}s"
            )
            if bool(self._args.publish_require_subscriber):
                raise RuntimeError(message)
            self.get_logger().warning(f"{message}; publishing anyway.")

        repeats = max(1, int(self._args.publish_repeat))
        wait_ack_s = float(getattr(self._args, "publish_wait_ack_s", 0.0))
        wait_for_ack = (
            wait_ack_s > 0.0
            and str(getattr(self._args, "publish_reliability", "best_effort"))
            .strip()
            .lower()
            == "reliable"
        )
        ack_timeout = RclpyDuration(seconds=max(0.0, wait_ack_s))
        for i in range(repeats):
            stamp = self._trajectory_stamp(float(start_time_delay_s))
            for msg in messages:
                msg.header.stamp = stamp
            for idx, pub in enumerate(publishers):
                pub.publish(messages[idx])
            if wait_for_ack:
                for idx, pub in enumerate(publishers):
                    if hasattr(pub, "wait_for_all_acked") and not pub.wait_for_all_acked(
                        ack_timeout
                    ):
                        self.get_logger().warning(
                            f"Timed out waiting for DDS acknowledgements on "
                            f"{normalized[idx][0]} after {wait_ack_s:.2f}s."
                        )
            if i + 1 < repeats:
                time.sleep(max(0.0, float(self._args.publish_period_s)))

        keep_alive_s = max(0.0, float(self._args.publish_keep_alive_s))
        if keep_alive_s > 0.0:
            time.sleep(keep_alive_s)

    def _wait_for_publisher_match(
        self,
        pub,
        topic: str,
        *,
        wait_subscriber_s: float,
    ) -> bool:
        deadline = None if float(wait_subscriber_s) < 0.0 else time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok():
            if pub.get_subscription_count() > 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _publish_joint_state_path(
        self,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        *,
        topic: str,
        dt: float,
    ) -> None:
        matched = self._wait_for_publisher_match(
            self._joint_state_cmd_pub,
            topic,
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )
        if not matched:
            self.get_logger().warning(
                f"No subscriber detected on {topic} after waiting "
                f"{max(0.0, float(self._args.publish_wait_subscriber_s)):.2f}s"
            )

        for idx, q in enumerate(path):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [str(name) for name in joint_names]
            msg.position = [float(v) for v in q]
            self._joint_state_cmd_pub.publish(msg)
            if idx + 1 < len(path):
                time.sleep(max(0.0, float(dt)))

        if path:
            last = JointState()
            last.header.stamp = self.get_clock().now().to_msg()
            last.name = [str(name) for name in joint_names]
            last.position = [float(v) for v in path[-1]]
            self._joint_state_cmd_pub.publish(last)

    def _wait_until_joint_positions(
        self,
        joint_names: Sequence[str],
        target_positions: Sequence[float],
        *,
        wait_s: float,
        tolerance: float,
        poll_period_s: float,
    ) -> tuple[bool, list[float], float]:
        deadline = None if float(wait_s) < 0.0 else time.monotonic() + float(wait_s)
        latest_positions = [float("nan") for _ in joint_names]
        latest_max_abs_err = float("inf")
        with self._joint_state_cv:
            while rclpy.ok():
                missing = [name for name in joint_names if name not in self._joint_state_by_name]
                if not missing:
                    latest_positions = [
                        float(self._joint_state_by_name[name]) for name in joint_names
                    ]
                    latest_max_abs_err = max(
                        abs(float(latest_positions[idx]) - float(target_positions[idx]))
                        for idx in range(len(joint_names))
                    )
                    if latest_max_abs_err <= float(tolerance):
                        return True, latest_positions, latest_max_abs_err
                if deadline is not None and time.monotonic() >= deadline:
                    return False, latest_positions, latest_max_abs_err
                self._joint_state_cv.wait(timeout=max(0.01, float(poll_period_s)))

        raise RuntimeError("rclpy shutdown while waiting for joint arrival")

    def _run_single_arm_init(
        self,
        arm: str,
        q_start_cspace: Sequence[float],
    ) -> SingleArmMotionPlan | None:
        q_goal_cspace = [float(v) for v in q_start_cspace]
        active_joint_names = _arm_joint_names(arm)
        name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}
        for joint_name in active_joint_names:
            q_goal_cspace[name_to_idx[joint_name]] = 0.0

        start_active = [float(q_start_cspace[name_to_idx[name]]) for name in active_joint_names]
        if max(abs(v) for v in start_active) <= float(_ZERO_GOAL_TOL):
            self.get_logger().info(
                f"[ARM_INIT] arm={arm} is already at zero goal within tol={_ZERO_GOAL_TOL:.1e}"
            )
            return None

        self.get_logger().info(
            f"[ARM_INIT] planning single-arm TB-RRT to zero pose for arm={arm}"
        )
        out = plan_single_arm_tbrrt_batch_conext(
            robot_yml=str(self._args.robot_yml),
            arm=arm,
            q_start=q_start_cspace,
            q_goals=[q_goal_cspace],
            world_yml=self._resolved_world_yml,
            cpu=bool(self._args.cpu),
            cfg=build_single_arm_tbrrt_config(self._args),
            joint_limit_yml=str(self._args.joint_limit_yml),
            block_k=int(self._args.tbrrt_block_k),
        )
        if not out.success or not out.path:
            raise RuntimeError(f"ARM_INIT single-arm TBRRT failed: {out.stats.extra}")

        self.get_logger().info(
            f"[ARM_INIT] arm={arm} path_len={len(out.path)} "
            f"iters={out.stats.iters} time={out.stats.time_sec:.3f}s"
        )
        return SingleArmMotionPlan(
            arm=arm,
            cspace_joint_names=list(CSPACE_JOINT_NAMES_14),
            active_joint_names=active_joint_names,
            q_start_cspace=[float(v) for v in q_start_cspace],
            q_goal_cspace=[float(v) for v in q_goal_cspace],
            raw_path=[[float(v) for v in row] for row in out.path],
            spline_path=[[float(v) for v in row] for row in out.path],
        )

    def _publish_single_arm_path(
        self,
        *,
        arm: str,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
    ) -> None:
        if str(self._args.publish_mode) == "real":
            topic = (
                str(self._args.real_left_topic)
                if arm == "left"
                else str(self._args.real_right_topic)
            )
            action = (
                str(self._args.real_left_action)
                if arm == "left"
                else str(self._args.real_right_action)
            )
            if bool(self._args.real_use_action):
                try:
                    self._send_follow_joint_trajectory_group(
                        [
                            (
                                self._get_arm_action_client(action),
                                action,
                                list(joint_names),
                                path,
                            )
                        ],
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                        start_time_delay_s=float(
                            getattr(self._args, "start_delay_s", 0.2)
                        ),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            self._publish_joint_trajectory_group(
                [
                    (
                        topic,
                        list(joint_names),
                        path,
                    )
                ],
                dt=float(self._args.publish_dt),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        self._publish_joint_state_path(
            joint_names,
            path,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
        )

    def _execute_single_arm_plan(self, plan: SingleArmMotionPlan) -> None:
        joint_names, active_path = build_active_joint_path(plan)
        goal_positions = [float(v) for v in active_path[-1]]
        max_attempts = _retry_attempt_limit(
            int(getattr(self._args, "arrival_max_retries", -1))
        )
        attempt_idx = 0
        last_err = float("inf")

        while max_attempts is None or attempt_idx < max_attempts:
            cmd_path = active_path
            if attempt_idx > 0:
                try:
                    current_positions = self._wait_for_joint_sample(
                        joint_names,
                        wait_s=float(self._args.joint_state_wait_s),
                    )
                    nearest_idx = _nearest_waypoint_index(current_positions, active_path)
                    cmd_path = _build_retry_path(
                        current_positions=current_positions,
                        original_path=active_path,
                    )
                    remaining_count = max(1, len(active_path) - nearest_idx - 1)
                    if nearest_idx == 0:
                        resume_desc = "no progress detected; re-publishing from the start"
                    else:
                        resume_desc = (
                            f"resume from waypoint {nearest_idx + 1}/{len(active_path)} with "
                            f"{remaining_count} remaining segment(s)"
                        )
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} "
                        f"for {plan.arm} arm init: {resume_desc}."
                    )
                except RuntimeError as exc:
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} "
                        f"for {plan.arm} arm init: failed to read current joints ({exc}); "
                        "re-publishing full path."
                    )
            else:
                self.get_logger().info(
                    "[TRAJ] single-arm init trajectory ready; publishing: "
                    f"arm={plan.arm} waypoints={len(cmd_path)} goal={goal_positions}"
                )

            try:
                self._publish_single_arm_path(
                    arm=plan.arm,
                    joint_names=joint_names,
                    path=cmd_path,
                )
            except RuntimeError as exc:
                attempt_idx += 1
                if max_attempts is not None and attempt_idx >= max_attempts:
                    raise RuntimeError(
                        f"Failed to publish {plan.arm} arm init command after "
                        f"{attempt_idx} attempt(s): {exc}"
                    ) from exc
                self.get_logger().warning(
                    f"[ARRIVAL] publish attempt {_format_attempt(attempt_idx - 1, max_attempts)} "
                    f"failed for {plan.arm} arm init: {exc}. Retrying."
                )
                continue

            wait_s = max(
                2.0,
                float(max(0, len(cmd_path) - 1)) * float(self._args.publish_dt) + 2.0,
            )
            configured_wait_s = float(getattr(self._args, "arrival_wait_s", -1.0))
            if configured_wait_s >= 0.0:
                wait_s = configured_wait_s
            arrived, _positions, max_abs_err = self._wait_until_joint_positions(
                joint_names,
                goal_positions,
                wait_s=wait_s,
                tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
                poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
            )
            if arrived:
                if attempt_idx > 0:
                    self.get_logger().info(
                        f"[ARRIVAL] confirmed after retry for {plan.arm} arm init: "
                        f"max_abs_err={max_abs_err:.6f}"
                    )
                return

            last_err = max_abs_err
            attempt_idx += 1
            self.get_logger().warning(
                f"[ARRIVAL] {plan.arm} arm init not at goal after attempt "
                f"{_format_attempt(attempt_idx - 1, max_attempts)}: "
                f"max_abs_err={max_abs_err:.6f}. "
                "Re-publishing toward the remaining path."
            )

        raise RuntimeError(
            f"Failed to confirm {plan.arm} arm init arrival after "
            f"{'infinite retry loop interruption' if max_attempts is None else f'{max_attempts} attempt(s)'}; "
            f"last max_abs_err={last_err:.6f}"
        )

    def _plan_dual_target_path(
        self,
        q_start_cspace: Sequence[float],
    ) -> tuple[list[list[float]], dict]:
        left_xyz, left_quat_wxyz, right_xyz, right_quat_wxyz = self._target_pose_args()

        self.get_logger().info(
            "[ARM_INIT] solving dual-arm target IK with ForceCuroboIK "
            f"left={_format_pose_arg(self._args.left_xyz, self._args.left_quat_xyzw)} "
            f"right={_format_pose_arg(self._args.right_xyz, self._args.right_quat_xyzw)}"
        )
        ik_out = self._get_force_ik_solver().solve_max_forward_force(
            left_xyz=left_xyz,
            left_quat_wxyz=left_quat_wxyz,
            right_xyz=right_xyz,
            right_quat_wxyz=right_quat_wxyz,
            q_start_cspace=q_start_cspace,
            forward_direction_base=[float(v) for v in self._args.forward_direction_base],
            num_trials=int(self._args.force_ik_num_trials),
            seed_noise_std=float(self._args.force_ik_seed_noise_std),
            random_seed=int(self._args.force_ik_random_seed),
        )
        if not ik_out.success or ik_out.q_cspace is None:
            raise RuntimeError("ForceCuroboIK failed to find a collision-free dual-arm IK solution")

        q_goal_dual = [float(v) for v in ik_out.q_cspace]
        self.get_logger().info(
            f"[ARM_INIT] dual-arm IK success score={ik_out.score:.6f} "
            f"left_force={ik_out.left_force_capacity:.6f} "
            f"right_force={ik_out.right_force_capacity:.6f} "
            f"valid={ik_out.valid_candidates}/{ik_out.tried_candidates}"
        )

        self.get_logger().info("[ARM_INIT] planning left arm to dual target with TB-RRT")
        left_out = _plan_single_arm_from_dual_goal(
            arm="left",
            robot_yml=str(self._args.robot_yml),
            joint_limit_yml=str(self._args.joint_limit_yml),
            q_start_cspace=q_start_cspace,
            q_goal_dual=q_goal_dual,
            world_yml=self._resolved_world_yml,
            cpu=bool(self._args.cpu),
            args=self._args,
        )
        self.get_logger().info(
            f"[ARM_INIT] left target path_len={len(left_out.path)} "
            f"iters={left_out.stats.iters} time={left_out.stats.time_sec:.3f}s"
        )

        self.get_logger().info("[ARM_INIT] planning right arm to dual target with TB-RRT")
        right_out = _plan_single_arm_from_dual_goal(
            arm="right",
            robot_yml=str(self._args.robot_yml),
            joint_limit_yml=str(self._args.joint_limit_yml),
            q_start_cspace=q_start_cspace,
            q_goal_dual=q_goal_dual,
            world_yml=self._resolved_world_yml,
            cpu=bool(self._args.cpu),
            args=self._args,
        )
        self.get_logger().info(
            f"[ARM_INIT] right target path_len={len(right_out.path)} "
            f"iters={right_out.stats.iters} time={right_out.stats.time_sec:.3f}s"
        )

        left_path_full = [[float(v) for v in row] for row in left_out.path]
        right_path_full = [[float(v) for v in row] for row in right_out.path]
        combined_path = _synchronize_single_arm_paths(
            q_start_cspace=q_start_cspace,
            left_path_full=left_path_full,
            right_path_full=right_path_full,
        )
        self.get_logger().info(
            f"[ARM_INIT] synchronized dual-arm target path_len={len(combined_path)}"
        )

        if bool(getattr(self._args, "validate_combined_path", True)):
            self.get_logger().info("[ARM_INIT] validating synchronized dual-arm path")
            _validate_dual_path(
                combined_path,
                robot_yml=str(self._args.robot_yml),
                cpu=bool(self._args.cpu),
                world_yml=self._resolved_world_yml,
            )
            self.get_logger().info("[ARM_INIT] synchronized dual-arm path is collision-free")

        plan_info = {
            "q_goal_cspace": q_goal_dual,
            "left_path_full": left_path_full,
            "right_path_full": right_path_full,
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
        return combined_path, plan_info

    def _publish_dual_target_path(
        self,
        q_path: Sequence[Sequence[float]],
    ) -> None:
        left_path = _project_full_path_to_active(
            q_path,
            active_joint_names=LEFT_JOINTS,
        )
        right_path = _project_full_path_to_active(
            q_path,
            active_joint_names=RIGHT_JOINTS,
        )

        if str(self._args.publish_mode) == "real":
            topic_commands = [
                (
                    str(self._args.real_left_topic),
                    list(LEFT_JOINTS),
                    left_path,
                ),
                (
                    str(self._args.real_right_topic),
                    list(RIGHT_JOINTS),
                    right_path,
                ),
            ]
            if bool(self._args.real_use_action):
                try:
                    action_commands = [
                        (
                            self._get_arm_action_client(
                                str(self._args.real_left_action)
                            ),
                            str(self._args.real_left_action),
                            list(LEFT_JOINTS),
                            left_path,
                        ),
                        (
                            self._get_arm_action_client(
                                str(self._args.real_right_action)
                            ),
                            str(self._args.real_right_action),
                            list(RIGHT_JOINTS),
                            right_path,
                        ),
                    ]
                    self._send_follow_joint_trajectory_group(
                        action_commands,
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                        start_time_delay_s=float(
                            getattr(self._args, "start_delay_s", 0.2)
                        ),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            self._publish_joint_trajectory_group(
                topic_commands,
                dt=float(self._args.publish_dt),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        self._publish_joint_state_path(
            list(CSPACE_JOINT_NAMES_14),
            q_path,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
        )

    def _execute_dual_target_path(
        self,
        q_path: Sequence[Sequence[float]],
    ) -> None:
        left_path = _project_full_path_to_active(
            q_path,
            active_joint_names=LEFT_JOINTS,
        )
        right_path = _project_full_path_to_active(
            q_path,
            active_joint_names=RIGHT_JOINTS,
        )
        left_goal = [float(v) for v in left_path[-1]]
        right_goal = [float(v) for v in right_path[-1]]
        max_attempts = _retry_attempt_limit(
            int(getattr(self._args, "arrival_max_retries", -1))
        )
        attempt_idx = 0
        last_failure = ""

        while max_attempts is None or attempt_idx < max_attempts:
            cmd_path = [[float(v) for v in row] for row in q_path]
            if attempt_idx > 0:
                try:
                    left_current = self._wait_for_joint_sample(
                        LEFT_JOINTS,
                        wait_s=float(self._args.joint_state_wait_s),
                    )
                    right_current = self._wait_for_joint_sample(
                        RIGHT_JOINTS,
                        wait_s=float(self._args.joint_state_wait_s),
                    )
                    left_nearest_idx = _nearest_waypoint_index(left_current, left_path)
                    right_nearest_idx = _nearest_waypoint_index(right_current, right_path)
                    cmd_path = _combine_active_joint_paths(
                        left_path=_build_retry_path(
                            current_positions=left_current,
                            original_path=left_path,
                        ),
                        right_path=_build_retry_path(
                            current_positions=right_current,
                            original_path=right_path,
                        ),
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
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} "
                        "for arm_init dual target: "
                        f"left at {left_desc}, right at {right_desc}."
                    )
                except RuntimeError as exc:
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} "
                        "for arm_init dual target: "
                        f"failed to read current joints ({exc}); re-publishing full path."
                    )
            else:
                self.get_logger().info(
                    "[TRAJ] dual target trajectory ready; publishing: "
                    f"waypoints={len(cmd_path)}"
                )

            try:
                self._publish_dual_target_path(cmd_path)
            except RuntimeError as exc:
                attempt_idx += 1
                if max_attempts is not None and attempt_idx >= max_attempts:
                    raise RuntimeError(
                        "Failed to publish arm_init dual target command after "
                        f"{attempt_idx} attempt(s): {exc}"
                    ) from exc
                self.get_logger().warning(
                    f"[ARRIVAL] publish attempt {_format_attempt(attempt_idx - 1, max_attempts)} "
                    f"failed for arm_init dual target: {exc}. Retrying."
                )
                continue

            wait_s = max(
                2.0,
                float(max(0, len(cmd_path) - 1)) * float(self._args.publish_dt) + 2.0,
            )
            configured_wait_s = float(getattr(self._args, "arrival_wait_s", -1.0))
            if configured_wait_s >= 0.0:
                wait_s = configured_wait_s

            failures: list[str] = []
            for arm_name, joint_names, goal in (
                ("left", LEFT_JOINTS, left_goal),
                ("right", RIGHT_JOINTS, right_goal),
            ):
                arrived, _positions, max_abs_err = self._wait_until_joint_positions(
                    joint_names,
                    goal,
                    wait_s=wait_s,
                    tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
                    poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
                )
                if not arrived:
                    failures.append(f"{arm_name}: max_abs_err={max_abs_err:.6f}")

            if not failures:
                if attempt_idx > 0:
                    self.get_logger().info(
                        "[ARRIVAL] arm_init dual target confirmed after retry."
                    )
                return

            last_failure = "; ".join(failures)
            attempt_idx += 1
            self.get_logger().warning(
                f"[ARRIVAL] arm_init dual target not confirmed after attempt "
                f"{_format_attempt(attempt_idx - 1, max_attempts)}: {last_failure}. "
                "Re-publishing toward the remaining path."
            )

        if max_attempts is None:
            raise RuntimeError(
                "Failed to confirm arm_init dual target arrival after "
                f"infinite retry loop interruption; {last_failure}"
            )
        raise RuntimeError(
            "Failed to confirm arm_init dual target arrival after "
            f"{max_attempts} attempt(s); {last_failure}"
        )

    def _process_request(self, arm: str) -> None:
        try:
            self.get_logger().info(f"Received arm_init_start arm={arm}")
            _publish_world_collision_for_mujoco(self._args, self._resolved_world_yml)

            q_start_cspace = self._wait_for_joint_sample(
                list(CSPACE_JOINT_NAMES_14),
                wait_s=float(self._args.joint_state_wait_s),
            )
            if _all_joints_zero(q_start_cspace):
                self.get_logger().info(
                    "[ARM_INIT] all joints are already at zero; skipping single-arm init "
                    "and starting dual-arm target planning immediately"
                )
            else:
                init_plan = self._run_single_arm_init(arm, q_start_cspace)
                if init_plan is not None:
                    self._execute_single_arm_plan(init_plan)

            q_after_init = self._wait_for_joint_sample(
                list(CSPACE_JOINT_NAMES_14),
                wait_s=float(self._args.joint_state_wait_s),
            )
            target_path, target_plan_info = self._plan_dual_target_path(q_after_init)
            self.get_logger().info(
                "[ARM_INIT] starting planned dual-arm target trajectory "
                f"path_len={len(target_path)}"
            )
            self._execute_dual_target_path(target_path)

            if self._args.save:
                payload = {
                    "mode": "arm_init",
                    "arm": arm,
                    "world_yml": self._resolved_world_yml,
                    "q_start_cspace": [float(v) for v in q_start_cspace],
                    "q_after_init_cspace": [float(v) for v in q_after_init],
                    "target": {
                        "left_xyz": [float(v) for v in self._args.left_xyz],
                        "left_quat_xyzw": [
                            float(v) for v in self._args.left_quat_xyzw
                        ],
                        "right_xyz": [float(v) for v in self._args.right_xyz],
                        "right_quat_xyzw": [
                            float(v) for v in self._args.right_quat_xyzw
                        ],
                    },
                    "target_plan": target_plan_info,
                    "combined_path": [[float(v) for v in row] for row in target_path],
                    "left_path": _project_full_path_to_active(
                        target_path,
                        active_joint_names=LEFT_JOINTS,
                    ),
                    "right_path": _project_full_path_to_active(
                        target_path,
                        active_joint_names=RIGHT_JOINTS,
                    ),
                    "publish_dt": float(self._args.publish_dt),
                }
                with open(str(self._args.save), "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self.get_logger().info(f"[ARM_INIT] saved execution payload to {self._args.save}")

            self._publish_finish()
            self.get_logger().info(f"[ARM_INIT] completed arm={arm}")
        except Exception as exc:
            self.get_logger().error(f"arm_init failed for arm={arm}: {exc}")


def main_arm_init(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    args, _unknown = build_parser().parse_known_args(argv_list)

    rclpy.init(args=argv_list)
    node = ArmInitNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ARM_INIT interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_init(argv)


if __name__ == "__main__":
    raise SystemExit(main())

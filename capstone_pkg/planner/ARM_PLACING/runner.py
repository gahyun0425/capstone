from __future__ import annotations

import argparse
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml

from capstone_pkg.kinematics.curobo_ik import (
    get_single_arm_ik,
    warmup_single_arm_ik_reachable,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
    build_active_joint_path,
    normalize_arm_name,
    plan_single_arm_motion,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
    _resolve_world_yml,
    build_single_arm_parser,
    build_single_arm_tbrrt_config,
)
from capstone_pkg.utils.config import CART_YAML, LEFT_JOINTS, RIGHT_JOINTS
from capstone_pkg.utils.world_collision_bridge import (
    WorldCuboid,
    make_world_collision_payload,
    load_world_cuboids,
)


_ARM_TARGETS = {
    "left": {
        "position": (0.5, 0.1, 1.0),
        "orientation_xyzw": (1.0, 0.0, 0.0, 0.0),
    },
    "right": {
        "position": (0.5, -0.1, 1.0),
        "orientation_xyzw": (1.0, 0.0, 0.0, 0.0),
    },
}
_CSPACE_JOINT_NAMES = list(LEFT_JOINTS) + list(RIGHT_JOINTS)
_SIM_GRIPPER_JOINT_NAMES = ["gripper_l_joint1", "gripper_r_joint1"]


def _duration_from_seconds(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((float(seconds) - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _command_qos(
    *,
    reliability: ReliabilityPolicy,
    durability: DurabilityPolicy,
    depth: int,
) -> QoSProfile:
    qos = QoSProfile(
        reliability=reliability,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
    )
    return qos


def _xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> list[float]:
    if len(quat_xyzw) != 4:
        raise ValueError("quat_xyzw must have length 4")
    x, y, z, w = [float(v) for v in quat_xyzw]
    return [w, x, y, z]


def _quat_multiply_wxyz(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(a) != 4 or len(b) != 4:
        raise ValueError("quaternion inputs must have length 4")
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _rotate_vec_by_quat_wxyz(vec_xyz: Sequence[float], quat_wxyz: Sequence[float]) -> list[float]:
    if len(vec_xyz) != 3:
        raise ValueError("vec_xyz must have length 3")
    q = [float(v) for v in quat_wxyz]
    q_conj = [q[0], -q[1], -q[2], -q[3]]
    v_quat = [0.0, float(vec_xyz[0]), float(vec_xyz[1]), float(vec_xyz[2])]
    rotated = _quat_multiply_wxyz(_quat_multiply_wxyz(q, v_quat), q_conj)
    return [float(rotated[1]), float(rotated[2]), float(rotated[3])]


def _normalize_xyzw(quat_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quat_xyzw) != 4:
        raise ValueError("quat_xyzw must have length 4")
    norm = math.sqrt(sum(float(v) * float(v) for v in quat_xyzw))
    if norm < 1.0e-9:
        raise ValueError("quat_xyzw norm is too small")
    x, y, z, w = [float(v) / norm for v in quat_xyzw]
    return x, y, z, w


def _copy_pose(pose: Pose) -> Pose:
    out = Pose()
    out.position.x = float(pose.position.x)
    out.position.y = float(pose.position.y)
    out.position.z = float(pose.position.z)
    out.orientation.x = float(pose.orientation.x)
    out.orientation.y = float(pose.orientation.y)
    out.orientation.z = float(pose.orientation.z)
    out.orientation.w = float(pose.orientation.w)
    return out


def _pose_orientation_xyzw(pose: Pose) -> list[float]:
    return [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _pose_position_xyz(pose: Pose) -> list[float]:
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]


def _cart_local_origin_x(cuboids: Sequence[WorldCuboid]) -> float:
    if not cuboids:
        raise ValueError("cart cuboids are empty")
    return min(float(c.pose[0]) - 0.5 * float(c.dims[0]) for c in cuboids)


def _cart_local_cuboids(base_cuboids: Sequence[WorldCuboid]) -> list[WorldCuboid]:
    x_origin = _cart_local_origin_x(base_cuboids)
    out: list[WorldCuboid] = []
    for cuboid in base_cuboids:
        local_pose = [
            float(cuboid.pose[0]) - float(x_origin),
            float(cuboid.pose[1]),
            float(cuboid.pose[2]),
            float(cuboid.pose[3]),
            float(cuboid.pose[4]),
            float(cuboid.pose[5]),
            float(cuboid.pose[6]),
        ]
        out.append(
            WorldCuboid(
                name=str(cuboid.name),
                dims=[float(v) for v in cuboid.dims],
                pose=local_pose,
            )
        )
    return out


def _transform_cart_cuboids(
    local_cuboids: Sequence[WorldCuboid],
    *,
    target_position_xyz: Sequence[float],
    target_orientation_xyzw: Sequence[float],
) -> list[WorldCuboid]:
    target_quat_wxyz = _xyzw_to_wxyz(_normalize_xyzw(target_orientation_xyzw))
    tx, ty, tz = [float(v) for v in target_position_xyz]
    out: list[WorldCuboid] = []
    for cuboid in local_cuboids:
        local_center = [float(cuboid.pose[0]), float(cuboid.pose[1]), float(cuboid.pose[2])]
        rotated_center = _rotate_vec_by_quat_wxyz(local_center, target_quat_wxyz)
        local_quat_wxyz = [float(v) for v in cuboid.pose[3:7]]
        world_quat_wxyz = _quat_multiply_wxyz(target_quat_wxyz, local_quat_wxyz)
        out.append(
            WorldCuboid(
                name=str(cuboid.name),
                dims=[float(v) for v in cuboid.dims],
                pose=[
                    tx + float(rotated_center[0]),
                    ty + float(rotated_center[1]),
                    tz + float(rotated_center[2]),
                    float(world_quat_wxyz[0]),
                    float(world_quat_wxyz[1]),
                    float(world_quat_wxyz[2]),
                    float(world_quat_wxyz[3]),
                ],
            )
        )
    return out


def _write_world_cuboids_yaml(cuboids: Sequence[WorldCuboid], *, prefix: str) -> str:
    payload = {
        "cuboid": {
            str(cuboid.name): {
                "dims": [float(v) for v in cuboid.dims],
                "pose": [float(v) for v in cuboid.pose],
            }
            for cuboid in cuboids
        }
    }
    tmp = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".yaml",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return str(tmp_path)


def _merge_world_cuboids(
    base_cuboids: Sequence[WorldCuboid],
    extra_cuboids: Sequence[WorldCuboid],
    *,
    extra_prefix: str,
) -> list[WorldCuboid]:
    out = [
        WorldCuboid(
            name=str(cuboid.name),
            dims=[float(v) for v in cuboid.dims],
            pose=[float(v) for v in cuboid.pose],
        )
        for cuboid in base_cuboids
    ]
    out.extend(
        WorldCuboid(
            name=f"{extra_prefix}{cuboid.name}",
            dims=[float(v) for v in cuboid.dims],
            pose=[float(v) for v in cuboid.pose],
        )
        for cuboid in extra_cuboids
    )
    return out


def _arm_joint_names(arm: str) -> list[str]:
    return list(LEFT_JOINTS if arm == "left" else RIGHT_JOINTS)


def _extract_joint_positions(
    q_cspace: Sequence[float],
    cspace_joint_names: Sequence[str],
    joint_names: Sequence[str],
) -> list[float]:
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in cspace state: {missing}")
    return [float(q_cspace[name_to_idx[name]]) for name in joint_names]


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


def build_parser() -> argparse.ArgumentParser:
    ap = build_single_arm_parser(
        default_world_yml=None,
        collision_models=None,
        default_collision_model=None,
    )
    ap.description = "ARM_PLACING sequence worker"
    ap.set_defaults(arrival_max_retries=-1)
    ap.add_argument("--arm_placing_start_topic", default="/arm_placing_start")
    ap.add_argument("--arm_placing_finish_topic", default="/arm_placing_finish")
    ap.add_argument("--collision_yaml", default=CART_YAML)
    ap.add_argument("--finish_publish_repeat", type=int, default=1)
    ap.add_argument("--finish_publish_period_s", type=float, default=0.05)
    ap.add_argument("--lift_action", default="/lift_controller/follow_joint_trajectory")
    ap.add_argument("--lift_joint_name", default="lift_joint")
    ap.add_argument("--lift_duration_s", type=float, default=10.0)
    ap.add_argument("--lift_action_server_wait_s", type=float, default=5.0)
    ap.add_argument("--lift_goal_done_timeout_s", type=float, default=30.0)
    ap.add_argument("--lift_down_delta_m", type=float, default=-0.15)
    ap.add_argument("--gripper_open_target", type=float, default=0.0)
    ap.add_argument("--gripper_duration_s", type=float, default=1.0)
    ap.add_argument("--gripper_open_wait_s", type=float, default=1.2)
    ap.add_argument("--final_arm_target_z_m", type=float, default=1.3)
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
    return ap


class ArmPlacingNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("arm_placing")
        self._args = args
        self._request_cv = threading.Condition()
        self._pending_request_arm: str | None = None
        self._request_active = False
        self._joint_state_cv = threading.Condition()
        self._joint_state_by_name: dict[str, float] = {}
        self._base_cart_cuboids = load_world_cuboids(str(args.collision_yaml))
        self._resolved_world_yml = _resolve_world_yml(
            args,
            collision_models=None,
            default_world_yml=None,
        )
        self._active_cart_world_yml = self._build_world_yml_with_cart_cuboids(
            self._base_cart_cuboids,
            prefix="capstone_cart_collision_spawn_",
        )
        self._run_startup_warmup()

        self._qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._finish_pub = self.create_publisher(
            Bool,
            str(args.arm_placing_finish_topic),
            self._qos_cmd,
        )
        self._start_sub = self.create_subscription(
            String,
            str(args.arm_placing_start_topic),
            self._start_callback,
            self._qos_cmd,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            str(args.joint_state_topic),
            self._joint_state_callback,
            _command_qos(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1,
            ),
        )
        self._world_collision_pub = self.create_publisher(
            String,
            str(args.world_collision_topic),
            _command_qos(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self._lift_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(args.lift_action),
        )
        self._left_arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(args.real_left_action),
        )
        self._right_arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(args.real_right_action),
        )
        self._left_gripper_pub = self.create_publisher(
            JointTrajectory,
            str(args.real_left_gripper_topic),
            self._qos_cmd,
        )
        self._right_gripper_pub = self.create_publisher(
            JointTrajectory,
            str(args.real_right_gripper_topic),
            self._qos_cmd,
        )
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
        traj_qos = _command_qos(
            reliability=reliability,
            durability=durability,
            depth=int(getattr(args, "publish_qos_depth", 1)),
        )
        self._traj_publishers: dict[str, object] = {
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
        self._joint_state_publishers: dict[str, object] = {
            str(args.publish_topic): self.create_publisher(
                JointState,
                str(args.publish_topic),
                _command_qos(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    depth=10,
                ),
            )
        }

        self.get_logger().info(
            "Listening on "
            f"{args.arm_placing_start_topic} and publishing finish on "
            f"{args.arm_placing_finish_topic}"
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
                f"Ignoring arm_placing_start with invalid arm={msg.data!r}"
            )
            return

        with self._request_cv:
            if self._request_active or self._pending_request_arm is not None:
                state = "active" if self._request_active else "pending"
                self.get_logger().warning(
                    f"Ignoring arm_placing_start arm={arm}: request already {state}"
                )
                return
            self._pending_request_arm = arm
            self._request_cv.notify()

        self.get_logger().info(
            f"Accepted arm_placing_start arm={arm}"
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
                f"Processing arm_placing_start arm={arm}"
            )
            try:
                self._process_request(arm)
            finally:
                with self._request_cv:
                    self._request_active = False
                    self._request_cv.notify_all()

    def _iter_startup_world_ymls(self) -> list[str | None]:
        candidates: list[str | None] = [
            self._resolved_world_yml,
            str(self._active_cart_world_yml),
        ]
        out: list[str | None] = []
        seen: set[str] = set()
        for item in candidates:
            normalized = None if item in (None, "", "none", "None") else str(item)
            key = "<none>" if normalized is None else normalized
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out

    def _run_startup_warmup(self) -> None:
        if not bool(getattr(self._args, "startup_warmup", True)):
            self.get_logger().info("Startup warmup disabled")
            return

        warmup_iters = max(0, int(getattr(self._args, "startup_warmup_iters", 1)))
        configured_batch_size = getattr(self._args, "startup_warmup_batch_size", None)
        if configured_batch_size is None:
            configured_batch_size = getattr(self._args, "ik_batch", 100)
        warmup_batch_size = max(1, int(configured_batch_size))
        if warmup_iters <= 0:
            self.get_logger().info(
                "Startup warmup skipped because startup_warmup_iters <= 0"
            )
            return

        from capstone_pkg.collision_check.collision import get_self_collision_checker

        worlds = self._iter_startup_world_ymls()
        t0 = time.monotonic()
        self.get_logger().info(
            "Starting ARM_PLACING warmup "
            f"(iters={warmup_iters}, batch_size={warmup_batch_size}, worlds={len(worlds)})"
        )
        for world_index, world_yml in enumerate(worlds):
            world_label = str(world_yml) if world_yml is not None else "none"
            self.get_logger().info(f"[warmup] world={world_label}")
            checker = get_self_collision_checker(
                str(self._args.robot_yml),
                cpu=bool(self._args.cpu),
                world_yml=world_yml,
            )
            try:
                _ = checker.check_single([0.0 for _ in checker.cspace_names])
            except Exception:
                pass

            for arm_index, arm in enumerate(("left", "right")):
                ik = get_single_arm_ik(
                    str(self._args.robot_yml),
                    arm=arm,
                    cpu=bool(self._args.cpu),
                    world_yml=world_yml,
                )
                warmup_single_arm_ik_reachable(
                    ik,
                    iters=warmup_iters,
                    batch_size=warmup_batch_size,
                    noise_std=float(getattr(self._args, "ik_seed_noise_std", 0.25)),
                    random_seed=int(getattr(self._args, "ik_seed", 0))
                    + (world_index * 17)
                    + arm_index,
                )
                self.get_logger().info(
                    f"[warmup] ready arm={arm} world={world_label}"
                )
        self.get_logger().info(
            f"ARM_PLACING warmup completed in {time.monotonic() - t0:.2f}s"
        )

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
                        _wrapped_joint_delta(latest_positions[i], float(target_positions[i]))
                        for i in range(len(joint_names))
                    )
                    if latest_max_abs_err <= float(tolerance):
                        return True, latest_positions, latest_max_abs_err
                if deadline is not None and time.monotonic() >= deadline:
                    return False, latest_positions, latest_max_abs_err
                self._joint_state_cv.wait(timeout=max(0.01, float(poll_period_s)))

        raise RuntimeError("rclpy shutdown while waiting for joint arrival")

    def _process_request(self, arm: str) -> None:
        try:
            initial_arm_pose = self._read_current_arm_pose(arm)
            initial_lift_position = self._read_current_lift_position()

            self.get_logger().info("Step 2: spawn cart collision model")
            self._publish_cart_collision_world(
                str(self._active_cart_world_yml),
                label="cart collision spawn",
            )

            placing_target_pose = self._build_arm_target_pose(arm)
            self.get_logger().info(
                "Step 3: move selected arm to placing target "
                f"arm={arm}, xyz={_pose_position_xyz(placing_target_pose)}"
            )
            self._move_arm_to_pose(
                arm,
                placing_target_pose,
                world_yml=str(self._active_cart_world_yml),
            )

            lower_lift_target = initial_lift_position + float(self._args.lift_down_delta_m)
            self.get_logger().info(
                f"Step 4: lower lift from {initial_lift_position:.3f} to {lower_lift_target:.3f}"
            )
            self._send_lift_goal(lower_lift_target)

            self.get_logger().info("Step 5: open both grippers")
            self._open_grippers()

            self.get_logger().info(
                f"Step 6: raise lift back to {initial_lift_position:.3f}"
            )
            self._send_lift_goal(initial_lift_position)

            final_target_pose = self._build_arm_target_pose(arm)
            final_target_pose.position.z = float(self._args.final_arm_target_z_m)
            self.get_logger().info(
                "Step 7: move selected arm to fixed xy/orientation with "
                f"z={final_target_pose.position.z:.3f}"
            )
            self._move_arm_to_pose(
                arm,
                final_target_pose,
                world_yml=str(self._active_cart_world_yml),
            )

            self.get_logger().info("Step 8: publish arm_placing_finish")
            self._publish_finish()
        except Exception as exc:
            self.get_logger().error(f"arm placing failed for arm={arm}: {exc}")

    def _build_arm_target_pose(self, arm: str) -> Pose:
        target = _ARM_TARGETS[arm]
        pose = Pose()
        pose.position.x = float(target["position"][0])
        pose.position.y = float(target["position"][1])
        pose.position.z = float(target["position"][2])
        pose.orientation.x = float(target["orientation_xyzw"][0])
        pose.orientation.y = float(target["orientation_xyzw"][1])
        pose.orientation.z = float(target["orientation_xyzw"][2])
        pose.orientation.w = float(target["orientation_xyzw"][3])
        return pose

    def _build_world_yml_with_cart_cuboids(
        self,
        cart_cuboids: Sequence[WorldCuboid],
        *,
        prefix: str,
    ) -> str:
        base_cuboids: list[WorldCuboid] = []
        if self._resolved_world_yml not in (None, "", "none", "None"):
            base_cuboids = load_world_cuboids(str(self._resolved_world_yml))
        merged = _merge_world_cuboids(
            base_cuboids,
            cart_cuboids,
            extra_prefix="cart_",
        )
        return _write_world_cuboids_yaml(
            merged,
            prefix=prefix,
        )

    def _publish_cart_collision_world(self, world_yml: str, *, label: str) -> None:
        if not bool(getattr(self._args, "publish_world_collision", True)):
            self.get_logger().info(
                f"Skipping MuJoCo world collision publish for {label}: publish_world_collision disabled"
            )
            return

        matched = self._wait_for_publisher_match(
            self._world_collision_pub,
            str(self._args.world_collision_topic),
            wait_subscriber_s=float(self._args.world_collision_wait_subscriber_s),
        )
        if not matched:
            self.get_logger().warning(
                f"No subscriber detected on {self._args.world_collision_topic} before publishing {label}"
            )

        msg = String()
        msg.data = make_world_collision_payload(str(world_yml))
        repeats = 3
        for i in range(repeats):
            self._world_collision_pub.publish(msg)
            if i + 1 < repeats:
                time.sleep(0.05)
        keep_alive_s = max(0.0, float(self._args.world_collision_keep_alive_s))
        if keep_alive_s > 0.0:
            time.sleep(keep_alive_s)

        cuboid_count = len(load_world_cuboids(str(world_yml)))
        self.get_logger().info(
            f"[WORLD] published {cuboid_count} collision cuboid(s) "
            f"to {self._args.world_collision_topic} for {label}"
        )

    def _wait_for_action_server(self, client: ActionClient, action_name: str) -> None:
        deadline = time.monotonic() + max(
            0.1,
            float(getattr(self._args, "action_wait_server_s", 2.0)),
        )
        while rclpy.ok():
            if client.wait_for_server(timeout_sec=0.2):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"action server not available: {action_name}")

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
                raise TimeoutError(f"action server not available: {action_name}")

    def _send_follow_joint_trajectory(
        self,
        client: ActionClient,
        *,
        action_name: str,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        dt: float,
        wait_server_s: float,
        wait_result_s: float,
    ) -> None:
        self._wait_for_action_server_with_timeout(
            client,
            action_name,
            wait_server_s=float(wait_server_s),
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = _build_joint_trajectory(path, joint_names, dt=float(dt))

        send_done = threading.Event()
        send_holder: dict[str, object] = {}
        send_future = client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda fut: (send_holder.setdefault("future", fut), send_done.set())
        )

        send_deadline = time.monotonic() + max(1.0, float(wait_server_s) + 1.0)
        while rclpy.ok():
            if send_done.wait(timeout=0.1):
                break
            if time.monotonic() >= send_deadline:
                raise RuntimeError(f"Timed out sending FollowJointTrajectory goal: {action_name}")

        send_result_future = send_holder.get("future", send_future)
        send_exc = send_result_future.exception()
        if send_exc is not None:
            raise RuntimeError(f"Failed to send goal on {action_name}: {send_exc}")

        goal_handle = send_result_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"FollowJointTrajectory goal rejected: {action_name}")

        result_done = threading.Event()
        result_holder: dict[str, object] = {}
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: (result_holder.setdefault("future", fut), result_done.set())
        )

        result_deadline = None if float(wait_result_s) < 0.0 else time.monotonic() + float(wait_result_s)
        while rclpy.ok():
            if result_done.wait(timeout=0.1):
                break
            if result_deadline is not None and time.monotonic() >= result_deadline:
                raise RuntimeError(
                    f"Timed out waiting for FollowJointTrajectory result: {action_name}"
                )

        result_wrapped_future = result_holder.get("future", result_future)
        result_exc = result_wrapped_future.exception()
        if result_exc is not None:
            raise RuntimeError(f"FollowJointTrajectory failed on {action_name}: {result_exc}")

        wrapped_result = result_wrapped_future.result()
        result = wrapped_result.result if wrapped_result is not None else None
        if result is None:
            raise RuntimeError(f"FollowJointTrajectory returned no result: {action_name}")
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            detail = result.error_string.strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"FollowJointTrajectory failed on {action_name} "
                f"with error_code {result.error_code}{suffix}"
            )

    def _get_traj_publisher(self, topic: str):
        pub = self._traj_publishers.get(topic)
        if pub is not None:
            return pub
        raise KeyError(f"No JointTrajectory publisher configured for topic: {topic}")

    def _get_joint_state_publisher(self, topic: str):
        pub = self._joint_state_publishers.get(topic)
        if pub is not None:
            return pub
        raise KeyError(f"No JointState publisher configured for topic: {topic}")

    def _wait_for_publisher_match(self, pub, topic: str, *, wait_subscriber_s: float) -> bool:
        deadline = None if float(wait_subscriber_s) < 0.0 else time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok():
            if pub.get_subscription_count() > 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _publish_joint_trajectory(
        self,
        *,
        topic: str,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        dt: float,
        start_time_delay_s: float,
    ) -> None:
        pub = self._get_traj_publisher(str(topic))
        msg = _build_joint_trajectory(path, joint_names, dt=float(dt))

        def _stamp_message() -> None:
            if float(start_time_delay_s) > 0.0:
                msg.header.stamp = (
                    self.get_clock().now() + RclpyDuration(seconds=float(start_time_delay_s))
                ).to_msg()
            else:
                msg.header.stamp = self.get_clock().now().to_msg()

        retry_until_subscriber = bool(
            getattr(self._args, "publish_retry_until_subscriber", True)
        )
        initial_wait_s = (
            0.0
            if retry_until_subscriber and float(self._args.publish_wait_subscriber_s) < 0.0
            else float(self._args.publish_wait_subscriber_s)
        )
        matched = self._wait_for_publisher_match(
            pub,
            str(topic),
            wait_subscriber_s=initial_wait_s,
        )
        if not matched and retry_until_subscriber:
            self.get_logger().warning(
                f"No matching subscribers on {topic}; re-publishing until one appears."
            )
            next_log_t = 0.0
            while rclpy.ok() and pub.get_subscription_count() == 0:
                now = time.monotonic()
                if next_log_t == 0.0 or now >= next_log_t:
                    self.get_logger().info(
                        f"Still waiting for at least 1 matching subscription(s) on {topic}..."
                    )
                    next_log_t = now + 1.0

                _stamp_message()
                pub.publish(msg)
                time.sleep(max(0.05, float(self._args.publish_period_s)))

            matched = pub.get_subscription_count() > 0
            if matched:
                self.get_logger().info(
                    f"Matched {pub.get_subscription_count()} subscription(s) on {topic} after retry loop."
                )

        if not matched:
            message = (
                f"No subscribers detected on {topic} after waiting "
                f"{max(0.0, float(self._args.publish_wait_subscriber_s)):.2f}s"
            )
            if bool(self._args.publish_require_subscriber):
                raise RuntimeError(message)
            self.get_logger().warning(f"{message}; publishing anyway.")

        repeats = max(1, int(self._args.publish_repeat))
        for i in range(repeats):
            _stamp_message()
            pub.publish(msg)
            if i + 1 < repeats:
                time.sleep(max(0.0, float(self._args.publish_period_s)))
        if float(self._args.publish_keep_alive_s) > 0.0:
            time.sleep(float(self._args.publish_keep_alive_s))

    def _publish_joint_state_path(
        self,
        *,
        topic: str,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        dt: float,
    ) -> None:
        pub = self._get_joint_state_publisher(str(topic))
        self._wait_for_publisher_match(
            pub,
            str(topic),
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )
        for idx, q in enumerate(path):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [str(name) for name in joint_names]
            msg.position = [float(v) for v in q]
            pub.publish(msg)
            if idx + 1 < len(path):
                time.sleep(max(0.0, float(dt)))

    def _publish_joint_state_command(
        self,
        *,
        topic: str,
        joint_names: Sequence[str],
        positions: Sequence[float],
    ) -> None:
        if len(joint_names) != len(positions):
            raise ValueError("joint_names and positions length mismatch")

        pub = self._get_joint_state_publisher(str(topic))
        self._wait_for_publisher_match(
            pub,
            str(topic),
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [str(name) for name in joint_names]
        msg.position = [float(v) for v in positions]
        pub.publish(msg)

    def _read_current_lift_position(self) -> float:
        values = self._wait_for_joint_sample(
            [str(self._args.lift_joint_name)],
            wait_s=float(self._args.joint_state_wait_s),
        )
        return float(values[0])

    def _read_current_arm_pose(self, arm: str) -> Pose:
        ik = get_single_arm_ik(
            str(self._args.robot_yml),
            arm=arm,
            cpu=bool(self._args.cpu),
            world_yml=self._resolved_world_yml,
        )
        q_cspace = self._wait_for_joint_sample(
            ik.cspace_joint_names,
            wait_s=float(self._args.joint_state_wait_s),
        )
        q_active = _extract_joint_positions(
            q_cspace,
            ik.cspace_joint_names,
            ik.active_joint_names,
        )

        import torch

        kin = ik.solver.fk(
            torch.tensor(
                [q_active],
                device=ik.device,
                dtype=torch.float32,
            )
        )
        pos = [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]
        quat_wxyz = [float(v) for v in kin.ee_quaternion[0].detach().cpu().tolist()]

        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.w = quat_wxyz[0]
        pose.orientation.x = quat_wxyz[1]
        pose.orientation.y = quat_wxyz[2]
        pose.orientation.z = quat_wxyz[3]
        return pose

    def _send_lift_goal(self, target_position: float) -> None:
        self._send_follow_joint_trajectory(
            self._lift_action_client,
            action_name=str(self._args.lift_action),
            joint_names=[str(self._args.lift_joint_name)],
            path=[[float(target_position)]],
            dt=max(1.0e-3, float(self._args.lift_duration_s)),
            wait_server_s=float(self._args.lift_action_server_wait_s),
            wait_result_s=float(self._args.lift_goal_done_timeout_s),
        )

    def _make_gripper_trajectory_msg(self, joint_name: str) -> JointTrajectory:
        msg = JointTrajectory()
        msg.joint_names = [str(joint_name)]
        point = JointTrajectoryPoint()
        point.positions = [float(self._args.gripper_open_target)]
        point.time_from_start = _duration_from_seconds(
            max(1.0e-3, float(self._args.gripper_duration_s))
        )
        msg.points = [point]
        return msg

    def _open_grippers(self) -> None:
        if str(self._args.publish_mode) != "real":
            self._publish_joint_state_command(
                topic=str(self._args.publish_topic),
                joint_names=_SIM_GRIPPER_JOINT_NAMES,
                positions=[
                    float(self._args.gripper_open_target),
                    float(self._args.gripper_open_target),
                ],
            )
            time.sleep(max(0.0, float(self._args.gripper_open_wait_s)))
            return

        left_topic = str(self._args.real_left_gripper_topic)
        right_topic = str(self._args.real_right_gripper_topic)

        self._wait_for_publisher_match(
            self._left_gripper_pub,
            left_topic,
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )
        self._wait_for_publisher_match(
            self._right_gripper_pub,
            right_topic,
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )

        left_msg = self._make_gripper_trajectory_msg("gripper_l_joint1")
        right_msg = self._make_gripper_trajectory_msg("gripper_r_joint1")

        self._left_gripper_pub.publish(left_msg)
        self._right_gripper_pub.publish(right_msg)
        time.sleep(max(0.0, float(self._args.gripper_open_wait_s)))

    def _move_arm_to_pose(self, arm: str, target_pose: Pose, *, world_yml: str | None) -> None:
        plan = plan_single_arm_motion(
            robot_yml=str(self._args.robot_yml),
            world_yml=world_yml,
            cpu=bool(self._args.cpu),
            arm=arm,
            target_xyz=_pose_position_xyz(target_pose),
            target_quat_xyzw=_pose_orientation_xyzw(target_pose),
            q_start_cspace=self._wait_for_joint_sample(
                _CSPACE_JOINT_NAMES,
                wait_s=float(self._args.joint_state_wait_s),
            ),
            joint_state_topic=str(self._args.joint_state_topic),
            joint_state_wait_s=float(self._args.joint_state_wait_s),
            use_current_joint_state_start=False,
            max_iters=int(self._args.max_iters),
            step=float(self._args.step),
            goal_bias=float(self._args.goal_bias),
            connect_threshold=float(self._args.connect_threshold),
            planner_backend=str(self._args.planner_backend),
            joint_limit_yml=str(self._args.joint_limit_yml),
            ik_batch=int(self._args.ik_batch),
            ik_seed_noise_std=float(self._args.ik_seed_noise_std),
            ik_seed_random_seed=int(self._args.ik_seed),
            ik_goal_dedupe_tol=float(self._args.ik_goal_dedupe_tol),
            tbrrt_cfg=build_single_arm_tbrrt_config(self._args),
            tbrrt_block_k=int(self._args.tbrrt_block_k),
            spline_dt=float(self._args.publish_dt),
        )
        self._execute_single_arm_plan(plan)

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
            action_name = (
                str(self._args.real_left_action)
                if arm == "left"
                else str(self._args.real_right_action)
            )

            if bool(self._args.real_use_action):
                try:
                    self._send_follow_joint_trajectory(
                        self._left_arm_action_client
                        if arm == "left"
                        else self._right_arm_action_client,
                        action_name=action_name,
                        joint_names=joint_names,
                        path=path,
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            self._publish_joint_trajectory(
                topic=topic,
                joint_names=joint_names,
                path=path,
                dt=float(self._args.publish_dt),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        self._publish_joint_state_path(
            topic=str(self._args.publish_topic),
            joint_names=joint_names,
            path=path,
            dt=float(self._args.publish_dt),
        )

    def _execute_single_arm_plan(self, plan) -> None:
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
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for {plan.arm} arm: "
                        f"{resume_desc}."
                    )
                except RuntimeError as exc:
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for {plan.arm} arm: "
                        f"failed to read current joints ({exc}); re-publishing full path."
                    )
            else:
                self.get_logger().info(
                    "[TRAJ] trajectory ready; publishing: "
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
                        f"Failed to publish {plan.arm} arm command after "
                        f"{attempt_idx} attempt(s): {exc}"
                    ) from exc
                self.get_logger().warning(
                    f"[ARRIVAL] publish attempt {_format_attempt(attempt_idx - 1, max_attempts)} "
                    f"failed for {plan.arm} arm: {exc}. Retrying."
                )
                continue

            wait_s = _resolve_arrival_wait_s(
                path_len=len(cmd_path),
                dt=float(self._args.publish_dt),
                configured_wait_s=float(getattr(self._args, "arrival_wait_s", -1.0)),
            )
            arrived, _current_positions, max_abs_err = self._wait_until_joint_positions(
                joint_names,
                goal_positions,
                wait_s=wait_s,
                tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
                poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
            )
            if arrived:
                if attempt_idx > 0:
                    self.get_logger().info(
                        f"[ARRIVAL] confirmed after retry for {plan.arm} arm: "
                        f"max_abs_err={max_abs_err:.6f}"
                    )
                return

            last_err = max_abs_err
            attempt_idx += 1
            self.get_logger().warning(
                f"[ARRIVAL] {plan.arm} arm not at goal after attempt "
                f"{_format_attempt(attempt_idx - 1, max_attempts)}: max_abs_err={max_abs_err:.6f}. "
                "Re-publishing toward the remaining path."
            )

        raise RuntimeError(
            f"Failed to confirm {plan.arm} arm arrival after "
            f"{'infinite retry loop interruption' if max_attempts is None else f'{max_attempts} attempt(s)'}; "
            f"last max_abs_err={last_err:.6f}"
        )

    def _publish_finish(self) -> None:
        repeat = max(1, int(self._args.finish_publish_repeat))
        period_s = max(0.0, float(self._args.finish_publish_period_s))
        msg = Bool()
        msg.data = True
        for _ in range(repeat):
            self._finish_pub.publish(msg)
            if period_s > 0.0:
                time.sleep(period_s)


def main_arm_placing(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    args, _unknown = build_parser().parse_known_args(argv_list)

    rclpy.init(args=argv_list)
    node: ArmPlacingNode | None = None
    try:
        node = ArmPlacingNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_placing(argv)


if __name__ == "__main__":
    raise SystemExit(main())

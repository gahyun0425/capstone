from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from grasp_msgs.msg import ObjectAlign, ObjectGrasp
from capstone_pkg.planner.arm_rrt_common.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_path,
    publish_joint_trajectory_group,
    read_joint_positions_once,
    send_joint_trajectory_action_group,
    wait_for_joint_positions,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
    SingleArmMotionPlan,
    normalize_arm_name,
    normalize_single_arm_planner_backend,
    plan_single_arm_motion,
)
from capstone_pkg.kinematics.curobo_ik import (
    get_single_arm_ik,
    warmup_single_arm_ik_reachable,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
    _merge_world_with_user_object,
    _publish_world_collision_for_mujoco,
    _resolve_world_yml,
    build_single_arm_tbrrt_config,
    build_single_arm_parser,
)
from capstone_pkg.utils.config import (
    LEFT_JOINTS,
    LONG_SHELF_YAML,
    RIGHT_JOINTS,
    SHELF_YAML,
)


_COLLISION_MODELS = {
    "long_shelf": LONG_SHELF_YAML,
    "shelf": SHELF_YAML,
    "shelf_1": SHELF_YAML,
    "shelf_2": LONG_SHELF_YAML,
}

_FIXED_ALIGN_QUATERNION_XYZW = (
    0.5,
    -0.5,
    0.5,
    -0.5,
)
_LEFT_SHELF_2_ALIGN_QUATERNION_XYZW = (
    0.5,
    0.5,
    0.5,
    0.5,
)
_SHELF_1_ALIGN_QUATERNION_XYZW = (
    1.0,
    0.0,
    0.0,
    0.0,
)
_SHELF_1_ALIGN_FIXED_Z_M = 1.2
_GRASP_OBJECT_POSE_X_OFFSET_M = 0.01
_GRASP_COLLISION_OBJECT_SIZE_X_M = 0.01
_GRASP_COLLISION_OBJECT_SIZE_Y_OFFSET_M = 0.0
_PREFERRED_GRASP_QUATERNION_XYZW = (
    0.7071067811865475,
    5.551115123125783e-17,
    0.7071067811865475,
    5.551115123125783e-17,
)
_GRASP_IK_CANDIDATE_BATCH_SIZE = 3
_GRASP_FALLBACK_Z_OFFSET_M = 0.03
_ZERO_JOINT_TOL = 1.0e-4


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


def _copy_point(point: Point) -> Point:
    out = Point()
    out.x = float(point.x)
    out.y = float(point.y)
    out.z = float(point.z)
    return out


def _copy_vector3(vec: Vector3) -> Vector3:
    out = Vector3()
    out.x = float(vec.x)
    out.y = float(vec.y)
    out.z = float(vec.z)
    return out


def _pose_position_xyz(pose: Pose) -> list[float]:
    return [float(pose.position.x), float(pose.position.y), float(pose.position.z)]


def _pose_orientation_xyzw(pose: Pose) -> list[float]:
    return [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _pose_orientation_wxyz(pose: Pose) -> list[float]:
    return [
        float(pose.orientation.w),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
    ]


def _vector3_xyz(vec: Vector3) -> list[float]:
    return [float(vec.x), float(vec.y), float(vec.z)]


def _normalize_orientation(quat: Quaternion) -> Quaternion:
    norm_sq = (
        float(quat.x) * float(quat.x)
        + float(quat.y) * float(quat.y)
        + float(quat.z) * float(quat.z)
        + float(quat.w) * float(quat.w)
    )
    out = Quaternion()
    if norm_sq <= 1.0e-12:
        out.w = 1.0
        return out

    inv_norm = norm_sq ** -0.5
    out.x = float(quat.x) * inv_norm
    out.y = float(quat.y) * inv_norm
    out.z = float(quat.z) * inv_norm
    out.w = float(quat.w) * inv_norm
    return out


def _build_align_target_pose(
    msg: ObjectAlign,
    *,
    selected_arm: str,
    fixed_x_m: float,
    lift_z_m: float,
) -> Pose:
    pose = Pose()
    shelf_type = str(msg.shelf_type).strip().lower()
    if shelf_type == "shelf_1":
        pose.position.x = float(msg.marker_position.x)
        pose.position.y = float(msg.marker_position.y)
        pose.position.z = float(_SHELF_1_ALIGN_FIXED_Z_M)
    else:
        pose.position.x = float(fixed_x_m)
        pose.position.y = float(msg.marker_position.y)
        pose.position.z = float(msg.marker_position.z) + float(lift_z_m)
    if shelf_type == "shelf_1":
        quat_xyzw = _SHELF_1_ALIGN_QUATERNION_XYZW
    elif shelf_type == "shelf_2" and normalize_arm_name(selected_arm) == "left":
        quat_xyzw = _LEFT_SHELF_2_ALIGN_QUATERNION_XYZW
    else:
        quat_xyzw = _FIXED_ALIGN_QUATERNION_XYZW
    pose.orientation = _normalize_orientation(
        Quaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        )
    )
    return pose


def _build_grasp_target_pose(msg: ObjectGrasp) -> Pose:
    pose = Pose()
    pose.position = _copy_point(msg.grasp_point)
    pose.orientation = _normalize_orientation(msg.grasp_pose.orientation)
    return pose


def _build_grasp_candidate_pose(point: Point, pose: Pose) -> Pose:
    out = Pose()
    out.position = _copy_point(point)
    out.orientation = _normalize_orientation(pose.orientation)
    return out


def _preferred_grasp_orientation() -> Quaternion:
    return _normalize_orientation(
        Quaternion(
            x=float(_PREFERRED_GRASP_QUATERNION_XYZW[0]),
            y=float(_PREFERRED_GRASP_QUATERNION_XYZW[1]),
            z=float(_PREFERRED_GRASP_QUATERNION_XYZW[2]),
            w=float(_PREFERRED_GRASP_QUATERNION_XYZW[3]),
        )
    )


def _build_object_center_fallback_pose(
    msg: ObjectGrasp,
    *,
    x_offset_m: float = 0.0,
    z_offset_m: float = 0.0,
) -> Pose:
    pose = Pose()
    pose.position = _copy_point(msg.object_pose.position)
    pose.position.x = float(pose.position.x) + float(x_offset_m)
    pose.position.z = float(pose.position.z) + float(z_offset_m)
    pose.orientation = _preferred_grasp_orientation()
    return pose


def _offset_pose_z(base_pose: Pose, delta_z_m: float) -> Pose:
    pose = _copy_pose(base_pose)
    pose.position.z = float(pose.position.z) + float(delta_z_m)
    return pose


def _build_post_grasp_staging_pose(arm: str, base_pose: Pose) -> Pose:
    pose = _copy_pose(base_pose)
    pose.position.x = 0.4
    pose.position.y = 0.2 if normalize_arm_name(arm) == "left" else -0.2
    return pose


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


def _build_zero_goal_cspace(
    *,
    q_start_cspace: Sequence[float],
    cspace_joint_names: Sequence[str],
    arm: str,
) -> list[float]:
    goal = [float(v) for v in q_start_cspace]
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    missing = [name for name in _arm_joint_names(arm) if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in cspace goal build: {missing}")
    for joint_name in _arm_joint_names(arm):
        goal[name_to_idx[joint_name]] = 0.0
    return goal


def _pad_path(path: Sequence[Sequence[float]], length: int) -> list[list[float]]:
    if not path:
        raise RuntimeError("Planner returned an empty path")
    out = [[float(v) for v in q] for q in path]
    while len(out) < length:
        out.append(list(out[-1]))
    return out


def _combine_active_joint_paths(
    *,
    cspace_joint_names: Sequence[str],
    selected_joint_names: Sequence[str],
    selected_path: Sequence[Sequence[float]],
    other_joint_names: Sequence[str],
    other_path: Sequence[Sequence[float]],
) -> list[list[float]]:
    total = max(len(selected_path), len(other_path))
    selected_sync = _pad_path(selected_path, total)
    other_sync = _pad_path(other_path, total)
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    full_path: list[list[float]] = []
    for selected_q, other_q in zip(selected_sync, other_sync):
        waypoint = [0.0 for _ in cspace_joint_names]
        for joint_name, joint_value in zip(selected_joint_names, selected_q):
            waypoint[name_to_idx[joint_name]] = float(joint_value)
        for joint_name, joint_value in zip(other_joint_names, other_q):
            waypoint[name_to_idx[joint_name]] = float(joint_value)
        full_path.append(waypoint)
    return full_path


def _build_combined_full_path(
    *,
    selected_plan: SingleArmMotionPlan,
    other_plan: SingleArmMotionPlan,
    selected_arm: str,
    other_arm: str,
) -> list[list[float]]:
    if not selected_plan.spline_path:
        raise RuntimeError("Selected-arm planner returned an empty spline path")
    if not other_plan.spline_path:
        raise RuntimeError("Other-arm planner returned an empty spline path")

    if list(selected_plan.cspace_joint_names) != list(other_plan.cspace_joint_names):
        raise RuntimeError("Selected/other arm plans use different cspace joint orders")

    cspace_joint_names = list(selected_plan.cspace_joint_names)
    selected_joint_names = _arm_joint_names(selected_arm)
    other_joint_names = _arm_joint_names(other_arm)
    selected_path = _build_active_joint_path(
        selected_plan.spline_path,
        cspace_joint_names,
        selected_joint_names,
    )
    other_path = _build_active_joint_path(
        other_plan.spline_path,
        cspace_joint_names,
        other_joint_names,
    )

    return _combine_active_joint_paths(
        cspace_joint_names=cspace_joint_names,
        selected_joint_names=selected_joint_names,
        selected_path=selected_path,
        other_joint_names=other_joint_names,
        other_path=other_path,
    )


def _build_active_joint_path(
    full_path: Sequence[Sequence[float]],
    cspace_joint_names: Sequence[str],
    joint_names: Sequence[str],
) -> list[list[float]]:
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in cspace path: {missing}")
    return [
        [float(q[name_to_idx[name]]) for name in joint_names]
        for q in full_path
    ]


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


def _joint_delta_l2(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if not a or not b:
        return float("inf")
    count = min(len(a), len(b))
    if count <= 0:
        return float("inf")
    return math.sqrt(
        sum(
            _wrapped_joint_delta(float(a[idx]), float(b[idx]))
            * _wrapped_joint_delta(float(a[idx]), float(b[idx]))
            for idx in range(count)
        )
    )


def _xyz_distance(a: Sequence[float], b: Sequence[float]) -> float:
    count = min(len(a), len(b))
    if count <= 0:
        return float("inf")
    return math.sqrt(
        sum(
            (float(a[idx]) - float(b[idx])) * (float(a[idx]) - float(b[idx]))
            for idx in range(count)
        )
    )


def _quaternion_angular_distance_xyzw(
    a_xyzw: Sequence[float],
    b_xyzw: Sequence[float],
) -> float:
    if len(a_xyzw) < 4 or len(b_xyzw) < 4:
        return float("inf")
    dot = sum(float(a_xyzw[idx]) * float(b_xyzw[idx]) for idx in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _iter_pose_chunks(
    poses: Sequence[Pose],
    sources: Sequence[str],
    *,
    chunk_size: int,
):
    for start_idx in range(0, len(poses), max(1, int(chunk_size))):
        end_idx = min(len(poses), start_idx + max(1, int(chunk_size)))
        yield start_idx, poses[start_idx:end_idx], sources[start_idx:end_idx]


def _is_zero_joint_goal(
    q_cspace: Sequence[float],
    cspace_joint_names: Sequence[str],
    *,
    arm: str,
    tol: float = _ZERO_JOINT_TOL,
) -> bool:
    arm_joint_names = _arm_joint_names(normalize_arm_name(arm))
    arm_joint_values = _extract_joint_positions(
        q_cspace,
        cspace_joint_names,
        arm_joint_names,
    )
    return all(abs(float(v)) <= float(tol) for v in arm_joint_values)


def _retry_attempt_limit(max_retries: int) -> int | None:
    retries = int(max_retries)
    if retries <= 0:
        return None
    return max(1, retries)


def _format_attempt(attempt_idx: int, max_attempts: int | None) -> str:
    if max_attempts is None:
        return f"{attempt_idx + 1}/inf"
    return f"{attempt_idx + 1}/{max_attempts}"


@dataclass
class AlignExecutionRecord:
    selected_arm: str
    other_arm: str
    shelf_type: str
    world_yml: str | None
    target_pose: Pose
    marker_position: Point
    q_start_cspace: list[float]
    full_path: list[list[float]]
    cspace_joint_names: list[str]


@dataclass
class GraspExecutionRecord:
    selected_arm: str
    world_yml: str | None
    object_pose: Pose
    object_size: Vector3
    grasp_pose: Pose
    lift_pose: Pose
    retreat_pose: Pose


@dataclass
class GraspIKSelection:
    pose: Pose
    q_goal_cspace: list[float]
    source: str
    candidate_index: int
    seed_index: int
    score: float
    align_position_distance_m: float
    joint_distance: float
    orientation_distance_rad: float


class ArmPickingCoordinator(Node):
    def __init__(self, args) -> None:
        super().__init__("arm_picking_align_server")
        self._args = args
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._state_lock = threading.Lock()
        self._sequence_active = False
        self._grasp_cv = threading.Condition()
        self._latest_grasp_msg: ObjectGrasp | None = None
        self._latest_grasp_seq = 0
        self._gripper_finish_cv = threading.Condition()
        self._latest_gripper_finish_arm = ""
        self._latest_gripper_finish_seq = 0

        self._run_startup_warmup()

        self._sub = self.create_subscription(
            ObjectAlign,
            str(args.object_align_topic),
            self._on_object_align,
            self.qos_cmd,
        )
        self._grasp_sub = self.create_subscription(
            ObjectGrasp,
            str(args.object_grasp_topic),
            self._on_object_grasp,
            self.qos_cmd,
        )
        self._gripper_finish_sub = self.create_subscription(
            String,
            str(args.gripper_finish_topic),
            self._on_gripper_finish,
            self.qos_cmd,
        )
        self._gripper_start_pub = self.create_publisher(String, str(args.gripper_start_topic), 10)
        self._arm_picking_finish_pub = self.create_publisher(
            Bool,
            str(args.arm_picking_finish_topic),
            self.qos_cmd,
        )
        self.get_logger().info(
            "ARM_PICKING align node ready: "
            f"align_sub={args.object_align_topic}, grasp_sub={args.object_grasp_topic}, "
            f"gripper_start_pub={args.gripper_start_topic}, "
            f"gripper_finish_sub={args.gripper_finish_topic}, arm_picking_finish_pub={args.arm_picking_finish_topic}"
        )

    def _set_active(self) -> bool:
        with self._state_lock:
            if self._sequence_active:
                return False
            self._sequence_active = True
            return True

    def _reset_active(self) -> None:
        with self._state_lock:
            self._sequence_active = False

    def _iter_startup_world_ymls(self) -> list[str | None]:
        candidates: list[str | None] = [
            _resolve_world_yml(
                self._args,
                collision_models=_COLLISION_MODELS,
                default_world_yml=None,
            ),
            *[str(path) for path in _COLLISION_MODELS.values()],
            None,
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
            self.get_logger().info("Startup warmup skipped because startup_warmup_iters <= 0")
            return

        worlds = self._iter_startup_world_ymls()
        t0 = time.monotonic()
        self.get_logger().info(
            "Starting ARM_PICKING warmup "
            f"(iters={warmup_iters}, batch_size={warmup_batch_size}, worlds={len(worlds)})"
        )
        for world_index, world_yml in enumerate(worlds):
            world_label = str(world_yml) if world_yml is not None else "none"
            self.get_logger().info(f"[warmup] world={world_label}")
            for arm_index, arm in enumerate(("left", "right")):
                ik = get_single_arm_ik(
                    self._args.robot_yml,
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
                self.get_logger().info(f"[warmup] ready arm={arm} world={world_label}")
        self.get_logger().info(
            f"ARM_PICKING warmup completed in {time.monotonic() - t0:.2f}s"
        )

    def _on_object_align(self, msg: ObjectAlign) -> None:
        if not self._set_active():
            self.get_logger().warning("Ignoring ObjectAlign: sequence already active")
            return

        worker = threading.Thread(
            target=self._execute_align,
            args=(msg,),
            daemon=True,
        )
        worker.start()

    def _on_object_grasp(self, msg: ObjectGrasp) -> None:
        label = str(getattr(msg, "label", "")).strip()
        try:
            arm = normalize_arm_name(msg.selected_arm)
        except ValueError:
            self.get_logger().warning(
                "Ignoring ObjectGrasp with invalid selected_arm="
                f"{msg.selected_arm!r} label={label!r}"
            )
            return

        with self._grasp_cv:
            self._latest_grasp_msg = msg
            self._latest_grasp_seq += 1
            self._grasp_cv.notify_all()
        self.get_logger().info(
            f"Received ObjectGrasp for arm={arm} label={label!r}"
        )

    def _on_gripper_finish(self, msg: String) -> None:
        arm_raw = str(msg.data).strip().lower()
        try:
            arm = normalize_arm_name(arm_raw)
        except ValueError:
            self.get_logger().warning(f"Ignoring gripper_finish with invalid arm={arm_raw!r}")
            return

        with self._gripper_finish_cv:
            self._latest_gripper_finish_arm = arm
            self._latest_gripper_finish_seq += 1
            self._gripper_finish_cv.notify_all()
        self.get_logger().info(f"Received gripper_finish for arm={arm}")

    def _resolve_world_yml_from_msg(self, msg: ObjectAlign) -> str | None:
        shelf_type = str(msg.shelf_type).strip()
        if shelf_type:
            mapped = _COLLISION_MODELS.get(shelf_type)
            if mapped is not None:
                return str(mapped)
            self.get_logger().warning(
                f"Unknown shelf_type '{shelf_type}'; falling back to CLI collision settings"
            )

        return _resolve_world_yml(
            self._args,
            collision_models=_COLLISION_MODELS,
            default_world_yml=None,
        )

    def _resolve_grasp_q_start_cspace(
        self,
        *,
        arm: str,
        world_yml: str | None,
    ) -> list[float]:
        normalized_arm = normalize_arm_name(arm)
        ik = get_single_arm_ik(
            self._args.robot_yml,
            arm=normalized_arm,
            cpu=bool(self._args.cpu),
            use_cuda_graph=False,
            world_yml=world_yml,
        )
        # IK retries and downstream planning should use the actual post-align robot state.
        return read_joint_positions_once(
            list(ik.cspace_joint_names),
            topic=str(self._args.joint_state_topic),
            wait_s=float(self._args.joint_state_wait_s),
        )

    def _collect_grasp_candidate_poses(
        self,
        msg: ObjectGrasp,
    ) -> tuple[list[Pose], list[str]]:
        grasp_points = list(getattr(msg, "grasp_points", []))
        grasp_poses = list(getattr(msg, "grasp_poses", []))
        if not grasp_points or not grasp_poses:
            return [], []

        candidate_count = min(len(grasp_points), len(grasp_poses))
        if len(grasp_points) != len(grasp_poses):
            self.get_logger().warning(
                "[GRASP IK] grasp_points/grasp_poses length mismatch: "
                f"points={len(grasp_points)} poses={len(grasp_poses)}; "
                f"using first {candidate_count} paired candidates."
            )

        candidate_poses: list[Pose] = []
        candidate_sources: list[str] = []
        for idx in range(candidate_count):
            candidate_poses.append(
                _build_grasp_candidate_pose(grasp_points[idx], grasp_poses[idx])
            )
            candidate_sources.append(f"grasp_candidate[{idx}]")
        return candidate_poses, candidate_sources

    def _compute_align_ee_position_xyz(
        self,
        *,
        ik,
        q_cspace: Sequence[float],
    ) -> list[float]:
        import torch

        q_active = _extract_joint_positions(
            q_cspace,
            ik.cspace_joint_names,
            ik.active_joint_names,
        )
        kin = ik.solver.fk(
            torch.tensor(
                [q_active],
                device=ik.device,
                dtype=torch.float32,
            )
        )
        return [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]

    def _score_grasp_ik_solution(
        self,
        *,
        arm: str,
        cspace_joint_names: Sequence[str],
        pose: Pose,
        q_start_cspace: Sequence[float],
        q_goal_cspace: Sequence[float],
        align_ee_xyz: Sequence[float],
    ) -> tuple[float, float, float, float]:
        normalized_arm = normalize_arm_name(arm)
        arm_joint_names = _arm_joint_names(normalized_arm)
        start_arm_q = _extract_joint_positions(
            q_start_cspace,
            cspace_joint_names,
            arm_joint_names,
        )
        goal_arm_q = _extract_joint_positions(
            q_goal_cspace,
            cspace_joint_names,
            arm_joint_names,
        )
        joint_distance = _joint_delta_l2(start_arm_q, goal_arm_q)
        align_distance_m = _xyz_distance(align_ee_xyz, _pose_position_xyz(pose))
        orientation_distance_rad = _quaternion_angular_distance_xyzw(
            _pose_orientation_xyzw(pose),
            list(_PREFERRED_GRASP_QUATERNION_XYZW),
        )
        # Prefer IKs that stay close to the post-align state and match the preferred grasp orientation.
        score = -(
            float(joint_distance)
            + (10.0 * float(align_distance_m))
            + float(orientation_distance_rad)
        )
        return (
            float(score),
            float(align_distance_m),
            float(joint_distance),
            float(orientation_distance_rad),
        )

    def _select_best_grasp_ik(
        self,
        *,
        arm: str,
        candidate_poses: Sequence[Pose],
        candidate_sources: Sequence[str],
        world_yml: str | None,
        q_start_cspace: Sequence[float],
    ) -> tuple[GraspIKSelection | None, int, int, int]:
        import torch

        from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
            _build_ik_seed_batch,
        )
        from capstone_pkg.utils.joint_limit import load_joint_limits_torch

        if len(candidate_poses) != len(candidate_sources):
            raise ValueError("candidate_poses and candidate_sources length mismatch")
        if not candidate_poses:
            return None, 0, 0, 0

        normalized_arm = normalize_arm_name(arm)
        q_start_list = [float(v) for v in q_start_cspace]
        ik = get_single_arm_ik(
            self._args.robot_yml,
            arm=normalized_arm,
            cpu=bool(self._args.cpu),
            use_cuda_graph=False,
            world_yml=world_yml,
        )
        joint_limits = load_joint_limits_torch(
            str(self._args.joint_limit_yml),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        seed_batch = _build_ik_seed_batch(
            q_start_list,
            batch_size=max(1, int(self._args.ik_batch)),
            noise_std=float(self._args.ik_seed_noise_std),
            random_seed=int(self._args.ik_seed),
            lower=joint_limits.lower.detach().cpu().numpy(),
            upper=joint_limits.upper.detach().cpu().numpy(),
        )
        align_ee_xyz = self._compute_align_ee_position_xyz(
            ik=ik,
            q_cspace=q_start_list,
        )
        arm_joint_names = _arm_joint_names(normalized_arm)
        start_arm_q = _extract_joint_positions(
            q_start_list,
            ik.cspace_joint_names,
            arm_joint_names,
        )

        solve_call_count = 0
        success_count = 0  # successful IK solutions across all seeds/candidates
        candidate_best: dict[int, tuple[float, GraspIKSelection]] = {}
        candidate_stats = [
            {
                "source": str(candidate_sources[idx]),
                "pose": _copy_pose(candidate_poses[idx]),
                "raw_successes": 0,
                "accepted_successes": 0,
                "zero_rejected": 0,
            }
            for idx in range(len(candidate_poses))
        ]
        for chunk_start, pose_chunk, source_chunk in _iter_pose_chunks(
            candidate_poses,
            candidate_sources,
            chunk_size=_GRASP_IK_CANDIDATE_BATCH_SIZE,
        ):
            xyz_batch: list[list[float]] = []
            quat_batch: list[list[float]] = []
            q_seed_batch: list[list[float]] = []
            batch_meta: list[tuple[int, int, Pose, str]] = []
            for seed_index, seed_q in enumerate(seed_batch):
                seed_q_list = [float(v) for v in seed_q]
                for local_idx, pose in enumerate(pose_chunk):
                    candidate_index = chunk_start + local_idx
                    xyz_batch.append(_pose_position_xyz(pose))
                    quat_batch.append(_pose_orientation_wxyz(pose))
                    q_seed_batch.append(list(seed_q_list))
                    batch_meta.append(
                        (
                            int(candidate_index),
                            int(seed_index),
                            pose,
                            str(source_chunk[local_idx]),
                        )
                    )

            solve_call_count += 1
            ik_outs = ik.solve_batch(
                xyz_batch,
                quat_batch,
                q_start_cspace=q_start_list,
                q_seed_cspace_batch=q_seed_batch,
            )
            for ik_out, (candidate_index, seed_index, pose, source) in zip(ik_outs, batch_meta):
                if not ik_out.success or ik_out.q_cspace is None:
                    continue
                candidate_stats[candidate_index]["raw_successes"] += 1
                if _is_zero_joint_goal(
                    ik_out.q_cspace,
                    ik.cspace_joint_names,
                    arm=normalized_arm,
                ):
                    candidate_stats[candidate_index]["zero_rejected"] += 1
                    self.get_logger().warning(
                        "[GRASP IK] rejected zero-joint IK solution: "
                        f"source={source} candidate_index={candidate_index} seed={seed_index}"
                    )
                    continue
                success_count += 1
                candidate_stats[candidate_index]["accepted_successes"] += 1
                goal_arm_q = _extract_joint_positions(
                    ik_out.q_cspace,
                    ik.cspace_joint_names,
                    arm_joint_names,
                )
                current_distance_l2 = _joint_delta_l2(start_arm_q, goal_arm_q)
                score, align_distance_m, joint_distance, orientation_distance_rad = self._score_grasp_ik_solution(
                    arm=normalized_arm,
                    cspace_joint_names=ik.cspace_joint_names,
                    pose=pose,
                    q_start_cspace=q_start_list,
                    q_goal_cspace=ik_out.q_cspace,
                    align_ee_xyz=align_ee_xyz,
                )
                selection = GraspIKSelection(
                    pose=_copy_pose(pose),
                    q_goal_cspace=[float(v) for v in ik_out.q_cspace],
                    source=source,
                    candidate_index=int(candidate_index),
                    seed_index=int(seed_index),
                    score=float(score),
                    align_position_distance_m=float(align_distance_m),
                    joint_distance=float(joint_distance),
                    orientation_distance_rad=float(orientation_distance_rad),
                )
                prev_entry = candidate_best.get(candidate_index)
                if (
                    prev_entry is None
                    or current_distance_l2 < prev_entry[0]
                    or (
                        math.isclose(current_distance_l2, prev_entry[0])
                        and selection.score > prev_entry[1].score
                    )
                ):
                    candidate_best[candidate_index] = (float(current_distance_l2), selection)

        total_seed_count = max(1, int(self._args.ik_batch))
        for candidate_index, stats in enumerate(candidate_stats):
            raw_successes = int(stats["raw_successes"])
            accepted_successes = int(stats["accepted_successes"])
            zero_rejected = int(stats["zero_rejected"])
            failures = max(0, total_seed_count - raw_successes)
            pose = stats["pose"]
            self.get_logger().info(
                "[GRASP IK] candidate summary: "
                f"source={stats['source']} candidate_index={candidate_index} "
                f"target_xyz={_pose_position_xyz(pose)} "
                f"target_quat_xyzw={_pose_orientation_xyzw(pose)} "
                f"raw_successes={raw_successes} accepted_successes={accepted_successes} "
                f"failures={failures} zero_rejected={zero_rejected}"
            )

        if not candidate_best:
            return None, solve_call_count, success_count, 0

        best_selection = max(
            (entry[1] for entry in candidate_best.values()),
            key=lambda selection: float(selection.score),
        )
        return best_selection, solve_call_count, success_count, len(candidate_best)

    def _resolve_grasp_target_selection(
        self,
        *,
        arm: str,
        grasp_msg: ObjectGrasp,
        world_yml: str | None,
    ) -> tuple[GraspIKSelection, list[float]]:
        q_start_cspace = self._resolve_grasp_q_start_cspace(
            arm=arm,
            world_yml=world_yml,
        )
        candidate_poses, candidate_sources = self._collect_grasp_candidate_poses(grasp_msg)

        if candidate_poses:
            selection, solve_call_count, success_count, candidate_success_count = self._select_best_grasp_ik(
                arm=arm,
                candidate_poses=candidate_poses,
                candidate_sources=candidate_sources,
                world_yml=world_yml,
                q_start_cspace=q_start_cspace,
            )
            self.get_logger().info(
                "[GRASP IK] evaluated grasp candidates in GPU batches: "
                f"arm={normalize_arm_name(arm)} candidates={len(candidate_poses)} "
                f"candidate_batch={_GRASP_IK_CANDIDATE_BATCH_SIZE} "
                f"seed_batch={max(1, int(self._args.ik_batch))} "
                f"solve_calls={solve_call_count} raw_successes={success_count} "
                f"candidate_successes={candidate_success_count}"
            )
            if selection is not None:
                self.get_logger().info(
                    "[GRASP IK] selected candidate after nearest-per-target IK filtering: "
                    f"source={selection.source} candidate_index={selection.candidate_index} "
                    f"seed={selection.seed_index} score={selection.score:.6f} "
                    f"align_distance_m={selection.align_position_distance_m:.4f} "
                    f"joint_distance={selection.joint_distance:.4f} "
                    f"orientation_distance_rad={selection.orientation_distance_rad:.4f} "
                    f"target_xyz={_pose_position_xyz(selection.pose)} "
                    f"target_quat_xyzw={_pose_orientation_xyzw(selection.pose)}"
                )
                return selection, q_start_cspace
            self.get_logger().warning(
                "[GRASP IK] no valid IK from grasp candidates; falling back to object center pose."
            )
        else:
            self.get_logger().warning(
                "[GRASP IK] no grasp candidates in message; falling back to object center pose."
            )

        fallback_attempts = (
            ("object_center", _build_object_center_fallback_pose(grasp_msg)),
            (
                f"object_center_z_plus_{_GRASP_FALLBACK_Z_OFFSET_M:.3f}",
                _build_object_center_fallback_pose(
                    grasp_msg,
                    z_offset_m=float(_GRASP_FALLBACK_Z_OFFSET_M),
                ),
            ),
        )
        for source, pose in fallback_attempts:
            selection, solve_call_count, success_count, candidate_success_count = self._select_best_grasp_ik(
                arm=arm,
                candidate_poses=[pose],
                candidate_sources=[source],
                world_yml=world_yml,
                q_start_cspace=q_start_cspace,
            )
            self.get_logger().info(
                "[GRASP IK] fallback attempt: "
                f"source={source} solve_calls={solve_call_count} "
                f"raw_successes={success_count} candidate_successes={candidate_success_count} "
                f"target_xyz={_pose_position_xyz(pose)}"
            )
            if selection is not None:
                self.get_logger().info(
                    "[GRASP IK] fallback selected target: "
                    f"source={selection.source} seed={selection.seed_index} "
                    f"score={selection.score:.6f}"
                )
                return selection, q_start_cspace

        raise RuntimeError("IK failed for all grasp candidates and object-center fallback targets")

    def _plan_selected_arm(
        self,
        *,
        stage: str,
        arm: str,
        target_pose: Pose,
        world_yml: str | None,
        q_start_cspace: Sequence[float] | None = None,
    ) -> SingleArmMotionPlan:
        t_start = time.perf_counter()
        plan = plan_single_arm_motion(
            robot_yml=self._args.robot_yml,
            arm=arm,
            target_xyz=_pose_position_xyz(target_pose),
            target_quat_xyzw=_pose_orientation_xyzw(target_pose),
            world_yml=world_yml,
            cpu=bool(self._args.cpu),
            joint_state_topic=self._args.joint_state_topic,
            joint_state_wait_s=float(self._args.joint_state_wait_s),
            use_current_joint_state_start=bool(self._args.use_current_joint_state_start),
            q_start_cspace=q_start_cspace,
            step=float(self._args.step),
            max_iters=int(self._args.max_iters),
            goal_bias=float(self._args.goal_bias),
            connect_threshold=float(self._args.connect_threshold),
            planner_backend=normalize_single_arm_planner_backend(self._args.planner_backend),
            joint_limit_yml=str(self._args.joint_limit_yml),
            ik_batch=int(self._args.ik_batch),
            ik_seed_noise_std=float(self._args.ik_seed_noise_std),
            ik_seed_random_seed=int(self._args.ik_seed),
            ik_goal_dedupe_tol=float(self._args.ik_goal_dedupe_tol),
            tbrrt_cfg=build_single_arm_tbrrt_config(self._args),
            tbrrt_block_k=int(self._args.tbrrt_block_k),
            spline_dt=max(0.001, float(self._args.publish_dt)),
        )
        planning_time_s = time.perf_counter() - t_start
        self.get_logger().info(
            "[TRAJ] trajectory planned: "
            f"stage={stage} arm={normalize_arm_name(arm)} waypoints={len(plan.spline_path)} "
            f"planning_time={planning_time_s:.3f}s "
            f"target_xyz={_pose_position_xyz(target_pose)}"
        )
        return plan

    def _plan_other_arm_zero(
        self,
        *,
        stage: str,
        arm: str,
        q_start_cspace: Sequence[float],
        cspace_joint_names: Sequence[str],
        world_yml: str | None,
    ) -> SingleArmMotionPlan:
        from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
            plan_single_arm_tbrrt_batch_conext,
        )

        normalized_arm = normalize_arm_name(arm)
        q_goal_cspace = _build_zero_goal_cspace(
            q_start_cspace=q_start_cspace,
            cspace_joint_names=cspace_joint_names,
            arm=normalized_arm,
        )
        t_start = time.perf_counter()
        out = plan_single_arm_tbrrt_batch_conext(
            robot_yml=self._args.robot_yml,
            arm=normalized_arm,
            q_start=q_start_cspace,
            q_goals=[q_goal_cspace],
            world_yml=world_yml,
            cpu=bool(self._args.cpu),
            cfg=build_single_arm_tbrrt_config(self._args),
            joint_limit_yml=str(self._args.joint_limit_yml),
            block_k=int(self._args.tbrrt_block_k),
        )
        if not out.success or not out.path:
            raise RuntimeError(f"Failed to plan zero-goal path for {normalized_arm} arm: {out.stats.extra}")

        spline_path = [[float(v) for v in q] for q in out.path]
        plan = SingleArmMotionPlan(
            arm=normalized_arm,
            cspace_joint_names=[str(name) for name in cspace_joint_names],
            active_joint_names=_arm_joint_names(normalized_arm),
            q_start_cspace=[float(v) for v in q_start_cspace],
            q_goal_cspace=[float(v) for v in q_goal_cspace],
            raw_path=[list(q) for q in spline_path],
            spline_path=[list(q) for q in spline_path],
        )
        planning_time_s = time.perf_counter() - t_start
        self.get_logger().info(
            "[TRAJ] trajectory planned: "
            f"stage={stage} arm={normalized_arm} waypoints={len(plan.spline_path)} "
            f"planning_time={planning_time_s:.3f}s goal=zero_pose"
        )
        return plan

    def _plan_selected_arm_to_q_goal(
        self,
        *,
        stage: str,
        arm: str,
        q_start_cspace: Sequence[float],
        q_goal_cspace: Sequence[float],
        world_yml: str | None,
    ) -> SingleArmMotionPlan:
        from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
            plan_single_arm_tbrrt_batch_conext,
        )

        normalized_arm = normalize_arm_name(arm)
        q_start_list = [float(v) for v in q_start_cspace]
        q_goal_list = [float(v) for v in q_goal_cspace]
        t_start = time.perf_counter()
        out = plan_single_arm_tbrrt_batch_conext(
            robot_yml=self._args.robot_yml,
            arm=normalized_arm,
            q_start=q_start_list,
            q_goals=[q_goal_list],
            world_yml=world_yml,
            cpu=bool(self._args.cpu),
            cfg=build_single_arm_tbrrt_config(self._args),
            joint_limit_yml=str(self._args.joint_limit_yml),
            block_k=int(self._args.tbrrt_block_k),
        )
        if not out.success or not out.path:
            raise RuntimeError(
                f"Failed to plan joint-goal path for {normalized_arm} arm: {out.stats.extra}"
            )

        ik = get_single_arm_ik(
            self._args.robot_yml,
            arm=normalized_arm,
            cpu=bool(self._args.cpu),
            world_yml=world_yml,
        )
        spline_path = [[float(v) for v in q] for q in out.path]
        plan = SingleArmMotionPlan(
            arm=normalized_arm,
            cspace_joint_names=list(ik.cspace_joint_names),
            active_joint_names=_arm_joint_names(normalized_arm),
            q_start_cspace=q_start_list,
            q_goal_cspace=q_goal_list,
            raw_path=[list(q) for q in spline_path],
            spline_path=[list(q) for q in spline_path],
        )
        planning_time_s = time.perf_counter() - t_start
        self.get_logger().info(
            "[TRAJ] joint-goal trajectory planned: "
            f"stage={stage} arm={normalized_arm} waypoints={len(plan.spline_path)} "
            f"planning_time={planning_time_s:.3f}s"
        )
        return plan

    def _plan_selected_arm_fixed_ee_z(
        self,
        *,
        stage: str,
        arm: str,
        target_pose: Pose,
        world_yml: str | None,
        q_start_cspace: Sequence[float],
    ) -> SingleArmMotionPlan:
        import torch

        from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
            _build_ik_seed_batch,
            _dedupe_q_candidates,
        )
        from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
            plan_single_arm_tbrrt_batch_conext_fixed_ee_z,
        )
        from capstone_pkg.utils.joint_limit import load_joint_limits_torch

        normalized_arm = normalize_arm_name(arm)
        t_start = time.perf_counter()
        ik = get_single_arm_ik(
            self._args.robot_yml,
            arm=normalized_arm,
            cpu=bool(self._args.cpu),
            world_yml=world_yml,
        )
        q_start_list = [float(v) for v in q_start_cspace]
        joint_limits = load_joint_limits_torch(
            str(self._args.joint_limit_yml),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        ik_seed_batch = _build_ik_seed_batch(
            q_start_list,
            batch_size=max(1, int(self._args.ik_batch)),
            noise_std=float(self._args.ik_seed_noise_std),
            random_seed=int(self._args.ik_seed),
            lower=joint_limits.lower.detach().cpu().numpy(),
            upper=joint_limits.upper.detach().cpu().numpy(),
        )
        target_xyz = _pose_position_xyz(target_pose)
        target_quat_wxyz = _pose_orientation_wxyz(target_pose)
        ik_outs = ik.solve_batch(
            [list(target_xyz) for _ in range(len(ik_seed_batch))],
            [list(target_quat_wxyz) for _ in range(len(ik_seed_batch))],
            q_start_cspace=q_start_list,
            q_seed_cspace_batch=ik_seed_batch,
        )
        cand_q = [
            list(out.q_cspace)
            for out in ik_outs
            if (
                out.success
                and out.q_cspace is not None
                and not _is_zero_joint_goal(
                    out.q_cspace,
                    ik.cspace_joint_names,
                    arm=normalized_arm,
                )
            )
        ]
        cand_q = _dedupe_q_candidates(
            cand_q,
            atol=float(self._args.ik_goal_dedupe_tol),
        )
        if not cand_q:
            raise RuntimeError("IK failed for fixed-z retreat target pose")

        out = plan_single_arm_tbrrt_batch_conext_fixed_ee_z(
            robot_yml=self._args.robot_yml,
            arm=normalized_arm,
            q_start=q_start_list,
            q_goals=cand_q,
            world_yml=world_yml,
            cpu=bool(self._args.cpu),
            cfg=build_single_arm_tbrrt_config(self._args),
            joint_limit_yml=str(self._args.joint_limit_yml),
            block_k=int(self._args.tbrrt_block_k),
        )
        if not out.success or not out.path:
            raise RuntimeError(
                f"Failed to plan fixed-z retreat path for {normalized_arm} arm: {out.stats.extra}"
            )

        spline_path = [[float(v) for v in q] for q in out.path]
        plan = SingleArmMotionPlan(
            arm=normalized_arm,
            cspace_joint_names=list(ik.cspace_joint_names),
            active_joint_names=_arm_joint_names(normalized_arm),
            q_start_cspace=[float(v) for v in q_start_list],
            q_goal_cspace=[float(v) for v in spline_path[-1]],
            raw_path=[list(q) for q in spline_path],
            spline_path=[list(q) for q in spline_path],
        )
        planning_time_s = time.perf_counter() - t_start
        self.get_logger().info(
            "[TRAJ] fixed-z retreat planned: "
            f"stage={stage} arm={normalized_arm} waypoints={len(plan.spline_path)} "
            f"planning_time={planning_time_s:.3f}s target_xyz={target_xyz}"
        )
        return plan

    def _publish_single_arm_joint_path(
        self,
        *,
        arm: str,
        joint_names: Sequence[str],
        joint_path: Sequence[Sequence[float]],
    ) -> None:
        if self._args.publish_mode == "real":
            if bool(self._args.real_use_action):
                command = JointTrajectoryCommand(
                    endpoint=(
                        str(self._args.real_left_action)
                        if arm == "left"
                        else str(self._args.real_right_action)
                    ),
                    joint_names=joint_names,
                    path=joint_path,
                    label=f"{arm}_arm",
                )
                try:
                    send_joint_trajectory_action_group(
                        [command],
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                        start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            command = JointTrajectoryCommand(
                endpoint=(
                    str(self._args.real_left_topic)
                    if arm == "left"
                    else str(self._args.real_right_topic)
                ),
                joint_names=joint_names,
                path=joint_path,
                label=f"{arm}_arm",
            )
            publish_joint_trajectory_group(
                [command],
                dt=float(self._args.publish_dt),
                wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
                require_subscriber=bool(self._args.publish_require_subscriber),
                retry_until_subscriber=bool(self._args.publish_retry_until_subscriber),
                publish_repeat=int(self._args.publish_repeat),
                publish_period_s=float(self._args.publish_period_s),
                wait_ack_s=float(self._args.publish_wait_ack_s),
                keep_alive_s=float(self._args.publish_keep_alive_s),
                reliability=str(getattr(self._args, "publish_reliability", "best_effort")),
                durability=(
                    "transient_local"
                    if bool(getattr(self._args, "publish_transient_local", False))
                    else str(getattr(self._args, "publish_durability", "volatile"))
                ),
                qos_depth=int(getattr(self._args, "publish_qos_depth", 1)),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        publish_joint_path(
            joint_path,
            joint_names,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )

    def _publish_single_arm_motion(
        self,
        *,
        plan: SingleArmMotionPlan,
    ) -> None:
        joint_names = _arm_joint_names(plan.arm)
        joint_path = _build_active_joint_path(
            plan.spline_path,
            plan.cspace_joint_names,
            joint_names,
        )
        self._publish_single_arm_joint_path(
            arm=plan.arm,
            joint_names=joint_names,
            joint_path=joint_path,
        )

    def _wait_for_single_arm(
        self,
        *,
        stage: str,
        plan: SingleArmMotionPlan,
    ) -> None:
        joint_names = _arm_joint_names(plan.arm)
        active_path = _build_active_joint_path(
            plan.spline_path,
            plan.cspace_joint_names,
            joint_names,
        )
        goal = [float(v) for v in active_path[-1]]
        max_attempts = _retry_attempt_limit(int(getattr(self._args, "arrival_max_retries", -1)))
        last_err = float("inf")
        attempt_idx = 0
        while max_attempts is None or attempt_idx < max_attempts:
            cmd_path = active_path
            if attempt_idx > 0:
                try:
                    current_positions = read_joint_positions_once(
                        joint_names,
                        topic=str(self._args.joint_state_topic),
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
                    f"stage={stage} arm={plan.arm} waypoints={len(cmd_path)} "
                    f"goal={goal}"
                )
            try:
                self._publish_single_arm_joint_path(
                    arm=plan.arm,
                    joint_names=joint_names,
                    joint_path=cmd_path,
                )
            except RuntimeError as exc:
                attempt_idx += 1
                if max_attempts is not None and attempt_idx >= max_attempts:
                    raise RuntimeError(
                        f"Failed to publish {plan.arm} arm command after {attempt_idx} attempt(s): {exc}"
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
            arrived, _current_positions, max_abs_err = wait_for_joint_positions(
                joint_names,
                goal,
                topic=str(self._args.joint_state_topic),
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

    def _wait_for_object_grasp(
        self,
        *,
        arm: str,
        min_seq: int,
        timeout_s: float,
    ) -> tuple[ObjectGrasp, int]:
        normalized_arm = normalize_arm_name(arm)
        deadline = None if timeout_s < 0.0 else time.monotonic() + float(timeout_s)
        with self._grasp_cv:
            while True:
                msg = self._latest_grasp_msg
                seq = int(self._latest_grasp_seq)
                if (
                    seq > int(min_seq)
                    and msg is not None
                    and normalize_arm_name(msg.selected_arm) == normalized_arm
                ):
                    return msg, seq
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(f"Timed out waiting for ObjectGrasp for arm={normalized_arm}")
                    self._grasp_cv.wait(timeout=remaining)
                else:
                    self._grasp_cv.wait()

    def _wait_for_gripper_finish(
        self,
        *,
        arm: str,
        min_seq: int,
        timeout_s: float,
    ) -> None:
        normalized_arm = normalize_arm_name(arm)
        deadline = None if timeout_s < 0.0 else time.monotonic() + float(timeout_s)
        with self._gripper_finish_cv:
            while True:
                if (
                    self._latest_gripper_finish_seq > int(min_seq)
                    and self._latest_gripper_finish_arm == normalized_arm
                ):
                    return
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(f"Timed out waiting for gripper_finish for arm={normalized_arm}")
                    self._gripper_finish_cv.wait(timeout=remaining)
                else:
                    self._gripper_finish_cv.wait()

    def _publish_motion_commands(
        self,
        *,
        plan: SingleArmMotionPlan,
        full_path: Sequence[Sequence[float]],
        selected_arm: str,
        other_arm: str,
    ) -> None:
        selected_joint_names = _arm_joint_names(selected_arm)
        other_joint_names = _arm_joint_names(other_arm)
        selected_path = _build_active_joint_path(
            full_path,
            plan.cspace_joint_names,
            selected_joint_names,
        )
        other_path = _build_active_joint_path(
            full_path,
            plan.cspace_joint_names,
            other_joint_names,
        )

        if self._args.publish_mode == "real":
            if bool(self._args.real_use_action):
                commands = [
                    JointTrajectoryCommand(
                        endpoint=(
                            str(self._args.real_left_action)
                            if selected_arm == "left"
                            else str(self._args.real_right_action)
                        ),
                        joint_names=selected_joint_names,
                        path=selected_path,
                        label=f"{selected_arm}_arm",
                    ),
                    JointTrajectoryCommand(
                        endpoint=(
                            str(self._args.real_left_action)
                            if other_arm == "left"
                            else str(self._args.real_right_action)
                        ),
                        joint_names=other_joint_names,
                        path=other_path,
                        label=f"{other_arm}_arm",
                    ),
                ]
                try:
                    send_joint_trajectory_action_group(
                        commands,
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                        start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            commands = [
                JointTrajectoryCommand(
                    endpoint=(
                        str(self._args.real_left_topic)
                        if selected_arm == "left"
                        else str(self._args.real_right_topic)
                    ),
                    joint_names=selected_joint_names,
                    path=selected_path,
                    label=f"{selected_arm}_arm",
                ),
                JointTrajectoryCommand(
                    endpoint=(
                        str(self._args.real_left_topic)
                        if other_arm == "left"
                        else str(self._args.real_right_topic)
                    ),
                    joint_names=other_joint_names,
                    path=other_path,
                    label=f"{other_arm}_arm",
                ),
            ]
            publish_joint_trajectory_group(
                commands,
                dt=float(self._args.publish_dt),
                wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
                require_subscriber=bool(self._args.publish_require_subscriber),
                retry_until_subscriber=bool(self._args.publish_retry_until_subscriber),
                publish_repeat=int(self._args.publish_repeat),
                publish_period_s=float(self._args.publish_period_s),
                wait_ack_s=float(self._args.publish_wait_ack_s),
                keep_alive_s=float(self._args.publish_keep_alive_s),
                reliability=str(getattr(self._args, "publish_reliability", "best_effort")),
                durability=(
                    "transient_local"
                    if bool(getattr(self._args, "publish_transient_local", False))
                    else str(getattr(self._args, "publish_durability", "volatile"))
                ),
                qos_depth=int(getattr(self._args, "publish_qos_depth", 1)),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        publish_joint_path(
            full_path,
            plan.cspace_joint_names,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )

    def _wait_for_both_arms(
        self,
        *,
        stage: str,
        plan: SingleArmMotionPlan,
        full_path: Sequence[Sequence[float]],
        selected_arm: str,
        other_arm: str,
    ) -> None:
        selected_joint_names = _arm_joint_names(selected_arm)
        other_joint_names = _arm_joint_names(other_arm)
        selected_path = _build_active_joint_path(
            full_path,
            plan.cspace_joint_names,
            selected_joint_names,
        )
        other_path = _build_active_joint_path(
            full_path,
            plan.cspace_joint_names,
            other_joint_names,
        )
        selected_goal = _build_active_joint_path(
            [full_path[-1]],
            plan.cspace_joint_names,
            selected_joint_names,
        )[0]
        other_goal = _build_active_joint_path(
            [full_path[-1]],
            plan.cspace_joint_names,
            other_joint_names,
        )[0]
        max_attempts = _retry_attempt_limit(int(getattr(self._args, "arrival_max_retries", -1)))
        attempt_idx = 0
        last_failure = ""
        while max_attempts is None or attempt_idx < max_attempts:
            cmd_full_path = [list(q) for q in full_path]
            if attempt_idx > 0:
                try:
                    selected_current = read_joint_positions_once(
                        selected_joint_names,
                        topic=str(self._args.joint_state_topic),
                        wait_s=float(self._args.joint_state_wait_s),
                    )
                    other_current = read_joint_positions_once(
                        other_joint_names,
                        topic=str(self._args.joint_state_topic),
                        wait_s=float(self._args.joint_state_wait_s),
                    )
                    selected_nearest_idx = _nearest_waypoint_index(selected_current, selected_path)
                    other_nearest_idx = _nearest_waypoint_index(other_current, other_path)
                    cmd_full_path = _combine_active_joint_paths(
                        cspace_joint_names=plan.cspace_joint_names,
                        selected_joint_names=selected_joint_names,
                        selected_path=_build_retry_path(
                            current_positions=selected_current,
                            original_path=selected_path,
                        ),
                        other_joint_names=other_joint_names,
                        other_path=_build_retry_path(
                            current_positions=other_current,
                            original_path=other_path,
                        ),
                    )
                    selected_desc = (
                        "no progress"
                        if selected_nearest_idx == 0
                        else f"waypoint {selected_nearest_idx + 1}/{len(selected_path)}"
                    )
                    other_desc = (
                        "no progress"
                        if other_nearest_idx == 0
                        else f"waypoint {other_nearest_idx + 1}/{len(other_path)}"
                    )
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for dual-arm align: "
                        f"{selected_arm} at {selected_desc}, {other_arm} at {other_desc}."
                    )
                except RuntimeError as exc:
                    self.get_logger().warning(
                        f"[ARRIVAL] retry {_format_attempt(attempt_idx, max_attempts)} for dual-arm align: "
                        f"failed to read current joints ({exc}); re-publishing full path."
                    )
            else:
                self.get_logger().info(
                    "[TRAJ] trajectory ready; publishing: "
                    f"stage={stage} selected_arm={selected_arm} other_arm={other_arm} "
                    f"waypoints={len(cmd_full_path)}"
                )

            try:
                self._publish_motion_commands(
                    plan=plan,
                    full_path=cmd_full_path,
                    selected_arm=selected_arm,
                    other_arm=other_arm,
                )
            except RuntimeError as exc:
                attempt_idx += 1
                if max_attempts is not None and attempt_idx >= max_attempts:
                    raise RuntimeError(
                        f"Failed to publish dual-arm align command after {attempt_idx} attempt(s): {exc}"
                    ) from exc
                self.get_logger().warning(
                    f"[ARRIVAL] publish attempt {_format_attempt(attempt_idx - 1, max_attempts)} "
                    f"failed for dual-arm align: {exc}. Retrying."
                )
                continue

            wait_s = _resolve_arrival_wait_s(
                path_len=len(cmd_full_path),
                dt=float(self._args.publish_dt),
                configured_wait_s=float(getattr(self._args, "arrival_wait_s", -1.0)),
            )
            failures: list[str] = []
            for arm_name, joint_names, goal in (
                (selected_arm, selected_joint_names, selected_goal),
                (other_arm, other_joint_names, other_goal),
            ):
                arrived, _current_positions, max_abs_err = wait_for_joint_positions(
                    joint_names,
                    goal,
                    topic=str(self._args.joint_state_topic),
                    wait_s=wait_s,
                    tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
                    poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
                )
                if not arrived:
                    failures.append(f"{arm_name}: max_abs_err={max_abs_err:.6f}")

            if not failures:
                if attempt_idx > 0:
                    self.get_logger().info("[ARRIVAL] dual-arm align confirmed after retry.")
                return

            last_failure = "; ".join(failures)
            attempt_idx += 1
            self.get_logger().warning(
                f"[ARRIVAL] dual-arm align not confirmed after attempt "
                f"{_format_attempt(attempt_idx - 1, max_attempts)}: {last_failure}. "
                "Re-publishing toward the remaining path."
            )

        raise RuntimeError(
            "Failed to confirm dual-arm align arrival after "
            f"{max_attempts} attempt(s); {last_failure}"
        )

    def _maybe_save_alignment(self, record: AlignExecutionRecord) -> None:
        save_path = str(getattr(self._args, "save", "") or "").strip()
        if not save_path:
            return

        payload = {
            "selected_arm": record.selected_arm,
            "other_arm": record.other_arm,
            "shelf_type": record.shelf_type,
            "world_yml": record.world_yml,
            "target_pose": {
                "position": {
                    "x": float(record.target_pose.position.x),
                    "y": float(record.target_pose.position.y),
                    "z": float(record.target_pose.position.z),
                },
                "orientation": {
                    "x": float(record.target_pose.orientation.x),
                    "y": float(record.target_pose.orientation.y),
                    "z": float(record.target_pose.orientation.z),
                    "w": float(record.target_pose.orientation.w),
                },
            },
            "marker_position": {
                "x": float(record.marker_position.x),
                "y": float(record.marker_position.y),
                "z": float(record.marker_position.z),
            },
            "q_start_cspace": list(record.q_start_cspace),
            "cspace_joint_names": list(record.cspace_joint_names),
            "path": list(record.full_path),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.get_logger().info(f"Saved ARM_PICKING align sequence to {save_path}")

    def _publish_gripper_start(self, arm: str) -> None:
        msg = String()
        msg.data = normalize_arm_name(arm)
        self._gripper_start_pub.publish(msg)
        self.get_logger().info(f"Published gripper_start for arm={msg.data}")

    def _publish_arm_picking_finish(self, arm: str, *, stage: str) -> None:
        msg = Bool()
        msg.data = True
        self._arm_picking_finish_pub.publish(msg)
        self.get_logger().info(
            f"Published arm_picking_finish=True for arm={normalize_arm_name(arm)} stage={stage}"
        )

    def _publish_arm_picking_finish_after_motion_complete(self, arm: str, *, stage: str) -> None:
        self.get_logger().info(
            "[ARM_PICKING] motion completed; publishing arm_picking_finish: "
            f"arm={normalize_arm_name(arm)} stage={stage}"
        )
        self._publish_arm_picking_finish(arm, stage=stage)

    def _execute_grasp_sequence(
        self,
        *,
        selected_arm: str,
        base_world_yml: str | None,
    ) -> None:
        grasp_seq = int(self._latest_grasp_seq)
        grasp_attempt = 0
        while True:
            grasp_msg, grasp_seq = self._wait_for_object_grasp(
                arm=selected_arm,
                min_seq=grasp_seq,
                timeout_s=float(self._args.object_grasp_wait_s),
            )
            try:
                object_pose = _copy_pose(grasp_msg.object_pose)
                object_pose.position.x = (
                    float(object_pose.position.x) + float(_GRASP_OBJECT_POSE_X_OFFSET_M)
                )
                object_size = _copy_vector3(grasp_msg.object_size)
                object_size.x = float(_GRASP_COLLISION_OBJECT_SIZE_X_M)
                object_size.y = float(object_size.y) - float(_GRASP_COLLISION_OBJECT_SIZE_Y_OFFSET_M)
                object_label = str(getattr(grasp_msg, "label", "")).strip()
                object_world_yml = _merge_world_with_user_object(
                    base_world_yml,
                    center_xyz=_pose_position_xyz(object_pose),
                    dims_xyz=_vector3_xyz(object_size),
                    quat_wxyz=_pose_orientation_wxyz(object_pose),
                    object_name=(
                        object_label
                        if object_label
                        else str(getattr(self._args, "grasp_object_name", "grasp_object"))
                    ),
                )
                _publish_world_collision_for_mujoco(self._args, object_world_yml)

                grasp_selection, grasp_q_start_cspace = self._resolve_grasp_target_selection(
                    arm=selected_arm,
                    grasp_msg=grasp_msg,
                    world_yml=object_world_yml,
                )
                grasp_target_pose = _copy_pose(grasp_selection.pose)
                grasp_plan = self._plan_selected_arm_to_q_goal(
                    stage="grasp",
                    arm=selected_arm,
                    q_start_cspace=grasp_q_start_cspace,
                    q_goal_cspace=grasp_selection.q_goal_cspace,
                    world_yml=object_world_yml,
                )
                self._wait_for_single_arm(stage="grasp", plan=grasp_plan)

                gripper_finish_seq = self._latest_gripper_finish_seq
                self._publish_gripper_start(selected_arm)
                self._wait_for_gripper_finish(
                    arm=selected_arm,
                    min_seq=gripper_finish_seq,
                    timeout_s=float(self._args.gripper_finish_wait_s),
                )

                lift_pose = _offset_pose_z(grasp_target_pose, float(self._args.post_grasp_lift_z_m))
                lift_plan = self._plan_selected_arm(
                    stage="lift",
                    arm=selected_arm,
                    target_pose=lift_pose,
                    world_yml=object_world_yml,
                    q_start_cspace=grasp_plan.q_goal_cspace,
                )
                self._wait_for_single_arm(stage="lift", plan=lift_plan)

                retreat_pose = _build_post_grasp_staging_pose(selected_arm, lift_pose)
                retreat_plan = self._plan_selected_arm_fixed_ee_z(
                    stage="retreat",
                    arm=selected_arm,
                    target_pose=retreat_pose,
                    world_yml=object_world_yml,
                    q_start_cspace=lift_plan.q_goal_cspace,
                )
                self._wait_for_single_arm(stage="retreat", plan=retreat_plan)

                self.get_logger().info(
                    "[ARM_PICKING] grasp sequence completed: "
                    f"label={object_label!r} "
                    f"grasp_source={grasp_selection.source} "
                    f"arm={selected_arm} object_center={_pose_position_xyz(object_pose)} "
                    f"object_size={_vector3_xyz(object_size)}"
                )
                self._publish_arm_picking_finish_after_motion_complete(selected_arm, stage="grasp")
                return
            except Exception as exc:
                grasp_attempt += 1
                self.get_logger().warning(
                    "[ARM_PICKING] grasp attempt failed; waiting for next ObjectGrasp: "
                    f"arm={normalize_arm_name(selected_arm)} "
                    f"attempt={grasp_attempt} seq={grasp_seq} error={exc}"
                )

    def _execute_align(self, msg: ObjectAlign) -> None:
        try:
            selected_arm = normalize_arm_name(msg.selected_arm)
            other_arm = "right" if selected_arm == "left" else "left"
            world_yml = self._resolve_world_yml_from_msg(msg)
            target_pose = _build_align_target_pose(
                msg,
                selected_arm=selected_arm,
                fixed_x_m=float(self._args.align_fixed_x_m),
                lift_z_m=float(self._args.align_lift_z_m),
            )

            self.get_logger().info(
                "[ARM_PICKING] start align: "
                f"aruco_id={int(msg.aruco_id)} arm={selected_arm} "
                f"shelf_type='{str(msg.shelf_type).strip()}' "
                f"target=({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f})"
            )
            _publish_world_collision_for_mujoco(self._args, world_yml)

            selected_plan = self._plan_selected_arm(
                stage="align_selected",
                arm=selected_arm,
                target_pose=target_pose,
                world_yml=world_yml,
            )
            other_plan = self._plan_other_arm_zero(
                stage="align_other_zero",
                arm=other_arm,
                q_start_cspace=selected_plan.q_start_cspace,
                cspace_joint_names=selected_plan.cspace_joint_names,
                world_yml=world_yml,
            )
            full_path = _build_combined_full_path(
                selected_plan=selected_plan,
                other_plan=other_plan,
                selected_arm=selected_arm,
                other_arm=other_arm,
            )
            self._wait_for_both_arms(
                stage="align",
                plan=selected_plan,
                full_path=full_path,
                selected_arm=selected_arm,
                other_arm=other_arm,
            )
            self._maybe_save_alignment(
                AlignExecutionRecord(
                    selected_arm=selected_arm,
                    other_arm=other_arm,
                    shelf_type=str(msg.shelf_type).strip(),
                    world_yml=world_yml,
                    target_pose=_copy_pose(target_pose),
                    marker_position=_copy_point(msg.marker_position),
                    q_start_cspace=list(selected_plan.q_start_cspace),
                    full_path=[list(q) for q in full_path],
                    cspace_joint_names=list(selected_plan.cspace_joint_names),
                )
            )
            self.get_logger().info("[ARM_PICKING] align sequence completed; waiting for ObjectGrasp")
            self._publish_arm_picking_finish_after_motion_complete(selected_arm, stage="align")
            self._execute_grasp_sequence(
                selected_arm=selected_arm,
                base_world_yml=world_yml,
            )
        except Exception as exc:
            self.get_logger().error(f"ARM_PICKING align sequence failed: {exc}")
        finally:
            self._reset_active()


def build_arm_picking_action_parser(argv: Sequence[str] | None = None):
    parser = build_single_arm_parser(
        default_world_yml=None,
        collision_models=_COLLISION_MODELS,
        default_collision_model="long_shelf",
    )
    parser.set_defaults(plot_path=False, arrival_max_retries=-1)
    parser.add_argument(
        "--object_align_topic",
        default="/object_align_result",
        help="topic name used to receive ObjectAlign messages from master",
    )
    parser.add_argument(
        "--arm_finish_topic",
        default="/arn_picking_finish",
        help="deprecated unused argument kept for launch compatibility",
    )
    parser.add_argument(
        "--object_grasp_topic",
        default="/object_grasp_result",
        help="topic name used to receive ObjectGrasp messages after arm alignment",
    )
    parser.add_argument(
        "--gripper_start_topic",
        default="/gripper_start",
        help="topic name used to notify gripper start for the selected arm",
    )
    parser.add_argument(
        "--gripper_finish_topic",
        default="/gripper_finish",
        help="topic name used to receive gripper completion for the selected arm",
    )
    parser.add_argument(
        "--arm_picking_finish_topic",
        default="/arm_picking_finish",
        help="topic name used to publish completion of the full arm picking sequence",
    )
    parser.add_argument(
        "--align_fixed_x_m",
        type=float,
        default=0.35,
        help="fixed target x position [m] used for non-shelf_1 align targets",
    )
    parser.add_argument(
        "--align_lift_z_m",
        type=float,
        default=0.15,
        help="z offset [m] added above marker_position.z for non-shelf_1 align targets",
    )
    parser.add_argument(
        "--post_grasp_lift_z_m",
        type=float,
        default=0.05,
        help="z offset [m] applied after gripper_finish before retreating",
    )
    parser.add_argument(
        "--object_grasp_wait_s",
        type=float,
        default=-1.0,
        help="wait time [s] for ObjectGrasp after arm_finish; -1 waits forever",
    )
    parser.add_argument(
        "--gripper_finish_wait_s",
        type=float,
        default=-1.0,
        help="wait time [s] for gripper_finish after gripper_start; -1 waits forever",
    )
    parser.add_argument(
        "--grasp_object_name",
        default="grasp_object",
        help="object name used when adding ObjectGrasp cuboid into the world collision model",
    )
    parser.add_argument(
        "--startup_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pre-initialize and warm up IK/collision objects before subscribing",
    )
    parser.add_argument(
        "--startup_warmup_iters",
        type=int,
        default=1,
        help="number of reachable IK warmup iterations to run per arm/world at startup",
    )
    parser.add_argument(
        "--startup_warmup_batch_size",
        type=int,
        default=None,
        help="batch size used for startup IK warmup solves; defaults to --ik_batch",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main_arm_picking_action_server(argv: Sequence[str] | None = None) -> int:
    args = build_arm_picking_action_parser(argv)
    rclpy.init()
    node = ArmPickingCoordinator(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("ARM_PICKING align node interrupted")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0

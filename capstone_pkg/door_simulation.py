from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import glfw
from geometry_msgs.msg import Pose2D, Twist
import mujoco
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from capstone_pkg.simulation import (
    DEFAULT_INIT_Q,
    GRIPPER_CLOSED_JOINTS,
    ROBOT_YAML,
    WorldCollisionIO,
    JointCmdIO,
    LiftTrajectoryActionIO,
    _dt_from_hz,
    _ensure_furniture_sim_paths,
    _find_world_box_geom_ids,
    _inject_world_box_slots,
    _make_mujoco_viewer,
    _viewer_running,
    apply_gripper_closed,
    apply_init_q_cspace,
    apply_world_collision_boxes,
    build_actuator_maps,
    build_joint_mapping,
    get_cspace_joint_names,
    make_ctrl_hold_from_qpos,
    read_q_cspace_from_mujoco,
)
from capstone_pkg.utils.world_collision_bridge import DEFAULT_WORLD_COLLISION_TOPIC


DOOR_MODEL = "/home/gaga/capstone_ws/src/capstone_pkg/models/door_ffw_sg2.xml"
BASE_VIRTUAL_JOINTS = ["base_x", "base_y", "base_yaw"]
DEFAULT_BASE_FREEJOINT = "floating_base"
WHEEL_RADIUS_M = 0.09
DOOR_HINGE_JOINT = "fridge_door_hinge_joint"
DEFAULT_DOOR_UNLOCK_GRIPPER_JOINT = "gripper_r_joint1"
DEFAULT_DOOR_UNLOCK_TOPIC = "/door_unlock"
GRIPPER_CLOSED_VALUE = 0.0

WHEEL_MODULES = [
    {
        "steer_joint": "left_wheel_steer_joint",
        "drive_joint": "left_wheel_drive_joint",
        "xy": (0.1371, 0.2554),
    },
    {
        "steer_joint": "right_wheel_steer_joint",
        "drive_joint": "right_wheel_drive_joint",
        "xy": (0.1371, -0.2554),
    },
    {
        "steer_joint": "rear_wheel_steer_joint",
        "drive_joint": "rear_wheel_drive_joint",
        "xy": (-0.2899, 0.0),
    },
]

WHEEL_DRIVE_ACTUATOR_NAMES = {
    "left_wheel_drive",
    "right_wheel_drive",
    "rear_wheel_drive",
    "left_wheel_drive_act",
    "right_wheel_drive_act",
    "rear_wheel_drive_act",
}

STATE_EXTRA_JOINTS = [
    "lift_joint",
    "head_joint1",
    "head_joint2",
]


@dataclass
class FreeJointInfo:
    name: str
    qpos_adr: int
    qvel_adr: int
    z_hold: float


@dataclass
class HingeJointInfo:
    name: str
    qpos_adr: int
    qvel_adr: int


class BaseCmdIO(Node):
    def __init__(self, twist_topic: str, pose_topic: str):
        super().__init__("door_mujoco_base_cmd_io")
        self._lock = threading.Lock()
        self._twist = (0.0, 0.0, 0.0)
        self._twist_rx = 0
        self._twist_t = 0.0
        self._pose: Optional[Tuple[float, float, float]] = None
        self._pose_rx = 0

        self.create_subscription(Twist, twist_topic, self._twist_cb, 10)
        self.create_subscription(Pose2D, pose_topic, self._pose_cb, 10)
        self.get_logger().info(f"Subscribed base Twist cmd: {twist_topic}")
        self.get_logger().info(f"Subscribed base Pose2D cmd: {pose_topic}")

    def _twist_cb(self, msg: Twist) -> None:
        with self._lock:
            self._twist = (
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            )
            self._twist_t = time.perf_counter()
            self._twist_rx += 1

    def _pose_cb(self, msg: Pose2D) -> None:
        with self._lock:
            self._pose = (
                float(msg.x),
                float(msg.y),
                float(msg.theta),
            )
            self._pose_rx += 1

    def get_twist(self) -> Tuple[float, float, float, int, float]:
        with self._lock:
            age = float("inf") if self._twist_rx <= 0 else time.perf_counter() - self._twist_t
            vx, vy, wz = self._twist
            return float(vx), float(vy), float(wz), int(self._twist_rx), age

    def get_pose(self) -> Tuple[Optional[Tuple[float, float, float]], int]:
        with self._lock:
            pose = None if self._pose is None else tuple(self._pose)
            return pose, int(self._pose_rx)


class DoorUnlockIO(Node):
    def __init__(self, topic: str):
        super().__init__("door_mujoco_unlock_io")
        self._lock = threading.Lock()
        self._unlock_rx = 0
        self.create_subscription(Bool, topic, self._cb, 10)
        self.get_logger().info(f"Subscribed door unlock: {topic}")

    def _cb(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        with self._lock:
            self._unlock_rx += 1

    def get_unlock_rx(self) -> int:
        with self._lock:
            return int(self._unlock_rx)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _quat_wxyz_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (float(qw) * float(qz) + float(qx) * float(qy))
    cosy_cosp = 1.0 - 2.0 * (float(qy) * float(qy) + float(qz) * float(qz))
    return math.atan2(siny_cosp, cosy_cosp)


def find_freejoint(model: mujoco.MjModel, joint_name: str) -> FreeJointInfo:
    for j_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if name != joint_name:
            continue
        if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"Joint '{joint_name}' exists but is not a freejoint")
        qadr = int(model.jnt_qposadr[j_id])
        dadr = int(model.jnt_dofadr[j_id])
        return FreeJointInfo(name=joint_name, qpos_adr=qadr, qvel_adr=dadr, z_hold=0.0)
    raise RuntimeError(f"Freejoint '{joint_name}' not found in MuJoCo model")


def find_hinge_joint(model: mujoco.MjModel, joint_name: str) -> Optional[HingeJointInfo]:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        return None
    if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise RuntimeError(f"Joint '{joint_name}' exists but is not a hinge joint")
    return HingeJointInfo(
        name=joint_name,
        qpos_adr=int(model.jnt_qposadr[j_id]),
        qvel_adr=int(model.jnt_dofadr[j_id]),
    )


def hold_hinge_position(data: mujoco.MjData, hinge: HingeJointInfo, position: float) -> None:
    data.qpos[hinge.qpos_adr] = float(position)
    data.qvel[hinge.qvel_adr] = 0.0


def disable_actuator_force(model: mujoco.MjModel, actuator_id: int) -> None:
    try:
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [0.0, 0.0]
    except Exception:
        pass


def get_base_pose2d(data: mujoco.MjData, freejoint: FreeJointInfo) -> Tuple[float, float, float]:
    qadr = freejoint.qpos_adr
    x = float(data.qpos[qadr + 0])
    y = float(data.qpos[qadr + 1])
    qw = float(data.qpos[qadr + 3])
    qx = float(data.qpos[qadr + 4])
    qy = float(data.qpos[qadr + 5])
    qz = float(data.qpos[qadr + 6])
    return x, y, _yaw_from_quat_wxyz(qw, qx, qy, qz)


def set_base_pose2d(
    data: mujoco.MjData,
    freejoint: FreeJointInfo,
    x: float,
    y: float,
    yaw: float,
    *,
    vx_world: float = 0.0,
    vy_world: float = 0.0,
    wz: float = 0.0,
) -> None:
    qadr = freejoint.qpos_adr
    dadr = freejoint.qvel_adr
    qw, qx, qy, qz = _quat_wxyz_from_yaw(yaw)
    data.qpos[qadr + 0] = float(x)
    data.qpos[qadr + 1] = float(y)
    data.qpos[qadr + 2] = float(freejoint.z_hold)
    data.qpos[qadr + 3] = float(qw)
    data.qpos[qadr + 4] = float(qx)
    data.qpos[qadr + 5] = float(qy)
    data.qpos[qadr + 6] = float(qz)
    data.qvel[dadr + 0] = float(vx_world)
    data.qvel[dadr + 1] = float(vy_world)
    data.qvel[dadr + 2] = 0.0
    data.qvel[dadr + 3] = 0.0
    data.qvel[dadr + 4] = 0.0
    data.qvel[dadr + 5] = float(wz)


def integrate_base_pose(
    x: float,
    y: float,
    yaw: float,
    vx_body: float,
    vy_body: float,
    wz: float,
    dt: float,
) -> Tuple[float, float, float, float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_world = c * vx_body - s * vy_body
    vy_world = s * vx_body + c * vy_body
    x_next = float(x) + vx_world * float(dt)
    y_next = float(y) + vy_world * float(dt)
    yaw_next = _wrap_pi(float(yaw) + float(wz) * float(dt))
    return x_next, y_next, yaw_next, vx_world, vy_world


def clamp_ctrl(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
    out = float(value)
    try:
        if int(model.actuator_ctrllimited[actuator_id]) != 0:
            lo = float(model.actuator_ctrlrange[actuator_id, 0])
            hi = float(model.actuator_ctrlrange[actuator_id, 1])
            if hi > lo:
                out = max(lo, min(hi, out))
    except Exception:
        pass
    return out


def clamp_velocity(value: float, limit_abs: float) -> float:
    limit_abs = abs(float(limit_abs))
    if limit_abs <= 0.0:
        return float(value)
    return max(-limit_abs, min(limit_abs, float(value)))


def build_state_joint_names(mapping: Dict[str, int], cspace_joint_names: List[str]) -> List[str]:
    out = list(cspace_joint_names)
    for name in STATE_EXTRA_JOINTS:
        if name in mapping and name not in out:
            out.append(name)
    return out


def zero_wheel_drive_hold(
    ctrl_hold: List[float],
    joint_to_act: Dict[str, int],
    model: mujoco.MjModel,
) -> None:
    for module in WHEEL_MODULES:
        act = joint_to_act.get(module["drive_joint"], None)
        if act is not None:
            ctrl_hold[act] = clamp_ctrl(model, act, 0.0)


def update_swerve_visual_controls(
    model: mujoco.MjModel,
    last_ctrl: List[float],
    joint_to_act: Dict[str, int],
    vx_body: float,
    vy_body: float,
    wz: float,
    wheel_radius_m: float,
) -> None:
    wheel_radius_m = max(1.0e-6, float(wheel_radius_m))
    for module in WHEEL_MODULES:
        x_i, y_i = module["xy"]
        vx_i = float(vx_body) - float(wz) * float(y_i)
        vy_i = float(vy_body) + float(wz) * float(x_i)
        speed = math.hypot(vx_i, vy_i)

        steer_act = joint_to_act.get(module["steer_joint"], None)
        drive_act = joint_to_act.get(module["drive_joint"], None)

        if steer_act is not None and speed > 1.0e-5:
            last_ctrl[steer_act] = clamp_ctrl(model, steer_act, math.atan2(vy_i, vx_i))
        if drive_act is not None:
            last_ctrl[drive_act] = clamp_ctrl(model, drive_act, speed / wheel_radius_m)


def publish_joint_state(
    pub_node: Node,
    pub,
    data: mujoco.MjData,
    mapping: Dict[str, int],
    joint_names_pub: List[str],
    *,
    base_pose: Optional[Tuple[float, float, float]],
    include_base_virtual_joints: bool,
) -> None:
    names = list(joint_names_pub)
    positions = read_q_cspace_from_mujoco(data, mapping, joint_names_pub)
    if include_base_virtual_joints and base_pose is not None:
        names.extend(BASE_VIRTUAL_JOINTS)
        positions.extend([float(base_pose[0]), float(base_pose[1]), float(base_pose[2])])

    msg = JointState()
    msg.header.stamp = pub_node.get_clock().now().to_msg()
    msg.name = names
    msg.position = positions
    pub.publish(msg)


def publish_base_pose(pub_node: Node, pub, base_pose: Tuple[float, float, float]) -> None:
    msg = Pose2D()
    msg.x = float(base_pose[0])
    msg.y = float(base_pose[1])
    msg.theta = float(base_pose[2])
    pub.publish(msg)


def main() -> None:
    if os.path.basename(__file__) == "mujoco_viewer.py":
        print("[WARN] Rename this file if it shadows the mujoco_viewer package.", file=sys.stderr)

    ap = argparse.ArgumentParser("Door-task MuJoCo simulation with kinematic base + both arms")
    ap.add_argument("--model", default=DOOR_MODEL)
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--sub_topic", default="/joint_states_cmd")
    ap.add_argument("--pub_topic", default="/joint_states")
    ap.add_argument("--cmd_vel_topic", default="/cmd_vel")
    ap.add_argument("--base_pose_cmd_topic", default="/base_pose_cmd")
    ap.add_argument("--base_pose_topic", default="/base_pose")

    ap.add_argument("--update_hz", type=float, default=100.0)
    ap.add_argument("--pub_hz", type=float, default=30.0)
    ap.add_argument("--render_hz", type=float, default=30.0)
    ap.add_argument("--idle_render_hz", type=float, default=2.0)
    ap.add_argument("--idle_after_s", type=float, default=0.2)
    ap.add_argument("--min_sleep_ms", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=5.0e-4)

    ap.add_argument("--render_only_when_moving", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_viewer", action="store_true", default=False)

    ap.add_argument("--init_q", nargs="+", type=float, default=DEFAULT_INIT_Q)
    ap.add_argument("--no_init_pose", action="store_true")
    ap.add_argument("--base_freejoint", default=DEFAULT_BASE_FREEJOINT)
    ap.add_argument("--base_z", type=float, default=None)
    ap.add_argument("--cmd_vel_timeout_s", type=float, default=0.35)
    ap.add_argument("--max_vx", type=float, default=0.4)
    ap.add_argument("--max_vy", type=float, default=0.4)
    ap.add_argument("--max_wz", type=float, default=0.8)
    ap.add_argument("--wheel_radius", type=float, default=WHEEL_RADIUS_M)
    ap.add_argument("--publish_base_in_joint_states", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC)
    ap.add_argument("--max_world_boxes", type=int, default=64)
    ap.add_argument("--world_box_group", type=int, default=0)
    ap.add_argument("--world_box_rgba", nargs=4, type=float, default=[0.95, 0.20, 0.05, 0.65])
    ap.add_argument("--world_box_physics_collision", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--lock_door_until_gripper_close", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--door_lock_angle", type=float, default=0.0)
    ap.add_argument("--door_unlock_topic", default=DEFAULT_DOOR_UNLOCK_TOPIC)
    ap.add_argument("--door_disable_actuator_after_unlock", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--unlock_door_on_gripper_close", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--door_unlock_gripper_joint", default=DEFAULT_DOOR_UNLOCK_GRIPPER_JOINT)
    ap.add_argument("--door_unlock_gripper_threshold", type=float, default=0.4)

    ap.add_argument("--lift_action_name", default="/lift_controller/follow_joint_trajectory")
    ap.add_argument("--lift_action_joint_name", default="lift_joint")
    ap.add_argument("--lift_action_goal_tolerance", type=float, default=0.02)
    ap.add_argument("--lift_action_timeout_margin_s", type=float, default=5.0)

    args = ap.parse_args()

    mjcf_path = os.path.abspath(str(args.model))
    max_world_boxes = max(0, int(args.max_world_boxes))
    world_box_group = max(0, min(5, int(args.world_box_group)))

    models_root = os.path.dirname(mjcf_path)
    _ensure_furniture_sim_paths(models_root)
    loaded_mjcf_path = _inject_world_box_slots(mjcf_path, max_world_boxes, world_box_group)

    model = mujoco.MjModel.from_xml_path(loaded_mjcf_path)
    if model.nu <= 0:
        raise RuntimeError("The MJCF has no actuators; arm commands require position actuators.")

    data = mujoco.MjData(model)
    mapping = build_joint_mapping(model)
    if not mapping:
        raise RuntimeError("No hinge/slide joints found in MJCF.")

    freejoint = find_freejoint(model, args.base_freejoint)
    door_hinge = find_hinge_joint(model, DOOR_HINGE_JOINT)
    freejoint.z_hold = (
        float(args.base_z)
        if args.base_z is not None
        else float(data.qpos[freejoint.qpos_adr + 2])
    )

    cspace_joint_names = get_cspace_joint_names(args.robot_yml)
    joint_names_pub = build_state_joint_names(mapping, cspace_joint_names)
    world_box_geom_ids = _find_world_box_geom_ids(model, max_world_boxes) if max_world_boxes > 0 else []

    if not args.no_init_pose:
        applied = apply_init_q_cspace(data, mapping, cspace_joint_names, args.init_q)
        gripper_applied = apply_gripper_closed(
            data,
            mapping,
            GRIPPER_CLOSED_JOINTS,
            GRIPPER_CLOSED_VALUE,
        )
        mujoco.mj_forward(model, data)
        print(f"[InitPose] applied {applied}/{len(cspace_joint_names)} arm joints")
        print(f"[GripperInit] closed {gripper_applied}/{len(GRIPPER_CLOSED_JOINTS)} joints")
    else:
        gripper_applied = apply_gripper_closed(
            data,
            mapping,
            GRIPPER_CLOSED_JOINTS,
            GRIPPER_CLOSED_VALUE,
        )
        mujoco.mj_forward(model, data)
        print("[InitPose] skipped")
        print(f"[GripperInit] closed {gripper_applied}/{len(GRIPPER_CLOSED_JOINTS)} joints")

    base_x, base_y, base_yaw = get_base_pose2d(data, freejoint)
    set_base_pose2d(data, freejoint, base_x, base_y, base_yaw)
    mujoco.mj_forward(model, data)

    act_names, joint_to_act = build_actuator_maps(model)
    ctrl_hold = make_ctrl_hold_from_qpos(model, data, act_names)
    zero_wheel_drive_hold(ctrl_hold, joint_to_act, model)
    door_hinge_act = joint_to_act.get(DOOR_HINGE_JOINT, None)
    door_unlock_gripper_joint = str(args.door_unlock_gripper_joint)
    door_locked = bool(args.lock_door_until_gripper_close and door_hinge is not None)

    def _unlock_door(reason: str) -> None:
        nonlocal door_locked
        door_locked = False
        if (
            bool(args.door_disable_actuator_after_unlock)
            and door_hinge_act is not None
        ):
            disable_actuator_force(model, door_hinge_act)
            print(f"[DoorLock] unlocked by {reason}; hinge actuator force disabled")
        else:
            print(f"[DoorLock] unlocked by {reason}")

    if door_locked:
        hold_hinge_position(data, door_hinge, float(args.door_lock_angle))
        if door_hinge_act is not None:
            ctrl_hold[door_hinge_act] = clamp_ctrl(model, door_hinge_act, float(args.door_lock_angle))
    data.ctrl[:] = ctrl_hold
    last_ctrl = list(ctrl_hold)

    print("=== Door MuJoCo simulation ===")
    print(f"MJCF                    : {mjcf_path}")
    print(f"Joint command topic      : {args.sub_topic}")
    print(f"Joint state topic        : {args.pub_topic}")
    print(f"Base cmd_vel topic       : {args.cmd_vel_topic}")
    print(f"Base pose cmd topic      : {args.base_pose_cmd_topic}")
    print(f"Base pose topic          : {args.base_pose_topic}")
    print(f"Base freejoint           : {args.base_freejoint}")
    print(f"Publish base in JointState: {args.publish_base_in_joint_states}")
    print(f"World collision topic    : {args.world_collision_topic}")
    print(f"World box slots          : {len(world_box_geom_ids)}")
    if door_locked:
        print(
            f"Door lock                : locked at {float(args.door_lock_angle):.3f} rad "
            f"until Bool(True) on {args.door_unlock_topic}"
        )
        if bool(args.unlock_door_on_gripper_close):
            print(
                f"Door gripper unlock      : enabled "
                f"({door_unlock_gripper_joint}>={float(args.door_unlock_gripper_threshold):.3f})"
            )
    else:
        print("Door lock                : disabled")
    print()
    print("[Command names]")
    print("  Left arm : arm_l_joint1 ... arm_l_joint7")
    print("  Right arm: arm_r_joint1 ... arm_r_joint7")
    print("  Gripper  : gripper_l_joint1, gripper_r_joint1")
    print("  Base     : geometry_msgs/Twist on /cmd_vel (linear.x, linear.y, angular.z)")
    print()

    rclpy.init()

    cmd_node = JointCmdIO(args.sub_topic)
    base_cmd_node = BaseCmdIO(args.cmd_vel_topic, args.base_pose_cmd_topic)
    door_unlock_node = DoorUnlockIO(args.door_unlock_topic) if door_hinge is not None else None
    world_node = WorldCollisionIO(args.world_collision_topic) if world_box_geom_ids else None
    lift_action_node = LiftTrajectoryActionIO(
        args.lift_action_name,
        args.lift_action_joint_name,
        actuator_available=args.lift_action_joint_name in joint_to_act,
        goal_tolerance=args.lift_action_goal_tolerance,
        timeout_margin_s=args.lift_action_timeout_margin_s,
    )

    state_node = Node("door_mujoco_state_pub")
    joint_pub = state_node.create_publisher(JointState, args.pub_topic, 10)
    base_pose_pub = state_node.create_publisher(Pose2D, args.base_pose_topic, 10)
    state_node.get_logger().info(f"Publishing JointState to: {args.pub_topic}")
    state_node.get_logger().info(f"Publishing base Pose2D to: {args.base_pose_topic}")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(cmd_node)
    executor.add_node(base_cmd_node)
    if door_unlock_node is not None:
        executor.add_node(door_unlock_node)
    if world_node is not None:
        executor.add_node(world_node)
    executor.add_node(lift_action_node)
    executor.add_node(state_node)

    def _spin_exec() -> None:
        try:
            executor.spin()
        finally:
            executor.shutdown()

    spin_thread = threading.Thread(target=_spin_exec, daemon=True)
    spin_thread.start()

    dt_update = _dt_from_hz(args.update_hz, 1.0 / 100.0)
    dt_pub = _dt_from_hz(args.pub_hz, 1.0 / 30.0)
    dt_render_move = _dt_from_hz(args.render_hz, 1.0 / 30.0)
    dt_render_idle = _dt_from_hz(args.idle_render_hz, 0.5)
    min_sleep = max(0.001, float(args.min_sleep_ms) / 1000.0)

    next_update = time.perf_counter()
    next_render = time.perf_counter()
    next_pub_wall = time.time()
    last_change_t = time.perf_counter()
    next_events = time.perf_counter()
    dt_events = 1.0 / 30.0
    last_update_t = time.perf_counter()

    last_joint_rx = 0
    last_pose_rx = 0
    last_world_rx = 0
    last_door_unlock_rx = 0

    viewer = None
    if not args.no_viewer:
        viewer = _make_mujoco_viewer(model, data)
        print("[VIEWER] created:", getattr(viewer, "window", None))
        for _ in range(3):
            viewer.render()
            glfw.poll_events()
            time.sleep(0.01)

        def _key_cb(window, key, scancode, action, mods) -> None:
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        try:
            glfw.set_key_callback(viewer.window, _key_cb)
        except Exception:
            pass

    try:
        while rclpy.ok() and (viewer is None or _viewer_running(viewer)):
            now = time.perf_counter()

            if world_node is not None:
                world_source, world_cuboids, world_rx = world_node.get_latest()
                if world_rx != last_world_rx:
                    last_world_rx = world_rx
                    shown = apply_world_collision_boxes(
                        model,
                        data,
                        world_box_geom_ids,
                        world_cuboids,
                        rgba=list(args.world_box_rgba),
                        enable_physics_collision=bool(args.world_box_physics_collision),
                    )
                    if len(world_cuboids) > len(world_box_geom_ids):
                        world_node.get_logger().warning(
                            f"Received {len(world_cuboids)} world cuboids, "
                            f"but only {len(world_box_geom_ids)} MuJoCo slots are available."
                        )
                    world_node.get_logger().info(
                        f"Displayed {shown}/{len(world_cuboids)} world collision cuboid(s)"
                        + (f" from {world_source}" if world_source else "")
                    )
                    last_change_t = time.perf_counter()
                    next_render = last_change_t

            if now >= next_update:
                if door_locked and door_unlock_node is not None:
                    door_unlock_rx = door_unlock_node.get_unlock_rx()
                    if door_unlock_rx != last_door_unlock_rx:
                        last_door_unlock_rx = door_unlock_rx
                        _unlock_door(str(args.door_unlock_topic))

                dt_elapsed = max(0.0, min(0.1, now - last_update_t))
                last_update_t = now
                if (now - next_update) > (3.0 * dt_update):
                    next_update = now + dt_update
                else:
                    next_update += dt_update

                cmd_dict, joint_rx = cmd_node.get_latest()
                if joint_rx != last_joint_rx:
                    last_joint_rx = joint_rx
                    for joint_name, q_des in cmd_dict.items():
                        if (
                            door_locked
                            and bool(args.unlock_door_on_gripper_close)
                            and joint_name == door_unlock_gripper_joint
                            and float(q_des) >= float(args.door_unlock_gripper_threshold)
                        ):
                            _unlock_door(f"{joint_name}={float(q_des):.3f}")
                        actuator_id = joint_to_act.get(joint_name, None)
                        if actuator_id is None:
                            continue
                        if act_names[actuator_id] in WHEEL_DRIVE_ACTUATOR_NAMES:
                            continue
                        last_ctrl[actuator_id] = clamp_ctrl(model, actuator_id, float(q_des))
                    last_change_t = time.perf_counter()

                lift_target = lift_action_node.get_active_target_position()
                if lift_target is not None:
                    lift_act = joint_to_act.get(args.lift_action_joint_name, None)
                    if lift_act is not None:
                        last_ctrl[lift_act] = clamp_ctrl(model, lift_act, float(lift_target))

                pose_cmd, pose_rx = base_cmd_node.get_pose()
                if pose_cmd is not None and pose_rx != last_pose_rx:
                    last_pose_rx = pose_rx
                    base_x, base_y, base_yaw = pose_cmd
                    base_yaw = _wrap_pi(base_yaw)
                    last_change_t = time.perf_counter()

                vx_body, vy_body, wz, _twist_rx, twist_age = base_cmd_node.get_twist()
                if twist_age > float(args.cmd_vel_timeout_s):
                    vx_body, vy_body, wz = 0.0, 0.0, 0.0
                vx_body = clamp_velocity(vx_body, args.max_vx)
                vy_body = clamp_velocity(vy_body, args.max_vy)
                wz = clamp_velocity(wz, args.max_wz)

                moving_base = (
                    abs(vx_body) > float(args.eps)
                    or abs(vy_body) > float(args.eps)
                    or abs(wz) > float(args.eps)
                )
                if moving_base:
                    base_x, base_y, base_yaw, vx_world, vy_world = integrate_base_pose(
                        base_x,
                        base_y,
                        base_yaw,
                        vx_body,
                        vy_body,
                        wz,
                        dt_elapsed,
                    )
                    last_change_t = time.perf_counter()
                else:
                    vx_world, vy_world = 0.0, 0.0

                update_swerve_visual_controls(
                    model,
                    last_ctrl,
                    joint_to_act,
                    vx_body,
                    vy_body,
                    wz,
                    float(args.wheel_radius),
                )
                if door_locked:
                    if door_hinge_act is not None:
                        last_ctrl[door_hinge_act] = clamp_ctrl(
                            model,
                            door_hinge_act,
                            float(args.door_lock_angle),
                        )
                    if door_hinge is not None:
                        hold_hinge_position(data, door_hinge, float(args.door_lock_angle))
                data.ctrl[:] = last_ctrl

                set_base_pose2d(
                    data,
                    freejoint,
                    base_x,
                    base_y,
                    base_yaw,
                    vx_world=vx_world,
                    vy_world=vy_world,
                    wz=wz,
                )

                dt_phys = float(model.opt.timestep)
                steps = max(1, int(round(dt_update / max(1.0e-9, dt_phys))))
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                    if door_locked and door_hinge is not None:
                        hold_hinge_position(data, door_hinge, float(args.door_lock_angle))
                    set_base_pose2d(
                        data,
                        freejoint,
                        base_x,
                        base_y,
                        base_yaw,
                        vx_world=vx_world,
                        vy_world=vy_world,
                        wz=wz,
                    )
                mujoco.mj_forward(model, data)

                lift_adr = mapping.get(args.lift_action_joint_name, None)
                if lift_adr is not None and 0 <= lift_adr < data.qpos.size:
                    lift_action_node.update_from_sim(float(data.qpos[lift_adr]))

                now_wall = time.time()
                if now_wall >= next_pub_wall:
                    base_pose = (base_x, base_y, base_yaw)
                    publish_joint_state(
                        state_node,
                        joint_pub,
                        data,
                        mapping,
                        joint_names_pub,
                        base_pose=base_pose,
                        include_base_virtual_joints=bool(args.publish_base_in_joint_states),
                    )
                    publish_base_pose(state_node, base_pose_pub, base_pose)
                    next_pub_wall = now_wall + dt_pub

            if viewer is not None:
                now_ev = time.perf_counter()
                if now_ev >= next_events:
                    try:
                        glfw.poll_events()
                    except Exception:
                        pass
                    next_events = now_ev + dt_events

            idle = (time.perf_counter() - last_change_t) > float(args.idle_after_s)
            if viewer is not None:
                do_render = True
                if args.render_only_when_moving and idle and not (args.idle_render_hz > 0.0):
                    do_render = False

                if do_render:
                    dt_render = dt_render_idle if idle else dt_render_move
                    if now >= next_render:
                        if (now - next_render) > (3.0 * dt_render):
                            next_render = now + dt_render
                        else:
                            next_render += dt_render
                        viewer.render()

            now2 = time.perf_counter()
            targets = [next_update]
            if viewer is not None:
                idle = (time.perf_counter() - last_change_t) > float(args.idle_after_s)
                do_render = not (args.render_only_when_moving and idle and not (args.idle_render_hz > 0.0))
                if do_render:
                    targets.append(next_render)
                targets.append(next_events)

            sleep_until = min(targets)
            sleep_time = sleep_until - now2

            if viewer is not None:
                if sleep_time > 0:
                    idle = (time.perf_counter() - last_change_t) > float(args.idle_after_s)
                    do_render = not (args.render_only_when_moving and idle and not (args.idle_render_hz > 0.0))
                    if not do_render:
                        try:
                            glfw.wait_events_timeout(min(0.2, max(0.0, sleep_time)))
                        except Exception:
                            time.sleep(min(0.02, sleep_time))
                    else:
                        time.sleep(max(min_sleep, sleep_time))
                else:
                    time.sleep(min_sleep)
            else:
                if sleep_time > 0:
                    time.sleep(max(min_sleep, sleep_time))
                else:
                    time.sleep(min_sleep)

    finally:
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass

        for node in [cmd_node, base_cmd_node, door_unlock_node, world_node, lift_action_node, state_node]:
            if node is None:
                continue
            try:
                executor.remove_node(node)
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        if loaded_mjcf_path != mjcf_path:
            try:
                os.unlink(loaded_mjcf_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()

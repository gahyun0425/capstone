from __future__ import annotations

import os
import sys

if os.environ.get("SPAWN_NO_FORCE_NVIDIA", "0").lower() not in ("1", "true", "yes", "y"):
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import argparse
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import glfw
import mujoco
import rclpy
from capstone_pkg.utils.world_collision_bridge import (
    DEFAULT_WORLD_COLLISION_TOPIC,
    WorldCuboid,
    parse_world_collision_payload,
)
from curobo.util_file import load_yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

MODEL = "/home/gaga/capstone_ws/src/capstone_pkg/models/ffw_sg2.xml"
ROBOT_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/test_curobo.yaml"
WORLD_BOX_PREFIX = "capstone_world_collision_box_"

DEFAULT_INIT_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GRIPPER_CLOSED_JOINTS = [
    "gripper_l_joint1", "gripper_l_joint2", "gripper_l_joint3", "gripper_l_joint4",
    "gripper_r_joint1", "gripper_r_joint2", "gripper_r_joint3", "gripper_r_joint4",
]
GRIPPER_CLOSED_VALUE = 0.5


def _safe_symlink(src: str, dst: str) -> None:
    if os.path.exists(dst) or os.path.islink(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(src, dst)
        print(f"[FIX] symlink: {dst} -> {src}")
    except FileExistsError:
        pass
    except OSError as e:
        print(f"[WARN] failed symlink {dst} -> {src}: {e}")


def _ensure_furniture_sim_paths(models_root: str) -> None:
    models_root = os.path.abspath(models_root)
    base_candidates = [
        os.path.join(models_root, "furniture_sim"),
        os.path.join(os.path.dirname(models_root), "furniture_sim"),
    ]
    base = None
    for cand in base_candidates:
        if os.path.isdir(cand):
            base = cand
            break
    if base is None:
        return

    # If MJCF is inside a subfolder (e.g., models/ffw_sg2), expose furniture_sim there.
    local_base = os.path.join(models_root, "furniture_sim")
    if not os.path.exists(local_base):
        _safe_symlink(base, local_base)

    # meshdir="assets" in ffw_sg2_world.xml expects assets under the MJCF folder.
    assets_src_candidates = [
        os.path.join(models_root, "assets"),
        os.path.join(os.path.dirname(models_root), "assets"),
    ]
    assets_src = None
    for cand in assets_src_candidates:
        if os.path.isdir(cand):
            assets_src = cand
            break
    if assets_src is not None:
        local_assets = os.path.join(models_root, "assets")
        if not os.path.exists(local_assets):
            _safe_symlink(assets_src, local_assets)

    dup = os.path.join(base, "furniture_sim")
    _safe_symlink(base, dup)

    common_src = os.path.join(base, "common")
    if not os.path.isdir(common_src):
        return

    for name in os.listdir(base):
        sub = os.path.join(base, name)
        if not os.path.isdir(sub):
            continue
        if name == "common":
            continue
        sub_common = os.path.join(sub, "common")
        _safe_symlink(common_src, sub_common)

    models_common = os.path.join(models_root, "common")
    if not os.path.exists(models_common):
        _safe_symlink(common_src, models_common)
    else:
        for subname in ["textures", "meshes", "materials"]:
            dst = os.path.join(models_common, subname)
            src = os.path.join(common_src, subname)
            if os.path.isdir(src) and not os.path.exists(dst):
                _safe_symlink(src, dst)


def _dt_from_hz(hz: float, default_dt: float) -> float:
    if hz is None:
        return default_dt
    hz = float(hz)
    if hz <= 0.0:
        return float("inf")
    return 1.0 / max(0.1, hz)


def _inject_world_box_slots(mjcf_path: str, max_boxes: int, box_group: int) -> str:
    if max_boxes <= 0:
        return mjcf_path
    box_group = max(0, min(5, int(box_group)))

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    models_root = os.path.dirname(os.path.abspath(mjcf_path))

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.abspath(os.path.join(models_root, meshdir)))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    for idx in range(max_boxes):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"{WORLD_BOX_PREFIX}{idx}",
                "type": "box",
                "size": "0.001 0.001 0.001",
                "pos": "0 0 -10",
                "quat": "1 0 0 0",
                "rgba": "0 0 0 0",
                "contype": "0",
                "conaffinity": "0",
                "group": str(box_group),
            },
        )

    tmp = tempfile.NamedTemporaryFile(
        prefix="capstone_mujoco_world_boxes_",
        suffix=".xml",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    return tmp_path


def _find_world_box_geom_ids(model: mujoco.MjModel, max_boxes: int) -> List[int]:
    geom_ids: List[int] = []
    for idx in range(max_boxes):
        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{WORLD_BOX_PREFIX}{idx}",
        )
        if gid < 0:
            raise RuntimeError(f"World collision box slot not found: {WORLD_BOX_PREFIX}{idx}")
        geom_ids.append(int(gid))
    return geom_ids


def _hide_world_box(model: mujoco.MjModel, geom_id: int) -> None:
    model.geom_pos[geom_id] = [0.0, 0.0, -10.0]
    model.geom_quat[geom_id] = [1.0, 0.0, 0.0, 0.0]
    model.geom_size[geom_id] = [0.001, 0.001, 0.001]
    model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
    model.geom_contype[geom_id] = 0
    model.geom_conaffinity[geom_id] = 0


def apply_world_collision_boxes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: List[int],
    cuboids: List[WorldCuboid],
    *,
    rgba: List[float],
    enable_physics_collision: bool,
) -> int:
    for geom_id in geom_ids:
        _hide_world_box(model, geom_id)

    active = min(len(cuboids), len(geom_ids))
    contype = 1 if enable_physics_collision else 0
    conaffinity = 1 if enable_physics_collision else 0

    for cuboid, geom_id in zip(cuboids[:active], geom_ids[:active]):
        x, y, z, qw, qx, qy, qz = cuboid.pose
        dx, dy, dz = cuboid.dims
        model.geom_pos[geom_id] = [float(x), float(y), float(z)]
        model.geom_quat[geom_id] = [float(qw), float(qx), float(qy), float(qz)]
        model.geom_size[geom_id] = [float(dx) * 0.5, float(dy) * 0.5, float(dz) * 0.5]
        model.geom_rgba[geom_id] = [float(v) for v in rgba]
        model.geom_contype[geom_id] = contype
        model.geom_conaffinity[geom_id] = conaffinity

    mujoco.mj_forward(model, data)
    return active


def get_cspace_joint_names(robot_yml: str) -> List[str]:
    cfg = load_yaml(robot_yml)
    robot_cfg: Dict[str, Any] = cfg["robot_cfg"]

    c1 = robot_cfg.get("cspace", {}) or {}
    names = c1.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    kin = robot_cfg.get("kinematics", {}) or {}
    c2 = (kin.get("cspace", {}) or {}).get("joint_names", None)
    if isinstance(c2, list) and c2:
        return [x for x in c2 if isinstance(x, str)]

    raise RuntimeError("YAML에서 cspace.joint_names를 찾을 수 없습니다.")


def build_joint_mapping(model: mujoco.MjModel) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for j_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if not name:
            continue
        j_type = model.jnt_type[j_id]
        qpos_adr = int(model.jnt_qposadr[j_id])
        if j_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            mapping[name] = qpos_adr
    return mapping


class JointCmdIO(Node):
    def __init__(self, sub_topic: str):
        super().__init__("mujoco_joint_cmd_io")
        self._lock = threading.Lock()
        self._latest = {}  # joint_name -> position
        self._rx = 0
        self.create_subscription(JointState, sub_topic, self._cb, 10)
        self.get_logger().info(f"Subscribed JointState cmd: {sub_topic}")

    def _cb(self, msg: JointState):
        d = {}
        for n, p in zip(list(msg.name), list(msg.position)):
            if isinstance(n, str):
                d[n] = float(p)
        with self._lock:
            self._latest = d
            self._rx += 1

    def get_latest(self) -> Tuple[Dict[str, float], int]:
        with self._lock:
            return dict(self._latest), int(self._rx)


class WorldCollisionIO(Node):
    def __init__(self, topic: str):
        super().__init__("mujoco_world_collision_io")
        self._lock = threading.Lock()
        self._latest_source = ""
        self._latest: List[WorldCuboid] = []
        self._rx = 0

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, topic, self._cb, qos)
        self.get_logger().info(f"Subscribed world collision cuboids: {topic}")

    def _cb(self, msg: String):
        try:
            source, cuboids = parse_world_collision_payload(msg.data)
        except Exception as exc:
            self.get_logger().error(f"Failed to parse world collision payload: {exc}")
            return
        with self._lock:
            self._latest_source = source
            self._latest = list(cuboids)
            self._rx += 1

    def get_latest(self) -> Tuple[str, List[WorldCuboid], int]:
        with self._lock:
            return self._latest_source, list(self._latest), int(self._rx)


class JointStateIO(Node):
    def __init__(self, sub_topic: str, pub_topic: str, joint_names_pub: List[str]):
        super().__init__("mujoco_jointstate_io")
        self._lock = threading.Lock()
        self._latest_names: List[str] = []
        self._latest_pos: List[float] = []
        self.joint_names_pub = list(joint_names_pub)

        self.create_subscription(JointState, sub_topic, self._cb, 10)
        self.get_logger().info(f"Subscribed JointState cmd: {sub_topic}")

        self.pub = self.create_publisher(JointState, pub_topic, 10)
        self.get_logger().info(f"Publishing MuJoCo state to: {pub_topic}")

    def _cb(self, msg: JointState):
        with self._lock:
            self._latest_names = list(msg.name)
            self._latest_pos = list(msg.position)

    def get_latest_cmd(self) -> Tuple[List[str], List[float]]:
        with self._lock:
            return list(self._latest_names), list(self._latest_pos)

    def publish_state(self, q_list: List[float]):
        if len(q_list) != len(self.joint_names_pub):
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names_pub)
        msg.position = list(q_list)
        self.pub.publish(msg)


def apply_init_q_cspace(
    data: mujoco.MjData,
    mapping: Dict[str, int],
    cspace_joint_names: List[str],
    init_q: List[float],
) -> int:
    if len(init_q) != len(cspace_joint_names):
        raise ValueError(f"init_q length mismatch: {len(init_q)} vs {len(cspace_joint_names)}")
    applied = 0
    for jn, v in zip(cspace_joint_names, init_q):
        adr = mapping.get(jn, None)
        if adr is None:
            continue
        data.qpos[adr] = float(v)
        applied += 1
    return applied


def apply_gripper_closed(
    data: mujoco.MjData,
    mapping: Dict[str, int],
    joint_names: List[str],
    value: float = 1.0,
) -> int:
    applied = 0
    for jn in joint_names:
        adr = mapping.get(jn, None)
        if adr is None:
            continue
        if 0 <= adr < data.qpos.size:
            data.qpos[adr] = float(value)
            applied += 1
    return applied


def apply_jointstate_qpos(
    data: mujoco.MjData,
    mapping: Dict[str, int],
    names: List[str],
    pos: List[float],
    eps: float,
) -> bool:
    if not names or not pos:
        return False
    changed = False
    for jn, v in zip(names, pos):
        adr = mapping.get(jn, None)
        if adr is None:
            continue
        v = float(v)
        if 0 <= adr < data.qpos.size:
            if abs(float(data.qpos[adr]) - v) > eps:
                data.qpos[adr] = v
                changed = True
    return changed


def read_q_cspace_from_mujoco(
    data: mujoco.MjData,
    mapping: Dict[str, int],
    cspace_joint_names: List[str],
) -> List[float]:
    out: List[float] = []
    for jn in cspace_joint_names:
        adr = mapping.get(jn, None)
        if adr is None or not (0 <= adr < data.qpos.size):
            out.append(0.0)
        else:
            out.append(float(data.qpos[adr]))
    return out


def _make_mujoco_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    import importlib

    mv = importlib.import_module("mujoco_viewer")
    if hasattr(mv, "MujocoViewer"):
        return mv.MujocoViewer(model, data)
    if hasattr(mv, "mujoco_viewer") and hasattr(mv.mujoco_viewer, "MujocoViewer"):
        return mv.mujoco_viewer.MujocoViewer(model, data)
    for mod_name in ("mujoco_viewer.mujoco_viewer", "mujoco_viewer.viewer"):
        try:
            sm = importlib.import_module(mod_name)
            if hasattr(sm, "MujocoViewer"):
                return sm.MujocoViewer(model, data)
        except Exception:
            pass
    raise AttributeError(f"mujoco_viewer에서 MujocoViewer를 못 찾음: {getattr(mv, '__file__', '?')}")


def _viewer_running(viewer) -> bool:
    try:
        return not glfw.window_should_close(viewer.window)
    except Exception:
        return True


def build_actuator_maps(model: mujoco.MjModel):
    act_names = []
    for a in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        act_names.append(n if n else f"act{a}")

    joint_to_act = {}
    for a in range(model.nu):
        j_id = int(model.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if jn:
            joint_to_act[jn] = a

    return act_names, joint_to_act


def make_ctrl_hold_from_qpos(model: mujoco.MjModel, data: mujoco.MjData, act_names: List[str]):
    """
    position actuator: ctrl = 현재 joint qpos (초기자세 유지)
    velocity actuator: ctrl = 0
    """
    ctrl_hold = [0.0] * model.nu

    def _clamp(a, v):
        try:
            if int(model.actuator_ctrllimited[a]) != 0:
                lo = float(model.actuator_ctrlrange[a, 0])
                hi = float(model.actuator_ctrlrange[a, 1])
                if hi > lo:
                    return max(lo, min(hi, v))
        except Exception:
            pass
        return v

    for a in range(model.nu):
        name = act_names[a]
        is_velocity = name in ("left_wheel_drive_act", "right_wheel_drive_act", "rear_wheel_drive_act")

        if is_velocity:
            ctrl_hold[a] = _clamp(a, 0.0)
            continue

        j_id = int(model.actuator_trnid[a, 0])
        qadr = int(model.jnt_qposadr[j_id])
        q = float(data.qpos[qadr])
        ctrl_hold[a] = _clamp(a, q)

    return ctrl_hold


def main():
    if os.path.basename(__file__) == "mujoco_viewer.py":
        print("[WARN] 파일명이 mujoco_viewer.py면 import 충돌 가능. spawn_mujoco.py로 바꾸는 걸 권장.", file=sys.stderr)

    ap = argparse.ArgumentParser("MuJoCo spawn (CPU-friendly / kinematic)")

    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--sub_topic", default="/joint_states_cmd")
    ap.add_argument("--pub_topic", default="/joint_states")

    ap.add_argument("--update_hz", type=float, default=60.0)
    ap.add_argument("--pub_hz", type=float, default=30.0)
    ap.add_argument("--render_hz", type=float, default=30.0)
    ap.add_argument("--idle_render_hz", type=float, default=2.0)
    ap.add_argument("--idle_after_s", type=float, default=0.2)
    ap.add_argument("--min_sleep_ms", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=5e-3)

    ap.add_argument("--render_only_when_moving", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_viewer", action="store_true", default=False)

    ap.add_argument("--init_q", nargs="+", type=float, default=DEFAULT_INIT_Q)
    ap.add_argument("--no_init_pose", action="store_true")
    ap.add_argument("--ctrl_topic", default="/ctrl_cmd")
    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC)
    ap.add_argument("--max_world_boxes", type=int, default=64)
    ap.add_argument("--world_box_group", type=int, default=0)
    ap.add_argument("--world_box_rgba", nargs=4, type=float, default=[0.95, 0.20, 0.05, 0.65])
    ap.add_argument("--world_box_physics_collision", action=argparse.BooleanOptionalAction, default=False)

    args = ap.parse_args()
    mjcf_path = MODEL
    max_world_boxes = max(0, int(args.max_world_boxes))
    world_box_group = max(0, min(5, int(args.world_box_group)))

    models_root = os.path.dirname(os.path.abspath(mjcf_path))
    _ensure_furniture_sim_paths(models_root)

    loaded_mjcf_path = _inject_world_box_slots(mjcf_path, max_world_boxes, world_box_group)

    model = mujoco.MjModel.from_xml_path(loaded_mjcf_path)
    if model.nu <= 0:
        raise RuntimeError(
            "이 MJCF에는 actuator가 없습니다 (model.nu == 0). "
            "ctrl로 움직이려면 MJCF에 <actuator>를 추가해야 합니다."
        )

    data = mujoco.MjData(model)
    mapping = build_joint_mapping(model)
    if not mapping:
        raise RuntimeError("No hinge/slide joints found in MJCF. (mapping is empty)")

    cspace_joint_names = get_cspace_joint_names(args.robot_yml)
    world_box_geom_ids = _find_world_box_geom_ids(model, max_world_boxes) if max_world_boxes > 0 else []

    print("=== MuJoCo spawn (CPU-friendly / kinematic) ===")
    print(f"MJCF               : {mjcf_path}")
    print(f"No viewer          : {args.no_viewer}")
    print(f"Render hz          : {args.render_hz}")
    print(f"Idle render hz     : {args.idle_render_hz}")
    print(f"Render only moving : {args.render_only_when_moving}")
    print(f"World collision topic : {args.world_collision_topic}")
    print(f"World box slots       : {len(world_box_geom_ids)}")
    print(f"World box group       : {world_box_group}")
    print(f"World box rgba        : {args.world_box_rgba}")
    print(f"World box physics col : {args.world_box_physics_collision}")
    print()

    if not args.no_init_pose:
        applied = apply_init_q_cspace(data, mapping, cspace_joint_names, args.init_q)
        gripper_applied = apply_gripper_closed(
            data,
            mapping,
            GRIPPER_CLOSED_JOINTS,
            GRIPPER_CLOSED_VALUE,
        )
        mujoco.mj_forward(model, data)
        print(f"[InitPose] applied {applied}/{len(cspace_joint_names)} joints")
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

    act_names, joint_to_act = build_actuator_maps(model)
    ctrl_hold = make_ctrl_hold_from_qpos(model, data, act_names)
    data.ctrl[:] = ctrl_hold

    rclpy.init()

    last_ctrl_rx = 0
    last_ctrl = list(ctrl_hold)
    last_world_rx = 0

    cmd_node = JointCmdIO(args.sub_topic)
    world_node = WorldCollisionIO(args.world_collision_topic) if world_box_geom_ids else None

    state_node = Node("mujoco_state_pub")
    pub = state_node.create_publisher(JointState, args.pub_topic, 10)
    state_node.get_logger().info(f"Publishing MuJoCo state to: {args.pub_topic}")

    executor = SingleThreadedExecutor()
    executor.add_node(cmd_node)
    if world_node is not None:
        executor.add_node(world_node)
    executor.add_node(state_node)

    def _spin_exec():
        try:
            executor.spin()
        finally:
            executor.shutdown()

    spin_thread = threading.Thread(target=_spin_exec, daemon=True)
    spin_thread.start()

    dt_update = _dt_from_hz(args.update_hz, 1.0 / 60.0)
    dt_pub = _dt_from_hz(args.pub_hz, 1.0 / 30.0)
    dt_render_move = _dt_from_hz(args.render_hz, 1.0 / 30.0)
    dt_render_idle = _dt_from_hz(args.idle_render_hz, 0.5)

    min_sleep = max(0.001, float(args.min_sleep_ms) / 1000.0)

    next_update = time.perf_counter()
    next_render = time.perf_counter()
    next_pub_wall = time.time()
    last_change_t = time.perf_counter()
    dt_events = 1.0 / 30.0
    next_events = time.perf_counter()

    viewer = None
    if not args.no_viewer:
        try:
            viewer = _make_mujoco_viewer(model, data)
            print("[VIEWER] created:", getattr(viewer, "window", None))
        except Exception as e:
            print(f"[ERROR] viewer create failed: {e}")
            raise

        for _ in range(3):
            try:
                viewer.render()
                glfw.poll_events()
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] initial render failed: {e}")
                break

        def _key_cb(window, key, scancode, action, mods):
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

            def publish_state(pub_node: Node, state_pub, joint_names_pub: List[str], q_list: List[float]):
                if len(q_list) != len(joint_names_pub):
                    return
                msg = JointState()
                msg.header.stamp = pub_node.get_clock().now().to_msg()
                msg.name = list(joint_names_pub)
                msg.position = list(q_list)
                state_pub.publish(msg)

            if now >= next_update:
                if (now - next_update) > (3.0 * dt_update):
                    next_update = now + dt_update
                else:
                    next_update += dt_update

                cmd_dict, rx = cmd_node.get_latest()

                if rx != last_ctrl_rx:
                    last_ctrl_rx = rx
                    last_ctrl = list(ctrl_hold)

                    vel_act_names = {"left_wheel_drive_act", "right_wheel_drive_act", "rear_wheel_drive_act"}

                    for jn, q_des in cmd_dict.items():
                        a = joint_to_act.get(jn, None)
                        if a is None:
                            continue

                        if act_names[a] in vel_act_names:
                            continue

                        if int(model.actuator_ctrllimited[a]) != 0:
                            lo = float(model.actuator_ctrlrange[a, 0])
                            hi = float(model.actuator_ctrlrange[a, 1])
                            if hi > lo:
                                q_des = max(lo, min(hi, q_des))

                        last_ctrl[a] = float(q_des)

                data.ctrl[:] = last_ctrl

                dt_phys = float(model.opt.timestep)
                steps = max(1, int(round(dt_update / max(1e-9, dt_phys))))
                for _ in range(steps):
                    mujoco.mj_step(model, data)

                last_change_t = time.perf_counter()

                now_wall = time.time()
                if now_wall >= next_pub_wall:
                    q_state = read_q_cspace_from_mujoco(data, mapping, cspace_joint_names)
                    publish_state(state_node, pub, cspace_joint_names, q_state)
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

        try:
            if executor is not None:
                try:
                    executor.remove_node(cmd_node)
                except Exception:
                    pass
                if world_node is not None:
                    try:
                        executor.remove_node(world_node)
                    except Exception:
                        pass
                try:
                    executor.remove_node(state_node)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            cmd_node.destroy_node()
        except Exception:
            pass
        if world_node is not None:
            try:
                world_node.destroy_node()
            except Exception:
                pass
        try:
            state_node.destroy_node()
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

from __future__ import annotations

import os
import sys

if os.environ.get("SPAWN_NO_FORCE_NVIDIA", "0").lower() not in ("1", "true", "yes", "y"):
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import argparse
import threading
import time
from typing import Any, Dict, List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image

import mujoco
import glfw

from curobo.util_file import load_yaml
from rclpy.executors import SingleThreadedExecutor

from capstone_pkg.utils.config import ROBOT_XML, ROBOT_YAML

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
    base = os.path.join(models_root, "furniture_sim")
    if not os.path.isdir(base):
        return

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
        # name/position 길이 다르면 zip이 짧은 쪽에 맞춰짐
        for n, p in zip(list(msg.name), list(msg.position)):
            if isinstance(n, str):
                d[n] = float(p)
        with self._lock:
            self._latest = d
            self._rx += 1

    def get_latest(self) -> Tuple[Dict[str, float], int]:
        with self._lock:
            return dict(self._latest), int(self._rx)



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


def apply_jointstate_qpos(data: mujoco.MjData, mapping: Dict[str, int], names: List[str], pos: List[float], eps: float) -> bool:
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


def read_q_cspace_from_mujoco(data: mujoco.MjData, mapping: Dict[str, int], cspace_joint_names: List[str]) -> List[float]:
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
    raise AttributeError(f"mujoco_viewer에서 MujocoViewer를 못 찾음: {getattr(mv,'__file__','?')}")


def _viewer_running(viewer) -> bool:
    try:
        return not glfw.window_should_close(viewer.window)
    except Exception:
        return True

def build_actuator_maps(model: mujoco.MjModel):
    # actuator idx -> name
    act_names = []
    for a in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        act_names.append(n if n else f"act{a}")

    # joint_name -> actuator idx (한 joint에 actuator 여러개면 마지막이 덮어씀)
    joint_to_act = {}
    for a in range(model.nu):
        j_id = int(model.actuator_trnid[a, 0])  # actuator가 걸린 joint id
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

    # actuator ctrlrange 있으면 clamp해주기
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

        # ✅ 너 MJCF에서 velocity actuator는 drive 3개뿐이라 이름으로 구분하는 게 제일 확실
        # (원하면 아래 set에 더 추가)
        is_velocity = name in ("left_wheel_drive_act", "right_wheel_drive_act", "rear_wheel_drive_act")

        if is_velocity:
            ctrl_hold[a] = _clamp(a, 0.0)
            continue

        # position actuator: 해당 joint의 현재 qpos를 목표로
        j_id = int(model.actuator_trnid[a, 0])
        qadr = int(model.jnt_qposadr[j_id])
        q = float(data.qpos[qadr])
        ctrl_hold[a] = _clamp(a, q)

    return ctrl_hold

def _make_image_msg(rgb_uint8, stamp, frame_id: str):
    """
    rgb_uint8: (H,W,3) uint8 RGB
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(rgb_uint8.shape[0])
    msg.width  = int(rgb_uint8.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb_uint8.shape[1] * 3)
    msg.data = rgb_uint8.tobytes()
    return msg


def _make_depth32_msg(depth_f32, stamp, frame_id: str):
    """
    depth_f32: (H,W) float32 depth (meters)
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(depth_f32.shape[0])
    msg.width  = int(depth_f32.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = int(depth_f32.shape[1] * 4)  # float32 = 4 bytes
    msg.data = depth_f32.tobytes()
    return msg

def main():
    if os.path.basename(__file__) == "mujoco_viewer.py":
        print("[WARN] 파일명이 mujoco_viewer.py면 import 충돌 가능. spawn_mujoco.py로 바꾸는 걸 권장.", file=sys.stderr)

    ap = argparse.ArgumentParser("MuJoCo spawn (CPU-friendly / kinematic)")

    ap.add_argument("--mjcf", default=ROBOT_XML)
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--sub_topic", default="/joint_states_cmd")
    ap.add_argument("--pub_topic", default="/joint_states")

    ap.add_argument("--update_hz", type=float, default=60.0)
    ap.add_argument("--pub_hz", type=float, default=30.0)
    ap.add_argument("--render_hz", type=float, default=30.0)      # ✅ 창 확인 위해 기본 30
    ap.add_argument("--idle_render_hz", type=float, default=2.0)  # ✅ 창 유지 위해 기본 2
    ap.add_argument("--idle_after_s", type=float, default=0.2)
    ap.add_argument("--min_sleep_ms", type=float, default=5.0)    # ✅ 창 반응성 위해 기본 5ms
    ap.add_argument("--eps", type=float, default=5e-3)

    ap.add_argument("--render_only_when_moving", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_viewer", action="store_true", default=False)

    ap.add_argument("--ctrl_topic", default="/ctrl_cmd")

    # -------- Camera publish (RealSense-style topics) --------
    ap.add_argument("--pub_color", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pub_depth", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cam_color_hz", type=float, default=30.0)

    ap.add_argument("--camera_left", default="camera_l")
    ap.add_argument("--camera_right", default="camera_r")

    ap.add_argument("--left_color_topic", default="/camera_l/color/image_raw")
    ap.add_argument("--right_color_topic", default="/camera_r/color/image_raw")
    ap.add_argument("--left_depth_topic", default="/camera_l/depth/image_raw")
    ap.add_argument("--right_depth_topic", default="/camera_r/depth/image_raw")

    # -------- ZED Mini publish (ZED-style topics) --------
    ap.add_argument("--pub_zed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zed_camera_left", default="zed_mini_left")
    ap.add_argument("--zed_camera_right", default="zed_mini_right")

    ap.add_argument("--zed_left_color_topic",  default="/zed_mini/zed_node/left/image_rect_color")
    ap.add_argument("--zed_right_color_topic", default="/zed_mini/zed_node/right/image_rect_color")
    ap.add_argument("--zed_depth_topic",       default="/zed_mini/zed_node/depth/depth_registered")
    args = ap.parse_args()

    models_root = os.path.dirname(os.path.abspath(args.mjcf))
    _ensure_furniture_sim_paths(models_root)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
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

    print("=== MuJoCo spawn (CPU-friendly / kinematic) ===")
    print(f"No viewer          : {args.no_viewer}")
    print(f"Render hz          : {args.render_hz}")
    print(f"Idle render hz     : {args.idle_render_hz}")
    print(f"Render only moving : {args.render_only_when_moving}")
    print()

    mujoco.mj_forward(model, data)

    act_names, joint_to_act = build_actuator_maps(model)

    # ✅ 2) (init qpos가 반영된) 현재 자세를 목표로 ctrl_hold 구성
    ctrl_hold = make_ctrl_hold_from_qpos(model, data, act_names)

    # ✅ 3) 시작부터 hold를 걸어둠
    data.ctrl[:] = ctrl_hold

    rclpy.init()

    last_ctrl_rx = 0
    last_ctrl = list(ctrl_hold)

    cmd_node = JointCmdIO(args.sub_topic) 

    # state publisher node
    state_node = Node("mujoco_state_pub")
    pub = state_node.create_publisher(JointState, args.pub_topic, 10)
    state_node.get_logger().info(f"Publishing MuJoCo state to: {args.pub_topic}")

    # ---------------- Camera publishers (Image) ----------------
    cam_color_pub_l = state_node.create_publisher(Image, args.left_color_topic, 10)
    cam_color_pub_r = state_node.create_publisher(Image, args.right_color_topic, 10)
    cam_depth_pub_l = state_node.create_publisher(Image, args.left_depth_topic, 10) if args.pub_depth else None
    cam_depth_pub_r = state_node.create_publisher(Image, args.right_depth_topic, 10) if args.pub_depth else None

    # MuJoCo renderer for offscreen camera rendering
    # width/height: 필요하면 argument로 뺄 수 있음
    cam_w, cam_h = 640, 480
    # camera id lookup (by name)
    cam_id_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_left)
    cam_id_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_right)
    if cam_id_left < 0:
        state_node.get_logger().error(f"Left camera not found in MJCF: {args.camera_left}")
    if cam_id_right < 0:
        state_node.get_logger().error(f"Right camera not found in MJCF: {args.camera_right}")

    # ---------------- ZED publishers (Image) ----------------
    zed_color_pub_l = state_node.create_publisher(Image, args.zed_left_color_topic, 10) if args.pub_zed else None
    zed_color_pub_r = state_node.create_publisher(Image, args.zed_right_color_topic, 10) if args.pub_zed else None
    zed_depth_pub   = state_node.create_publisher(Image, args.zed_depth_topic, 10) if (args.pub_zed and args.pub_depth) else None

    zed_cam_id_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.zed_camera_left)
    zed_cam_id_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.zed_camera_right)

    if args.pub_zed and zed_cam_id_l < 0:
        state_node.get_logger().error(f"ZED LEFT camera not found in MJCF: {args.zed_camera_left}")
    if args.pub_zed and zed_cam_id_r < 0:
        state_node.get_logger().error(f"ZED RIGHT camera not found in MJCF: {args.zed_camera_right}")
    
    # executor 하나에 노드 둘 다 등록하고, spin은 스레드 1개만
    executor = SingleThreadedExecutor()
    executor.add_node(cmd_node)
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
    dt_events = 1.0 / 30.0          # 이벤트 처리 30Hz면 충분
    next_events = time.perf_counter()
    dt_cam = _dt_from_hz(args.cam_color_hz, 1.0 / 30.0)
    next_cam_pub = time.perf_counter()


    viewer = None
    if not args.no_viewer:
        try:
            viewer = _make_mujoco_viewer(model, data)
            print("[VIEWER] created:", getattr(viewer, "window", None))
            # viewer 만든 직후(초기 렌더 for-loop 전에) 추가
            renderer = mujoco.Renderer(model, width=cam_w, height=cam_h)
        except Exception as e:
            print(f"[ERROR] viewer create failed: {e}")
            raise

        # ✅ 창이 “확실히 뜨도록” 시작 시 강제로 몇 프레임 렌더
        for _ in range(3):
            try:
                viewer.render()
                glfw.poll_events()
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] initial render failed: {e}")
                break

        # ESC로 종료
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

            def publish_state(pub_node: Node, pub, joint_names_pub: List[str], q_list: List[float]):
                if len(q_list) != len(joint_names_pub):
                    return
                msg = JointState()
                msg.header.stamp = pub_node.get_clock().now().to_msg()
                msg.name = list(joint_names_pub)
                msg.position = list(q_list)
                pub.publish(msg)


            # --- update tick ---
            if now >= next_update:
                if (now - next_update) > (3.0 * dt_update):
                    next_update = now + dt_update
                else:
                    next_update += dt_update

                cmd_dict, rx = cmd_node.get_latest()

                # ✅ 새 JointState 명령이 오면: ctrl_hold를 베이스로 깔고, 들어온 joint만 덮어쓴 ctrl 벡터 생성
                if rx != last_ctrl_rx:
                    last_ctrl_rx = rx
                    last_ctrl = list(ctrl_hold)  # 기본은 init 유지

                    # drive(velocity) actuator는 항상 0 유지(원하면 JointState로도 속도명령 따로 처리 가능)
                    vel_act_names = {"left_wheel_drive_act", "right_wheel_drive_act", "rear_wheel_drive_act"}

                    for jn, q_des in cmd_dict.items():
                        a = joint_to_act.get(jn, None)
                        if a is None:
                            continue  # 이 joint는 actuator가 없거나 이름 불일치

                        # velocity actuator는 JointState로 위치명령 주면 의미 없으니 무시(=0 유지)
                        if act_names[a] in vel_act_names:
                            continue

                        # ctrlrange clamp
                        if int(model.actuator_ctrllimited[a]) != 0:
                            lo = float(model.actuator_ctrlrange[a, 0])
                            hi = float(model.actuator_ctrlrange[a, 1])
                            if hi > lo:
                                q_des = max(lo, min(hi, q_des))

                        last_ctrl[a] = float(q_des)

                # ✅ 항상 last_ctrl로 구동 (한 번 명령 오면 계속 유지됨)
                data.ctrl[:] = last_ctrl


                # substep
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

            # --- cheap viewer events (throttled) ---
            if viewer is not None:
                now_ev = time.perf_counter()
                if now_ev >= next_events:
                    try:
                        glfw.poll_events()
                    except Exception:
                        pass
                    next_events = now_ev + dt_events

            # --- render tick (idle-aware) ---
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
                        did_render = True

            # --- camera publish tick (RealSense + ZED share same timing) ---
            now_cam = time.perf_counter()
            if now_cam >= next_cam_pub:
                if (now_cam - next_cam_pub) > (3.0 * dt_cam):
                    next_cam_pub = now_cam + dt_cam
                else:
                    next_cam_pub += dt_cam

                # -------- RealSense-style cameras (camera_l / camera_r) --------
                if args.pub_color and (cam_id_left >= 0) and (cam_id_right >= 0):
                    renderer.update_scene(data, camera=cam_id_left)
                    rgb_l = renderer.render()
                    stamp = state_node.get_clock().now().to_msg()
                    cam_color_pub_l.publish(_make_image_msg(rgb_l, stamp, args.camera_left))

                    renderer.update_scene(data, camera=cam_id_right)
                    rgb_r = renderer.render()
                    stamp = state_node.get_clock().now().to_msg()
                    cam_color_pub_r.publish(_make_image_msg(rgb_r, stamp, args.camera_right))

                    if args.pub_depth and (cam_depth_pub_l is not None) and (cam_depth_pub_r is not None):
                        renderer.update_scene(data, camera=cam_id_left)
                        depth_l = renderer.render(depth=True)
                        stamp = state_node.get_clock().now().to_msg()
                        cam_depth_pub_l.publish(_make_depth32_msg(depth_l.astype("float32"), stamp, args.camera_left))

                        renderer.update_scene(data, camera=cam_id_right)
                        depth_r = renderer.render(depth=True)
                        stamp = state_node.get_clock().now().to_msg()
                        cam_depth_pub_r.publish(_make_depth32_msg(depth_r.astype("float32"), stamp, args.camera_right))

                # -------- ZED stereo cameras (zed_mini_left / zed_mini_right) --------
                if args.pub_zed and (zed_cam_id_l >= 0) and (zed_cam_id_r >= 0) and (zed_color_pub_l is not None) and (zed_color_pub_r is not None):
                    # left
                    renderer.update_scene(data, camera=zed_cam_id_l)
                    zed_rgb_l = renderer.render()
                    stamp = state_node.get_clock().now().to_msg()
                    zed_color_pub_l.publish(_make_image_msg(zed_rgb_l, stamp, args.zed_camera_left))

                    # right
                    renderer.update_scene(data, camera=zed_cam_id_r)
                    zed_rgb_r = renderer.render()
                    stamp = state_node.get_clock().now().to_msg()
                    zed_color_pub_r.publish(_make_image_msg(zed_rgb_r, stamp, args.zed_camera_right))

                    # depth (옵션: left 기준 depth)
                    if args.pub_depth and (zed_depth_pub is not None):
                        renderer.update_scene(data, camera=zed_cam_id_l)
                        zed_depth = renderer.render(depth=True)
                        stamp = state_node.get_clock().now().to_msg()
                        zed_depth_pub.publish(_make_depth32_msg(zed_depth.astype("float32"), stamp, args.zed_camera_left))

                # ✅ 컨텍스트 복구: viewer가 검게 되는 문제 방지 (한 번만)
                if viewer is not None:
                    try:
                        glfw.make_context_current(viewer.window)
                    except Exception:
                        pass

            # --- sleep (avoid busy loop) ---
            now2 = time.perf_counter()

            targets = [next_update]

            if viewer is not None:
                # 렌더 예정이 있으면 그 시각도 타겟에 포함
                idle = (time.perf_counter() - last_change_t) > float(args.idle_after_s)
                do_render = not (args.render_only_when_moving and idle and not (args.idle_render_hz > 0.0))
                if do_render:
                    targets.append(next_render)
                # 이벤트 처리 타이밍도 포함(너무 늦게 처리하면 창이 뻑뻑해짐)
                targets.append(next_events)

            sleep_until = min(targets)
            sleep_time = sleep_until - now2

            if viewer is not None:
                # idle 렌더를 안 하는 경우: wait_events_timeout으로 CPU 사용 최소화 + 창 응답 유지
                if sleep_time > 0:
                    idle = (time.perf_counter() - last_change_t) > float(args.idle_after_s)
                    do_render = not (args.render_only_when_moving and idle and not (args.idle_render_hz > 0.0))
                    if not do_render:
                        # 이벤트가 오면 즉시 깨어남
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
            renderer.close()
        except Exception:
            pass
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass

        # executor / nodes 정리
        try:
            if executor is not None:
                try:
                    executor.remove_node(cmd_node)     # ✅ ctrl_node -> cmd_node
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
        try:
            state_node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass




if __name__ == "__main__":
    main()
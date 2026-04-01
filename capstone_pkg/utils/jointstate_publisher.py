from __future__ import annotations

import time
from typing import List, Optional, Sequence
import xml.etree.ElementTree as ET

import torch
from curobo.util_file import load_yaml


def get_cspace_joint_names(robot_yml: str) -> List[str]:
    cfg = load_yaml(robot_yml)
    robot_cfg = cfg["robot_cfg"] if isinstance(cfg, dict) and "robot_cfg" in cfg else cfg

    c1 = robot_cfg.get("cspace", {}) or {}
    names = c1.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    kin = robot_cfg.get("kinematics", {}) or {}
    c2 = (kin.get("cspace", {}) or {}).get("joint_names", None)
    if isinstance(c2, list) and c2:
        return [x for x in c2 if isinstance(x, str)]

    raise RuntimeError("YAML에서 cspace.joint_names를 찾을 수 없습니다.")


# -----------------------------
# Gripper helpers
# -----------------------------
def parse_gripper_joint_names_from_mujoco_xml(xml_path: str) -> List[str]:
    """
    MuJoCo XML에서 name이 'gripper_'로 시작하는 joint들을 찾아 반환.
    못 찾으면 fallback을 반환.
    """
    fallback = [
        "gripper_l_joint1", "gripper_l_joint2", "gripper_l_joint3", "gripper_l_joint4",
        "gripper_r_joint1", "gripper_r_joint2", "gripper_r_joint3", "gripper_r_joint4",
    ]
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        names: List[str] = []
        for j in root.iter("joint"):
            n = j.attrib.get("name", "")
            if n.startswith("gripper_"):
                names.append(n)

        def key(nm: str):
            side = 0 if "_l_" in nm else 1
            num = int("".join([c for c in nm if c.isdigit()]) or "0")
            return (side, num, nm)

        names = sorted(set(names), key=key)
        return names if names else fallback
    except Exception:
        return fallback


# -----------------------------
# Publishers
# -----------------------------

def publish_q_path_as_jointstate_keep_gripper_closed(
    *,
    q_path: torch.Tensor,            # (T, D_arm)  (보통 14)
    robot_yml: str,
    mujoco_xml: str,
    close_value: float = 1.0,
    cmd_topic: str = "/joint_states_cmd",
    hz: float = 30.0,
    repeat: int = 1,
    start_delay_s: float = 0.2,
    hold_last_s: float = 0.0,
    node_name: str = "step2_path_publisher_with_gripper",
    # ✅ fallback 옵션: cmd_topic이 gripper를 안 먹는 경우 gripper_cmd를 keepalive로 같이 쏠 수 있음
    gripper_cmd_topic: Optional[str] = None,   # 예: "/gripper_cmd"
    gripper_dim: int = 2,
):
    """
    팔 경로를 publish할 때 gripper joint를 같이 포함해서 "닫힌 채로 이동"하도록 만든 버전.

    동작:
    - JointState(cmd_topic)에 name = [cspace_joint_names + gripper_joints]
    - position = [q_arm + close_value*len(gripper_joints)]
    - (옵션) gripper_cmd_topic을 주면 Float64MultiArray([close_value]*gripper_dim)를 hz로 keepalive publish도 병행
      -> cmd_topic이 gripper를 무시하는 시스템에서도 닫힘 유지에 도움.
    """
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    arm_joint_names = get_cspace_joint_names(robot_yml)  # 보통 14개
    gripper_joints = parse_gripper_joint_names_from_mujoco_xml(mujoco_xml)

    q_cpu = q_path.detach().to("cpu")
    if q_cpu.dim() != 2:
        raise ValueError(f"q_path must be (T,D), got shape={tuple(q_cpu.shape)}")
    if q_cpu.shape[1] != len(arm_joint_names):
        raise ValueError(
            f"DOF mismatch: q_path D={q_cpu.shape[1]} vs arm_joint_names={len(arm_joint_names)}"
        )

    # output message fields
    names_out = list(arm_joint_names) + list(gripper_joints)
    grip_pos = [float(close_value)] * len(gripper_joints)

    dt = 1.0 / max(1e-6, float(hz))

    rclpy.init()
    node = Node(node_name)

    pub_js = node.create_publisher(JointState, cmd_topic, 10)

    pub_grip = None
    grip_msg = None
    if isinstance(gripper_cmd_topic, str) and gripper_cmd_topic:
        pub_grip = node.create_publisher(Float64MultiArray, gripper_cmd_topic, 10)
        grip_msg = Float64MultiArray()
        grip_msg.data = [float(close_value)] * int(gripper_dim)

    if start_delay_s > 0:
        time.sleep(float(start_delay_s))

    def _publish_one(q_arm_row: Sequence[float]):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = names_out
        msg.position = list(map(float, q_arm_row)) + grip_pos
        pub_js.publish(msg)

        # (옵션) keepalive
        if pub_grip is not None and grip_msg is not None:
            pub_grip.publish(grip_msg)

    # warmup
    _publish_one(q_cpu[0].tolist())
    rclpy.spin_once(node, timeout_sec=0.0)

    for _ in range(int(max(1, repeat))):
        for t in range(q_cpu.shape[0]):
            _publish_one(q_cpu[t].tolist())
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)

    if hold_last_s > 0:
        t_end = time.time() + float(hold_last_s)
        last = q_cpu[-1].tolist()
        while time.time() < t_end and rclpy.ok():
            _publish_one(last)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)

    node.destroy_node()
    rclpy.shutdown()

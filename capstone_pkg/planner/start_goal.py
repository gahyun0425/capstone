#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

import math
import time
import numpy as np
import torch
import rclpy

from capstone_pkg.utils.config import LEFT_GRIPPER, RIGHT_GRIPPER
from capstone_pkg.warmup.warmup_all import WarmupBundle, init_all_and_warmup, now, fmt_s
from capstone_pkg.kinematics.curobo_ik import solve_batch_bimanual
from capstone_pkg.constraint_projection.constraint import RigidConstraint, build_bimanual_fk_robotworld
from capstone_pkg.collision_check.collision import get_self_collision_checker


# -----------------------------
# quaternion helpers (wxyz)
# -----------------------------
def _quat_conj_wxyz(q: List[float]) -> List[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_mul_wxyz(a: List[float], b: List[float]) -> List[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_normalize_wxyz(q: List[float]) -> List[float]:
    n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if n2 < 1e-18:
        return [1.0, 0.0, 0.0, 0.0]
    inv = 1.0 / math.sqrt(n2)
    return [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]


def _quat_rotate_wxyz(q: List[float], v: List[float]) -> List[float]:
    """R(q) * v (q: wxyz, v: xyz)"""
    qn = _quat_normalize_wxyz(q)
    vq = [0.0, v[0], v[1], v[2]]
    out = _quat_mul_wxyz(_quat_mul_wxyz(qn, vq), _quat_conj_wxyz(qn))
    return [out[1], out[2], out[3]]


def _quat_make_continuous_w_pos_w(q: List[float]) -> List[float]:
    """q와 -q는 같은 회전이므로, w>=0로 맞춰서 출력/최적화 튐을 완화"""
    if q[0] < 0.0:
        return [-q[0], -q[1], -q[2], -q[3]]
    return q


# -----------------------------
# math / input helpers
# -----------------------------
def euler_rpy_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
    """roll-pitch-yaw (deg) -> quat (wxyz)"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _quat_make_continuous_w_pos_w(_quat_normalize_wxyz([qw, qx, qy, qz]))


def quat_wxyz_to_euler_rpy_deg(q: List[float]) -> List[float]:
    """quat (wxyz) -> roll-pitch-yaw (deg) using the same ZYX convention."""
    qw, qx, qy, qz = _quat_normalize_wxyz(q)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _parse_line_floats(line: str, n: int) -> List[float]:
    toks = [t for t in line.replace(",", " ").split() if t.strip()]
    if len(toks) != n:
        raise ValueError(f"{n}개 숫자를 입력해야 합니다. got={len(toks)} line='{line}'")
    return [float(x) for x in toks]


def prompt_target_left_only(
    left_pos: Optional[List[float]] = None,
    left_rpy_deg: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    터미널에서 LEFT 목표만 입력받는 헬퍼.
    (이 함수가 싫으면, get_start_and_goal_from_topic_and_ik 호출 전에
     left_pos/left_quat를 외부에서 만들어 _auto_right_target_from_left만 써도 됨)
    """
    if left_pos is not None or left_rpy_deg is not None:
        if left_pos is None or left_rpy_deg is None:
            raise ValueError("left_pos와 left_rpy_deg는 함께 지정되어야 합니다.")
        if len(left_pos) != 3 or len(left_rpy_deg) != 3:
            raise ValueError("left_pos와 left_rpy_deg는 각각 3개여야 합니다.")
        pos = [float(v) for v in left_pos]
        rpy_deg = [float(v) for v in left_rpy_deg]
        print("\n[LEFT] target 입력 생략: CLI로 고정 목표 사용")
        print(f"  pos xyz (m)     : {pos}")
        print(f"  rpy (deg)       : {rpy_deg}")
    else:
        print("\n[LEFT] target 입력 (RIGHT는 초기 상대관계로 자동 계산)")
        pos = _parse_line_floats(input("  pos xyz (m)     : ").strip(), 3)
        rpy_deg = _parse_line_floats(input("  rpy (deg)       : ").strip(), 3)
    quat = euler_rpy_deg_to_quat_wxyz(rpy_deg[0], rpy_deg[1], rpy_deg[2])
    return pos, quat


# -----------------------------
# ROS JointState grabber
# -----------------------------
class JointStateGrabber(Node):
    def __init__(self, topic: str, joint_names: List[str]):
        super().__init__("ik_qstart_jointstate_grabber")
        self.joint_names = list(joint_names)
        self._latest: Optional[List[float]] = None
        self._last_msg_names: List[str] = []
        self._last_missing: List[str] = []
        self._rx_count = 0

        self.create_subscription(JointState, topic, self._cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing JointState: {topic}")

    def _cb(self, msg: JointState):
        self._rx_count += 1
        if not msg.name or not msg.position:
            return
        if len(msg.name) != len(msg.position):
            return

        self._last_msg_names = [n for n in msg.name if isinstance(n, str)]
        pos_map = {n: msg.position[i] for i, n in enumerate(msg.name) if isinstance(n, str)}
        missing = [jn for jn in self.joint_names if jn not in pos_map]
        if missing:
            self._last_missing = missing
            return

        self._last_missing = []
        self._latest = [float(pos_map[jn]) for jn in self.joint_names]

    def get_latest(self) -> Optional[List[float]]:
        return self._latest

    def get_debug_snapshot(self) -> Tuple[int, List[str], List[str]]:
        return self._rx_count, list(self._last_msg_names), list(self._last_missing)


def get_q_start_from_jointstate(topic: str, joint_names: List[str], timeout_sec: float = 3.0) -> List[float]:
    """
    JointState 토픽에서 joint_names 순서로 q_start(cspace)를 수집.
    - rclpy.init/shutdown을 내부에서 처리하므로, "이미 init된 노드 환경"에서는
      이 함수를 직접 쓰기보다는 Node를 재사용하는 구조로 바꾸는 게 더 안전할 수 있음.
    """
    rclpy.init()
    node = JointStateGrabber(topic, joint_names)

    q_start = None
    debug_snapshot = (0, [], [])
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            q_start = node.get_latest()
            if q_start is not None:
                break
            if (time.time() - t0) > timeout_sec:
                break
    finally:
        debug_snapshot = node.get_debug_snapshot()
        node.destroy_node()
        rclpy.shutdown()

    if q_start is None:
        rx_count, last_msg_names, last_missing = debug_snapshot
        detail = ""
        if rx_count == 0:
            detail = " JointState 메시지 자체를 한 번도 못 받았습니다."
        elif last_missing:
            detail = (
                f" JointState는 받았지만 필요한 joint가 없습니다."
                f" missing={last_missing}, msg_names={last_msg_names}"
            )
        raise TimeoutError(f"q_start를 JointState로 못 받았습니다. topic={topic}.{detail}")
    return q_start


# -----------------------------
# start_goal core helpers
# -----------------------------
def _auto_right_target_from_left(
    *,
    robot_yml: str,
    left_ee: str,
    right_ee: str,
    q_start_cspace: List[float],
    left_pos: List[float],
    left_quat_wxyz: List[float],
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    """초기(q_start)에서 얻은 상대변환을 고정해, LEFT 목표만으로 RIGHT 목표 pose를 자동 생성"""
    q_ref = torch.tensor(q_start_cspace, device=device, dtype=torch.float32)

    c = RigidConstraint(
        robot_yml=robot_yml,
        left_ee=left_ee,
        right_ee=right_ee,
        q_ref=q_ref,
        device=device,
        dtype=torch.float32,
        mode="se3",
    )

    p_rel_ref = [float(x) for x in c.p_rel_ref.detach().cpu().tolist()]  # (3,)
    q_rel_ref = [float(x) for x in c.q_rel_ref.detach().cpu().tolist()]  # (4,) wxyz

    rot_p = _quat_rotate_wxyz(left_quat_wxyz, p_rel_ref)
    right_pos = [left_pos[0] + rot_p[0], left_pos[1] + rot_p[1], left_pos[2] + rot_p[2]]

    right_quat = _quat_mul_wxyz(left_quat_wxyz, q_rel_ref)
    right_quat = _quat_make_continuous_w_pos_w(_quat_normalize_wxyz(right_quat))

    return right_pos, right_quat


def _get_current_left_pose(
    *,
    robot_yml: str,
    left_ee: str,
    right_ee: str,
    q_start_cspace: List[float],
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    fk = build_bimanual_fk_robotworld(
        robot_yml=robot_yml,
        left_ee=left_ee,
        right_ee=right_ee,
        device=device,
        dtype=torch.float32,
    )

    q_ref = torch.tensor(q_start_cspace, device=device, dtype=torch.float32).view(1, -1)
    p_left, q_left = fk.fk_left_pose(q_ref)
    return (
        [float(x) for x in p_left.squeeze(0).detach().cpu().tolist()],
        [float(x) for x in q_left.squeeze(0).detach().cpu().tolist()],
    )


def _apply_planar_xy_target_filter(
    *,
    left_pos: List[float],
    left_quat_wxyz: List[float],
    current_left_pos: List[float],
    current_left_quat_wxyz: List[float],
) -> Tuple[List[float], List[float]]:
    current_rpy_deg = quat_wxyz_to_euler_rpy_deg(current_left_quat_wxyz)
    target_rpy_deg = quat_wxyz_to_euler_rpy_deg(left_quat_wxyz)

    filtered_pos = [float(left_pos[0]), float(left_pos[1]), float(current_left_pos[2])]
    filtered_rpy_deg = [float(current_rpy_deg[0]), float(current_rpy_deg[1]), float(target_rpy_deg[2])]
    filtered_quat = euler_rpy_deg_to_quat_wxyz(
        filtered_rpy_deg[0],
        filtered_rpy_deg[1],
        filtered_rpy_deg[2],
    )

    print("[planar_xy] using current left z / roll / pitch and target yaw only")
    print("[planar_xy] left target xyz (filtered) =", [round(x, 6) for x in filtered_pos])
    print("[planar_xy] left target rpy (filtered) =", [round(x, 6) for x in filtered_rpy_deg])
    return filtered_pos, filtered_quat


# -----------------------------
# public API (import해서 쓰는 함수)
# -----------------------------
def get_start_and_goal_from_topic_and_ik(
    *,
    robot_yml: str,
    jointstate_topic: str,
    joint_names: List[str],
    timeout_sec: float = 5.0,
    target_left_xyz: Optional[List[float]] = None,
    target_left_rpy_deg: Optional[List[float]] = None,
    left_ee: str = LEFT_GRIPPER,
    right_ee: str = RIGHT_GRIPPER,
    tries: int = 10,
    select: Literal["first_free", "min_penetration"] = "first_free",
    device_str: str = "cuda",
    bundle: Optional[WarmupBundle] = None,
    world_yml: Optional[str] = None,
    fail_if_start_in_collision: bool = False,
    ik_warmup_iters: int = 3,
    cuda_ctx_warmup_iters: int = 2,
    selfcol_warmup_iters: int = 2,
    robotworld_warmup_iters: int = 2,
    ik_batch: int = 100,
    topk: int = 16,
    planar_xy: bool = False,
) -> Tuple[List[float], List[List[float]], float, float]:
    """
    JointState(q_start) -> (LEFT target 입력) -> RIGHT target 자동 -> batch IK -> collision-free 필터 -> topK 반환

    return: (q_start_list, q_goal_topk_list, best_penetration, t_pose_input_done)
    """
    # 1) q_start from ROS
    q_start = get_q_start_from_jointstate(
        topic=jointstate_topic,
        joint_names=joint_names,
        timeout_sec=timeout_sec,
    )
    print("[start_goal] q_start (cspace) =", q_start)

    # 2) warmup bundle 확보
    if bundle is None:
        device = torch.device("cuda") if (device_str == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
        bundle = init_all_and_warmup(
            robot_yml=robot_yml,
            left_ee=left_ee,
            right_ee=right_ee,
            device=device,
            ik_warmup_iters=ik_warmup_iters,
            cuda_ctx_warmup_iters=cuda_ctx_warmup_iters,
            selfcol_warmup_iters=selfcol_warmup_iters,
            robotworld_warmup_iters=robotworld_warmup_iters,
            log_prefix="[start_goal]",
        )

    solver = bundle.ik_solver
    device = bundle.device

    # 3) world 포함 collision checker 강제
    cpu = (device.type == "cpu")
    checker = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)

    if world_yml is None:
        print("[start_goal][WARN] world_yml=None → world collision을 체크하지 않습니다.")
    else:
        print(f"[start_goal] world_yml = {world_yml}")

    # 4) q_start collision 체크
    in_col_s, d_self_s, d_world_s = checker.check_single(q_start)
    pen_s = max(float(d_self_s), float(d_world_s))
    print(
        f"[start_goal] q_start collision: "
        f"self={d_self_s:.6f} world={d_world_s:.6f} pen={pen_s:.6f} -> "
        f"{'IN COLLISION' if in_col_s else 'FREE'}"
    )
    if in_col_s and d_self_s > 0.0 and hasattr(checker, "print_self_collision_link_pairs"):
        checker.print_self_collision_link_pairs(
            q_start,
            topk=5,
            prefix="[start_goal][DEBUG][q_start]",
        )
    if in_col_s and fail_if_start_in_collision:
        raise RuntimeError("[start_goal] q_start가 world/self collision 상태입니다. (fail_if_start_in_collision=True)")

    # 5) LEFT 목표 입력
    print("[start_goal] ✅ 모든 준비 완료. 이제 LEFT 목표만 입력하세요. (RIGHT는 자동)")
    l_pos, l_quat = prompt_target_left_only(target_left_xyz, target_left_rpy_deg)

    if planar_xy:
        cur_left_pos, cur_left_quat = _get_current_left_pose(
            robot_yml=robot_yml,
            left_ee=left_ee,
            right_ee=right_ee,
            q_start_cspace=q_start,
            device=device,
        )
        print("[planar_xy] current left xyz =", [round(x, 6) for x in cur_left_pos])
        print("[planar_xy] current left rpy =", [round(x, 6) for x in quat_wxyz_to_euler_rpy_deg(cur_left_quat)])
        l_pos, l_quat = _apply_planar_xy_target_filter(
            left_pos=l_pos,
            left_quat_wxyz=l_quat,
            current_left_pos=cur_left_pos,
            current_left_quat_wxyz=cur_left_quat,
        )

    # 6) RIGHT 목표 자동 생성
    r_pos, r_quat = _auto_right_target_from_left(
        robot_yml=robot_yml,
        left_ee=left_ee,
        right_ee=right_ee,
        q_start_cspace=q_start,
        left_pos=l_pos,
        left_quat_wxyz=l_quat,
        device=device,
    )

    print("[AUTO-RIGHT] pos (m)        =", [round(x, 6) for x in r_pos])
    print("[AUTO-RIGHT] quat (wxyz)    =", [round(x, 6) for x in r_quat])

    t_pose_input_done = time.perf_counter()

    # -----------------------------
    # batch IK -> collision-free -> rank -> topK
    # -----------------------------
    t_ik0 = now(device)

    left_xyz_batch = [l_pos for _ in range(int(ik_batch))]
    left_quat_batch = [l_quat for _ in range(int(ik_batch))]
    right_xyz_batch = [r_pos for _ in range(int(ik_batch))]
    right_quat_batch = [r_quat for _ in range(int(ik_batch))]

    ts0 = now(device)
    outs = solve_batch_bimanual(
        solver,
        left_xyz_batch,
        left_quat_batch,
        right_xyz_batch,
        right_quat_batch,
        q_start_cspace=q_start,
        parallel_cuda_streams=True,
    )
    ts1 = now(device)

    cand_q: List[List[float]] = []
    for o in outs:
        if (not o.success) or (o.q_cspace is None):
            continue
        cand_q.append(list(o.q_cspace))

    if len(cand_q) == 0:
        raise RuntimeError("[start_goal] ❌ batch IK success=0")

    tc0 = now(device)
    free_q: List[List[float]] = []
    free_pen: List[float] = []
    rejected_pen: List[Tuple[float, float, float, List[float]]] = []
    for q in cand_q:
        in_col, d_self_max, d_world_max = checker.check_single(q)
        pen = max(float(d_self_max), float(d_world_max))
        if in_col:
            rejected_pen.append((pen, float(d_self_max), float(d_world_max), list(q)))
            continue
        free_q.append(q)
        free_pen.append(pen)
    tc1 = now(device)

    if len(free_q) == 0:
        print(f"[start_goal][DEBUG] batch IK success={len(cand_q)} but all candidates were rejected by collision filtering.")
        if rejected_pen:
            best_pen_rej, best_self_rej, best_world_rej, best_q_rej = min(rejected_pen, key=lambda x: x[0])
            print(
                "[start_goal][DEBUG] best rejected candidate: "
                f"self={best_self_rej:.6f} world={best_world_rej:.6f} pen={best_pen_rej:.6f}"
            )
            if best_self_rej > 0.0 and hasattr(checker, "print_self_collision_link_pairs"):
                checker.print_self_collision_link_pairs(
                    best_q_rej,
                    topk=5,
                    prefix="[start_goal][DEBUG][best_rejected]",
                )
        raise RuntimeError("[start_goal] ❌ batch IK collision-free=0")

    q_start_np = np.asarray(q_start, dtype=np.float64)

    scored = []
    for q, pen in zip(free_q, free_pen):
        q_np = np.asarray(q, dtype=np.float64)
        d = float(np.linalg.norm(q_np - q_start_np))
        scored.append((d, pen, q))

    scored.sort(key=lambda x: x[0])  # distance ascending
    scored = scored[: min(int(topk), len(scored))]

    q_goal_topk = [q for (d, pen, q) in scored]
    best_pen = float(scored[0][1])

    t_ik1 = now(device)

    print("\n================ IK-ONLY TIME (start_goal) ================")
    print(f"[TIME][IK-only] total (batch solve+validate+rank) : {fmt_s(t_ik1 - t_ik0)}")
    print(f"[TIME][IK-only] batch_solve_total                 : {fmt_s(ts1 - ts0)}")
    print(f"[TIME][IK-only] validate_total                    : {fmt_s(tc1 - tc0)}")
    print(f"[start_goal] batch IK success={len(cand_q)} free={len(free_q)} topk={len(q_goal_topk)}")
    print("===========================================================\n")

    return q_start, q_goal_topk, float(best_pen), float(t_pose_input_done)


# -----------------------------
# (optional) kept dataclass (not used now)
# -----------------------------
@dataclass
class SelectedSolution:
    q_cspace: List[float]
    d_self_max: float
    attempt_idx: int

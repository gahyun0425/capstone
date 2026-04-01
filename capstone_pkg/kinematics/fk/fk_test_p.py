#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinocchio FK example (URDF + EE frames 그대로 사용)
- 코드 안에 왼팔/오른팔 각도값을 미리 넣어둠 (터미널 입력 X)
- ❗부호 보정(반전) 없이: URDF 모델이 정의한 그대로 각도를 q에 넣고 FK 수행
- ✅ EE frame pose를 base_link 기준으로 출력 (base_link -> EE)
"""

import numpy as np
import pinocchio as pin

from capstone_pkg.utils.config import ROBOT_URDF, BASE_FRAME, LEFT_EE_FRAME, RIGHT_EE_FRAME, LEFT_JOINTS, RIGHT_JOINTS

# -----------------------------
# ✅ 여기서 각도값을 "코드 상"에 입력해두면 됨
# 단위: degree (USE_DEG=False면 rad로 넣으면 됨)
# -----------------------------
USE_DEG = False

LEFT_ANGLES = [-2.462579097321958, -0.7563391772717571, -1.1751908326761291, 1.6275928025513056, -0.40875074419563934, -0.7880831536874251, -0.3147234619836178]       # arm_l_joint1..7
RIGHT_ANGLES = [-2.462579097322224, 0.75633917727171, 1.175190832675681, 1.6275928025513082, 0.4087507441955061, -0.7880831536871611, 0.31472346198366996]    # arm_r_joint1..7


# -----------------------------
# 유틸
# -----------------------------
def _frame_id_or_raise(model: pin.Model, frame_name: str) -> int:
    fid = model.getFrameId(frame_name)
    if fid == len(model.frames):
        raise RuntimeError(f"Frame '{frame_name}' not found.")
    return fid

def _joint_id_or_raise(model: pin.Model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    if jid == 0:
        raise RuntimeError(f"Joint '{joint_name}' not found.")
    return jid

def set_arm_q(model: pin.Model, q_model: np.ndarray, joint_names: list[str], angles_rad: list[float]):
    """
    joint_names 순서대로 angles_rad(URDF 모델 기준 rad)를 그대로 q_model에 반영.
    """
    if len(angles_rad) != len(joint_names):
        raise ValueError(f"angles length({len(angles_rad)}) != joint_names length({len(joint_names)})")

    for jname, ang in zip(joint_names, angles_rad):
        jid = _joint_id_or_raise(model, jname)
        j = model.joints[jid]
        if j.nq != 1:
            raise RuntimeError(f"Joint '{jname}' has nq={j.nq}; this example assumes 1DoF joints.")
        q_model[j.idx_q] = float(ang)

def se3_to_pretty(M: pin.SE3):
    p = M.translation
    R = M.rotation
    rpy = pin.rpy.matrixToRpy(R)  # rad
    return {
        "xyz": p.tolist(),
        "rpy_deg": (np.rad2deg(np.array(rpy))).tolist(),
    }


def main():
    model = pin.buildModelFromUrdf(ROBOT_URDF)
    data = model.createData()

    base_fid = _frame_id_or_raise(model, BASE_FRAME)
    left_fid = _frame_id_or_raise(model, LEFT_EE_FRAME)
    right_fid = _frame_id_or_raise(model, RIGHT_EE_FRAME)

    # 각도 단위 처리
    if USE_DEG:
        left_angles_rad = np.deg2rad(np.array(LEFT_ANGLES, dtype=float)).tolist()
        right_angles_rad = np.deg2rad(np.array(RIGHT_ANGLES, dtype=float)).tolist()
    else:
        left_angles_rad = [float(x) for x in LEFT_ANGLES]
        right_angles_rad = [float(x) for x in RIGHT_ANGLES]

    # q 구성 (neutral에서 시작)
    q_model = pin.neutral(model).copy()
    set_arm_q(model, q_model, LEFT_JOINTS, left_angles_rad)
    set_arm_q(model, q_model, RIGHT_JOINTS, right_angles_rad)

    # FK
    pin.forwardKinematics(model, data, q_model)
    pin.updateFramePlacements(model, data)

    oMbase  = data.oMf[base_fid]
    oMleft  = data.oMf[left_fid]
    oMright = data.oMf[right_fid]

    # ✅ base_link 기준 (base_link -> EE)
    bMleft  = oMbase.actInv(oMleft)
    bMright = oMbase.actInv(oMright)

    L = se3_to_pretty(bMleft)
    R = se3_to_pretty(bMright)

    print("=== Pinocchio FK (base_link 기준: base_link -> EE) ===")
    print(f"URDF: {ROBOT_URDF}")
    print(f"BASE_FRAME   : {BASE_FRAME}")
    print(f"LEFT_EE_FRAME : {LEFT_EE_FRAME}")
    print(f"RIGHT_EE_FRAME: {RIGHT_EE_FRAME}")
    print("")
    print(f"LEFT angles ({'deg' if USE_DEG else 'rad'}):  {LEFT_ANGLES}")
    print(f"RIGHT angles ({'deg' if USE_DEG else 'rad'}): {RIGHT_ANGLES}")
    print("")
    print(f"[LEFT]  xyz(base) = {L['xyz']}")
    print(f"[LEFT]  rpy(deg, base) = {L['rpy_deg']}")
    print(f"[RIGHT] xyz(base) = {R['xyz']}")
    print(f"[RIGHT] rpy(deg, base) = {R['rpy_deg']}")


if __name__ == "__main__":
    main()

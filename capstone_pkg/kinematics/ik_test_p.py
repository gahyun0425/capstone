#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinocchio IK example (URDF 그대로)
- ❗부호 보정(반전) 없이: URDF 모델이 정의한 그대로 IK 업데이트
- ✅ 목표 pose를 base_link 기준으로 입력/해석 (base_link -> EE)
- 왼팔/오른팔 각각 frame 목표를 맞추는 반복 IK (log6 + damped least squares)
"""

import argparse
import ctypes
import os
import tempfile
import sys
import time
import numpy as np
from numpy.linalg import norm, solve

from capstone_pkg.utils.config import ROBOT_URDF, BASE_FRAME, LEFT_EE_FRAME, RIGHT_EE_FRAME, LEFT_PREFIX, RIGHT_PREFIX

def _prefer_venv_pinocchio():
    """
    ROS 환경이 섞인 셸에서도 venv의 cmeel pinocchio/eigenpy를 우선 사용한다.
    """
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_site = os.path.join(sys.prefix, "lib", pyver, "site-packages")
    cmeel_site = os.path.join(venv_site, "cmeel.prefix", "lib", pyver, "site-packages")
    cmeel_lib = os.path.join(venv_site, "cmeel.prefix", "lib")

    if os.path.isdir(cmeel_site):
        if cmeel_site in sys.path:
            sys.path.remove(cmeel_site)
        sys.path.insert(0, cmeel_site)

    if os.path.isdir(venv_site):
        if venv_site in sys.path:
            sys.path.remove(venv_site)
        sys.path.insert(0, venv_site)

    if os.path.isdir(cmeel_site):
        ros_py_paths = []
        for p in list(sys.path):
            if "/opt/ros/" in p and p.endswith("site-packages"):
                ros_py_paths.append(p)
        for p in ros_py_paths:
            sys.path.remove(p)

    libeigenpy = os.path.join(cmeel_lib, "libeigenpy.so")
    if os.path.isfile(libeigenpy):
        try:
            ctypes.CDLL(libeigenpy, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_prefer_venv_pinocchio()
import pinocchio as pin

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12


def rpy_deg_to_rot(rpy_deg):
    rpy_rad = np.deg2rad(np.array(rpy_deg, dtype=float))
    return pin.rpy.rpyToMatrix(rpy_rad)


def rot_to_rpy_deg(R):
    rpy = pin.rpy.matrixToRpy(R)  # rad
    return np.rad2deg(np.array(rpy)).tolist()


def select_arm_velocity_indices(model, arm_prefix):
    idxs = []
    for jid, jname in enumerate(model.names):
        if jid == 0:
            continue
        if jname.startswith(arm_prefix):
            j = model.joints[jid]
            idxs.extend(list(range(j.idx_v, j.idx_v + j.nv)))
    idxs = sorted(set(idxs))
    if len(idxs) == 0:
        raise RuntimeError(f"arm_prefix='{arm_prefix}' 로 시작하는 조인트를 못 찾았어.")
    return idxs


def select_arm_configuration_indices(model, arm_prefix):
    idxs = []
    for jid, jname in enumerate(model.names):
        if jid == 0:
            continue
        if jname.startswith(arm_prefix):
            j = model.joints[jid]
            idxs.extend(list(range(j.idx_q, j.idx_q + j.nq)))
    idxs = sorted(set(idxs))
    if len(idxs) == 0:
        raise RuntimeError(f"arm_prefix='{arm_prefix}' 로 시작하는 조인트를 못 찾았어.")
    return idxs


def init_meshcat_visualizer(model, urdf_path, open_browser):
    try:
        from pinocchio.visualize import MeshcatVisualizer
    except Exception as exc:
        raise RuntimeError(
            "MeshcatVisualizer import 실패. 'pip install meshcat pin' 설치 후 다시 실행해줘."
        ) from exc

    urdf_for_vis = _prepare_urdf_for_meshcat(urdf_path)

    try:
        collision_model = pin.buildGeomFromUrdf(model, urdf_for_vis, pin.GeometryType.COLLISION)
        visual_model = pin.buildGeomFromUrdf(model, urdf_for_vis, pin.GeometryType.VISUAL)
    except Exception as exc:
        raise RuntimeError(
            "URDF visual/collision geometry 로드 실패. URDF 경로/mesh 경로를 확인해줘."
        ) from exc

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=open_browser)
    viz.loadViewerModel("ik_result")
    return viz


def _prepare_urdf_for_meshcat(urdf_path):
    """
    Meshcat 로딩 실패를 유발하는 깨진 절대 mesh 경로를 임시 URDF에서 교체한다.
    """
    broken_d405_uri = (
        "file:///root/ros2_ws/install/realsense2_description/"
        "share/realsense2_description/meshes/d405.stl"
    )
    fallback_mesh = (
        "/home/gaga/capstone_ws/src/ai_worker/ffw_description/"
        "meshes/common/follower/zedm.stl"
    )

    if not os.path.isfile(urdf_path):
        return urdf_path

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    if broken_d405_uri not in urdf_text:
        return urdf_path

    if not os.path.isfile(fallback_mesh):
        print(f"[Meshcat][WARN] fallback mesh 없음: {fallback_mesh}")
        return urdf_path

    patched = urdf_text.replace(broken_d405_uri, f"file://{fallback_mesh}")
    if patched == urdf_text:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False, encoding="utf-8")
    with tmp:
        tmp.write(patched)
    print(f"[Meshcat] 임시 URDF 생성: {tmp.name}")
    print(f"[Meshcat] d405.stl -> {fallback_mesh} 로 대체")
    return tmp.name


def solve_ik_one_arm(
    model, data, q0_model,
    base_frame_id, ee_frame_id,
    bMdes,                 # ✅ base_link 기준 목표 (base_link -> EE)
    active_v_idx,
    eps=1e-4, IT_MAX=1000, DT=1e-1, damp=1e-12, verbose=True,
    position_only=False
):
    """
    반복 IK:
      - 매 반복에서 현재 oMbase를 구하고,
      - 목표를 oMdes = oMbase * bMdes 로 world(o) 기준으로 변환한 뒤,
      - oMee와의 log6 오차를 줄이는 방식.

    이렇게 하면 base_link가 움직이는 모델(부유/이동 베이스)이라도
    'base_link 기준 목표'가 일관되게 유지됨.
    """
    q_model = q0_model.copy()

    i = 0
    success = False
    err = np.zeros(6)

    while True:
        pin.forwardKinematics(model, data, q_model)
        pin.updateFramePlacements(model, data)

        oMbase = data.oMf[base_frame_id]
        oMee = data.oMf[ee_frame_id]

        # ✅ base_link 기준 목표를 world(o)로 변환
        oMdes = oMbase * bMdes

        iMd = oMee.actInv(oMdes)
        err = pin.log6(iMd).vector

        if position_only:
            err[3:6] = 0.0

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            break

        J6 = pin.computeFrameJacobian(model, data, q_model, ee_frame_id, pin.ReferenceFrame.LOCAL)
        J6 = -pin.Jlog6(iMd.inverse()) @ J6

        if position_only:
            J6[3:6, :] = 0.0

        # active dof만 사용
        J_active = J6[:, active_v_idx]  # 6 x na

        v_active = -J_active.T @ solve(
            J_active @ J_active.T + damp * np.eye(6), err
        )

        v_model = np.zeros(model.nv)
        v_model[active_v_idx] = v_active

        q_model = pin.integrate(model, q_model, v_model * DT)

        if verbose and (i % 10 == 0):
            print(f"{i}: error = {err.T}")

        i += 1

    return success, q_model, err


def _parse_vec3(line):
    vals = [float(x) for x in line.replace(",", " ").split()]
    if len(vals) != 3:
        raise ValueError("반드시 3개 값 입력: 'a b c'")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str, default=ROBOT_URDF)
    ap.add_argument("--base_frame", type=str, default=BASE_FRAME)
    ap.add_argument("--eps", type=float, default=eps)
    ap.add_argument("--itmax", type=int, default=IT_MAX)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--damp", type=float, default=damp)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--meshcat", action="store_true", help="IK 결과를 Meshcat으로 시각화")
    ap.add_argument("--meshcat_open", action="store_true", help="Meshcat 웹 브라우저 자동 오픈")
    ap.add_argument(
        "--meshcat_sleep",
        type=float,
        default=2.0,
        help="LEFT/RIGHT/COMBINED 표시 사이 대기 시간(초)",
    )
    args = ap.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    base_fid = model.getFrameId(args.base_frame)
    if base_fid == len(model.frames):
        raise RuntimeError(f"BASE_FRAME='{args.base_frame}' 못 찾음")

    left_fid = model.getFrameId(LEFT_EE_FRAME)
    right_fid = model.getFrameId(RIGHT_EE_FRAME)
    if left_fid == len(model.frames):
        raise RuntimeError(f"LEFT_EE_FRAME='{LEFT_EE_FRAME}' 못 찾음")
    if right_fid == len(model.frames):
        raise RuntimeError(f"RIGHT_EE_FRAME='{RIGHT_EE_FRAME}' 못 찾음")

    q0_model = pin.neutral(model)

    left_active_v  = select_arm_velocity_indices(model, LEFT_PREFIX)
    right_active_v = select_arm_velocity_indices(model, RIGHT_PREFIX)
    left_active_q = select_arm_configuration_indices(model, LEFT_PREFIX)
    right_active_q = select_arm_configuration_indices(model, RIGHT_PREFIX)

    viz = None
    if args.meshcat:
        viz = init_meshcat_visualizer(model, args.urdf, args.meshcat_open)
        viz.display(q0_model)
        print("[Meshcat] 초기 자세 표시 완료")

    print("Base frame:", args.base_frame)
    print("Left active v idx:", left_active_v)
    print("Right active v idx:", right_active_v)

    verbose = (not args.quiet)

    # ✅ 기본 목표도 base_link 기준으로 잡음
    default_left_xyz  = [1.0,  0.3, 1.0]
    default_right_xyz = [1.0, -0.3, 1.0]

    print("\n입력 형식 (모두 base_link 기준):")
    print("  xyz  : x y z   (meters)   [base_link 좌표]")
    print("  rpy  : roll pitch yaw  (degrees) [base_link 기준 자세]")
    print("  rpy는 엔터 치면 '현재 EE 자세 유지(=base_link 기준)'로 들어감")
    print("  모드: p=position-only(자세 무시), f=full(자세 포함)")
    print("  q 입력하면 종료\n")

    while True:
        cmd = input("계속할래? (Enter=입력/실행, q=quit) > ").strip().lower()
        if cmd == "q":
            break

        mode = input("모드 선택 (p=pos only, f=full) [기본 f] > ").strip().lower()
        position_only = (mode == "p")

        # 현재 자세(기본 rpy로 사용) - q0 기준, ✅ base_link 기준으로 계산
        pin.forwardKinematics(model, data, q0_model)
        pin.updateFramePlacements(model, data)
        oMbase0 = data.oMf[base_fid]
        bMleft0  = oMbase0.actInv(data.oMf[left_fid])
        bMright0 = oMbase0.actInv(data.oMf[right_fid])
        cur_left_rpy_deg  = rot_to_rpy_deg(bMleft0.rotation)
        cur_right_rpy_deg = rot_to_rpy_deg(bMright0.rotation)

        # LEFT
        print(f"\n[LEFT] 기본 xyz(base)={default_left_xyz}")
        lxyz_line = input("LEFT xyz (Enter=기본값) > ").strip()
        lxyz = default_left_xyz if lxyz_line == "" else _parse_vec3(lxyz_line)

        print(f"[LEFT] 기본 rpy(deg, base)=현재자세 {cur_left_rpy_deg}")
        lrpy_line = input("LEFT rpy(deg) (Enter=현재자세 유지) > ").strip()
        lrpy = cur_left_rpy_deg if lrpy_line == "" else _parse_vec3(lrpy_line)

        # RIGHT
        print(f"\n[RIGHT] 기본 xyz(base)={default_right_xyz}")
        rxyz_line = input("RIGHT xyz (Enter=기본값) > ").strip()
        rxyz = default_right_xyz if rxyz_line == "" else _parse_vec3(rxyz_line)

        print(f"[RIGHT] 기본 rpy(deg, base)=현재자세 {cur_right_rpy_deg}")
        rrpy_line = input("RIGHT rpy(deg) (Enter=현재자세 유지) > ").strip()
        rrpy = cur_right_rpy_deg if rrpy_line == "" else _parse_vec3(rrpy_line)

        # ✅ 목표를 base_link 기준 SE3로 생성 (base_link -> EE)
        bMdes_left  = pin.SE3(rpy_deg_to_rot(lrpy), np.array(lxyz, dtype=float))
        bMdes_right = pin.SE3(rpy_deg_to_rot(rrpy), np.array(rxyz, dtype=float))

        print("\n=== LEFT ARM IK (base_link 기준 목표) ===")
        succ_L, qL_model, errL = solve_ik_one_arm(
            model, data, q0_model,
            base_fid, left_fid,
            bMdes_left,
            left_active_v,
            eps=args.eps, IT_MAX=args.itmax, DT=args.dt, damp=args.damp,
            verbose=verbose, position_only=position_only
        )
        print("Convergence achieved! (LEFT)" if succ_L else "Warning: not converged (LEFT)")
        print(f"LEFT result (q_model): {qL_model.flatten().tolist()}")
        print(f"LEFT final error: {errL.T}")

        print("\n=== RIGHT ARM IK (base_link 기준 목표) ===")
        succ_R, qR_model, errR = solve_ik_one_arm(
            model, data, q0_model,
            base_fid, right_fid,
            bMdes_right,
            right_active_v,
            eps=args.eps, IT_MAX=args.itmax, DT=args.dt, damp=args.damp,
            verbose=verbose, position_only=position_only
        )
        print("Convergence achieved! (RIGHT)" if succ_R else "Warning: not converged (RIGHT)")
        print(f"RIGHT result (q_model): {qR_model.flatten().tolist()}")
        print(f"RIGHT final error: {errR.T}")

        if viz is not None:
            pause_sec = max(0.0, float(args.meshcat_sleep))
            q_combined = q0_model.copy()
            q_combined[left_active_q] = qL_model[left_active_q]
            q_combined[right_active_q] = qR_model[right_active_q]

            print("[Meshcat] LEFT 결과 표시")
            viz.display(qL_model)
            time.sleep(pause_sec)

            print("[Meshcat] RIGHT 결과 표시")
            viz.display(qR_model)
            time.sleep(pause_sec)

            print("[Meshcat] COMBINED(양팔) 결과 표시")
            viz.display(q_combined)

    print("\n종료!")


if __name__ == "__main__":
    main()

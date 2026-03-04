# bimanual_ik_test.py
from __future__ import annotations
from typing import List, Optional

import argparse
import json
import math

from capstone_pkg.kinematics.curobo_ik import FastBimanualIK
from capstone_pkg.utils.config import ROBOT_YAML


def _read_vec(name: str, n: int, example: str) -> List[float]:
    """
    터미널에서 벡터 입력 받기.
    허용 입력:
      - 0.1,0.2,0.3
      - 0.1 0.2 0.3
      - [0.1, 0.2, 0.3]
    """
    while True:
        s = input(f"{name} ({n} floats) ex) {example} > ").strip()
        try:
            if s.startswith("["):
                arr = json.loads(s)
                if not isinstance(arr, list):
                    raise ValueError("not a list")
                out = [float(x) for x in arr]
            else:
                if "," in s:
                    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                else:
                    parts = [p.strip() for p in s.split() if p.strip() != ""]
                out = [float(x) for x in parts]

            if len(out) != n:
                raise ValueError(f"length must be {n}, got {len(out)}")
            return out
        except Exception as e:
            print(f"[INPUT ERROR] {e}. Try again.")


def _read_optional_list(name: str) -> Optional[List[float]]:
    """JSON list로 입력 받거나 Enter로 스킵."""
    s = input(f"{name} (JSON list) or press Enter to skip > ").strip()
    if s == "":
        return None
    arr = json.loads(s)
    if not isinstance(arr, list):
        raise ValueError(f"{name} must be a JSON list")
    return [float(x) for x in arr]


def quat_wxyz_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
    """
    roll/pitch/yaw (deg) -> quaternion [w, x, y, z]
    (roll=X, pitch=Y, yaw=Z, 순서는 yaw->pitch->roll 적용, 흔히 쓰는 RPY)
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    # Z(Y(X)) = yaw-pitch-roll
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y_ = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return [w, x, y_, z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)

    # solver options
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--num_seeds", type=int, default=20)
    ap.add_argument("--rot_th", type=float, default=0.05)
    ap.add_argument("--pos_th", type=float, default=0.005)
    ap.add_argument("--no_cuda_graph", action="store_true")

    args = ap.parse_args()

    print(
        "[INFO] First cuRobo run may take several minutes for CUDA JIT "
        "compilation (kinematics_fused_cu/geom_cu/lbfgs_step_cu/...)."
    )

    # ✅ solver 1회 생성
    ik = FastBimanualIK(
        args.robot_yml,
        cpu=bool(args.cpu),
        num_seeds=int(args.num_seeds),
        rotation_threshold=float(args.rot_th),
        position_threshold=float(args.pos_th),
        use_cuda_graph=(not args.no_cuda_graph),
    )

    print("\n=== Enter bimanual target poses ===")
    print("- xyz: meters (3 floats)")
    print("- rpy: degrees (roll, pitch, yaw) (3 floats)\n")

    # ✅ 터미널 입력: xyz + rpy(deg)
    left_xyz = _read_vec("left_xyz", 3, "0.45,0.25,0.90")
    left_rpy_deg = _read_vec("left_rpy_deg (roll,pitch,yaw)", 3, "0,0,0")
    left_quat = quat_wxyz_from_rpy_deg(*left_rpy_deg)

    right_xyz = _read_vec("right_xyz", 3, "0.45,-0.25,0.90")
    right_rpy_deg = _read_vec("right_rpy_deg (roll,pitch,yaw)", 3, "0,0,0")
    right_quat = quat_wxyz_from_rpy_deg(*right_rpy_deg)

    print("\n[INFO] Converted quaternion (wxyz)")
    print(f"  left_quat_wxyz : {left_quat}")
    print(f"  right_quat_wxyz: {right_quat}")

    # (선택) 시작 관절값
    q_start_cspace: Optional[List[float]] = None
    try:
        q_start_cspace = _read_optional_list("q_start_cspace (YAML cspace joint order)")
    except Exception as e:
        print(f"[WARN] q_start_cspace ignored due to input error: {e}")
        q_start_cspace = None

    # ✅ IK solve
    out = ik.solve(
        left_xyz=left_xyz,
        left_quat_wxyz=left_quat,
        right_xyz=right_xyz,
        right_quat_wxyz=right_quat,
        q_start_cspace=q_start_cspace,
    )

    print("\n=== Bimanual IK Result ===")
    print(f"success: {out.success}")
    if not out.success:
        return

    # 결과 출력(ROS publish 제거)
    if (out.q_cspace is None) or (out.cspace_joint_names is None):
        print("[ERROR] out.q_cspace / out.cspace_joint_names is None.")
        return

    names = list(out.cspace_joint_names)
    positions = list(out.q_cspace)

    print(f"\n- q_cspace ({len(names)} joints) in YAML cspace joint order:")
    print(f"  names[0:14] = {names[:14]}")
    print(f"  pos  [0:14] = {positions[:14]}")
    # 필요하면 전체도 출력:
    # for n, q in zip(names, positions):
    #     print(f"{n:30s} {q: .6f}")

    print("[DONE] IK solved")


if __name__ == "__main__":
    main()

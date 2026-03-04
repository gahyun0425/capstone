from __future__ import annotations

import argparse
import json
from typing import List

from capstone_pkg.utils.config import ROBOT_YAML, LEFT_JOINTS, RIGHT_JOINTS
from capstone_pkg.kinematics.curobo_ik import SingleArmIK
from .input_utils import read_vec, xyzw_to_wxyz
from .birrt import plan_birrt_jointspace
from .path_publisher import publish_joint_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--cpu", action="store_true", help="force CPU (no CUDA)")
    ap.add_argument("--max_iters", type=int, default=6000)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--goal_bias", type=float, default=0.10)
    ap.add_argument("--connect_threshold", type=float, default=0.20)
    ap.add_argument("--save", default="", help="optional path to save result json")
    ap.add_argument("--publish_path", action=argparse.BooleanOptionalAction, default=True, help="publish planned path as JointState commands")
    ap.add_argument("--publish_topic", default="/joint_states_cmd", help="target topic for JointState command stream")
    ap.add_argument("--publish_dt", type=float, default=0.1, help="publish period [s] between waypoints")
    args = ap.parse_args()

    arm = input("Plan which arm? (left/right): ").strip().lower()
    if arm not in ("left", "right"):
        print("[ERROR] arm must be 'left' or 'right'")
        return 2

    xyz = read_vec("Target xyz (m)", 3, "0.4 0.2 1.65")
    q_xyzw = read_vec("Target quat (xyzw)", 4, "0 0 0.7071 0.7071")
    quat_wxyz = xyzw_to_wxyz(q_xyzw)

    # start configuration: keep others fixed at zero (you can later replace with /joint_states if needed)
    # cspace joint order is taken from robot yaml inside IK module.
    q_start_cspace: List[float] = None  # will be constructed after IK init

    print("\n[1/2] Solving IK with cuRobo...")
    ik = SingleArmIK(args.robot_yml, arm=arm, cpu=args.cpu)
    q_start_cspace = [0.0 for _ in ik.cspace_joint_names]

    ik_out = ik.solve(xyz, quat_wxyz, q_start_cspace=q_start_cspace)
    if not ik_out.success or ik_out.q_cspace is None:
        print("[IK] Failed or in collision.")
        return 1
    q_goal = ik_out.q_cspace
    print("[IK] success.\n")

    active = LEFT_JOINTS if arm == "left" else RIGHT_JOINTS

    print("[2/2] Running BiRRT (joint-space)...")
    ok, path = plan_birrt_jointspace(
        robot_yml=args.robot_yml,
        q_start=q_start_cspace,
        q_goal=q_goal,
        active_joint_names=active,
        cspace_joint_names=ik.cspace_joint_names,
        cpu=args.cpu,
        step=args.step,
        max_iters=args.max_iters,
        goal_bias=args.goal_bias,
        connect_threshold=args.connect_threshold,
    )

    if not ok:
        print("[BiRRT] Failed to find a path.")
        return 1

    print(f"[BiRRT] Success! path_len={len(path)}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "path": path, "cspace_joint_names": ik.cspace_joint_names}, f, indent=2)
        print(f"[BiRRT] Saved: {args.save}")

    if args.publish_path:
        print(f"[3/3] Publishing path -> {args.publish_topic} (dt={args.publish_dt:.3f}s)")
        name_to_idx = {n: i for i, n in enumerate(ik.cspace_joint_names)}
        pub_names = [n for n in active if n in name_to_idx]
        if not pub_names:
            print("[PUBLISH] no active joints found in cspace_joint_names; skip publish.")
            return 1

        pub_path = [[float(q[name_to_idx[n]]) for n in pub_names] for q in path]
        publish_joint_path(
            pub_path,
            pub_names,
            topic=args.publish_topic,
            dt=args.publish_dt,
        )
        print("[PUBLISH] done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

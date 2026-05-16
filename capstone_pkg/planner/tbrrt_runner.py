from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import torch

from capstone_pkg.utils.config import (
    ROBOT_YAML,
    JOINT_LIMIT,
    ROBOT_XML,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    CSPACE_JOINT_NAMES_14,
    LEFT_JOINTS,
    RIGHT_JOINTS,
)
from capstone_pkg.utils.world_collision_bridge import (
    DEFAULT_WORLD_COLLISION_TOPIC,
    publish_world_collision_yaml,
)
from capstone_pkg.planner.start_goal import get_start_and_goal_from_topic_and_ik
from capstone_pkg.planner.arm_rrt_common.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_trajectory_group,
    send_joint_trajectory_action_group,
)
from capstone_pkg.utils.joint_limit import load_joint_limits_torch
from capstone_pkg.constraint_projection.constraint import RigidConstraint
from capstone_pkg.constraint_projection.projection import ManifoldProjectorTorch
from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.planner.tbrrt import TBRRTConfig
from capstone_pkg.planner.tbrrt.batch import plan_tbrrt_extcon_batch_conext
from capstone_pkg.utils.jointstate_publisher import publish_q_path_as_jointstate_keep_gripper_closed


def build_tbrrt_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_yml", type=str, default=ROBOT_YAML)
    ap.add_argument("--world_yml", type=str, default=None, help="world collision yaml; None이면 world collision 비활성화")
    ap.add_argument("--joint_limit_yml", type=str, default=JOINT_LIMIT)
    ap.add_argument("--jointstate_topic", type=str, default="/joint_states")
    ap.add_argument("--cmd_topic", type=str, default="/joint_states_cmd")
    ap.add_argument("--publish_mode", choices=("joint_state", "real"), default="joint_state")
    ap.add_argument("--mujoco_xml", type=str, default=ROBOT_XML)
    ap.add_argument("--left_ee", type=str, default=LEFT_EE_FRAME)
    ap.add_argument("--right_ee", type=str, default=RIGHT_EE_FRAME)
    ap.add_argument("--goal_topk", type=int, default=16)
    ap.add_argument("--ik_batch", type=int, default=100, help="number of CuRobo seed trials used to build goal-region candidates")
    ap.add_argument("--ik_seed_noise_std", type=float, default=0.25, help="Gaussian std [rad] for perturbing q_start into CuRobo IK seeds")
    ap.add_argument("--ik_seed", type=int, default=0, help="random seed used for CuRobo IK seed perturbations")
    ap.add_argument("--ik_goal_dedupe_tol", type=float, default=1.0e-3, help="merge IK goals whose joint-space distance is within this tolerance")
    ap.add_argument("--select_goal", type=str, default="first_free", choices=["first_free", "min_penetration"])
    ap.add_argument("--fail_if_start_in_collision", action="store_true")
    ap.add_argument("--target_left_xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--target_left_rpy_deg", type=float, nargs=3, required=True, metavar=("R", "P", "Y"))
    ap.add_argument(
        "--planar_xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep x-y planar motion: current left z and roll/pitch are preserved, yaw is allowed",
    )
    ap.add_argument("--rigid_orientation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lock_z", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--step_size", type=float, default=0.1)
    ap.add_argument("--goal_threshold", type=float, default=0.10)
    ap.add_argument("--EM", type=float, default=0.02)
    ap.add_argument("--E_conn", type=float, default=0.025)
    ap.add_argument("--ts_radius", type=float, default=1.1)
    ap.add_argument("--p_uniform", type=float, default=0.25)
    ap.add_argument("--goal_bias", type=float, default=0.15)
    ap.add_argument("--block_k", type=int, default=512)
    ap.add_argument("--goal_rerank_topk", type=int, default=3)
    ap.add_argument("--goal_rerank_interp_points", type=int, default=6)
    ap.add_argument("--connect_max_steps", type=int, default=10)
    ap.add_argument("--connect_stagnation_steps", type=int, default=4)
    ap.add_argument("--connect_stagnation_progress_ratio", type=float, default=0.10)
    ap.add_argument("--connect_stagnation_escape", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--failed_connection_cooldown", type=int, default=8)
    ap.add_argument("--connect_bridge_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--connect_bridge_near_threshold", type=float, default=0.2)
    ap.add_argument("--connect_bridge_curvature_threshold", type=float, default=1.5)
    ap.add_argument("--connect_bridge_max_attempts_per_iter", type=int, default=1)
    ap.add_argument("--connect_bridge_reset_cooldown", type=int, default=1)
    ap.add_argument("--connection_segment_precheck", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--failed_connection_region_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--failed_connection_region_radius", type=float, default=0.05)
    ap.add_argument("--failed_connection_region_max", type=int, default=2048)
    ap.add_argument("--failed_edge_region_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--failed_edge_region_radius", type=float, default=0.05)
    ap.add_argument("--failed_edge_region_max", type=int, default=4096)
    ap.add_argument("--failed_edge_region_retarget_threshold", type=int, default=4)
    ap.add_argument("--failed_edge_region_escape_threshold", type=int, default=16)
    ap.add_argument("--failed_edge_region_cooldown", type=int, default=8)
    ap.add_argument("--connection_lazy_prealloc_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--connection_lazy_prealloc_max_candidates", type=int, default=0)
    ap.add_argument("--connection_lazy_prealloc_max_path_points", type=int, default=512)
    ap.add_argument("--connection_lazy_edge_branch_ban_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--connection_lazy_edge_subtree_ban_max_nodes", type=int, default=256)
    ap.add_argument("--escape_spawn_blocks", type=int, default=5)
    ap.add_argument("--escape_extend_steps", type=int, default=5)
    ap.add_argument("--escape_fuse_trees", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max_iters", type=int, default=500000)
    ap.add_argument("--time_limit_sec", type=float, default=900.0)
    ap.add_argument("--proj_iters", type=int, default=60)
    ap.add_argument("--proj_tol", type=float, default=1e-3)
    ap.add_argument("--proj_damping", type=float, default=0.0)
    ap.add_argument("--proj_step", type=float, default=1.0)
    ap.add_argument("--proj_fd_eps", type=float, default=1e-3)
    ap.add_argument("--edge_step_q", type=float, default=0.03)
    ap.add_argument("--edge_max_steps", type=int, default=128)
    ap.add_argument("--svd_tol", type=float, default=1e-6)
    ap.add_argument("--ts_bias_volume", type=float, default=0.1)
    ap.add_argument("--ts_bias_curvature", type=float, default=1.0)
    ap.add_argument("--ts_bias_nodecount", type=float, default=0.8)
    ap.add_argument("--ts_bias_collision", type=float, default=3.0)
    ap.add_argument("--ts_curv_eps", type=float, default=1e-3)
    ap.add_argument("--discard_overlap_max_tries", type=int, default=8)
    ap.add_argument("--dynamic_domain_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ts_domain_expand_ratio", type=float, default=1.25)
    ap.add_argument("--ts_domain_shrink_ratio", type=float, default=0.55)
    ap.add_argument("--ts_domain_expand_frac", type=float, default=0.9)
    ap.add_argument("--ts_domain_shrink_frac", type=float, default=0.7)
    ap.add_argument("--ts_domain_min", type=float, default=0.04)
    ap.add_argument("--ts_domain_max", type=float, default=3.0)
    ap.add_argument("--enable_halfspace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--enable_overlap_discard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shortcut_smoothing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shortcut_smoothing_iters", type=int, default=80)
    ap.add_argument("--shortcut_smoothing_min_skip", type=int, default=1)
    ap.add_argument("--spline_interpolation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--spline_step_q", type=float, default=0.03)
    ap.add_argument("--spline_max_steps_per_segment", type=int, default=32)
    ap.add_argument("--spline_max_points", type=int, default=2048)
    ap.add_argument("--spline_fallback_to_input", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--topp_enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--topp_max_velocity", type=float, default=1.0)
    ap.add_argument("--topp_max_acceleration", type=float, default=2.0)
    ap.add_argument("--topp_safety_scale", type=float, default=1.05)
    ap.add_argument("--topp_max_iterations", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--start_delay_s", type=float, default=0.2)
    ap.add_argument("--hold_last_s", type=float, default=0.0)
    ap.add_argument("--close_value", type=float, default=1.0)
    ap.add_argument("--real_left_topic", default="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory")
    ap.add_argument("--real_right_topic", default="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory")
    ap.add_argument("--real_left_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_left/joint_trajectory")
    ap.add_argument("--real_right_gripper_topic", default="/leader/joint_trajectory_command_broadcaster_gripper_right/joint_trajectory")
    ap.add_argument("--real_use_action", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--real_action_fallback_to_topic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--real_left_action", default="/leader/joint_trajectory_command_broadcaster_left/follow_joint_trajectory")
    ap.add_argument("--real_right_action", default="/leader/joint_trajectory_command_broadcaster_right/follow_joint_trajectory")
    ap.add_argument("--real_left_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_left/follow_joint_trajectory")
    ap.add_argument("--real_right_gripper_action", default="/leader/joint_trajectory_command_broadcaster_gripper_right/follow_joint_trajectory")
    ap.add_argument("--action_wait_server_s", type=float, default=2.0)
    ap.add_argument("--action_wait_result_s", type=float, default=-1.0)
    ap.add_argument("--publish_wait_subscriber_s", type=float, default=5.0)
    ap.add_argument("--publish_require_subscriber", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--publish_retry_until_subscriber", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--publish_repeat", type=int, default=8)
    ap.add_argument("--publish_period_s", type=float, default=0.03)
    ap.add_argument("--publish_wait_ack_s", type=float, default=0.0)
    ap.add_argument("--publish_keep_alive_s", type=float, default=1.0)
    ap.add_argument("--publish_reliability", choices=("reliable", "best_effort"), default="best_effort")
    ap.add_argument("--publish_durability", choices=("volatile", "transient_local"), default="volatile")
    ap.add_argument("--publish_qos_depth", type=int, default=1)
    ap.add_argument("--publish_transient_local", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--attach_object", action="store_true", default=False)
    ap.add_argument("--publish_world_collision", action=argparse.BooleanOptionalAction, default=True, help="publish selected world collision yaml to MuJoCo simulation")
    ap.add_argument("--world_collision_topic", default=DEFAULT_WORLD_COLLISION_TOPIC, help="MuJoCo world collision cuboid topic")
    ap.add_argument("--world_collision_wait_subscriber_s", type=float, default=1.0, help="wait for MuJoCo subscriber before publishing world collision")
    ap.add_argument("--world_collision_keep_alive_s", type=float, default=0.5, help="keep world collision publisher alive after publishing")
    return ap


def _rerank_q_goals_for_batch_conext(
    *,
    q_start: list[float],
    q_goals: list[list[float]],
    projector: ManifoldProjectorTorch,
    checker,
    device: torch.device,
    topk: int,
    interp_points: int = 10,
) -> list[list[float]]:
    if len(q_goals) <= 1:
        return q_goals

    interp_points = max(2, int(interp_points))
    topk = max(1, int(topk))

    with torch.no_grad():
        q_start_t = torch.tensor(q_start, device=device, dtype=torch.float32).view(1, -1)
        q_goals_t = torch.tensor(q_goals, device=device, dtype=torch.float32)
        if q_goals_t.ndim != 2:
            return q_goals

        g_count, q_dim = q_goals_t.shape
        alphas = torch.linspace(0.0, 1.0, interp_points, device=device, dtype=torch.float32).view(1, interp_points, 1)
        q_interp = q_start_t.unsqueeze(1) + alphas * (q_goals_t.unsqueeze(1) - q_start_t.unsqueeze(1))
        q_interp_flat = q_interp.reshape(g_count * interp_points, q_dim)

        proj_out = projector.project_batch(q_interp_flat)
        q_proj = proj_out.q_proj.reshape(g_count, interp_points, q_dim)
        proj_success = proj_out.success_mask.reshape(g_count, interp_points)
        proj_iters = proj_out.iters.reshape(g_count, interp_points).to(torch.float32)

        col_out = checker.check_batch(q_proj.reshape(g_count * interp_points, q_dim))
        in_col = col_out.in_collision.reshape(g_count, interp_points)

        joint_dist = torch.linalg.norm(q_goals_t - q_start_t, dim=1)
        avg_proj_iters = proj_iters.mean(dim=1)
        proj_fail_ratio = (~proj_success).to(torch.float32).mean(dim=1)
        collision_ratio = in_col.to(torch.float32).mean(dim=1)

        score = (
            1.0 * joint_dist
            + 0.5 * avg_proj_iters
            + 3.0 * proj_fail_ratio
            + 2.0 * collision_ratio
        )

        order = torch.argsort(score)
        keep = order[: min(topk, g_count)].detach().cpu().tolist()

        print(
            "[goal_rerank] batch_conext precheck: "
            f"candidates={g_count} interp_points={interp_points} selected={len(keep)}"
        )
        for rank, idx in enumerate(keep[: min(5, len(keep))], start=1):
            print(
                f"[goal_rerank] #{rank} idx={idx} "
                f"score={float(score[idx].item()):.4f} "
                f"joint={float(joint_dist[idx].item()):.4f} "
                f"proj_iter={float(avg_proj_iters[idx].item()):.4f} "
                f"proj_fail={float(proj_fail_ratio[idx].item()):.4f} "
                f"col={float(collision_ratio[idx].item()):.4f}"
            )

        return [list(q_goals[idx]) for idx in keep]


def main_tbrrt(argv: Sequence[str] | None = None) -> int:
    args = build_tbrrt_parser().parse_args(list(argv) if argv is not None else None)
    t0 = time.time()
    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float32
    _world_yml = None if args.world_yml in (None, "", "none", "None") else str(args.world_yml)

    if bool(args.publish_world_collision):
        if _world_yml is None:
            print("[tbrrt][WORLD] no world_yml; skip MuJoCo world collision publish.")
        else:
            try:
                count = publish_world_collision_yaml(
                    _world_yml,
                    topic=str(args.world_collision_topic),
                    wait_subscriber_s=float(args.world_collision_wait_subscriber_s),
                    keep_alive_s=float(args.world_collision_keep_alive_s),
                    node_name="tbrrt_world_collision_publisher",
                )
                print(
                    f"[tbrrt][WORLD] published {count} collision cuboid(s) "
                    f"to MuJoCo topic {args.world_collision_topic}"
                )
            except Exception as exc:
                print(f"[tbrrt][WORLD][WARN] failed to publish collision cuboids to MuJoCo: {exc}")

    q_start, q_goals, best_pen, t_pose_done = get_start_and_goal_from_topic_and_ik(
        robot_yml=str(args.robot_yml),
        jointstate_topic=str(args.jointstate_topic),
        joint_names=list(CSPACE_JOINT_NAMES_14),
        target_left_xyz=list(args.target_left_xyz),
        target_left_rpy_deg=list(args.target_left_rpy_deg),
        left_ee=str(args.left_ee),
        right_ee=str(args.right_ee),
        select=args.select_goal,
        device_str=("cuda" if device.type == "cuda" else "cpu"),
        world_yml=_world_yml,
        fail_if_start_in_collision=bool(args.fail_if_start_in_collision),
        topk=int(args.goal_topk),
        planar_xy=bool(args.planar_xy),
        ik_batch=int(args.ik_batch),
        ik_seed_noise_std=float(args.ik_seed_noise_std),
        ik_seed_random_seed=int(args.ik_seed),
        ik_goal_dedupe_tol=float(args.ik_goal_dedupe_tol),
    )
    print(f"[tbrrt] got start + {len(q_goals)} goals. best_pen={best_pen:.6f} t_pose_done={t_pose_done:.3f}s")

    checker = get_self_collision_checker(str(args.robot_yml), cpu=(device.type == "cpu"), world_yml=_world_yml)

    if bool(args.attach_object):
        if not hasattr(checker, "_build_q_active_from_cspace"):
            raise RuntimeError("SelfCollisionChecker has no _build_q_active_from_cspace()")
        q_start_c = torch.tensor(q_start, device=device, dtype=torch.float32).view(1, -1)
        q_model = checker._build_q_active_from_cspace(q_start_c)[0]
        checker.attach_mujoco_object_to_robot(
            mujoco_xml_path=str(args.mujoco_xml),
            q_model_order=q_model,
            link_name=str(args.left_ee),
            name_prefix="att_",
            disable_in_world=False,
        )
        print("[tbrrt] attached mujoco object to robot")

    jl = load_joint_limits_torch(str(args.joint_limit_yml), device=device, dtype=dtype)
    q_ref = torch.tensor(q_start, device=device, dtype=dtype)
    lock_z = bool(args.lock_z or args.planar_xy)
    if bool(args.planar_xy):
        print("[tbrrt] planar_xy enabled: enforcing left z lock and yaw-only absolute rotation")
    if bool(args.planar_xy) and bool(args.rigid_orientation):
        print("[tbrrt][WARN] rigid_orientation is also enabled, so yaw will be locked too.")
    constraint = RigidConstraint(
        robot_yml=str(args.robot_yml),
        left_ee=str(args.left_ee),
        right_ee=str(args.right_ee),
        q_ref=q_ref,
        device=device,
        dtype=dtype,
        mode="se3",
        rigid_orientation=bool(args.rigid_orientation),
        lock_z=lock_z,
        planar_xy=bool(args.planar_xy),
    )
    projector = ManifoldProjectorTorch(
        constraint=constraint,
        limits=jl,
        max_iters=int(args.proj_iters),
        tol=float(args.proj_tol),
        fd_eps=float(args.proj_fd_eps),
        damping=float(args.proj_damping),
        step_size=float(args.proj_step),
    )

    q_goals = _rerank_q_goals_for_batch_conext(
        q_start=q_start,
        q_goals=q_goals,
        projector=projector,
        checker=checker,
        device=device,
        topk=int(args.goal_rerank_topk),
        interp_points=int(args.goal_rerank_interp_points),
    )

    cfg = TBRRTConfig(
        step_size=float(args.step_size),
        goal_threshold=float(args.goal_threshold),
        EM=float(args.EM),
        E_conn=float(args.E_conn),
        ts_radius=float(args.ts_radius),
        p_uniform=float(args.p_uniform),
        goal_bias=float(args.goal_bias),
        connect_max_steps=int(args.connect_max_steps),
        connect_stagnation_steps=int(args.connect_stagnation_steps),
        connect_stagnation_progress_ratio=float(args.connect_stagnation_progress_ratio),
        connect_stagnation_escape=bool(args.connect_stagnation_escape),
        failed_connection_cooldown=int(args.failed_connection_cooldown),
        connect_bridge_enable=bool(args.connect_bridge_enable),
        connect_bridge_near_threshold=float(args.connect_bridge_near_threshold),
        connect_bridge_curvature_threshold=float(args.connect_bridge_curvature_threshold),
        connect_bridge_max_attempts_per_iter=int(args.connect_bridge_max_attempts_per_iter),
        connect_bridge_reset_cooldown=int(args.connect_bridge_reset_cooldown),
        connection_segment_precheck=bool(args.connection_segment_precheck),
        failed_connection_region_enable=bool(args.failed_connection_region_enable),
        failed_connection_region_radius=float(args.failed_connection_region_radius),
        failed_connection_region_max=int(args.failed_connection_region_max),
        failed_edge_region_enable=bool(args.failed_edge_region_enable),
        failed_edge_region_radius=float(args.failed_edge_region_radius),
        failed_edge_region_max=int(args.failed_edge_region_max),
        failed_edge_region_retarget_threshold=int(args.failed_edge_region_retarget_threshold),
        failed_edge_region_escape_threshold=int(args.failed_edge_region_escape_threshold),
        failed_edge_region_cooldown=int(args.failed_edge_region_cooldown),
        connection_lazy_prealloc_enable=bool(args.connection_lazy_prealloc_enable),
        connection_lazy_prealloc_max_candidates=int(args.connection_lazy_prealloc_max_candidates),
        connection_lazy_prealloc_max_path_points=int(args.connection_lazy_prealloc_max_path_points),
        connection_lazy_edge_branch_ban_enable=bool(args.connection_lazy_edge_branch_ban_enable),
        connection_lazy_edge_subtree_ban_max_nodes=int(args.connection_lazy_edge_subtree_ban_max_nodes),
        escape_spawn_blocks=max(1, int(args.escape_spawn_blocks)),
        escape_extend_steps=max(1, int(args.escape_extend_steps)),
        escape_fuse_trees=bool(args.escape_fuse_trees),
        max_iters=int(args.max_iters),
        time_limit_sec=float(args.time_limit_sec),
        ts_bias_volume=float(args.ts_bias_volume),
        ts_bias_curvature=float(args.ts_bias_curvature),
        ts_bias_nodecount=float(args.ts_bias_nodecount),
        ts_bias_collision=float(args.ts_bias_collision),
        ts_curv_eps=float(args.ts_curv_eps),
        discard_overlap_max_tries=int(args.discard_overlap_max_tries),
        dynamic_domain_enable=bool(args.dynamic_domain_enable),
        ts_domain_expand_ratio=float(args.ts_domain_expand_ratio),
        ts_domain_shrink_ratio=float(args.ts_domain_shrink_ratio),
        ts_domain_expand_frac=float(args.ts_domain_expand_frac),
        ts_domain_shrink_frac=float(args.ts_domain_shrink_frac),
        ts_domain_min=float(args.ts_domain_min),
        ts_domain_max=float(args.ts_domain_max),
        edge_step_q=float(args.edge_step_q),
        edge_max_steps=int(args.edge_max_steps),
        shortcut_smoothing=bool(args.shortcut_smoothing),
        shortcut_smoothing_iters=int(args.shortcut_smoothing_iters),
        shortcut_smoothing_min_skip=int(args.shortcut_smoothing_min_skip),
        spline_interpolation=bool(args.spline_interpolation),
        spline_step_q=float(args.spline_step_q),
        spline_max_steps_per_segment=int(args.spline_max_steps_per_segment),
        spline_max_points=int(args.spline_max_points),
        spline_fallback_to_input=bool(args.spline_fallback_to_input),
        topp_enable=bool(args.topp_enable),
        topp_max_velocity=float(args.topp_max_velocity),
        topp_max_acceleration=float(args.topp_max_acceleration),
        topp_output_dt=(1.0 / max(1e-6, float(args.hz))),
        topp_safety_scale=float(args.topp_safety_scale),
        topp_max_iterations=int(args.topp_max_iterations),
        svd_tol=float(args.svd_tol),
        enable_halfspace=bool(args.enable_halfspace),
        enable_overlap_discard=bool(args.enable_overlap_discard),
        seed=(None if int(args.seed) < 0 else int(args.seed)),
    )

    out = plan_tbrrt_extcon_batch_conext(
        q_start=q_start,
        q_goals=q_goals,
        cfg=cfg,
        checker=checker,
        projector=projector,
        joint_limits=jl,
        device=device,
        block_K=int(args.block_k),
    )
    if not out.success:
        print(f"[tbrrt] planning failed: {out.stats.extra}")
        print(f"[tbrrt] total wall time = {time.time() - t0:.3f}s")
        return 1

    print("\n[tbrrt] SUCCESS")
    print(f"  iters={out.stats.iters} nodesA={out.stats.nodes_A} nodesB={out.stats.nodes_B} ts={out.stats.ts_count} time={out.stats.time_sec:.3f}s")
    if out.path is not None:
        print(f"  path_len={len(out.path)}")

    traj_dt = 1.0 / max(1e-6, float(args.hz))
    path_rows = [[float(v) for v in row] for row in out.path]
    left_dof = len(LEFT_JOINTS)
    left_path = [row[:left_dof] for row in path_rows]
    right_path = [row[left_dof:] for row in path_rows]

    def _send_real_commands(
        *,
        topic_commands: list[JointTrajectoryCommand],
        action_commands: list[JointTrajectoryCommand],
        dt: float,
        start_time_delay_s: float,
        label: str,
    ) -> str:
        if args.real_use_action:
            try:
                action_targets = ", ".join(cmd.endpoint for cmd in action_commands)
                print(
                    f"[{label}] Sending FollowJointTrajectory -> {action_targets} "
                    f"(dt={dt:.3f}s)"
                )
                send_joint_trajectory_action_group(
                    action_commands,
                    dt=dt,
                    wait_server_s=float(args.action_wait_server_s),
                    wait_result_s=float(args.action_wait_result_s),
                    start_time_delay_s=float(start_time_delay_s),
                )
                return "action"
            except RuntimeError as exc:
                print(f"[ACTION] {exc}")
                if not args.real_action_fallback_to_topic:
                    raise
                topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
                print(f"[ACTION] Falling back to JointTrajectory topics -> {topic_targets}")

        topic_targets = ", ".join(cmd.endpoint for cmd in topic_commands)
        print(f"[{label}] Publishing JointTrajectory -> {topic_targets} (dt={dt:.3f}s)")
        publish_joint_trajectory_group(
            topic_commands,
            dt=dt,
            wait_subscriber_s=float(args.publish_wait_subscriber_s),
            require_subscriber=bool(args.publish_require_subscriber),
            retry_until_subscriber=bool(args.publish_retry_until_subscriber),
            publish_repeat=int(args.publish_repeat),
            publish_period_s=float(args.publish_period_s),
            wait_ack_s=float(args.publish_wait_ack_s),
            keep_alive_s=float(args.publish_keep_alive_s),
            reliability=str(getattr(args, "publish_reliability", "best_effort")),
            durability=(
                "transient_local"
                if bool(getattr(args, "publish_transient_local", False))
                else str(getattr(args, "publish_durability", "volatile"))
            ),
            qos_depth=int(getattr(args, "publish_qos_depth", 1)),
            start_time_delay_s=float(start_time_delay_s),
        )
        return "topic"

    if args.publish_mode == "real":
        gripper_dt = min(0.1, traj_dt) if traj_dt > 0.0 else 0.1
        gripper_topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_gripper_topic),
                joint_names=["gripper_l_joint1"],
                path=[[float(args.close_value)]],
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_gripper_topic),
                joint_names=["gripper_r_joint1"],
                path=[[float(args.close_value)]],
            ),
        ]
        gripper_action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_gripper_action),
                joint_names=["gripper_l_joint1"],
                path=[[float(args.close_value)]],
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_gripper_action),
                joint_names=["gripper_r_joint1"],
                path=[[float(args.close_value)]],
            ),
        ]
        _send_real_commands(
            topic_commands=gripper_topic_commands,
            action_commands=gripper_action_commands,
            dt=gripper_dt,
            start_time_delay_s=0.0,
            label="GRIPPER",
        )

        arm_topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_topic),
                joint_names=list(LEFT_JOINTS),
                path=left_path,
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_topic),
                joint_names=list(RIGHT_JOINTS),
                path=right_path,
            ),
        ]
        arm_action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_action),
                joint_names=list(LEFT_JOINTS),
                path=left_path,
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_action),
                joint_names=list(RIGHT_JOINTS),
                path=right_path,
            ),
        ]
        _send_real_commands(
            topic_commands=arm_topic_commands,
            action_commands=arm_action_commands,
            dt=traj_dt,
            start_time_delay_s=float(args.start_delay_s),
            label="REAL",
        )
    else:
        q_path = torch.tensor(out.path, device=device, dtype=dtype)
        publish_q_path_as_jointstate_keep_gripper_closed(
            q_path=q_path,
            robot_yml=str(args.robot_yml),
            mujoco_xml=str(args.mujoco_xml),
            close_value=float(args.close_value),
            cmd_topic=str(args.cmd_topic),
            hz=float(args.hz),
            repeat=int(args.repeat),
            start_delay_s=float(args.start_delay_s),
            hold_last_s=float(args.hold_last_s),
            node_name="tbrrt_path_publisher",
        )

    print("  q_start(path) =", out.path[0])
    print("  q_goal(path)  =", out.path[-1])
    print(f"[tbrrt] total wall time = {time.time() - t0:.3f}s")
    return 0


# Backward-compatible entrypoint expected by capstone_pkg.main
run_tbrrt = main_tbrrt

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import rclpy
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


_PKG_ROOT = Path(__file__).resolve().parents[3]
DOOR_MODEL_XML = str(_PKG_ROOT / "models" / "door_ffw_sg2.xml")
DOOR_COLLISION_YAML = str(_PKG_ROOT / "models" / "door_collision.yaml")
HANDLE_TARGET_X_OFFSET_M = -0.02
HANDLE_TARGET_Z_OFFSET_M = 0.0
DETECTED_HANDLE_TARGET_X_OFFSET_M = -0.02
DETECTED_HANDLE_TARGET_Z_M = 1.10

DEFAULT_DOOR_UNLOCK_TOPIC = "/door_unlock"
DOOR_HINGE_JOINT = "fridge_door_hinge_joint"
DOOR_MOVING_COLLISION_NAMES = (
    "fridge_door_glass",
    "fridge_door_right_frame",
    "fridge_door_left_frame",
    "fridge_door_top_frame",
    "fridge_door_bottom_frame",
    "fridge_handle_bar",
    "fridge_handle_top_mount",
    "fridge_handle_bottom_mount",
)
DOOR_GRASP_CONTACT_COLLISION_NAMES = (
    "fridge_handle_bar",
    "fridge_handle_top_mount",
    "fridge_handle_bottom_mount",
)
DOOR_BASE_CLEARANCE_COLLISION_NAMES = tuple(
    name
    for name in DOOR_MOVING_COLLISION_NAMES
    if name not in DOOR_GRASP_CONTACT_COLLISION_NAMES
)
HANDLE_QUAT_XYZW = (
    0.7071067811865475,
    5.551115123125783e-17,
    0.7071067811865475,
    5.551115123125783e-17,
)

_COLLISION_MODELS: Mapping[str, str] = {
    "door": DOOR_COLLISION_YAML,
}


@dataclass(frozen=True)
class DoorModelFrames:
    hinge_xyz: tuple[float, float, float]
    handle_frame_xyz: tuple[float, float, float]
    grasp_target_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class DoorClosePlan:
    arm: str
    joint_names: list[str]
    path: list[list[float]]
    door_alphas_rad: list[float]
    base_poses: list[list[float]] | None
    desired_ee_poses: list[Any] | None
    close_end_cspace_joint_names: list[str]
    close_end_q_cspace: list[float]
    world_yml: str | None


def _parse_xyz_attr(value: str | None) -> list[float]:
    if value is None or not str(value).strip():
        return [0.0, 0.0, 0.0]
    parts = [float(v) for v in str(value).split()]
    if len(parts) != 3:
        raise RuntimeError(f"expected 3 xyz values, got {value!r}")
    return parts


def _add_xyz(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a[i]) + float(b[i]) for i in range(3))


def _load_door_model_frames(mujoco_xml: str) -> DoorModelFrames:
    import xml.etree.ElementTree as ET

    root = ET.parse(str(mujoco_xml)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"door model has no worldbody: {mujoco_xml}")

    sites: dict[str, tuple[float, float, float]] = {}
    geoms: dict[str, tuple[float, float, float]] = {}

    def walk_body(body, parent_xyz: Sequence[float]) -> None:
        body_xyz = _add_xyz(parent_xyz, _parse_xyz_attr(body.get("pos")))
        for child in body:
            if child.tag == "site":
                name = child.get("name")
                if name:
                    sites[str(name)] = _add_xyz(body_xyz, _parse_xyz_attr(child.get("pos")))
            elif child.tag == "geom":
                name = child.get("name")
                if name:
                    geoms[str(name)] = _add_xyz(body_xyz, _parse_xyz_attr(child.get("pos")))
            elif child.tag == "body":
                walk_body(child, body_xyz)

    for body in worldbody.findall("body"):
        walk_body(body, (0.0, 0.0, 0.0))

    try:
        hinge_xyz = sites["fridge_hinge_axis_site"]
        handle_frame_xyz = sites["fridge_handle_center_site"]
    except KeyError as exc:
        raise RuntimeError(f"door model missing required site {exc.args[0]!r}") from exc

    return DoorModelFrames(
        hinge_xyz=hinge_xyz,
        handle_frame_xyz=handle_frame_xyz,
        grasp_target_xyz=(
            handle_frame_xyz[0] + HANDLE_TARGET_X_OFFSET_M,
            handle_frame_xyz[1],
            handle_frame_xyz[2] + HANDLE_TARGET_Z_OFFSET_M,
        ),
    )


_DOOR_MODEL_FRAMES = _load_door_model_frames(DOOR_MODEL_XML)
HANDLE_CENTER_XYZ = _DOOR_MODEL_FRAMES.grasp_target_xyz
HANDLE_FRAME_XYZ = _DOOR_MODEL_FRAMES.handle_frame_xyz
DOOR_HINGE_XYZ = _DOOR_MODEL_FRAMES.hinge_xyz


def _build_arm_door_parser():
    from capstone_pkg.planner.arm_rrt_common.single_arm_runner import build_single_arm_parser

    ap = build_single_arm_parser(
        default_world_yml=DOOR_COLLISION_YAML,
        collision_models=_COLLISION_MODELS,
        default_collision_model="door",
    )
    ap.description = "Plan a single-arm TB-RRT motion to the showcase refrigerator handle."
    ap.add_argument(
        "--arm",
        choices=("left", "right"),
        default="right",
        help="arm used to reach the refrigerator handle",
    )
    ap.add_argument(
        "--target_xyz",
        nargs=3,
        type=float,
        default=list(HANDLE_CENTER_XYZ),
        help="target handle center xyz in world/base frame",
    )
    ap.add_argument(
        "--target_quat_xyzw",
        nargs=4,
        type=float,
        default=list(HANDLE_QUAT_XYZW),
        help="target end-effector quaternion in xyzw order",
    )
    ap.add_argument(
        "--handle_xyz",
        nargs=3,
        type=float,
        default=list(HANDLE_FRAME_XYZ),
        help="closed-door handle frame xyz in world/base frame",
    )
    ap.add_argument(
        "--handle_quat_xyzw",
        nargs=4,
        type=float,
        default=None,
        help="closed-door handle frame quaternion in xyzw order; default uses --target_quat_xyzw",
    )
    ap.add_argument(
        "--door_open_handle_radial_offset_m",
        type=float,
        default=0.11,
        help="radial correction applied to the opening EE target; positive moves away from the hinge",
    )
    ap.add_argument(
        "--door_open_orientation_lag_deg",
        type=float,
        default=0.0,
        help="use a smaller door angle for opening EE orientation than for handle position; useful when the physical door lags the commanded hinge",
    )
    ap.add_argument(
        "--close_gripper_after_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="close the selected gripper after the arm path is published",
    )
    ap.add_argument(
        "--open_door_after_grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after grasping, unlock the simulated door and follow the handle arc",
    )
    ap.add_argument("--door_open_angle_deg", type=float, default=80.0)
    ap.add_argument("--door_open_steps", type=int, default=25)
    ap.add_argument("--door_open_dt", type=float, default=0.1)
    ap.add_argument(
        "--door_open_ik_batch",
        type=int,
        default=8,
        help="number of IK seed trials per opening waypoint",
    )
    ap.add_argument(
        "--door_open_ik_seed_noise_std",
        type=float,
        default=0.20,
        help="Gaussian std [rad] for perturbing the previous opening waypoint as IK seeds",
    )
    ap.add_argument(
        "--door_open_front_ik_batch_multiplier",
        type=int,
        default=4,
        help="multiply IK seed trials for post-open/front waypoints without relaxing IK error thresholds",
    )
    ap.add_argument("--door_open_ik_seed", type=int, default=11)
    ap.add_argument(
        "--door_open_ik_max_pos_m",
        type=float,
        default=0.02,
        help="reject opening IK candidates whose FK position error exceeds this value",
    )
    ap.add_argument(
        "--door_open_ik_max_rot_deg",
        type=float,
        default=15.0,
        help="reject opening IK candidates whose FK orientation error exceeds this value",
    )
    ap.add_argument(
        "--door_open_orientation_constraint",
        choices=("door_relative", "rigid_grasp"),
        default="door_relative",
        help="opening hand orientation target: door_relative rotates the closed-grasp EE orientation with the door; rigid_grasp preserves full handle-to-EE pose",
    )
    ap.add_argument(
        "--door_open_base_assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample a simple mobile-base pose schedule during door opening",
    )
    ap.add_argument("--door_open_base_start_deg", type=float, default=0.0)
    ap.add_argument("--door_open_base_end_x", type=float, default=-0.20)
    ap.add_argument("--door_open_base_end_y", type=float, default=-0.10)
    ap.add_argument("--door_open_base_end_yaw_deg", type=float, default=-10.0)
    ap.add_argument("--door_open_base_x_span", type=float, default=0.25)
    ap.add_argument("--door_open_base_y_span", type=float, default=0.15)
    ap.add_argument("--door_open_base_yaw_span_deg", type=float, default=15.0)
    ap.add_argument(
        "--door_open_base_max_candidates",
        type=int,
        default=32,
        help="maximum base pose candidates evaluated per opening waypoint; <=0 disables the cap",
    )
    ap.add_argument(
        "--door_open_beam_width",
        type=int,
        default=1,
        help="number of base+right-arm partial paths kept per opening waypoint; 1 is greedy",
    )
    ap.add_argument(
        "--door_open_base_planner",
        choices=("beam", "astar", "graph"),
        default="graph",
        help="base/arm opening planner: beam keeps IK in the layer search, graph runs a fast base-door S1 planner before IK, astar keeps the old IK-in-loop A*",
    )
    ap.add_argument(
        "--door_open_graph_s2_local_repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when the graph-selected base pose fails opening IK, try nearby base candidates at the same door angle before failing",
    )
    ap.add_argument(
        "--door_open_graph_s2_repair_max_candidates",
        type=int,
        default=96,
        help="maximum base candidates tried by the S2 local repair at a failed graph waypoint",
    )
    ap.add_argument(
        "--door_open_graph_s2_repair_xy_step_m",
        type=float,
        default=0.04,
        help="xy sampling step for S2 local repair; kept separate from the S1 graph step to avoid excessive IK calls",
    )
    ap.add_argument(
        "--door_open_astar_max_expansions",
        type=int,
        default=1200,
        help="maximum A* graph-node expansions for the coordinated opening path",
    )
    ap.add_argument(
        "--door_open_astar_layer_keep",
        type=int,
        default=96,
        help="maximum queued A* states kept per door-angle layer; <=0 disables this pruning",
    )
    ap.add_argument(
        "--door_open_astar_queue_keep",
        type=int,
        default=512,
        help="maximum queued A* states kept globally; <=0 disables this pruning",
    )
    ap.add_argument(
        "--door_open_astar_heuristic_weight",
        type=float,
        default=1.0,
        help="weight for the admissible base-distance heuristic used by A*",
    )
    ap.add_argument("--door_open_graph_xy_step_m", type=float, default=0.04)
    ap.add_argument("--door_open_graph_yaw_step_deg", type=float, default=5.0)
    ap.add_argument(
        "--door_open_graph_lambda_step_deg",
        type=float,
        default=1.0,
        help="door-angle discretization for Lambda(s), the feasible angle set used by the S1 graph constraint",
    )
    ap.add_argument("--door_open_graph_max_expansions", type=int, default=25000)
    ap.add_argument("--door_open_graph_bound_margin_m", type=float, default=0.12)
    ap.add_argument("--door_open_graph_bound_margin_yaw_deg", type=float, default=15.0)
    ap.add_argument("--door_open_graph_base_radius_m", type=float, default=0.30)
    ap.add_argument("--door_open_graph_base_height_m", type=float, default=0.45)
    ap.add_argument("--door_open_graph_reach_shoulder_xyz", nargs=3, type=float, default=[0.0, -0.25, 1.0])
    ap.add_argument("--door_open_graph_reach_min_m", type=float, default=0.20)
    ap.add_argument("--door_open_graph_reach_max_m", type=float, default=0.95)
    ap.add_argument("--door_open_graph_reach_z_min_m", type=float, default=0.65)
    ap.add_argument("--door_open_graph_reach_z_max_m", type=float, default=1.35)
    ap.add_argument("--door_open_graph_reach_nominal_m", type=float, default=0.55)
    ap.add_argument("--door_open_front_goal_x", type=float, default=None)
    ap.add_argument("--door_open_front_goal_y", type=float, default=None)
    ap.add_argument("--door_open_front_goal_yaw_deg", type=float, default=0.0)
    ap.add_argument("--door_open_front_goal_tol_m", type=float, default=0.02)
    ap.add_argument(
        "--door_open_front_goal_x_tol_m",
        type=float,
        default=None,
        help="x tolerance for the post-opening base goal; default uses --door_open_front_goal_tol_m",
    )
    ap.add_argument(
        "--door_open_front_goal_y_tol_m",
        type=float,
        default=0.06,
        help="y tolerance for the post-opening base goal while keeping the base on the shifted traversal line",
    )
    ap.add_argument("--door_open_front_goal_tol_deg", type=float, default=20.0)
    ap.add_argument(
        "--door_open_front_detour",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="split the final forward move into an open-door detour while keeping the handle grasped",
    )
    ap.add_argument(
        "--door_open_front_detour_y",
        type=float,
        default=None,
        help="detour y pose used after the door is open; default picks the side away from the open door",
    )
    ap.add_argument(
        "--door_open_front_detour_y_backoff_m",
        type=float,
        default=0.05,
        help="legacy option retained for CLI compatibility; automatic traversal now targets --door_open_front_goal_y",
    )
    ap.add_argument("--door_open_front_detour_y_steps", type=int, default=4)
    ap.add_argument(
        "--door_open_front_clear_y_steps",
        type=int,
        default=4,
        help="number of post-detour y-return waypoints before advancing forward through the open door",
    )
    ap.add_argument("--door_open_front_detour_x_steps", type=int, default=8)
    ap.add_argument(
        "--door_open_front_door_clearance_m",
        type=float,
        default=0.0,
        help="minimum clearance from the moving door collision model during post-open/front motion",
    )
    ap.add_argument(
        "--post_open_base_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append a base-only forward move after the door opening publish path",
    )
    ap.add_argument(
        "--post_open_base_forward_m",
        type=float,
        default=0.15,
        help="world-x distance to move after the constrained door opening path",
    )
    ap.add_argument(
        "--post_open_base_steps",
        type=int,
        default=8,
        help="number of appended base-only publish samples after opening",
    )
    ap.add_argument(
        "--post_open_release_ingress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after opening the door, release the handle, retreat the right gripper, zero the right arm, then drive the base inward",
    )
    ap.add_argument(
        "--post_open_retreat_z_m",
        type=float,
        default=0.05,
        help="right-gripper local +Z retreat distance after releasing the door handle",
    )
    ap.add_argument("--post_open_retreat_steps", type=int, default=5)
    ap.add_argument("--post_open_release_wait_s", type=float, default=0.3)
    ap.add_argument("--post_open_retreat_wait_s", type=float, default=0.0)
    ap.add_argument("--post_open_zero_wait_s", type=float, default=0.0)
    ap.add_argument(
        "--post_open_lateral_distance_m",
        type=float,
        default=0.15,
        help="base body-y distance to move after the right arm is zeroed and before ingress",
    )
    ap.add_argument(
        "--post_open_lateral_speed_mps",
        type=float,
        default=0.10,
        help="base body-y speed for the post-open lateral move before ingress",
    )
    ap.add_argument(
        "--post_open_ingress_distance_m",
        type=float,
        default=0.50,
        help="base body-x distance to drive inward after the right arm is zeroed",
    )
    ap.add_argument(
        "--post_open_ingress_yaw_deg",
        type=float,
        default=-10.0,
        help="base yaw delta to apply during the post-open inward drive",
    )
    ap.add_argument(
        "--post_open_ingress_speed_mps",
        type=float,
        default=0.10,
        help="base body-x speed for the post-open inward drive",
    )
    ap.add_argument(
        "--post_open_final_lateral_distance_m",
        type=float,
        default=-0.15,
        help="base body-y distance to move after the post-open inward drive",
    )
    ap.add_argument(
        "--post_open_final_lateral_speed_mps",
        type=float,
        default=0.10,
        help="base body-y speed for the post-open lateral move after ingress",
    )
    ap.add_argument(
        "--door_open_base_publish_interp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="densify simulated opening publish samples so base pose commands do not jump",
    )
    ap.add_argument("--door_open_base_publish_step_m", type=float, default=0.02)
    ap.add_argument("--door_open_base_publish_step_deg", type=float, default=3.0)
    ap.add_argument(
        "--door_open_base_publish_smooth_window",
        type=int,
        default=3,
        help="optional centered moving-average window over real opening base poses before cmd_vel conversion; 0 disables",
    )
    ap.add_argument(
        "--door_open_topp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="spline and TOPP-retime the coordinated arm+base real opening path",
    )
    ap.add_argument(
        "--door_open_topp_spline_mode",
        choices=("cubic", "linear"),
        default="linear",
        help="path interpolation used before TOPP; linear avoids cubic overshoot on short arm paths",
    )
    ap.add_argument("--door_open_topp_spline_step", type=float, default=0.05)
    ap.add_argument("--door_open_topp_arm_max_velocity", type=float, default=1.0)
    ap.add_argument("--door_open_topp_arm_max_acceleration", type=float, default=2.0)
    ap.add_argument("--door_open_topp_base_max_linear_accel", type=float, default=2.0)
    ap.add_argument("--door_open_topp_base_max_angular_accel", type=float, default=2.0)
    ap.add_argument("--door_open_topp_safety_scale", type=float, default=1.05)
    ap.add_argument("--door_open_topp_max_iterations", type=int, default=30)
    ap.add_argument(
        "--door_open_topp_max_duration_s",
        type=float,
        default=0.0,
        help="optional cap for retimed opening duration; <=0 leaves TOPP duration unconstrained",
    )
    ap.add_argument(
        "--door_open_wbc_qp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run a velocity-level QP rollout over the planned opening base+arm path before publishing",
    )
    ap.add_argument(
        "--door_open_wbc_qp_backend",
        choices=("auto", "osqp", "scipy"),
        default="osqp",
        help="QP backend: auto uses OSQP when installed, otherwise scipy SLSQP",
    )
    ap.add_argument(
        "--door_open_wbc_qp_hard_constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="treat the gripper/handle velocity task as a hard equality constraint; soft weighted tracking is the default",
    )
    ap.add_argument("--door_open_wbc_qp_kp_pos", type=float, default=2.0)
    ap.add_argument("--door_open_wbc_qp_kp_rot", type=float, default=2.0)
    ap.add_argument("--door_open_wbc_qp_task_weight", type=float, default=100.0)
    ap.add_argument("--door_open_wbc_qp_base_ref_weight", type=float, default=1.0)
    ap.add_argument("--door_open_wbc_qp_joint_ref_weight", type=float, default=1.0)
    ap.add_argument("--door_open_wbc_qp_joint_reg_weight", type=float, default=1.0e-3)
    ap.add_argument("--door_open_base_pose_topic", default="/base_pose_cmd")
    ap.add_argument(
        "--real_base_cmd_vel_topic",
        default="/cmd_vel",
        help="real FFW-SG2 mobile-base command topic; ROBOTIS AI Worker swerve_drive_controller listens on /cmd_vel",
    )
    ap.add_argument("--real_base_max_linear_mps", type=float, default=0.35)
    ap.add_argument("--real_base_max_angular_rps", type=float, default=0.60)
    ap.add_argument(
        "--real_base_cmd_rate_hz",
        type=float,
        default=100.0,
        help="rate used to stream repeated /cmd_vel samples for real base opening motion",
    )
    ap.add_argument("--real_base_stop_duration_s", type=float, default=0.5)
    ap.add_argument(
        "--door_open_arm_republish_if_static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in real opening mode, republish the arm trajectory if joint_states do not show arm motion shortly after publishing",
    )
    ap.add_argument("--door_open_arm_republish_attempts", type=int, default=2)
    ap.add_argument("--door_open_arm_motion_check_s", type=float, default=1.0)
    ap.add_argument("--door_open_arm_motion_threshold_rad", type=float, default=0.005)
    ap.add_argument("--door_unlock_topic", default=DEFAULT_DOOR_UNLOCK_TOPIC)
    ap.add_argument(
        "--oneshot",
        action="store_true",
        help="run once from CLI target arguments instead of waiting for arm_door_start and door_handle_align_result",
    )
    ap.add_argument("--arm_door_start_topic", default="/arm_door_start")
    ap.add_argument("--arm_door_finish_topic", default="/arm_door_finish")
    ap.add_argument("--arm_picking_finish_topic", default="/arm_picking_finish")
    ap.add_argument("--one_step_finish_topic", default="/one_step_finish")
    ap.add_argument(
        "--post_one_step_door_close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after arm_door_finish, wait for one_step_finish and back "
            "the base up to close the door"
        ),
    )
    ap.add_argument(
        "--post_one_step_reverse_distance_m",
        type=float,
        default=0.50,
        help="base reverse distance after one_step_finish",
    )
    ap.add_argument(
        "--post_one_step_reverse_speed_mps",
        type=float,
        default=0.10,
        help="base reverse speed after one_step_finish",
    )
    ap.add_argument(
        "--post_one_step_align_yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "align the mobile-base yaw to --post_one_step_target_yaw_deg "
            "before the post-one-step reverse move"
        ),
    )
    ap.add_argument(
        "--post_one_step_base_pose_topic",
        default="/base_pose",
        help="Pose2D feedback topic used for post-one-step absolute yaw alignment",
    )
    ap.add_argument("--post_one_step_target_yaw_deg", type=float, default=0.0)
    ap.add_argument("--post_one_step_yaw_tolerance_deg", type=float, default=2.0)
    ap.add_argument("--post_one_step_yaw_timeout_s", type=float, default=8.0)
    ap.add_argument("--post_one_step_base_pose_wait_s", type=float, default=1.0)
    ap.add_argument("--post_one_step_yaw_kp", type=float, default=1.5)
    ap.add_argument("--post_one_step_yaw_min_speed_rps", type=float, default=0.05)
    ap.add_argument("--door_handle_align_topic", default="/door_handle_align_result")
    ap.add_argument("--finish_publish_repeat", type=int, default=1)
    ap.add_argument("--finish_publish_period_s", type=float, default=0.05)
    ap.add_argument(
        "--detected_handle_target_x_offset_m",
        type=float,
        default=DETECTED_HANDLE_TARGET_X_OFFSET_M,
        help="x-axis offset from detected handle center to the grasp target",
    )
    ap.add_argument(
        "--detected_handle_target_z_m",
        type=float,
        default=DETECTED_HANDLE_TARGET_Z_M,
        help="fixed z used for detected handle center and grasp target",
    )
    ap.add_argument(
        "--startup_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pre-initialize and warm up IK/collision objects before processing start topics",
    )
    ap.add_argument(
        "--startup_warmup_iters",
        type=int,
        default=1,
        help="number of reachable IK warmup iterations to run at startup",
    )
    ap.add_argument(
        "--startup_warmup_batch_size",
        type=int,
        default=None,
        help="batch size used for startup IK warmup solves; defaults to --ik_batch",
    )
    ap.add_argument(
        "--door_open_sync_hinge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in simulation JointState mode, publish fridge door hinge angle with the arm opening path",
    )
    ap.add_argument("--door_hinge_joint", default=DOOR_HINGE_JOINT)
    ap.add_argument(
        "--validate_opening_fk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run FK on the generated opening path and report door-relative/rigid grasp constraint error",
    )
    ap.add_argument(
        "--opening_fk_log",
        default="",
        help="optional JSON path for per-waypoint opening FK validation",
    )
    ap.add_argument("--opening_fk_warn_pos_m", type=float, default=0.02)
    ap.add_argument("--opening_fk_warn_rot_deg", type=float, default=15.0)
    ap.add_argument(
        "--opening_fk_fail_on_warn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail before publishing if opening FK constraint error exceeds warning thresholds",
    )
    ap.add_argument(
        "--validate_opening_dynamic_collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate each opening waypoint against a world model whose door cuboids are rotated to that waypoint angle",
    )
    ap.add_argument(
        "--door_open_collision_aware_base_selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject opening IK/base candidates that collide with the dynamic door world while selecting the path",
    )
    ap.add_argument(
        "--door_open_collision_lag_deg",
        type=float,
        default=0.0,
        help="also check dynamic door collision with the door this many degrees less open than commanded",
    )
    ap.add_argument(
        "--opening_dynamic_world_yml",
        default="same",
        help="'same' uses --world_yml as the closed-door source; 'none' disables dynamic opening collision validation; otherwise path to yaml",
    )
    ap.add_argument(
        "--opening_dynamic_ignore",
        nargs="*",
        default=list(DOOR_GRASP_CONTACT_COLLISION_NAMES),
        help="moving cuboid names to remove during opening collision validation, usually grasp contact geometry",
    )
    ap.add_argument(
        "--opening_dynamic_collision_margin_m",
        type=float,
        default=0.0,
        help="extra required clearance for opening dynamic collision validation",
    )
    ap.add_argument(
        "--opening_dynamic_collision_log",
        default="",
        help="optional JSON path for per-waypoint dynamic opening collision validation",
    )
    ap.add_argument(
        "--door_open_world_yml",
        default="none",
        help="'none' disables static closed-door world collision during opening IK; 'same' reuses --world_yml; otherwise path to yaml",
    )
    ap.set_defaults(gripper_target=1.1, gripper_delay_s=0.0)
    return ap


def _publish_real_path(args, arm: str, joint_names, path, *, dt_override: float | None = None) -> None:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        publish_joint_trajectory,
        send_joint_trajectory_action,
    )

    dt = float(args.publish_dt) if dt_override is None else float(dt_override)
    topic = args.real_left_topic if arm == "left" else args.real_right_topic
    action_name = args.real_left_action if arm == "left" else args.real_right_action

    if args.real_use_action:
        try:
            print(f"[PUBLISH] FollowJointTrajectory -> {action_name}")
            send_joint_trajectory_action(
                path,
                joint_names,
                action_name=action_name,
                dt=dt,
                wait_server_s=args.action_wait_server_s,
                wait_result_s=args.action_wait_result_s,
            )
            return
        except RuntimeError as exc:
            print(f"[PUBLISH][ACTION] {exc}")
            if not args.real_action_fallback_to_topic:
                raise
            print(f"[PUBLISH][ACTION] Falling back to JointTrajectory topic -> {topic}")

    publish_joint_trajectory(
        path,
        joint_names,
        topic=topic,
        dt=dt,
        wait_subscriber_s=args.publish_wait_subscriber_s,
        require_subscriber=args.publish_require_subscriber,
        retry_until_subscriber=args.publish_retry_until_subscriber,
        publish_repeat=args.publish_repeat,
        publish_period_s=args.publish_period_s,
        wait_ack_s=args.publish_wait_ack_s,
        keep_alive_s=args.publish_keep_alive_s,
        reliability=str(getattr(args, "publish_reliability", "reliable")),
        durability=(
            "transient_local"
            if bool(getattr(args, "publish_transient_local", False))
            else str(getattr(args, "publish_durability", "volatile"))
        ),
        qos_depth=int(getattr(args, "publish_qos_depth", 10)),
        start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
        zero_header_stamp=bool(getattr(args, "real_zero_trajectory_stamp", True)),
    )


def _pad_joint_path_to_length(
    path: Sequence[Sequence[float]],
    length: int,
) -> list[list[float]]:
    if not path:
        raise RuntimeError("cannot pad an empty joint path")
    out = [[float(v) for v in q] for q in path]
    while len(out) < int(length):
        out.append(list(out[-1]))
    return out


def _plan_left_arm_zero_for_reach(
    args,
    *,
    cspace_joint_names: Sequence[str],
    q_start_cspace: Sequence[float],
    world_yml: str | None,
) -> tuple[list[str], list[list[float]]]:
    from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
        build_single_arm_tbrrt_config,
    )
    from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
        plan_single_arm_tbrrt_batch_conext,
    )
    from capstone_pkg.utils.config import LEFT_JOINTS

    left_joint_names = [str(name) for name in LEFT_JOINTS]
    cspace_names = [str(name) for name in cspace_joint_names]
    q_start = [float(v) for v in q_start_cspace]
    if len(cspace_names) != len(q_start):
        raise RuntimeError("left zero planning cspace names and start state length mismatch")

    q_goal = _build_arm_zero_goal_cspace(
        q_start_cspace=q_start,
        cspace_joint_names=cspace_names,
        active_joint_names=left_joint_names,
    )
    left_idx = {
        name: cspace_names.index(name)
        for name in left_joint_names
        if name in cspace_names
    }
    if len(left_idx) != len(left_joint_names):
        missing = [name for name in left_joint_names if name not in left_idx]
        raise RuntimeError(f"Missing left joints in cspace state: {missing}")

    if all(abs(q_start[left_idx[name]]) <= 1.0e-4 for name in left_joint_names):
        print("[ARM_DOOR][LEFT_ARM_ZERO] left arm already at zero; sending hold command")
        return left_joint_names, [[0.0 for _ in left_joint_names]]

    print("[ARM_DOOR][LEFT_ARM_ZERO] planning left arm path to zero joints")
    t_start = time.perf_counter()
    out = plan_single_arm_tbrrt_batch_conext(
        robot_yml=str(args.robot_yml),
        arm="left",
        q_start=q_start,
        q_goals=[q_goal],
        world_yml=world_yml,
        cpu=bool(args.cpu),
        cfg=build_single_arm_tbrrt_config(args),
        joint_limit_yml=str(args.joint_limit_yml),
        block_k=int(args.tbrrt_block_k),
    )
    if not out.success or not out.path:
        raise RuntimeError(f"Failed to plan left arm zero path: {out.stats.extra}")

    _, left_path = _active_path_from_cspace(
        cspace_names,
        left_joint_names,
        [[float(v) for v in q] for q in out.path],
    )
    print(
        "[ARM_DOOR][LEFT_ARM_ZERO] planned "
        f"waypoints={len(left_path)} planning_time={time.perf_counter() - t_start:.3f}s"
    )
    return left_joint_names, left_path


def _publish_reach_path_with_left_zero(
    args,
    arm: str,
    joint_names,
    path,
    *,
    cspace_joint_names: Sequence[str] | None = None,
    q_start_cspace: Sequence[float] | None = None,
    world_yml: str | None = None,
) -> int:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        JointTrajectoryCommand,
        publish_joint_path,
        publish_joint_trajectory_group,
        send_joint_trajectory_action_group,
    )

    normalized_arm = str(arm).strip().lower()
    if normalized_arm != "right":
        if args.publish_mode == "real":
            _publish_real_path(args, arm, joint_names, path)
        else:
            publish_joint_path(
                path,
                joint_names,
                topic=str(args.publish_topic),
                dt=float(args.publish_dt),
                wait_subscriber_s=float(args.publish_wait_subscriber_s),
            )
        return len(path)

    right_joint_names = [str(name) for name in joint_names]
    right_path = [[float(v) for v in q] for q in path]
    if cspace_joint_names is None or q_start_cspace is None:
        raise RuntimeError(
            "right-arm reach publishing requires cspace start state for left zero planning"
        )
    left_joint_names, left_zero_path = _plan_left_arm_zero_for_reach(
        args,
        cspace_joint_names=cspace_joint_names,
        q_start_cspace=q_start_cspace,
        world_yml=world_yml,
    )
    sync_len = max(len(left_zero_path), len(right_path))
    left_zero_path = _pad_joint_path_to_length(left_zero_path, sync_len)
    right_path = _pad_joint_path_to_length(right_path, sync_len)

    if args.publish_mode == "real":
        action_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_action),
                joint_names=left_joint_names,
                path=left_zero_path,
                label="left_arm_zero",
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_action),
                joint_names=right_joint_names,
                path=right_path,
                label="right_arm_reach",
            ),
        ]
        topic_commands = [
            JointTrajectoryCommand(
                endpoint=str(args.real_left_topic),
                joint_names=left_joint_names,
                path=left_zero_path,
                label="left_arm_zero",
            ),
            JointTrajectoryCommand(
                endpoint=str(args.real_right_topic),
                joint_names=right_joint_names,
                path=right_path,
                label="right_arm_reach",
            ),
        ]
        if args.real_use_action:
            try:
                print(
                    "[PUBLISH] FollowJointTrajectory group -> "
                    f"{args.real_left_action}, {args.real_right_action}"
                )
                send_joint_trajectory_action_group(
                    action_commands,
                    dt=args.publish_dt,
                    wait_server_s=args.action_wait_server_s,
                    wait_result_s=args.action_wait_result_s,
                    start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
                )
                return sync_len
            except RuntimeError as exc:
                print(f"[PUBLISH][ACTION] {exc}")
                if not args.real_action_fallback_to_topic:
                    raise
                print(
                    "[PUBLISH][ACTION] Falling back to JointTrajectory topics -> "
                    f"{args.real_left_topic}, {args.real_right_topic}"
                )

        publish_joint_trajectory_group(
            topic_commands,
            dt=args.publish_dt,
            wait_subscriber_s=args.publish_wait_subscriber_s,
            require_subscriber=args.publish_require_subscriber,
            retry_until_subscriber=args.publish_retry_until_subscriber,
            publish_repeat=args.publish_repeat,
            publish_period_s=args.publish_period_s,
            wait_ack_s=args.publish_wait_ack_s,
            keep_alive_s=args.publish_keep_alive_s,
            reliability=str(getattr(args, "publish_reliability", "reliable")),
            durability=(
                "transient_local"
                if bool(getattr(args, "publish_transient_local", False))
                else str(getattr(args, "publish_durability", "volatile"))
            ),
            qos_depth=int(getattr(args, "publish_qos_depth", 10)),
            start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
            zero_header_stamp=bool(getattr(args, "real_zero_trajectory_stamp", True)),
        )
        return sync_len

    combined_joint_names = left_joint_names + right_joint_names
    combined_path = [
        list(left_q) + list(right_q)
        for left_q, right_q in zip(left_zero_path, right_path)
    ]
    print(
        "[ARM_DOOR] Publishing reach JointState path with left arm zero -> "
        f"{args.publish_topic} "
        f"(dt={float(args.publish_dt):.3f}s)"
    )
    publish_joint_path(
        combined_path,
        combined_joint_names,
        topic=str(args.publish_topic),
        dt=float(args.publish_dt),
        wait_subscriber_s=float(args.publish_wait_subscriber_s),
    )
    return sync_len


def _publish_gripper_target(args, arm: str, target: float, *, label: str) -> None:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        publish_joint_path,
        publish_joint_trajectory,
        send_joint_trajectory_action,
    )

    gripper_joint = "gripper_l_joint1" if arm == "left" else "gripper_r_joint1"
    target = float(target)

    if args.publish_mode == "real":
        topic = args.real_left_gripper_topic if arm == "left" else args.real_right_gripper_topic
        action_name = args.real_left_gripper_action if arm == "left" else args.real_right_gripper_action
        if args.real_use_action:
            try:
                print(
                    f"[ARM_DOOR][GRIPPER][{label}] FollowJointTrajectory -> {action_name} "
                    f"({gripper_joint}={target:.3f})"
                )
                send_joint_trajectory_action(
                    [[target]],
                    [gripper_joint],
                    action_name=action_name,
                    dt=0.1,
                    wait_server_s=args.action_wait_server_s,
                    wait_result_s=args.action_wait_result_s,
                )
                return
            except RuntimeError as exc:
                print(f"[ARM_DOOR][GRIPPER][{label}][ACTION] {exc}")
                if not args.real_action_fallback_to_topic:
                    raise
                print(f"[ARM_DOOR][GRIPPER][{label}][ACTION] Falling back to topic -> {topic}")

        print(
            f"[ARM_DOOR][GRIPPER][{label}] JointTrajectory -> {topic} "
            f"({gripper_joint}={target:.3f})"
        )
        publish_joint_trajectory(
            [[target]],
            [gripper_joint],
            topic=topic,
            dt=0.1,
            wait_subscriber_s=args.publish_wait_subscriber_s,
            require_subscriber=args.publish_require_subscriber,
            retry_until_subscriber=args.publish_retry_until_subscriber,
            publish_repeat=args.publish_repeat,
            publish_period_s=args.publish_period_s,
            wait_ack_s=args.publish_wait_ack_s,
            keep_alive_s=args.publish_keep_alive_s,
            reliability=str(getattr(args, "publish_reliability", "reliable")),
            durability=(
                "transient_local"
                if bool(getattr(args, "publish_transient_local", False))
                else str(getattr(args, "publish_durability", "volatile"))
            ),
            qos_depth=int(getattr(args, "publish_qos_depth", 10)),
            start_time_delay_s=float(getattr(args, "start_delay_s", 0.2)),
            zero_header_stamp=bool(getattr(args, "real_zero_trajectory_stamp", True)),
        )
        return

    print(
        f"[ARM_DOOR][GRIPPER][{label}] JointState -> {args.publish_topic} "
        f"({gripper_joint}={target:.3f})"
    )
    publish_joint_path(
        [[target]],
        [gripper_joint],
        topic=str(args.publish_topic),
        dt=0.1,
        wait_subscriber_s=float(args.publish_wait_subscriber_s),
    )


def _publish_gripper_close(args, arm: str) -> None:
    _publish_gripper_target(args, arm, float(args.gripper_target), label="CLOSE")


def _publish_gripper_open(args, arm: str) -> None:
    _publish_gripper_target(args, arm, 0.0, label="OPEN")


def _build_arm_zero_goal_cspace(
    *,
    q_start_cspace: Sequence[float],
    cspace_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> list[float]:
    goal = [float(v) for v in q_start_cspace]
    name_to_idx = {str(name): idx for idx, name in enumerate(cspace_joint_names)}
    missing = [str(name) for name in active_joint_names if str(name) not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in zero-goal cspace build: {missing}")
    for joint_name in active_joint_names:
        goal[name_to_idx[str(joint_name)]] = 0.0
    return goal


def _read_or_fallback_cspace_start(
    args,
    *,
    cspace_joint_names: Sequence[str],
    fallback_q_cspace: Sequence[float],
) -> list[float]:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import read_joint_positions_once

    try:
        return read_joint_positions_once(
            [str(name) for name in cspace_joint_names],
            topic=str(args.joint_state_topic),
            wait_s=max(0.05, float(args.joint_state_wait_s)),
        )
    except Exception as exc:
        print(
            "[ARM_DOOR][RIGHT_ARM_ZERO][WARN] "
            f"could not read current cspace from {args.joint_state_topic}: {exc}; "
            "using cached close-end state"
        )
        return [float(v) for v in fallback_q_cspace]


def _plan_and_publish_right_arm_zero(
    args,
    *,
    start_cspace_joint_names: Sequence[str] | None = None,
    fallback_start_q_cspace: Sequence[float] | None = None,
    world_yml: str | None = None,
) -> float:
    from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
    from capstone_pkg.planner.arm_rrt_common.single_arm_motion import normalize_arm_name
    from capstone_pkg.planner.arm_rrt_common.single_arm_runner import build_single_arm_tbrrt_config
    from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
        plan_single_arm_tbrrt_batch_conext,
    )
    from capstone_pkg.utils.config import RIGHT_JOINTS

    arm = normalize_arm_name("right")
    active_joint_names = [str(name) for name in RIGHT_JOINTS]
    if start_cspace_joint_names is None:
        ik = get_single_arm_ik(
            str(args.robot_yml),
            arm=arm,
            cpu=bool(args.cpu),
            world_yml=world_yml,
        )
        cspace_joint_names = [str(name) for name in ik.cspace_joint_names]
    else:
        cspace_joint_names = [str(name) for name in start_cspace_joint_names]
    fallback_q = (
        [0.0 for _ in cspace_joint_names]
        if fallback_start_q_cspace is None
        else [float(v) for v in fallback_start_q_cspace]
    )
    q_start = _read_or_fallback_cspace_start(
        args,
        cspace_joint_names=cspace_joint_names,
        fallback_q_cspace=fallback_q,
    )
    q_goal = _build_arm_zero_goal_cspace(
        q_start_cspace=q_start,
        cspace_joint_names=cspace_joint_names,
        active_joint_names=active_joint_names,
    )
    print("[ARM_DOOR][RIGHT_ARM_ZERO] planning right arm path to zero joints")
    t_start = time.perf_counter()
    out = plan_single_arm_tbrrt_batch_conext(
        robot_yml=str(args.robot_yml),
        arm=arm,
        q_start=q_start,
        q_goals=[q_goal],
        world_yml=world_yml,
        cpu=bool(args.cpu),
        cfg=build_single_arm_tbrrt_config(args),
        joint_limit_yml=str(args.joint_limit_yml),
        block_k=int(args.tbrrt_block_k),
    )
    if not out.success or not out.path:
        raise RuntimeError(f"Failed to plan right arm zero path: {out.stats.extra}")
    cspace_path = [[float(v) for v in q] for q in out.path]
    joint_names, active_path = _active_path_from_cspace(
        cspace_joint_names,
        active_joint_names,
        cspace_path,
    )
    planning_time_s = time.perf_counter() - t_start
    print(
        "[ARM_DOOR][RIGHT_ARM_ZERO] planned "
        f"waypoints={len(active_path)} planning_time={planning_time_s:.3f}s"
    )
    publish_duration_s = max(0.0, float(len(active_path) - 1) * float(args.publish_dt))
    if args.publish_mode == "real":
        _publish_real_path(args, "right", joint_names, active_path)
        return publish_duration_s

    from capstone_pkg.planner.arm_rrt_common.path_publisher import publish_joint_path

    print(
        f"[ARM_DOOR][RIGHT_ARM_ZERO] JointState path -> {args.publish_topic} "
        f"({len(joint_names)} joints)"
    )
    publish_joint_path(
        active_path,
        joint_names,
        topic=str(args.publish_topic),
        dt=max(0.1, float(args.publish_dt)),
        wait_subscriber_s=float(args.publish_wait_subscriber_s),
    )
    return publish_duration_s


def _plan_and_publish_right_gripper_local_z_retreat(
    args,
    *,
    start_cspace_joint_names: Sequence[str],
    start_q_cspace: Sequence[float],
    retreat_z_m: float,
) -> tuple[list[str], list[float], float]:
    from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
    from capstone_pkg.planner.arm_rrt_common.single_arm_motion import _build_ik_seed_batch
    from capstone_pkg.utils.joint_limit import load_joint_limits_torch
    import numpy as np
    import torch

    cspace_joint_names = [str(name) for name in start_cspace_joint_names]
    q_start = [float(v) for v in start_q_cspace]
    if len(cspace_joint_names) != len(q_start):
        raise RuntimeError("post-open retreat cspace names and q length mismatch")

    ik = get_single_arm_ik(
        str(args.robot_yml),
        arm="right",
        cpu=bool(args.cpu),
        world_yml=None,
        use_cuda_graph=False,
    )
    q_active = _extract_active_from_cspace(
        q_start,
        cspace_joint_names,
        ik.active_joint_names,
    )
    with torch.no_grad():
        kin = ik.solver.fk(
            torch.tensor([q_active], device=ik.device, dtype=torch.float32)
        )
    ee_xyz = [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]
    ee_quat_wxyz = [float(v) for v in kin.ee_quaternion[0].detach().cpu().tolist()]
    local_z_base = _quat_rotate_wxyz(ee_quat_wxyz, [0.0, 0.0, 1.0])
    target_xyz = [
        ee_xyz[i] + float(retreat_z_m) * local_z_base[i]
        for i in range(3)
    ]

    jl = load_joint_limits_torch(
        str(args.joint_limit_yml),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    seed_batch = _build_ik_seed_batch(
        q_start,
        batch_size=max(1, int(getattr(args, "door_open_ik_batch", 8))),
        noise_std=max(0.0, float(getattr(args, "door_open_ik_seed_noise_std", 0.20))),
        random_seed=int(getattr(args, "door_open_ik_seed", 11)) + 50021,
        lower=jl.lower.detach().cpu().numpy(),
        upper=jl.upper.detach().cpu().numpy(),
    )
    outs = ik.solve_batch(
        [list(target_xyz) for _ in seed_batch],
        [list(ee_quat_wxyz) for _ in seed_batch],
        q_start_cspace=q_start,
        q_seed_cspace_batch=seed_batch,
    )
    candidates = [out.q_cspace for out in outs if out.success and out.q_cspace is not None]
    if not candidates:
        raise RuntimeError(
            "failed to solve post-open gripper local +Z retreat IK: "
            f"start_xyz={[round(v, 4) for v in ee_xyz]} "
            f"target_xyz={[round(v, 4) for v in target_xyz]}"
        )
    q_goal = min(
        ([float(v) for v in q] for q in candidates),
        key=lambda q: float(np.linalg.norm(np.asarray(q) - np.asarray(q_start))),
    )

    steps = max(1, int(getattr(args, "post_open_retreat_steps", 5)))
    cspace_path = []
    for idx in range(steps + 1):
        t = float(idx) / float(steps)
        cspace_path.append([
            q_start[j] + (q_goal[j] - q_start[j]) * t
            for j in range(len(q_start))
        ])
    joint_names, active_path = _active_path_from_cspace(
        cspace_joint_names,
        ik.controlled_joint_names,
        cspace_path,
    )
    dt = max(0.1, float(args.publish_dt))
    duration_s = max(0.0, float(len(active_path) - 1) * dt)
    print(
        "[ARM_DOOR][POST_OPEN][RETREAT] "
        f"right gripper local +Z {float(retreat_z_m):.3f}m "
        f"target_xyz={[round(v, 4) for v in target_xyz]} "
        f"waypoints={len(active_path)}"
    )
    if args.publish_mode == "real":
        _publish_real_path(args, "right", joint_names, active_path, dt_override=dt)
    else:
        from capstone_pkg.planner.arm_rrt_common.path_publisher import publish_joint_path

        publish_joint_path(
            active_path,
            joint_names,
            topic=str(args.publish_topic),
            dt=dt,
            wait_subscriber_s=float(args.publish_wait_subscriber_s),
        )
    return cspace_joint_names, q_goal, duration_s


def _publish_base_body_move(
    args,
    *,
    axis: str,
    distance_m: float,
    speed_mps: float,
    label: str,
    yaw_delta_deg: float = 0.0,
) -> None:
    axis = str(axis).lower()
    if axis not in {"x", "y"}:
        raise RuntimeError(f"unsupported base body move axis: {axis}")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node(f"arm_door_{label}_base_move")
    pub = node.create_publisher(Twist, str(args.real_base_cmd_vel_topic), 10)
    wait_deadline = time.monotonic() + max(0.0, float(args.publish_wait_subscriber_s))
    while rclpy.ok() and pub.get_subscription_count() == 0 and time.monotonic() < wait_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    distance = float(distance_m)
    max_linear = max(1.0e-3, abs(float(args.real_base_max_linear_mps)))
    speed = min(max_linear, max(1.0e-3, abs(float(speed_mps))))
    signed_speed = math.copysign(speed, distance)
    duration_s = abs(distance) / speed if abs(distance) > 1.0e-9 else 0.0
    yaw_delta_rad = math.radians(float(yaw_delta_deg))
    max_angular = max(1.0e-3, abs(float(args.real_base_max_angular_rps)))
    angular_z = 0.0 if duration_s <= 1.0e-9 else max(-max_angular, min(max_angular, yaw_delta_rad / duration_s))
    cmd_period_s = 1.0 / max(1.0, float(args.real_base_cmd_rate_hz))
    print(
        f"[ARM_DOOR][POST_OPEN][BASE] {label}: "
        f"body_{axis}={distance:+.3f}m speed={signed_speed:+.3f}m/s "
        f"yaw_delta={math.degrees(angular_z * duration_s):+.1f}deg "
        f"topic={args.real_base_cmd_vel_topic}"
    )

    msg = Twist()
    if axis == "x":
        msg.linear.x = float(signed_speed)
    else:
        msg.linear.y = float(signed_speed)
    msg.angular.z = float(angular_z)
    end_time = time.monotonic() + duration_s
    while rclpy.ok() and time.monotonic() < end_time:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        remaining = end_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(cmd_period_s, remaining))

    stop_msg = Twist()
    stop_end = time.monotonic() + max(0.0, float(args.real_base_stop_duration_s))
    while rclpy.ok() and time.monotonic() < stop_end:
        pub.publish(stop_msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(min(0.05, max(0.001, cmd_period_s)))

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()


def _run_post_open_release_retreat_zero_ingress(
    args,
    *,
    cspace_joint_names: Sequence[str],
    q_cspace_end: Sequence[float],
) -> None:
    if not bool(getattr(args, "post_open_release_ingress", True)):
        return

    print("[ARM_DOOR][POST_OPEN] release handle, retreat gripper, zero right arm, ingress base")
    _publish_gripper_open(args, "right")
    release_wait_s = max(0.0, float(getattr(args, "post_open_release_wait_s", 0.3)))
    if release_wait_s > 0.0:
        time.sleep(release_wait_s)

    retreat_names, retreat_q_end, retreat_duration_s = _plan_and_publish_right_gripper_local_z_retreat(
        args,
        start_cspace_joint_names=cspace_joint_names,
        start_q_cspace=q_cspace_end,
        retreat_z_m=float(getattr(args, "post_open_retreat_z_m", 0.05)),
    )
    retreat_wait_s = max(
        retreat_duration_s,
        max(0.0, float(getattr(args, "post_open_retreat_wait_s", 0.0))),
    )
    if retreat_wait_s > 0.0:
        print(f"[ARM_DOOR][POST_OPEN][RETREAT] waiting {retreat_wait_s:.2f}s")
        time.sleep(retreat_wait_s)

    zero_duration_s = _plan_and_publish_right_arm_zero(
        args,
        start_cspace_joint_names=retreat_names,
        fallback_start_q_cspace=retreat_q_end,
        world_yml=None,
    )
    zero_wait_s = max(
        zero_duration_s,
        max(0.0, float(getattr(args, "post_open_zero_wait_s", 0.0))),
    )
    if zero_wait_s > 0.0:
        print(f"[ARM_DOOR][POST_OPEN][ZERO] waiting {zero_wait_s:.2f}s")
        time.sleep(zero_wait_s)

    _publish_base_body_move(
        args,
        axis="y",
        distance_m=float(getattr(args, "post_open_lateral_distance_m", 0.15)),
        speed_mps=float(getattr(args, "post_open_lateral_speed_mps", 0.10)),
        label="lateral",
    )

    _publish_base_body_move(
        args,
        axis="x",
        distance_m=float(getattr(args, "post_open_ingress_distance_m", 0.50)),
        speed_mps=float(getattr(args, "post_open_ingress_speed_mps", 0.10)),
        label="ingress",
        yaw_delta_deg=float(getattr(args, "post_open_ingress_yaw_deg", -10.0)),
    )

    _publish_base_body_move(
        args,
        axis="y",
        distance_m=float(getattr(args, "post_open_final_lateral_distance_m", -0.15)),
        speed_mps=float(getattr(args, "post_open_final_lateral_speed_mps", 0.10)),
        label="final_lateral",
    )


def _xyzw_to_wxyz(q_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = [float(v) for v in q_xyzw]
    return [w, x, y, z]


def _quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> list[float]:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_conj_wxyz(q: Sequence[float]) -> list[float]:
    w, x, y, z = [float(v) for v in q]
    return [w, -x, -y, -z]


def _quat_rotate_wxyz(q: Sequence[float], vec_xyz: Sequence[float]) -> list[float]:
    qv = [0.0, float(vec_xyz[0]), float(vec_xyz[1]), float(vec_xyz[2])]
    out = _quat_mul_wxyz(_quat_mul_wxyz(q, qv), _quat_conj_wxyz(q))
    return [out[1], out[2], out[3]]


def _pose_mul(
    a_xyz: Sequence[float],
    a_quat_wxyz: Sequence[float],
    b_xyz: Sequence[float],
    b_quat_wxyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    b_rot = _quat_rotate_wxyz(a_quat_wxyz, b_xyz)
    xyz = [float(a_xyz[i]) + b_rot[i] for i in range(3)]
    quat = _quat_mul_wxyz(a_quat_wxyz, b_quat_wxyz)
    return xyz, quat


def _pose_inv(
    xyz: Sequence[float],
    quat_wxyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    q_inv = _quat_conj_wxyz(quat_wxyz)
    p_inv = _quat_rotate_wxyz(q_inv, [-float(xyz[0]), -float(xyz[1]), -float(xyz[2])])
    return p_inv, q_inv


def _yaw_quat_wxyz(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _yaw_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in q_wxyz]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(vec_xyz: Sequence[float], yaw_rad: float) -> list[float]:
    x, y, z = [float(v) for v in vec_xyz]
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return [c * x - s * y, s * x + c * y, z]


def _door_handle_pose_at(
    *,
    alpha_rad: float,
    closed_xyz: Sequence[float],
    closed_quat_wxyz: Sequence[float],
    door_hinge_xyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    hinge = [float(v) for v in door_hinge_xyz]
    rel = [float(closed_xyz[i]) - hinge[i] for i in range(3)]
    rel_rot = _rotate_z(rel, alpha_rad)
    xyz = [hinge[i] + rel_rot[i] for i in range(3)]
    quat = _quat_mul_wxyz(_yaw_quat_wxyz(alpha_rad), closed_quat_wxyz)
    return xyz, quat


def _offset_point_radially_from_hinge(
    point_xyz: Sequence[float],
    hinge_xyz: Sequence[float],
    offset_m: float,
) -> list[float]:
    point = [float(v) for v in point_xyz]
    hinge = [float(v) for v in hinge_xyz]
    offset = float(offset_m)
    if abs(offset) <= 1.0e-12:
        return point

    dx = point[0] - hinge[0]
    dy = point[1] - hinge[1]
    norm_xy = math.hypot(dx, dy)
    if norm_xy <= 1.0e-12:
        raise RuntimeError("cannot apply door handle radial offset: handle and hinge overlap in XY")
    return [
        point[0] + offset * dx / norm_xy,
        point[1] + offset * dy / norm_xy,
        point[2],
    ]


def _world_pose_to_base_pose(
    world_xyz: Sequence[float],
    world_quat_wxyz: Sequence[float],
    base_pose_xyyaw: Sequence[float],
) -> tuple[list[float], list[float]]:
    bx, by, byaw = [float(v) for v in base_pose_xyyaw]
    rel_world = [
        float(world_xyz[0]) - bx,
        float(world_xyz[1]) - by,
        float(world_xyz[2]),
    ]
    base_xyz = _rotate_z(rel_world, -byaw)
    base_quat = _quat_mul_wxyz(_quat_conj_wxyz(_yaw_quat_wxyz(byaw)), world_quat_wxyz)
    return base_xyz, base_quat


def _transform_points_by_base_pose(points_xyz, base_pose_xyyaw: Sequence[float]):
    import torch

    bx, by, byaw = [float(v) for v in base_pose_xyyaw]
    c = math.cos(byaw)
    s = math.sin(byaw)
    rot = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=points_xyz.device,
        dtype=points_xyz.dtype,
    )
    trans = torch.tensor([bx, by, 0.0], device=points_xyz.device, dtype=points_xyz.dtype)
    return points_xyz @ rot.transpose(0, 1) + trans


def _base_assist_fraction(args, alpha_rad: float) -> float:
    start = math.radians(float(args.door_open_base_start_deg))
    goal = max(start + 1.0e-6, math.radians(float(args.door_open_angle_deg)))
    return max(0.0, min(1.0, (float(alpha_rad) - start) / (goal - start)))


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _base_assist_center_spans(args, alpha_rad: float) -> tuple[list[float], list[float]]:
    frac = _base_assist_fraction(args, alpha_rad)
    center = [
        float(args.door_open_base_end_x) * frac,
        float(args.door_open_base_end_y) * frac,
        math.radians(float(args.door_open_base_end_yaw_deg)) * frac,
    ]
    spans = [
        abs(float(args.door_open_base_x_span)) * frac,
        abs(float(args.door_open_base_y_span)) * frac,
        math.radians(abs(float(args.door_open_base_yaw_span_deg))) * frac,
    ]
    return center, spans


def _base_pose_candidates_for_alpha(
    args,
    *,
    alpha_rad: float,
    previous_base_pose: Sequence[float],
) -> list[list[float]]:
    if not bool(args.door_open_base_assist):
        return [[0.0, 0.0, 0.0]]

    center, spans = _base_assist_center_spans(args, alpha_rad)
    center_x, center_y, center_yaw = center
    x_span, y_span, yaw_span = spans

    x_offsets = [0.0, -0.5 * x_span, 0.5 * x_span, -x_span, x_span]
    y_offsets = [0.0, -0.5 * y_span, 0.5 * y_span, -y_span, y_span]
    yaw_offsets = [0.0, -0.5 * yaw_span, 0.5 * yaw_span, -yaw_span, yaw_span]
    candidates: list[list[float]] = []
    seen: set[tuple[int, int, int]] = set()

    def _add(pose: Sequence[float]) -> None:
        key = tuple(int(round(float(v) * 1000.0)) for v in pose)
        if key in seen:
            return
        seen.add(key)
        candidates.append([float(pose[0]), float(pose[1]), float(pose[2])])

    _add(previous_base_pose)
    _add([center_x, center_y, center_yaw])
    for dx in x_offsets:
        for dy in y_offsets:
            for dyaw in yaw_offsets:
                _add([center_x + dx, center_y + dy, center_yaw + dyaw])

    max_candidates = int(getattr(args, "door_open_base_max_candidates", 32))
    if max_candidates > 0 and len(candidates) > max_candidates:
        anchors = candidates[:2]
        rest = candidates[2:]

        def _rank(pose: Sequence[float]) -> float:
            dx_prev = float(pose[0]) - float(previous_base_pose[0])
            dy_prev = float(pose[1]) - float(previous_base_pose[1])
            dyaw_prev = _wrap_pi(float(pose[2]) - float(previous_base_pose[2]))
            dx_center = float(pose[0]) - center_x
            dy_center = float(pose[1]) - center_y
            dyaw_center = _wrap_pi(float(pose[2]) - center_yaw)
            return (
                0.35 * math.sqrt(dx_prev * dx_prev + dy_prev * dy_prev + dyaw_prev * dyaw_prev)
                + math.sqrt(dx_center * dx_center + dy_center * dy_center + dyaw_center * dyaw_center)
            )

        rest.sort(key=_rank)
        candidates = anchors + rest[: max(0, max_candidates - len(anchors))]
    return candidates


def _base_pose_graph_candidates_for_alpha(args, *, alpha_rad: float) -> list[list[float]]:
    center, _ = _base_assist_center_spans(args, alpha_rad)
    return _base_pose_candidates_for_alpha(
        args,
        alpha_rad=alpha_rad,
        previous_base_pose=center,
    )


def _base_pose_goal_span_distance(
    args,
    *,
    base_pose: Sequence[float],
    goal_alpha_rad: float,
) -> float:
    if not bool(args.door_open_base_assist):
        return 0.0
    center, spans = _base_assist_center_spans(args, goal_alpha_rad)
    dx = max(0.0, abs(float(base_pose[0]) - center[0]) - spans[0])
    dy = max(0.0, abs(float(base_pose[1]) - center[1]) - spans[1])
    dyaw = max(0.0, abs(_wrap_pi(float(base_pose[2]) - center[2])) - spans[2])
    return math.sqrt(dx * dx + dy * dy + dyaw * dyaw)


def _interp_base_pose(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    t = max(0.0, min(1.0, float(t)))
    ax, ay, ayaw = [float(v) for v in a]
    bx, by, byaw = [float(v) for v in b]
    return [
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        _wrap_pi(ayaw + _wrap_pi(byaw - ayaw) * t),
    ]


def _densify_opening_publish_samples(
    publish_path: Sequence[Sequence[float]],
    base_poses: Sequence[Sequence[float]],
    *,
    max_base_step_m: float,
    max_base_step_yaw_rad: float,
) -> tuple[list[list[float]], list[list[float]]]:
    if len(publish_path) != len(base_poses):
        raise RuntimeError("opening base pose path length does not match arm path length")
    if len(publish_path) <= 1:
        return (
            [[float(v) for v in q] for q in publish_path],
            [[float(v) for v in pose] for pose in base_poses],
        )

    max_base_step_m = max(1.0e-6, float(max_base_step_m))
    max_base_step_yaw_rad = max(1.0e-6, abs(float(max_base_step_yaw_rad)))
    dense_path: list[list[float]] = [[float(v) for v in publish_path[0]]]
    dense_base: list[list[float]] = [[float(v) for v in base_poses[0]]]

    for idx in range(1, len(publish_path)):
        q0 = [float(v) for v in publish_path[idx - 1]]
        q1 = [float(v) for v in publish_path[idx]]
        b0 = [float(v) for v in base_poses[idx - 1]]
        b1 = [float(v) for v in base_poses[idx]]
        if len(q0) != len(q1):
            raise RuntimeError("opening publish path joint dimension changed")

        base_xy_step = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
        base_yaw_step = abs(_wrap_pi(b1[2] - b0[2]))
        n = max(
            1,
            int(math.ceil(base_xy_step / max_base_step_m)),
            int(math.ceil(base_yaw_step / max_base_step_yaw_rad)),
        )
        for sub_idx in range(1, n + 1):
            t = float(sub_idx) / float(n)
            dense_path.append([q0[j] + (q1[j] - q0[j]) * t for j in range(len(q0))])
            dense_base.append(_interp_base_pose(b0, b1, t))

    return dense_path, dense_base


def _append_post_open_base_forward_publish_samples(
    args: argparse.Namespace,
    publish_path: Sequence[Sequence[float]],
    door_alphas_rad: Sequence[float],
    base_poses: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    path_out = [[float(v) for v in q] for q in publish_path]
    alpha_out = [float(v) for v in door_alphas_rad]
    base_out = [[float(v) for v in pose] for pose in base_poses]

    if not bool(getattr(args, "post_open_base_enable", True)):
        return path_out, alpha_out, base_out
    if not path_out or not base_out:
        return path_out, alpha_out, base_out
    if len(path_out) != len(alpha_out) or len(path_out) != len(base_out):
        raise RuntimeError("post-open base publish path lengths do not match")

    forward_m = float(getattr(args, "post_open_base_forward_m", 0.0))
    if abs(forward_m) <= 1.0e-9:
        return path_out, alpha_out, base_out

    steps = max(1, int(getattr(args, "post_open_base_steps", 1)))
    start_base = base_out[-1]
    reference_x = base_out[0][0]
    goal_base = [reference_x + forward_m, start_base[1], start_base[2]]
    hold_q = path_out[-1]
    hold_alpha = alpha_out[-1]
    for step_idx in range(1, steps + 1):
        t = float(step_idx) / float(steps)
        path_out.append(list(hold_q))
        alpha_out.append(float(hold_alpha))
        base_out.append(_interp_base_pose(start_base, goal_base, t))

    print(
        "[ARM_DOOR][OPEN][POST_BASE] "
        f"appended base-only world-x move to opening_start_x{forward_m:+.3f}m "
        f"from x={start_base[0]:.3f} via reference x={reference_x:.3f} "
        f"to x={goal_base[0]:.3f} "
        f"(steps={steps})"
    )
    return path_out, alpha_out, base_out


def _smooth_opening_base_poses(
    base_poses: Sequence[Sequence[float]],
    *,
    window: int,
) -> list[list[float]]:
    poses = [[float(v) for v in pose] for pose in base_poses]
    if len(poses) <= 2:
        return poses
    window = int(window)
    if window <= 1:
        return poses
    if window % 2 == 0:
        window += 1

    yaw_unwrapped = [poses[0][2]]
    for pose in poses[1:]:
        yaw_unwrapped.append(yaw_unwrapped[-1] + _wrap_pi(float(pose[2]) - yaw_unwrapped[-1]))

    half = window // 2
    smooth: list[list[float]] = []
    for idx in range(len(poses)):
        lo = max(0, idx - half)
        hi = min(len(poses), idx + half + 1)
        count = float(hi - lo)
        smooth.append(
            [
                sum(poses[j][0] for j in range(lo, hi)) / count,
                sum(poses[j][1] for j in range(lo, hi)) / count,
                _wrap_pi(sum(yaw_unwrapped[j] for j in range(lo, hi)) / count),
            ]
        )

    smooth[0] = list(poses[0])
    smooth[-1] = list(poses[-1])
    return smooth


def _unwrap_base_yaws(base_poses: Sequence[Sequence[float]]) -> list[list[float]]:
    poses = [[float(v) for v in pose] for pose in base_poses]
    if not poses:
        return []
    out = [[poses[0][0], poses[0][1], poses[0][2]]]
    for pose in poses[1:]:
        prev_yaw = out[-1][2]
        out.append([pose[0], pose[1], prev_yaw + _wrap_pi(pose[2] - prev_yaw)])
    return out


def _dedupe_numeric_path(path: Sequence[Sequence[float]], *, eps: float = 1.0e-9) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in path]
    if not rows:
        return []
    out = [rows[0]]
    for row in rows[1:]:
        if len(row) != len(out[-1]):
            raise RuntimeError("numeric path dimension changed")
        dist = math.sqrt(sum((row[i] - out[-1][i]) ** 2 for i in range(len(row))))
        if dist > float(eps):
            out.append(row)
    if len(out) == 1 and len(rows) > 1:
        out.append(rows[-1])
    return out


def _linear_interpolate_numeric_path(
    path: Sequence[Sequence[float]],
    *,
    step: float,
) -> list[list[float]]:
    rows = _dedupe_numeric_path(path)
    if len(rows) <= 1:
        return rows
    step = max(1.0e-4, float(step))
    out = [list(rows[0])]
    for q0, q1 in zip(rows[:-1], rows[1:]):
        if len(q0) != len(q1):
            raise RuntimeError("linear interpolation path dimension changed")
        dist = math.sqrt(sum((q1[i] - q0[i]) ** 2 for i in range(len(q0))))
        n = max(1, int(math.ceil(dist / step)))
        for sub_idx in range(1, n + 1):
            t = float(sub_idx) / float(n)
            out.append([q0[i] + (q1[i] - q0[i]) * t for i in range(len(q0))])
    return out


def _retime_opening_arm_base_path(
    args,
    arm_path: Sequence[Sequence[float]],
    base_poses: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]], list[float]]:
    from capstone_pkg.planner.tbrrt.postprocess import topp_retime_path
    import torch

    arm_rows = [[float(v) for v in q] for q in arm_path]
    base_rows = _unwrap_base_yaws(base_poses)
    if len(arm_rows) != len(base_rows):
        raise RuntimeError("opening TOPP path length does not match base pose length")
    if len(arm_rows) < 2:
        raise RuntimeError("opening TOPP path must contain at least two waypoints")

    arm_dim = len(arm_rows[0])
    if arm_dim <= 0:
        raise RuntimeError("opening TOPP arm path has no joints")
    combined: list[list[float]] = []
    for idx, (q, base_pose) in enumerate(zip(arm_rows, base_rows)):
        if len(q) != arm_dim:
            raise RuntimeError(f"opening TOPP arm path dimension changed at idx={idx}")
        combined.append(list(q) + list(base_pose))

    spline_mode = str(getattr(args, "door_open_topp_spline_mode", "cubic")).strip().lower()
    if spline_mode == "cubic":
        from capstone_pkg.planner.arm_rrt_common.spline import spline_interpolate_path

        spline_step = max(1.0e-4, float(args.door_open_topp_spline_step))
        spline_path = spline_interpolate_path(combined, dt=spline_step)
    elif spline_mode == "linear":
        spline_path = _linear_interpolate_numeric_path(
            combined,
            step=max(1.0e-4, float(args.door_open_topp_spline_step)),
        )
    else:
        raise RuntimeError(f"unknown door_open_topp_spline_mode={spline_mode!r}")
    if len(spline_path) < 2:
        raise RuntimeError("opening TOPP spline output has fewer than two points")

    max_duration = float(getattr(args, "door_open_topp_max_duration_s", 0.0))
    duration_cap = max_duration if max_duration > 0.0 else None
    max_linear = max(1.0e-4, float(args.real_base_max_linear_mps))
    max_angular = max(1.0e-4, float(args.real_base_max_angular_rps))
    vmax = (
        [max(1.0e-4, float(args.door_open_topp_arm_max_velocity))] * arm_dim
        + [max_linear, max_linear, max_angular]
    )
    amax = (
        [max(1.0e-4, float(args.door_open_topp_arm_max_acceleration))] * arm_dim
        + [
            max(1.0e-4, float(args.door_open_topp_base_max_linear_accel)),
            max(1.0e-4, float(args.door_open_topp_base_max_linear_accel)),
            max(1.0e-4, float(args.door_open_topp_base_max_angular_accel)),
        ]
    )
    trajectory = topp_retime_path(
        torch.tensor(spline_path, dtype=torch.float32),
        max_velocity=vmax,
        max_acceleration=amax,
        output_dt=max(1.0e-3, float(args.door_open_dt)),
        max_duration_sec=duration_cap,
        safety_scale=max(1.0, float(args.door_open_topp_safety_scale)),
        max_iterations=max(1, int(args.door_open_topp_max_iterations)),
    )

    q_all = trajectory.q.detach().cpu().tolist()
    qdot_all = trajectory.qdot.detach().cpu().tolist()
    qddot_all = trajectory.qddot.detach().cpu().tolist()
    times = [float(v) for v in trajectory.t.detach().cpu().tolist()]

    retimed_arm = [[float(v) for v in row[:arm_dim]] for row in q_all]
    retimed_base = [
        [float(row[arm_dim]), float(row[arm_dim + 1]), _wrap_pi(float(row[arm_dim + 2]))]
        for row in q_all
    ]
    arm_velocities = [[float(v) for v in row[:arm_dim]] for row in qdot_all]
    arm_accelerations = [[float(v) for v in row[:arm_dim]] for row in qddot_all]
    if arm_velocities:
        arm_velocities[0] = [0.0] * arm_dim
        arm_velocities[-1] = [0.0] * arm_dim
    if arm_accelerations:
        arm_accelerations[0] = [0.0] * arm_dim
        arm_accelerations[-1] = [0.0] * arm_dim

    print(
        "[ARM_DOOR][OPEN][REAL][TOPP] "
        f"raw={len(arm_rows)} mode={spline_mode} interp={len(spline_path)} output={len(retimed_arm)} "
        f"duration={trajectory.duration_sec:.3f}s dt={float(args.door_open_dt):.3f}s "
        f"max_v={trajectory.max_abs_velocity:.3f} max_a={trajectory.max_abs_acceleration:.3f} "
        f"time_scale={trajectory.time_scale:.3f}"
    )
    return retimed_arm, retimed_base, arm_velocities, arm_accelerations, times


def _clamp_abs(value: float, limit: float) -> float:
    limit = abs(float(limit))
    if limit <= 0.0:
        return float(value)
    return max(-limit, min(limit, float(value)))


def _base_pose_delta_to_body_twist(
    a: Sequence[float],
    b: Sequence[float],
    *,
    dt: float,
    max_linear_mps: float,
    max_angular_rps: float,
) -> tuple[float, float, float, bool]:
    dt = max(1.0e-6, float(dt))
    ax, ay, ayaw = [float(v) for v in a]
    bx, by, byaw = [float(v) for v in b]
    dx_world = bx - ax
    dy_world = by - ay
    c = math.cos(-ayaw)
    s = math.sin(-ayaw)
    vx = (c * dx_world - s * dy_world) / dt
    vy = (s * dx_world + c * dy_world) / dt
    wz = _wrap_pi(byaw - ayaw) / dt

    vx_clamped = _clamp_abs(vx, max_linear_mps)
    vy_clamped = _clamp_abs(vy, max_linear_mps)
    wz_clamped = _clamp_abs(wz, max_angular_rps)
    clamped = (
        abs(vx_clamped - vx) > 1.0e-9
        or abs(vy_clamped - vy) > 1.0e-9
        or abs(wz_clamped - wz) > 1.0e-9
    )
    return vx_clamped, vy_clamped, wz_clamped, clamped


def _publish_base_stop_with_node(node: Node, pub, args, *, cmd_period_s: float) -> None:
    stop_msg = Twist()
    stop_end = time.monotonic() + max(0.0, float(args.real_base_stop_duration_s))
    while rclpy.ok() and time.monotonic() < stop_end:
        pub.publish(stop_msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(min(0.05, max(0.001, float(cmd_period_s))))


def _publish_post_one_step_base_reset(args) -> None:
    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("arm_door_post_one_step_base_reset")
    pub = node.create_publisher(Twist, str(args.real_base_cmd_vel_topic), 10)
    wait_deadline = (
        time.monotonic() + max(0.0, float(args.publish_wait_subscriber_s))
    )
    while (
        rclpy.ok()
        and pub.get_subscription_count() == 0
        and time.monotonic() < wait_deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.05)

    cmd_period_s = 1.0 / max(1.0, float(args.real_base_cmd_rate_hz))
    try:
        if bool(getattr(args, "post_one_step_align_yaw", True)):
            _publish_post_one_step_yaw_align_with_node(
                args,
                node=node,
                pub=pub,
                cmd_period_s=cmd_period_s,
            )
        _publish_post_one_step_reverse_with_node(
            args,
            node=node,
            pub=pub,
            cmd_period_s=cmd_period_s,
        )
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()


def _publish_post_one_step_yaw_align_with_node(
    args,
    *,
    node: Node,
    pub,
    cmd_period_s: float,
) -> None:
    pose_box: dict[str, Any] = {"pose": None}

    def _on_pose(msg: Pose2D) -> None:
        pose_box["pose"] = msg

    pose_topic = str(getattr(args, "post_one_step_base_pose_topic", "/base_pose"))
    sub = node.create_subscription(Pose2D, pose_topic, _on_pose, 10)
    wait_deadline = time.monotonic() + max(
        0.0,
        float(getattr(args, "post_one_step_base_pose_wait_s", 1.0)),
    )
    while (
        rclpy.ok()
        and pose_box["pose"] is None
        and time.monotonic() < wait_deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.05)

    if pose_box["pose"] is None:
        node.get_logger().warning(
            "Post one_step yaw alignment skipped: "
            f"no Pose2D feedback received on {pose_topic}"
        )
        node.destroy_subscription(sub)
        return

    target_yaw = math.radians(
        float(getattr(args, "post_one_step_target_yaw_deg", 0.0))
    )
    tolerance = math.radians(
        max(0.0, float(getattr(args, "post_one_step_yaw_tolerance_deg", 2.0)))
    )
    timeout_s = max(0.0, float(getattr(args, "post_one_step_yaw_timeout_s", 8.0)))
    kp = max(0.0, float(getattr(args, "post_one_step_yaw_kp", 1.5)))
    min_speed = max(
        0.0,
        abs(float(getattr(args, "post_one_step_yaw_min_speed_rps", 0.05))),
    )
    max_angular = max(1.0e-3, abs(float(args.real_base_max_angular_rps)))
    align_deadline = time.monotonic() + timeout_s
    last_err = 0.0

    node.get_logger().info(
        "Post one_step yaw alignment: "
        f"target={math.degrees(target_yaw):.1f}deg "
        f"tolerance={math.degrees(tolerance):.1f}deg "
        f"feedback={pose_topic}"
    )

    while rclpy.ok() and time.monotonic() < align_deadline:
        rclpy.spin_once(node, timeout_sec=0.0)
        pose = pose_box["pose"]
        if pose is None:
            break
        err = _wrap_pi(target_yaw - float(pose.theta))
        last_err = err
        if abs(err) <= tolerance:
            node.get_logger().info(
                "Post one_step yaw alignment complete: "
                f"error={math.degrees(err):.2f}deg"
            )
            break
        cmd = Twist()
        wz = max(-max_angular, min(max_angular, kp * err))
        if min_speed > 0.0 and abs(wz) < min_speed:
            wz = math.copysign(min(min_speed, max_angular), err)
        cmd.angular.z = float(wz)
        pub.publish(cmd)
        time.sleep(min(max(0.001, float(cmd_period_s)), 0.05))
    else:
        node.get_logger().warning(
            "Post one_step yaw alignment timed out: "
            f"last_error={math.degrees(last_err):.2f}deg"
        )

    _publish_base_stop_with_node(node, pub, args, cmd_period_s=cmd_period_s)
    node.destroy_subscription(sub)


def _publish_post_one_step_reverse_with_node(
    args,
    *,
    node: Node,
    pub,
    cmd_period_s: float,
) -> None:
    distance_m = max(0.0, float(args.post_one_step_reverse_distance_m))
    speed_mps = max(1.0e-3, abs(float(args.post_one_step_reverse_speed_mps)))
    max_linear = max(1.0e-3, abs(float(args.real_base_max_linear_mps)))
    speed_mps = min(speed_mps, max_linear)
    duration_s = distance_m / speed_mps if distance_m > 0.0 else 0.0
    node.get_logger().info(
        "Post one_step base reverse: "
        f"{distance_m:.3f}m at {speed_mps:.3f}m/s on {args.real_base_cmd_vel_topic}"
    )

    reverse_msg = Twist()
    reverse_msg.linear.x = -speed_mps
    end_time = time.monotonic() + duration_s
    while rclpy.ok() and time.monotonic() < end_time:
        pub.publish(reverse_msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        remaining = end_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(float(cmd_period_s), remaining))

    _publish_base_stop_with_node(node, pub, args, cmd_period_s=cmd_period_s)


def _base_path_distance(base_poses: Sequence[Sequence[float]]) -> tuple[float, float]:
    poses = [[float(v) for v in pose] for pose in base_poses]
    if len(poses) < 2:
        return 0.0, 0.0
    xy = 0.0
    yaw = 0.0
    for prev, cur in zip(poses[:-1], poses[1:]):
        xy += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        yaw += abs(_wrap_pi(cur[2] - prev[2]))
    return xy, yaw


def _print_base_path_summary(label: str, base_poses: Sequence[Sequence[float]]) -> None:
    poses = [[float(v) for v in pose] for pose in base_poses]
    if not poses:
        print(f"{label} base_path=empty")
        return
    start = poses[0]
    end = poses[-1]
    xy_len, yaw_len = _base_path_distance(poses)
    print(
        f"{label} waypoints={len(poses)} "
        f"start={['%.3f' % v for v in start]} "
        f"end={['%.3f' % v for v in end]} "
        f"delta_xy={math.hypot(end[0] - start[0], end[1] - start[1]):.3f}m "
        f"path_xy={xy_len:.3f}m "
        f"delta_yaw={math.degrees(_wrap_pi(end[2] - start[2])):.1f}deg "
        f"path_yaw={math.degrees(yaw_len):.1f}deg"
    )


def _joint_delta_abs(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def _read_joint_sample_with_node(
    node,
    joint_names: Sequence[str],
    *,
    topic: str,
    wait_s: float,
) -> list[float] | None:
    import rclpy
    from sensor_msgs.msg import JointState

    names = [str(name) for name in joint_names]
    latest_by_name: dict[str, float] = {}

    def _cb(msg: JointState) -> None:
        name_to_idx = {str(name): idx for idx, name in enumerate(msg.name)}
        for name in names:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(msg.position):
                continue
            latest_by_name[name] = float(msg.position[idx])

    sub = node.create_subscription(JointState, str(topic), _cb, 10)
    deadline = time.monotonic() + max(0.0, float(wait_s))
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if all(name in latest_by_name for name in names):
                return [latest_by_name[name] for name in names]
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_subscription(sub)
    return None


def _wait_for_joint_motion_with_node(
    node,
    joint_names: Sequence[str],
    baseline: Sequence[float],
    *,
    topic: str,
    wait_s: float,
    threshold_rad: float,
) -> tuple[bool, float]:
    import rclpy
    from sensor_msgs.msg import JointState

    names = [str(name) for name in joint_names]
    baseline_values = [float(v) for v in baseline]
    latest_by_name: dict[str, float] = {}
    max_delta = 0.0

    def _cb(msg: JointState) -> None:
        nonlocal max_delta
        name_to_idx = {str(name): idx for idx, name in enumerate(msg.name)}
        for name in names:
            idx = name_to_idx.get(name)
            if idx is None or idx >= len(msg.position):
                continue
            latest_by_name[name] = float(msg.position[idx])
        if all(name in latest_by_name for name in names):
            current = [latest_by_name[name] for name in names]
            max_delta = max(
                max_delta,
                max(_joint_delta_abs(current[i], baseline_values[i]) for i in range(len(names))),
            )

    sub = node.create_subscription(JointState, str(topic), _cb, 10)
    deadline = time.monotonic() + max(0.0, float(wait_s))
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if max_delta >= float(threshold_rad):
                return True, float(max_delta)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_subscription(sub)
    return max_delta >= float(threshold_rad), float(max_delta)


def _build_timed_joint_trajectory(
    *,
    joint_names: Sequence[str],
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    accelerations: Sequence[Sequence[float]],
    times: Sequence[float],
):
    from capstone_pkg.planner.arm_rrt_common.path_publisher import _duration_from_seconds
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    if not positions:
        raise RuntimeError("timed JointTrajectory path is empty")
    if len(positions) != len(times):
        raise RuntimeError("timed JointTrajectory positions/time length mismatch")
    if velocities and len(velocities) != len(positions):
        raise RuntimeError("timed JointTrajectory velocity length mismatch")
    if accelerations and len(accelerations) != len(positions):
        raise RuntimeError("timed JointTrajectory acceleration length mismatch")

    names = [str(name) for name in joint_names]
    msg = JointTrajectory()
    msg.joint_names = names
    last_t = 0.0
    for idx, q in enumerate(positions):
        if len(q) != len(names):
            raise RuntimeError(f"timed JointTrajectory point {idx} dimension mismatch")
        point_t = max(last_t, float(times[idx]))
        last_t = point_t
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        if velocities:
            p.velocities = [float(v) for v in velocities[idx]]
        if accelerations:
            p.accelerations = [float(v) for v in accelerations[idx]]
        p.time_from_start = _duration_from_seconds(point_t)
        msg.points.append(p)
    return msg


def _resolve_opening_world_yml(raw_value: str, resolved_world_yml: str | None) -> str | None:
    raw = str(raw_value).strip()
    if raw.lower() in ("", "none", "null", "false", "off", "0"):
        return None
    if raw.lower() == "same":
        return resolved_world_yml
    return raw


def _load_world_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(str(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"world yaml must contain a mapping: {path}")
    return raw


def _write_temp_world_yaml(world: Mapping[str, Any]) -> str:
    import yaml

    tmp = tempfile.NamedTemporaryFile(
        prefix="arm_door_opening_world_",
        suffix=".yaml",
        dir="/tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    with tmp:
        yaml.safe_dump(dict(world), tmp, allow_unicode=True, sort_keys=False)
    return str(tmp.name)


def _command_qos(
    *,
    reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE,
    durability: DurabilityPolicy = DurabilityPolicy.TRANSIENT_LOCAL,
    depth: int = 1,
) -> QoSProfile:
    return QoSProfile(
        reliability=reliability,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
    )


def _make_shifted_door_world_yaml(
    *,
    source_world_yml: str,
    detected_handle_center_xyz: Sequence[float],
) -> tuple[str, list[float], list[float]]:
    world = _load_world_yaml(str(source_world_yml))
    cuboids = world.get("cuboid", {})
    if not isinstance(cuboids, dict):
        raise RuntimeError("door collision yaml cuboid entry must be a mapping")

    handle_center = [float(v) for v in detected_handle_center_xyz]
    model_handle = [float(v) for v in HANDLE_FRAME_XYZ]
    delta = [handle_center[i] - model_handle[i] for i in range(3)]
    shifted = copy.deepcopy(dict(world))
    shifted_cuboids = shifted.get("cuboid", {})
    if not isinstance(shifted_cuboids, dict):
        raise RuntimeError("door collision yaml cuboid entry must be a mapping")
    for name, item in shifted_cuboids.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"cuboid '{name}' must be a mapping")
        pose = item.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) != 7:
            raise RuntimeError(f"cuboid '{name}' pose must be length 7")
        item["pose"] = [
            float(pose[0]) + delta[0],
            float(pose[1]) + delta[1],
            float(pose[2]) + delta[2],
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
            float(pose[6]),
        ]

    shifted["cuboid"] = shifted_cuboids
    shifted_world_yml = _write_temp_world_yaml(shifted)
    shifted_hinge_xyz = [
        float(DOOR_HINGE_XYZ[i]) + delta[i]
        for i in range(3)
    ]
    return shifted_world_yml, shifted_hinge_xyz, delta


def _quat_wxyz_to_rotmat(q_wxyz: Sequence[float]) -> list[list[float]]:
    w, x, y, z = [float(v) for v in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1.0e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _rotate_door_cuboid_pose(
    pose_wxyz: Sequence[float],
    *,
    alpha_rad: float,
    door_hinge_xyz: Sequence[float],
) -> list[float]:
    if len(pose_wxyz) != 7:
        raise RuntimeError(f"cuboid pose must be [x,y,z,w,x,y,z], got {pose_wxyz}")
    hinge = [float(v) for v in door_hinge_xyz]
    xyz = [float(v) for v in pose_wxyz[:3]]
    quat = [float(v) for v in pose_wxyz[3:7]]
    rel = [xyz[i] - hinge[i] for i in range(3)]
    rel_rot = _rotate_z(rel, alpha_rad)
    xyz_rot = [hinge[i] + rel_rot[i] for i in range(3)]
    quat_rot = _quat_mul_wxyz(_yaw_quat_wxyz(alpha_rad), quat)
    return xyz_rot + quat_rot


def _make_opening_dynamic_world(
    *,
    source_world: Mapping[str, Any],
    alpha_rad: float,
    door_hinge_xyz: Sequence[float],
    ignore_names: Sequence[str],
) -> dict[str, Any]:
    import copy

    world = copy.deepcopy(dict(source_world))
    cuboids = world.get("cuboid", {})
    if not isinstance(cuboids, dict):
        raise RuntimeError("world yaml cuboid entry must be a mapping")

    ignore = {str(name) for name in ignore_names}
    for name in DOOR_MOVING_COLLISION_NAMES:
        if name not in cuboids:
            continue
        if name in ignore:
            cuboids.pop(name, None)
            continue
        item = cuboids.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"cuboid '{name}' must be a mapping")
        if "pose" not in item:
            raise RuntimeError(f"cuboid '{name}' has no pose")
        item["pose"] = _rotate_door_cuboid_pose(
            item["pose"],
            alpha_rad=alpha_rad,
            door_hinge_xyz=door_hinge_xyz,
        )

    world["cuboid"] = cuboids
    return world


def _make_opening_dynamic_world_yaml(
    *,
    source_world_yml: str,
    alpha_rad: float,
    door_hinge_xyz: Sequence[float],
    ignore_names: Sequence[str],
) -> str:
    world = _make_opening_dynamic_world(
        source_world=_load_world_yaml(source_world_yml),
        alpha_rad=alpha_rad,
        door_hinge_xyz=door_hinge_xyz,
        ignore_names=ignore_names,
    )
    return _write_temp_world_yaml(world)


def _opening_collision_alpha_samples(args, alpha_rad: float) -> list[float]:
    samples = [float(alpha_rad)]
    lag = max(0.0, math.radians(float(getattr(args, "door_open_collision_lag_deg", 0.0))))
    if lag > 1.0e-9:
        samples.append(max(0.0, float(alpha_rad) - lag))

    out: list[float] = []
    seen: set[int] = set()
    for sample in samples:
        key = int(round(float(sample) * 1.0e6))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(sample))
    return out


def _resolve_dynamic_opening_world_yml(args, resolved_world_yml: str | None) -> str | None:
    return _resolve_opening_world_yml(
        str(args.opening_dynamic_world_yml),
        resolved_world_yml,
    )


def _active_path_from_cspace(
    cspace_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
    cspace_path: Sequence[Sequence[float]],
) -> tuple[list[str], list[list[float]]]:
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    joint_names = [name for name in active_joint_names if name in name_to_idx]
    if not joint_names:
        raise RuntimeError("No active joints found in opening cspace path")
    path = [
        [float(q[name_to_idx[name]]) for name in joint_names]
        for q in cspace_path
    ]
    return joint_names, path


def _extract_active_from_cspace(
    q_cspace: Sequence[float],
    cspace_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> list[float]:
    name_to_idx = {name: idx for idx, name in enumerate(cspace_joint_names)}
    return [float(q_cspace[name_to_idx[name]]) for name in active_joint_names]


def _quat_angle_error_rad(a_wxyz: Sequence[float], b_wxyz: Sequence[float]) -> float:
    dot = sum(float(a_wxyz[i]) * float(b_wxyz[i]) for i in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _pack_cuboids_for_sphere_check(
    cuboids: Mapping[str, Any],
    *,
    device,
    dtype=None,
) -> tuple[list[str], Any, Any, Any] | None:
    import torch

    names: list[str] = []
    centers: list[list[float]] = []
    halves: list[list[float]] = []
    rotations: list[list[list[float]]] = []
    for name, cuboid in cuboids.items():
        if not isinstance(cuboid, Mapping):
            continue
        dims = cuboid.get("dims")
        pose = cuboid.get("pose")
        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            raise RuntimeError(f"cuboid dims must be length 3: {dims}")
        if not isinstance(pose, (list, tuple)) or len(pose) != 7:
            raise RuntimeError(f"cuboid pose must be length 7: {pose}")
        names.append(str(name))
        centers.append([float(v) for v in pose[:3]])
        halves.append([0.5 * float(v) for v in dims])
        rotations.append(_quat_wxyz_to_rotmat(pose[3:7]))

    if not names:
        return None
    tensor_dtype = dtype if dtype is not None else torch.float32
    return (
        names,
        torch.tensor(centers, device=device, dtype=tensor_dtype),
        torch.tensor(halves, device=device, dtype=tensor_dtype),
        torch.tensor(rotations, device=device, dtype=tensor_dtype),
    )


def _cuboid_sphere_pack_max_violation(
    spheres,
    cuboid_pack: tuple[list[str], Any, Any, Any] | None,
    *,
    margin_m: float,
) -> tuple[float, str]:
    import torch

    if cuboid_pack is None:
        return 0.0, ""

    names, centers, halves, rotations = cuboid_pack
    if len(names) <= 0:
        return 0.0, ""

    sphere_xyz = spheres[:, :3]
    sphere_r = spheres[:, 3]
    rel = sphere_xyz.unsqueeze(0) - centers.unsqueeze(1)
    local = torch.matmul(rel, rotations)
    q = torch.abs(local) - halves.unsqueeze(1)
    outside = torch.clamp(q, min=0.0)
    outside_dist = torch.linalg.norm(outside, dim=-1)
    inside_dist = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    signed_dist = outside_dist + inside_dist
    violation = sphere_r.unsqueeze(0) + float(margin_m) - signed_dist
    per_cuboid = torch.max(violation, dim=1).values
    max_violation, max_idx = torch.max(per_cuboid, dim=0)
    return (
        float(max_violation.detach().cpu().item()),
        names[int(max_idx.detach().cpu().item())],
    )


def _opening_candidate_collision(
    *,
    checker,
    q_cspace: Sequence[float],
    base_pose: Sequence[float],
    cuboids: Mapping[str, Any],
    cuboid_sphere_pack=None,
    cuboid_sphere_packs: Sequence[Any] | None = None,
    margin_m: float,
    device,
) -> tuple[float, float, str]:
    import torch

    q_t = torch.tensor([list(q_cspace)], device=device, dtype=torch.float32)
    with torch.no_grad():
        q_active = checker._build_q_active_from_cspace(q_t)
        state = checker.robot_world_self_only.get_kinematics(q_active)
        x_sph = checker._extract_spheres(checker.robot_world_self_only, state, q_active)
        d_self = checker.robot_world_self_only.get_self_collision_distance(x_sph)

    self_pen = float(d_self.view(d_self.shape[0], -1).max(dim=1).values[0].detach().cpu().item())
    if x_sph.dim() == 4:
        spheres = x_sph[0, 0, :, :].clone()
    elif x_sph.dim() == 3:
        spheres = x_sph[0, :, :].clone()
    else:
        raise RuntimeError(f"Unexpected sphere tensor shape: {tuple(x_sph.shape)}")

    spheres[:, :3] = _transform_points_by_base_pose(spheres[:, :3], base_pose)
    if cuboid_sphere_packs is not None:
        packs = list(cuboid_sphere_packs)
    elif cuboid_sphere_pack is not None:
        packs = [cuboid_sphere_pack]
    else:
        cuboid_sphere_pack = _pack_cuboids_for_sphere_check(
            cuboids,
            device=device,
            dtype=spheres.dtype,
        )
        packs = [cuboid_sphere_pack]

    world_violation = -float("inf")
    obstacle = ""
    for pack in packs:
        violation, pack_obstacle = _cuboid_sphere_pack_max_violation(
            spheres,
            pack,
            margin_m=margin_m,
        )
        if violation > world_violation:
            world_violation = float(violation)
            obstacle = str(pack_obstacle)
    if world_violation == -float("inf"):
        world_violation = 0.0
    return float(world_violation), float(self_pen), obstacle


def _base_door_clearance_violation(
    *,
    base_pose: Sequence[float],
    cuboids: Mapping[str, Any],
    clearance_m: float,
    base_radius_m: float,
    base_height_m: float,
    door_names: Sequence[str],
) -> tuple[float, str]:
    clearance_m = max(0.0, float(clearance_m))
    if clearance_m <= 1.0e-9:
        return 0.0, ""

    bx = float(base_pose[0])
    by = float(base_pose[1])
    base_radius = max(0.0, float(base_radius_m))
    base_height = max(0.0, float(base_height_m))
    door_name_set = {str(name) for name in door_names}
    worst_violation = -float("inf")
    worst_name = ""

    for name, cuboid in cuboids.items():
        if str(name) not in door_name_set or not isinstance(cuboid, Mapping):
            continue
        dims = cuboid.get("dims")
        pose = cuboid.get("pose")
        if not isinstance(dims, Sequence) or not isinstance(pose, Sequence):
            continue
        if len(dims) != 3 or len(pose) != 7:
            continue

        cx, cy, cz = [float(v) for v in pose[:3]]
        half_x = 0.5 * float(dims[0])
        half_y = 0.5 * float(dims[1])
        half_z = 0.5 * float(dims[2])
        if cz - half_z > base_height or cz + half_z < 0.0:
            continue

        yaw = _yaw_from_quat_wxyz(pose[3:7])
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        dx = bx - cx
        dy = by - cy
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        qx = abs(local_x) - half_x
        qy = abs(local_y) - half_y
        outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        signed_dist = outside + inside
        actual_clearance = signed_dist - base_radius
        violation = clearance_m - actual_clearance
        if violation > worst_violation:
            worst_violation = float(violation)
            worst_name = str(name)

    if worst_violation == -float("inf"):
        return 0.0, ""
    return max(0.0, float(worst_violation)), worst_name


def _build_door_opening_cspace_path(
    args,
    *,
    arm: str,
    q_start_cspace: Sequence[float],
    grasp_ee_xyz: Sequence[float],
    grasp_ee_quat_xyzw: Sequence[float],
    closed_handle_xyz: Sequence[float],
    closed_handle_quat_xyzw: Sequence[float],
    door_hinge_xyz: Sequence[float],
    resolved_world_yml: str | None,
) -> tuple[list[str], list[str], list[list[float]], list[float], list[tuple[list[float], list[float]]], list[list[float]]]:
    from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
    from capstone_pkg.planner.arm_rrt_common.single_arm_motion import _build_ik_seed_batch
    from capstone_pkg.utils.joint_limit import load_joint_limits_torch
    import numpy as np
    import torch

    if arm != "right":
        raise RuntimeError("door opening is currently implemented for the right arm only")

    steps = max(2, int(args.door_open_steps))
    goal_alpha = math.radians(float(args.door_open_angle_deg))
    grasp_ee_quat_wxyz = _xyzw_to_wxyz(grasp_ee_quat_xyzw)
    closed_handle_quat_wxyz = _xyzw_to_wxyz(closed_handle_quat_xyzw)
    inv_grasp_ee_xyz, inv_grasp_ee_quat = _pose_inv(
        grasp_ee_xyz,
        grasp_ee_quat_wxyz,
    )
    ee_handle_xyz, ee_handle_quat = _pose_mul(
        inv_grasp_ee_xyz,
        inv_grasp_ee_quat,
        closed_handle_xyz,
        closed_handle_quat_wxyz,
    )
    inv_ee_handle_xyz, inv_ee_handle_quat = _pose_inv(
        ee_handle_xyz,
        ee_handle_quat,
    )
    opening_world_yml = _resolve_opening_world_yml(
        str(args.door_open_world_yml),
        resolved_world_yml,
    )
    ik = get_single_arm_ik(
        str(args.robot_yml),
        arm=arm,
        cpu=bool(args.cpu),
        world_yml=opening_world_yml,
        use_cuda_graph=False,
    )
    jl = load_joint_limits_torch(
        str(args.joint_limit_yml),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    joint_lower = jl.lower.detach().cpu().numpy()
    joint_upper = jl.upper.detach().cpu().numpy()
    ik_batch = max(1, int(args.door_open_ik_batch))
    ik_noise = max(0.0, float(args.door_open_ik_seed_noise_std))
    max_pos_err = float(args.door_open_ik_max_pos_m)
    max_rot_err = math.radians(float(args.door_open_ik_max_rot_deg))
    orientation_constraint = str(
        getattr(args, "door_open_orientation_constraint", "door_relative")
    ).strip().lower()
    if orientation_constraint not in ("door_relative", "rigid_grasp"):
        raise RuntimeError(f"unknown door_open_orientation_constraint={orientation_constraint!r}")
    orientation_lag = max(
        0.0,
        math.radians(float(getattr(args, "door_open_orientation_lag_deg", 0.0))),
    )
    ee_radial_offset = float(getattr(args, "door_open_handle_radial_offset_m", 0.0))
    print(
        f"[ARM_DOOR][OPEN][ORI] constraint={orientation_constraint} "
        f"lag_deg={math.degrees(orientation_lag):.1f}"
    )
    print(f"[ARM_DOOR][OPEN][EE] radial_offset_m={ee_radial_offset:.3f}")

    collision_checker = None
    collision_source_world: dict[str, Any] | None = None
    collision_device = None
    collision_ignore_names = [str(name) for name in getattr(args, "opening_dynamic_ignore", [])]
    collision_margin_m = float(args.opening_dynamic_collision_margin_m)
    if bool(args.door_open_collision_aware_base_selection):
        source_world_yml = _resolve_dynamic_opening_world_yml(args, resolved_world_yml)
        if source_world_yml is not None:
            from capstone_pkg.collision_check.collision import get_self_collision_checker

            collision_source_world = _load_world_yaml(str(source_world_yml))
            collision_checker = get_self_collision_checker(
                str(args.robot_yml),
                cpu=bool(args.cpu),
                world_yml=None,
            )
            collision_device = collision_checker.tensor_args.device
            print(
                "[ARM_DOOR][OPEN][COLLISION] candidate-aware selection enabled "
                f"(world_yml={source_world_yml}, margin={collision_margin_m:.4f}m, "
                f"lag={float(getattr(args, 'door_open_collision_lag_deg', 0.0)):.1f}deg)"
            )

    planner_mode = str(getattr(args, "door_open_base_planner", "beam")).lower().strip()
    if planner_mode not in ("astar", "beam", "graph"):
        raise RuntimeError(f"unknown door_open_base_planner={planner_mode!r}")

    if planner_mode == "graph" and not bool(args.door_open_base_assist):
        raise RuntimeError("--door_open_base_planner graph requires --door_open_base_assist")

    graph_base_poses: list[list[float]] | None = None
    graph_phases: list[int] | None = None
    graph_repair_candidates: list[list[list[float]] | None] | None = None
    if planner_mode == "graph":
        from capstone_pkg.planner.ARM_DOOR.base_door_graph import (
            BaseDoorGraphConfig,
            PHASE_FRONT,
            PHASE_OPENING,
            plan_base_door_graph,
        )

        graph_source_world = collision_source_world
        if graph_source_world is None:
            source_world_yml = _resolve_dynamic_opening_world_yml(args, resolved_world_yml)
            if source_world_yml is not None:
                graph_source_world = _load_world_yaml(str(source_world_yml))
        front_goal_x = (
            float(args.door_open_base_end_x)
            if getattr(args, "door_open_front_goal_x", None) is None
            else float(args.door_open_front_goal_x)
        )
        front_goal_y = (
            float(args.door_open_base_end_y)
            if getattr(args, "door_open_front_goal_y", None) is None
            else float(args.door_open_front_goal_y)
        )
        front_goal_yaw = (
            math.radians(float(args.door_open_base_end_yaw_deg))
            if getattr(args, "door_open_front_goal_yaw_deg", None) is None
            else math.radians(float(args.door_open_front_goal_yaw_deg))
        )
        front_goal_x_tol = (
            float(args.door_open_front_goal_tol_m)
            if getattr(args, "door_open_front_goal_x_tol_m", None) is None
            else float(args.door_open_front_goal_x_tol_m)
        )
        front_goal_y_tol = (
            float(args.door_open_front_goal_tol_m)
            if getattr(args, "door_open_front_goal_y_tol_m", None) is None
            else float(args.door_open_front_goal_y_tol_m)
        )
        graph_front_goal_x = front_goal_x
        graph_front_goal_y = front_goal_y
        graph_front_goal_yaw = front_goal_yaw
        graph_front_goal_x_tol = front_goal_x_tol
        graph_front_goal_y_tol = front_goal_y_tol
        if bool(getattr(args, "door_open_front_detour", True)):
            graph_front_goal_x = float(args.door_open_base_end_x)
            graph_front_goal_y = float(args.door_open_base_end_y)
            graph_front_goal_yaw = math.radians(float(args.door_open_base_end_yaw_deg))
            graph_front_goal_x_tol = max(front_goal_x_tol, float(args.door_open_graph_xy_step_m))
            graph_front_goal_y_tol = max(front_goal_y_tol, float(args.door_open_graph_xy_step_m))
        graph_cfg = BaseDoorGraphConfig(
            steps=steps,
            goal_alpha_rad=goal_alpha,
            lambda_step_rad=max(1.0e-4, math.radians(float(args.door_open_graph_lambda_step_deg))),
            base_motion_start_alpha_rad=math.radians(float(args.door_open_base_start_deg)),
            closed_handle_xyz=tuple(float(v) for v in closed_handle_xyz),
            door_hinge_xyz=tuple(float(v) for v in door_hinge_xyz),
            start_base_xyyaw=(0.0, 0.0, 0.0),
            opening_end_xyyaw=(
                float(args.door_open_base_end_x),
                float(args.door_open_base_end_y),
                math.radians(float(args.door_open_base_end_yaw_deg)),
            ),
            opening_span_xyyaw=(
                abs(float(args.door_open_base_x_span)),
                abs(float(args.door_open_base_y_span)),
                math.radians(abs(float(args.door_open_base_yaw_span_deg))),
            ),
            front_goal_xyyaw=(graph_front_goal_x, graph_front_goal_y, graph_front_goal_yaw),
            front_goal_tol_m=float(args.door_open_front_goal_tol_m),
            front_goal_x_tol_m=max(0.0, graph_front_goal_x_tol),
            front_goal_y_tol_m=max(0.0, graph_front_goal_y_tol),
            front_goal_tol_yaw_rad=math.radians(float(args.door_open_front_goal_tol_deg)),
            xy_step_m=max(1.0e-3, float(args.door_open_graph_xy_step_m)),
            yaw_step_rad=max(1.0e-3, math.radians(float(args.door_open_graph_yaw_step_deg))),
            max_expansions=max(1, int(args.door_open_graph_max_expansions)),
            layer_keep=max(1, int(args.door_open_astar_layer_keep)),
            max_candidates_per_layer=int(args.door_open_base_max_candidates),
            bound_margin_m=max(0.0, float(args.door_open_graph_bound_margin_m)),
            bound_margin_yaw_rad=max(
                0.0,
                math.radians(float(args.door_open_graph_bound_margin_yaw_deg)),
            ),
            base_radius_m=max(0.0, float(args.door_open_graph_base_radius_m)),
            base_height_m=max(0.0, float(args.door_open_graph_base_height_m)),
            collision_margin_m=max(0.0, float(args.opening_dynamic_collision_margin_m)),
            reach_shoulder_xyz=tuple(float(v) for v in args.door_open_graph_reach_shoulder_xyz),
            reach_min_m=max(0.0, float(args.door_open_graph_reach_min_m)),
            reach_max_m=max(0.0, float(args.door_open_graph_reach_max_m)),
            reach_z_min_m=float(args.door_open_graph_reach_z_min_m),
            reach_z_max_m=float(args.door_open_graph_reach_z_max_m),
            reach_nominal_m=max(0.0, float(args.door_open_graph_reach_nominal_m)),
            world=graph_source_world,
            moving_collision_names=tuple(DOOR_MOVING_COLLISION_NAMES),
            ignore_collision_names=tuple(collision_ignore_names),
        )
        graph_plan = plan_base_door_graph(graph_cfg)
        if not graph_plan.base_poses or len(graph_plan.base_poses) != len(graph_plan.door_alphas_rad):
            raise RuntimeError("base-door graph returned an empty or inconsistent path")
        graph_base_poses = [[float(v) for v in pose] for pose in graph_plan.base_poses]
        graph_phases = [int(v) for v in graph_plan.phases]
        planned_alphas = [float(v) for v in graph_plan.door_alphas_rad]
        graph_repair_candidates = [None for _ in graph_base_poses]
        if bool(getattr(args, "door_open_front_detour", True)):
            opening_indices = [
                idx for idx, phase in enumerate(graph_phases)
                if int(phase) == PHASE_OPENING
            ]
            if opening_indices:
                last_open_idx = opening_indices[-1]
                start_base = [float(v) for v in graph_base_poses[last_open_idx]]
                if getattr(args, "door_open_front_detour_y", None) is None:
                    detour_y = front_goal_y
                else:
                    detour_y = float(args.door_open_front_detour_y)
                detour_y = max(front_goal_y - front_goal_y_tol, min(front_goal_y + front_goal_y_tol, detour_y))

                y_steps = max(1, int(getattr(args, "door_open_front_detour_y_steps", 3)))
                clear_y_steps = max(1, int(getattr(args, "door_open_front_clear_y_steps", 3)))
                x_steps = max(1, int(getattr(args, "door_open_front_detour_x_steps", 5)))
                front_waypoints: list[list[float]] = [list(start_base)]
                y_target = [start_base[0], detour_y, start_base[2]]
                for step_idx in range(1, y_steps + 1):
                    front_waypoints.append(_interp_base_pose(start_base, y_target, step_idx / float(y_steps)))
                traversal_y = detour_y
                forward_y = detour_y
                forward_yaw = front_goal_yaw
                clear_y_target = [start_base[0], traversal_y, start_base[2]]
                for step_idx in range(1, clear_y_steps + 1):
                    front_waypoints.append(_interp_base_pose(y_target, clear_y_target, step_idx / float(clear_y_steps)))
                x_target = [front_goal_x, forward_y, forward_yaw]
                for step_idx in range(1, x_steps + 1):
                    front_waypoints.append(_interp_base_pose(clear_y_target, x_target, step_idx / float(x_steps)))

                repair_step = max(
                    graph_cfg.xy_step_m,
                    float(getattr(args, "door_open_graph_s2_repair_xy_step_m", 0.04)),
                )

                def _front_detour_repair_grid(center_pose: Sequence[float]) -> list[list[float]]:
                    center = [float(center_pose[0]), float(center_pose[1]), float(center_pose[2])]
                    x_span = repair_step
                    yaw_span = graph_cfg.front_goal_tol_yaw_rad
                    final_x_step = abs(center[0] - front_goal_x) <= front_goal_x_tol
                    out: list[list[float]] = []
                    seen: set[tuple[int, int, int]] = set()

                    def _axis_values(center_value: float, span: float, step: float) -> list[float]:
                        span = max(0.0, float(span))
                        step = max(1.0e-6, float(step))
                        if span <= 1.0e-9:
                            return [float(center_value)]
                        count = max(1, int(math.ceil((2.0 * span) / step)))
                        values = [float(center_value)]
                        for sample_idx in range(count + 1):
                            values.append(float(center_value) - span + (2.0 * span * sample_idx / float(count)))
                        deduped: list[float] = []
                        axis_seen: set[int] = set()
                        for value in values:
                            key = int(round(float(value) * 1000.0))
                            if key in axis_seen:
                                continue
                            axis_seen.add(key)
                            deduped.append(float(value))
                        return deduped

                    x_values = _axis_values(
                        front_goal_x if final_x_step else center[0],
                        front_goal_x_tol if final_x_step else x_span,
                        repair_step,
                    )
                    forward_x_step = center[0] > start_base[0] + 0.5 * repair_step
                    y_values = _axis_values(
                        front_goal_y if final_x_step else center[1],
                        front_goal_y_tol if (final_x_step or forward_x_step) else repair_step,
                        repair_step,
                    )
                    yaw_count = max(1, int(math.ceil((2.0 * yaw_span) / graph_cfg.yaw_step_rad)))
                    for x in x_values:
                        for y in y_values:
                            for yaw_idx in range(yaw_count + 1):
                                yaw = center[2] - yaw_span + (2.0 * yaw_span * yaw_idx / float(yaw_count))
                                pose = [float(x), float(y), _wrap_pi(yaw)]
                                if final_x_step:
                                    allowed_x_span = front_goal_x_tol
                                else:
                                    allowed_x_span = abs(center[0] - front_goal_x) + x_span
                                if abs(pose[0] - front_goal_x) > allowed_x_span:
                                    continue
                                if abs(pose[1] - front_goal_y) > front_goal_y_tol:
                                    continue
                                if abs(_wrap_pi(pose[2] - front_goal_yaw)) > graph_cfg.front_goal_tol_yaw_rad:
                                    continue
                                key = (
                                    int(round(pose[0] * 1000.0)),
                                    int(round(pose[1] * 1000.0)),
                                    int(round(pose[2] * 1000.0)),
                                )
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(pose)
                    out.sort(
                        key=lambda pose: (
                            (0.0 if final_x_step and pose[1] <= center[1] else 0.02)
                            + math.hypot(pose[0] - center[0], pose[1] - center[1])
                            + 0.25 * abs(_wrap_pi(pose[2] - center[2]))
                        )
                    )
                    max_candidates = int(getattr(args, "door_open_graph_s2_repair_max_candidates", 96))
                    if max_candidates > 0:
                        out = out[:max_candidates]
                    return out

                front_waypoint_phases = [PHASE_OPENING] + [PHASE_FRONT for _ in front_waypoints[1:]]
                graph_base_poses = graph_base_poses[: last_open_idx + 1] + front_waypoints
                graph_phases = graph_phases[: last_open_idx + 1] + front_waypoint_phases
                planned_alphas = planned_alphas[: last_open_idx + 1] + [goal_alpha for _ in front_waypoints]
                graph_repair_candidates = (
                    graph_repair_candidates[: last_open_idx + 1]
                    + [_front_detour_repair_grid(pose) for pose in front_waypoints]
                )
                print(
                    "[ARM_DOOR][OPEN][FRONT-DETOUR] "
                    f"enabled waypoints={len(front_waypoints)} "
                    f"start={['%.3f' % v for v in start_base]} "
                    f"detour_y={detour_y:.3f} "
                    f"clear_y={traversal_y:.3f} "
                    f"forward_y={forward_y:.3f} "
                    f"goal={['%.3f' % v for v in [front_goal_x, forward_y, forward_yaw]]}"
                )
        phase_counts = {
            "a1_opening": sum(1 for phase in graph_phases if int(phase) == PHASE_OPENING),
            "a2_traversal": sum(1 for phase in graph_phases if int(phase) == PHASE_FRONT),
        }
        print(
            "[ARM_DOOR][OPEN][S1-GRAPH] done "
            "constraint=2010-ICRA-Lambda-overlap "
            f"waypoints={len(planned_alphas)} cost={graph_plan.cost:.3f} "
            f"expanded={graph_plan.expansions} generated={graph_plan.generated} "
            f"rejected_collision={graph_plan.rejected_collision} "
            f"rejected_reach={graph_plan.rejected_reach} "
            f"rejected_lambda_overlap={graph_plan.rejected_lambda_overlap} "
            f"graph_goal={['%.3f' % v for v in graph_cfg.front_goal_xyyaw]} "
            f"traversal_goal={['%.3f' % v for v in [front_goal_x, front_goal_y, front_goal_yaw]]} "
            f"front_goal_tol_xy=({graph_cfg.front_goal_x_tol_m:.3f}, {graph_cfg.front_goal_y_tol_m:.3f}) "
            f"area_counts={phase_counts}"
        )
    else:
        planned_alphas = [
            goal_alpha * float(idx) / float(steps)
            for idx in range(1, steps + 1)
        ]

    alphas: list[float] = []
    desired_ee_poses: list[tuple[list[float], list[float]]] = []
    step_targets: list[dict[str, Any]] = []
    for idx, alpha in enumerate(planned_alphas, start=1):
        xyz, _ = _door_handle_pose_at(
            alpha_rad=alpha,
            closed_xyz=closed_handle_xyz,
            closed_quat_wxyz=closed_handle_quat_wxyz,
            door_hinge_xyz=door_hinge_xyz,
        )
        orientation_alpha = max(0.0, float(alpha) - orientation_lag)
        quat_wxyz = _quat_mul_wxyz(
            _yaw_quat_wxyz(orientation_alpha),
            closed_handle_quat_wxyz,
        )
        ee_xyz, ee_quat_wxyz = _pose_mul(
            xyz,
            quat_wxyz,
            inv_ee_handle_xyz,
            inv_ee_handle_quat,
        )
        if abs(ee_radial_offset) > 1.0e-12:
            offset_frac = min(1.0, max(0.0, float(alpha) / max(1.0e-9, goal_alpha)))
            ee_xyz = _offset_point_radially_from_hinge(
                ee_xyz,
                door_hinge_xyz,
                ee_radial_offset * offset_frac,
            )
        if orientation_constraint == "door_relative":
            ee_quat_wxyz = _quat_mul_wxyz(
                _yaw_quat_wxyz(orientation_alpha),
                grasp_ee_quat_wxyz,
            )
        collision_cuboid_samples: list[Mapping[str, Any]] = []
        collision_sphere_packs: list[Any] = []
        if collision_checker is not None and collision_source_world is not None:
            for collision_alpha in _opening_collision_alpha_samples(args, float(alpha)):
                dynamic_world = _make_opening_dynamic_world(
                    source_world=collision_source_world,
                    alpha_rad=float(collision_alpha),
                    door_hinge_xyz=door_hinge_xyz,
                    ignore_names=collision_ignore_names,
                )
                cuboids = dynamic_world.get("cuboid", {}) or {}
                if not isinstance(cuboids, dict):
                    raise RuntimeError("dynamic world cuboid entry must be a mapping")
                collision_cuboid_samples.append(cuboids)
                collision_sphere_packs.append(
                    _pack_cuboids_for_sphere_check(
                        cuboids,
                        device=collision_device,
                        dtype=torch.float32,
                    )
                )

        base_graph_candidates = (
            _base_pose_graph_candidates_for_alpha(args, alpha_rad=float(alpha))
            if planner_mode == "astar"
            else None
        )
        alphas.append(float(alpha))
        desired_ee_poses.append((list(ee_xyz), list(ee_quat_wxyz)))
        step_targets.append(
            {
                "idx": int(idx),
                "alpha": float(alpha),
                "handle_xyz": list(xyz),
                "ee_xyz": list(ee_xyz),
                "ee_quat_wxyz": list(ee_quat_wxyz),
                "collision_cuboid_samples": collision_cuboid_samples,
                "collision_sphere_packs": collision_sphere_packs,
                "base_graph_candidates": base_graph_candidates,
                "fixed_base_pose": None if graph_base_poses is None else graph_base_poses[idx - 1],
                "phase": None if graph_phases is None else graph_phases[idx - 1],
                "repair_base_candidates": (
                    None
                    if graph_repair_candidates is None
                    else graph_repair_candidates[idx - 1]
                ),
            }
        )

    target_count = len(step_targets)

    collision_rejected = 0
    collision_checked = 0
    node_counter = 1

    def _base_pose_step_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(
            (float(b[0]) - float(a[0])) ** 2
            + (float(b[1]) - float(a[1])) ** 2
            + (_wrap_pi(float(b[2]) - float(a[2]))) ** 2
        )

    def _opening_heuristic(completed_idx: int, base_pose: Sequence[float]) -> float:
        if int(completed_idx) >= target_count:
            return 0.0
        weight = max(
            0.0,
            min(1.0, float(getattr(args, "door_open_astar_heuristic_weight", 1.0))),
        )
        return weight * 0.5 * _base_pose_goal_span_distance(
            args,
            base_pose=base_pose,
            goal_alpha_rad=goal_alpha,
        )

    def _state_key(state: Mapping[str, Any]) -> tuple[Any, ...]:
        base_pose = [float(v) for v in state["base_pose"]]
        q_cspace = [float(v) for v in state["q_cspace"]]
        base_key = (
            int(round(base_pose[0] / 0.01)),
            int(round(base_pose[1] / 0.01)),
            int(round(base_pose[2] / math.radians(1.0))),
        )
        q_key = tuple(int(round(v / 0.05)) for v in q_cspace)
        return (int(state["idx"]), base_key, q_key)

    def _best_any_detail(best_any: Mapping[str, Any] | None) -> str:
        if best_any is None:
            return ""
        world_v = best_any.get("world_violation", None)
        self_p = best_any.get("self_penetration", None)
        base_v = best_any.get("base_clearance_violation", None)
        collision_detail = ""
        if world_v is not None or self_p is not None or base_v is not None:
            collision_detail = (
                f", best_world_violation={float(world_v or 0.0):.4f}m"
                f"({best_any.get('world_obstacle', '')}), "
                f"best_self_penetration={float(self_p or 0.0):.4f}m"
                f", best_base_clearance_violation={float(base_v or 0.0):.4f}m"
            )
        return (
            f", best_pos_err={float(best_any['pos_err']):.4f}m, "
            f"best_rot_err={math.degrees(float(best_any['rot_err'])):.2f}deg, "
            f"best_joint_step={float(best_any['joint_step']):.3f}rad, "
            f"best_base_pose={[round(float(v), 4) for v in best_any['base_pose']]}"
            f"{collision_detail}"
        )

    def _raise_opening_failure(
        *,
        target: Mapping[str, Any],
        best_any: Mapping[str, Any] | None,
        prefix: str,
    ) -> None:
        raise RuntimeError(
            f"{prefix} at alpha={math.degrees(float(target['alpha'])):.1f} deg, "
            f"handle_xyz={[round(float(v), 4) for v in target['handle_xyz']]}, "
            f"ee_xyz={[round(float(v), 4) for v in target['ee_xyz']]}"
            f"{_best_any_detail(best_any)}"
        )

    def _front_goal_repair_candidates(
        target: Mapping[str, Any],
        previous_base_pose: Sequence[float],
        fixed_base_pose: Sequence[float],
    ) -> list[list[float]] | None:
        if int(target.get("phase", -1)) != PHASE_FRONT:
            return None
        candidates: list[list[float]] = []
        anchors: list[list[float]] = []
        seen: set[tuple[int, int, int]] = set()

        def _add(raw_pose: Sequence[float], *, anchor: bool = False) -> None:
            pose = [float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])]
            dx = abs(pose[0] - graph_cfg.front_goal_xyyaw[0])
            dy = abs(pose[1] - graph_cfg.front_goal_xyyaw[1])
            dyaw = abs(_wrap_pi(pose[2] - graph_cfg.front_goal_xyyaw[2]))
            if dx > graph_cfg.front_goal_x_tol_m:
                return
            if dy > graph_cfg.front_goal_y_tol_m:
                return
            if dyaw > graph_cfg.front_goal_tol_yaw_rad:
                return
            key = (
                int(round(pose[0] * 1000.0)),
                int(round(pose[1] * 1000.0)),
                int(round(pose[2] * 1000.0)),
            )
            if key in seen:
                return
            seen.add(key)
            if anchor:
                anchors.append(pose)
            else:
                candidates.append(pose)

        center_x, center_y, center_yaw = graph_cfg.front_goal_xyyaw
        min_x = center_x - graph_cfg.front_goal_x_tol_m
        max_x = center_x + graph_cfg.front_goal_x_tol_m
        min_y = center_y - graph_cfg.front_goal_y_tol_m
        max_y = center_y + graph_cfg.front_goal_y_tol_m
        min_yaw = center_yaw - graph_cfg.front_goal_tol_yaw_rad
        max_yaw = center_yaw + graph_cfg.front_goal_tol_yaw_rad

        _add(fixed_base_pose, anchor=True)
        _add(previous_base_pose, anchor=True)
        for x in (center_x, min_x, max_x):
            for y in (center_y, min_y, max_y, float(previous_base_pose[1])):
                for yaw in (center_yaw, min_yaw, max_yaw):
                    _add([x, y, yaw], anchor=True)

        def _sample_axis(center: float, span: float, step: float) -> list[float]:
            span = max(0.0, float(span))
            step = max(1.0e-6, float(step))
            count = max(1, int(math.ceil((2.0 * span) / step)))
            values = [float(center)]
            for sample_idx in range(count + 1):
                value = float(center) - span + (2.0 * span * sample_idx / float(count))
                values.append(value)
            return values

        x_values = _sample_axis(
            graph_cfg.front_goal_xyyaw[0],
            graph_cfg.front_goal_x_tol_m,
            max(graph_cfg.xy_step_m, float(getattr(args, "door_open_graph_s2_repair_xy_step_m", 0.04))),
        )
        y_values = _sample_axis(
            graph_cfg.front_goal_xyyaw[1],
            graph_cfg.front_goal_y_tol_m,
            max(graph_cfg.xy_step_m, float(getattr(args, "door_open_graph_s2_repair_xy_step_m", 0.04))),
        )
        yaw_values = _sample_axis(
            graph_cfg.front_goal_xyyaw[2],
            graph_cfg.front_goal_tol_yaw_rad,
            graph_cfg.yaw_step_rad,
        )
        for x in x_values:
            for y in y_values:
                for yaw in yaw_values:
                    _add([x, y, yaw])
        candidates.sort(
            key=lambda pose: (
                math.hypot(
                    float(pose[0]) - float(fixed_base_pose[0]),
                    float(pose[1]) - float(fixed_base_pose[1]),
                )
                + 0.25 * abs(_wrap_pi(float(pose[2]) - float(fixed_base_pose[2]))),
                math.hypot(
                    float(pose[0]) - float(previous_base_pose[0]),
                    float(pose[1]) - float(previous_base_pose[1]),
                ),
            )
        )
        max_candidates = int(getattr(args, "door_open_graph_s2_repair_max_candidates", 96))
        if max_candidates <= 0:
            return anchors + candidates
        remaining = max(0, max_candidates - len(anchors))
        return anchors + candidates[:remaining]

    def _expand_opening_parent(
        parent: Mapping[str, Any],
        target: Mapping[str, Any],
        *,
        parent_tag: int,
        base_candidates: Sequence[Sequence[float]] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
        nonlocal collision_checked, collision_rejected, node_counter
        idx = int(target["idx"])
        alpha = float(target["alpha"])
        ee_xyz = [float(v) for v in target["ee_xyz"]]
        ee_quat_wxyz = [float(v) for v in target["ee_quat_wxyz"]]
        collision_cuboid_samples = list(target["collision_cuboid_samples"])
        collision_sphere_packs = list(target.get("collision_sphere_packs", []))
        raw_target_phase = target.get("phase", -1)
        target_phase = -1 if raw_target_phase is None else int(raw_target_phase)
        target_collision_margin_m = collision_margin_m
        target_base_clearance_m = 0.0
        if target_phase == 2:
            target_base_clearance_m = max(
                0.0,
                float(getattr(args, "door_open_front_door_clearance_m", 0.05)),
            )
        target["collision_margin_m"] = float(target_collision_margin_m)
        target["base_clearance_m"] = float(target_base_clearance_m)
        parent_q = [float(v) for v in parent["q_cspace"]]
        parent_base_pose = [float(v) for v in parent["base_pose"]]
        if base_candidates is None:
            base_candidates = _base_pose_candidates_for_alpha(
                args,
                alpha_rad=alpha,
                previous_base_pose=parent_base_pose,
            )
        target_ik_batch = ik_batch
        if target_phase == 2 or alpha >= goal_alpha - 1.0e-9:
            target_ik_batch = max(
                target_ik_batch,
                ik_batch * max(1, int(getattr(args, "door_open_front_ik_batch_multiplier", 3))),
            )
        q_seed_batch = _build_ik_seed_batch(
            parent_q,
            batch_size=target_ik_batch,
            noise_std=ik_noise,
            random_seed=int(args.door_open_ik_seed) + idx * 997 + parent_tag * 37,
            lower=joint_lower,
            upper=joint_upper,
        )

        children: list[dict[str, Any]] = []
        best_any: dict[str, Any] | None = None
        for base_pose_raw in base_candidates:
            base_pose = [float(v) for v in base_pose_raw]
            ee_xyz_base, ee_quat_base = _world_pose_to_base_pose(
                ee_xyz,
                ee_quat_wxyz,
                base_pose,
            )
            ik_outs = ik.solve_batch(
                [list(ee_xyz_base) for _ in range(len(q_seed_batch))],
                [list(ee_quat_base) for _ in range(len(q_seed_batch))],
                q_start_cspace=parent_q,
                q_seed_cspace_batch=q_seed_batch,
            )

            for out in ik_outs:
                if not out.success or out.q_cspace is None:
                    continue
                q_cand = [float(v) for v in out.q_cspace]
                q_active = _extract_active_from_cspace(
                    q_cand,
                    ik.cspace_joint_names,
                    ik.active_joint_names,
                )
                with torch.no_grad():
                    kin = ik.solver.fk(
                        torch.tensor([q_active], device=ik.device, dtype=torch.float32)
                    )
                fk_xyz = [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]
                fk_quat = [float(v) for v in kin.ee_quaternion[0].detach().cpu().tolist()]
                pos_err = math.sqrt(sum((fk_xyz[i] - ee_xyz_base[i]) ** 2 for i in range(3)))
                rot_err = _quat_angle_error_rad(fk_quat, ee_quat_base)
                joint_step = float(np.linalg.norm(np.asarray(q_cand) - np.asarray(parent_q)))
                base_step = _base_pose_step_distance(parent_base_pose, base_pose)
                candidate: dict[str, Any] = {
                    "pos_err": float(pos_err),
                    "rot_err": float(rot_err),
                    "joint_step": float(joint_step),
                    "base_step": float(base_step),
                    "world_violation": None,
                    "self_penetration": None,
                    "base_clearance_violation": None,
                    "world_obstacle": "",
                    "q_cspace": q_cand,
                    "base_pose": list(base_pose),
                }
                if pos_err <= max_pos_err and rot_err <= max_rot_err:
                    collision_free = True
                    if collision_cuboid_samples:
                        collision_checked += 1
                        candidate_sphere_packs = collision_sphere_packs
                        if not candidate_sphere_packs:
                            candidate_sphere_packs = [
                                _pack_cuboids_for_sphere_check(
                                    cuboids,
                                    device=collision_device,
                                    dtype=torch.float32,
                                )
                                for cuboids in collision_cuboid_samples
                            ]
                        world_violation, self_pen, obstacle = _opening_candidate_collision(
                            checker=collision_checker,
                            q_cspace=q_cand,
                            base_pose=base_pose,
                            cuboids=collision_cuboid_samples[0],
                            cuboid_sphere_packs=candidate_sphere_packs,
                            margin_m=target_collision_margin_m,
                            device=collision_device,
                        )
                        base_clearance_violation = -float("inf")
                        base_obstacle = ""
                        for cuboids in collision_cuboid_samples:
                            cand_base_violation, cand_base_obstacle = _base_door_clearance_violation(
                                base_pose=base_pose,
                                cuboids=cuboids,
                                clearance_m=target_base_clearance_m,
                                base_radius_m=float(args.door_open_graph_base_radius_m),
                                base_height_m=float(args.door_open_graph_base_height_m),
                                door_names=DOOR_BASE_CLEARANCE_COLLISION_NAMES,
                            )
                            if cand_base_violation > base_clearance_violation:
                                base_clearance_violation = float(cand_base_violation)
                                base_obstacle = str(cand_base_obstacle)
                        if base_clearance_violation > max(world_violation, self_pen):
                            obstacle = str(base_obstacle)
                        candidate["world_violation"] = float(world_violation)
                        candidate["self_penetration"] = float(self_pen)
                        candidate["base_clearance_violation"] = max(0.0, float(base_clearance_violation))
                        candidate["world_obstacle"] = str(obstacle)
                        if (
                            world_violation > 0.0
                            or self_pen > 0.0
                            or float(candidate["base_clearance_violation"]) > 0.0
                        ):
                            collision_rejected += 1
                            collision_free = False
                    if collision_free:
                        local_cost = joint_step + 0.5 * base_step + 20.0 * pos_err + rot_err
                        children.append(
                            {
                                "idx": idx,
                                "node_id": int(node_counter),
                                "cost": float(parent["cost"]) + float(local_cost),
                                "q_cspace": q_cand,
                                "base_pose": list(base_pose),
                                "path": list(parent["path"]) + [q_cand],
                                "base_path": list(parent["base_path"]) + [list(base_pose)],
                            }
                        )
                        node_counter += 1

                candidate_score = (
                    float(parent["cost"])
                    + candidate["pos_err"]
                    + candidate["rot_err"]
                    + 0.01 * candidate["joint_step"]
                    + 0.05 * candidate["base_step"]
                    + 10.0 * max(0.0, float(candidate["world_violation"] or 0.0))
                    + 10.0 * max(0.0, float(candidate["self_penetration"] or 0.0))
                    + 10.0 * max(0.0, float(candidate["base_clearance_violation"] or 0.0))
                )
                if best_any is None or candidate_score < float(best_any.get("_score", float("inf"))):
                    candidate["_score"] = candidate_score
                    best_any = candidate

        return children, best_any, len(base_candidates)

    final_state: dict[str, Any] | None = None
    if planner_mode == "graph":
        print(f"[ARM_DOOR][OPEN][S2-IK] solving fixed S1 graph path ({target_count} waypoint(s))")
        parent: dict[str, Any] = {
            "idx": 0,
            "node_id": 0,
            "cost": 0.0,
            "q_cspace": [float(v) for v in q_start_cspace],
            "base_pose": [0.0, 0.0, 0.0],
            "path": [],
            "base_path": [],
        }
        for target in step_targets:
            idx = int(target["idx"])
            fixed_base_pose = target.get("fixed_base_pose")
            if fixed_base_pose is None:
                raise RuntimeError("graph planner target is missing fixed base pose")
            checked_before = collision_checked
            rejected_before = collision_rejected
            children, best_any, _ = _expand_opening_parent(
                parent,
                target,
                parent_tag=idx,
                base_candidates=[fixed_base_pose],
            )
            repaired = False
            repair_base_count = 0
            if not children:
                if bool(getattr(args, "door_open_graph_s2_local_repair", True)):
                    repair_candidates = target.get("repair_base_candidates")
                    if repair_candidates is None:
                        repair_candidates = _front_goal_repair_candidates(
                            target,
                            parent["base_pose"],
                            fixed_base_pose,
                        )
                    print(
                        "[ARM_DOOR][OPEN][S2-IK][REPAIR] "
                        f"fixed graph base failed at idx={idx}/{target_count}; "
                        f"trying {len(repair_candidates) if repair_candidates is not None else 'auto'} nearby base candidates"
                    )
                    children, repair_best, repair_base_count = _expand_opening_parent(
                        parent,
                        target,
                        parent_tag=idx + 10000,
                        base_candidates=repair_candidates,
                    )
                    repaired = bool(children)
                    if repair_best is not None and (
                        best_any is None
                        or float(repair_best.get("_score", float("inf")))
                        < float(best_any.get("_score", float("inf")))
                    ):
                        best_any = repair_best
                if not children:
                    _raise_opening_failure(
                        target=target,
                        best_any=best_any,
                        prefix="opening graph S2 IK failed",
                )
            children.sort(key=lambda item: float(item["cost"]))
            parent = children[0]
            selected_base_pose = [float(v) for v in parent["base_pose"]]
            checked_this = collision_checked - checked_before
            rejected_this = collision_rejected - rejected_before
            phase_value = target.get("phase")
            if phase_value == 1:
                area_label = "a=1_opening"
            elif phase_value == 2:
                area_label = "a=2_traversal"
            else:
                area_label = f"a={phase_value}"
            print(
                "[ARM_DOOR][OPEN][S2-IK] "
                f"idx={idx}/{target_count} alpha={math.degrees(float(target['alpha'])):.1f}deg "
                f"phase={phase_value}({area_label}) "
                f"margin={float(target.get('collision_margin_m', 0.0)):.3f} "
                f"base_clearance={float(target.get('base_clearance_m', 0.0)):.3f} "
                f"collision_rejected={rejected_this}/{checked_this} "
                f"base={[round(float(v), 4) for v in selected_base_pose]}"
                + (
                    f" repaired_from={[round(float(v), 4) for v in fixed_base_pose]} "
                    f"repair_candidates={repair_base_count}"
                    if repaired
                    else ""
                )
            )
        final_state = parent
    elif planner_mode == "beam":
        beam_width = max(1, int(getattr(args, "door_open_beam_width", 1)))
        print(f"[ARM_DOOR][OPEN][BEAM] enabled beam_width={beam_width}")
        beam_states: list[dict[str, Any]] = [
            {
                "idx": 0,
                "node_id": 0,
                "cost": 0.0,
                "q_cspace": [float(v) for v in q_start_cspace],
                "base_pose": [0.0, 0.0, 0.0],
                "path": [],
                "base_path": [],
            }
        ]

        for target in step_targets:
            idx = int(target["idx"])
            next_states: list[dict[str, Any]] = []
            best_any: dict[str, Any] | None = None
            checked_before = collision_checked
            rejected_before = collision_rejected
            print(
                "[ARM_DOOR][OPEN][CAND] "
                f"idx={idx}/{target_count} alpha={math.degrees(float(target['alpha'])):.1f}deg "
                f"beam_in={len(beam_states)} ik_batch={ik_batch}"
            )
            total_base_candidates = 0
            for parent_idx, parent in enumerate(beam_states):
                children, local_best, base_count = _expand_opening_parent(
                    parent,
                    target,
                    parent_tag=parent_idx,
                    base_candidates=None,
                )
                next_states.extend(children)
                total_base_candidates += base_count
                if local_best is not None and (
                    best_any is None
                    or float(local_best.get("_score", float("inf")))
                    < float(best_any.get("_score", float("inf")))
                ):
                    best_any = local_best

            if not next_states:
                _raise_opening_failure(
                    target=target,
                    best_any=best_any,
                    prefix="opening IK failed",
                )

            next_states.sort(key=lambda item: float(item["cost"]))
            beam_states = next_states[:beam_width]
            selected_base_pose = [float(v) for v in beam_states[0]["base_pose"]]
            checked_this = collision_checked - checked_before
            rejected_this = collision_rejected - rejected_before
            print(
                "[ARM_DOOR][OPEN][CAND] "
                f"idx={idx}/{target_count} base_candidates={total_base_candidates} "
                f"accepted={len(next_states)} beam_out={len(beam_states)} "
                f"collision_rejected={rejected_this}/{checked_this} "
                f"selected_base={[round(v, 4) for v in selected_base_pose]}"
            )

        final_state = beam_states[0]
    else:
        max_expansions = max(1, int(getattr(args, "door_open_astar_max_expansions", 1200)))
        layer_keep = int(getattr(args, "door_open_astar_layer_keep", 96))
        queue_keep = int(getattr(args, "door_open_astar_queue_keep", 512))
        total_base_nodes = sum(
            len(target["base_graph_candidates"] or [])
            for target in step_targets
        )
        print(
            "[ARM_DOOR][OPEN][ASTAR] enabled "
            f"layers={target_count} base_nodes={total_base_nodes} "
            f"max_expansions={max_expansions} layer_keep={layer_keep} queue_keep={queue_keep}"
        )
        start_state: dict[str, Any] = {
            "idx": 0,
            "node_id": 0,
            "cost": 0.0,
            "q_cspace": [float(v) for v in q_start_cspace],
            "base_pose": [0.0, 0.0, 0.0],
            "path": [],
            "base_path": [],
        }
        open_heap: list[tuple[float, float, int, dict[str, Any]]] = []
        heap_counter = 0

        def _push_astar(state: dict[str, Any]) -> None:
            nonlocal heap_counter
            h = _opening_heuristic(int(state["idx"]), state["base_pose"])
            heapq.heappush(
                open_heap,
                (
                    float(state["cost"]) + h,
                    float(state["cost"]),
                    heap_counter,
                    state,
                ),
            )
            heap_counter += 1

        best_cost_by_key: dict[tuple[Any, ...], float] = {
            _state_key(start_state): 0.0,
        }
        queued_by_layer: dict[int, int] = {}
        best_any_global: dict[str, Any] | None = None
        _push_astar(start_state)

        expansions = 0
        while open_heap and expansions < max_expansions:
            f_score, g_score, _, parent = heapq.heappop(open_heap)
            parent_key = _state_key(parent)
            if float(parent["cost"]) > best_cost_by_key.get(parent_key, float("inf")) + 1.0e-9:
                continue
            if int(parent["idx"]) >= target_count:
                final_state = parent
                break

            next_idx = int(parent["idx"]) + 1
            target = step_targets[next_idx - 1]
            checked_before = collision_checked
            rejected_before = collision_rejected
            if expansions == 0 or expansions % 20 == 0:
                print(
                    "[ARM_DOOR][OPEN][ASTAR] "
                    f"expand={expansions} layer={int(parent['idx'])}->{next_idx}/{target_count} "
                    f"queue={len(open_heap)} f={f_score:.3f} g={g_score:.3f} "
                    f"base={[round(float(v), 4) for v in parent['base_pose']]}"
                )

            children, local_best, base_count = _expand_opening_parent(
                parent,
                target,
                parent_tag=int(parent.get("node_id", expansions)),
                base_candidates=target["base_graph_candidates"],
            )
            if local_best is not None and (
                best_any_global is None
                or float(local_best.get("_score", float("inf")))
                < float(best_any_global.get("_score", float("inf")))
            ):
                best_any_global = local_best

            children.sort(
                key=lambda state: float(state["cost"])
                + _opening_heuristic(int(state["idx"]), state["base_pose"])
            )
            pushed = 0
            for child in children:
                child_idx = int(child["idx"])
                if layer_keep > 0 and queued_by_layer.get(child_idx, 0) >= layer_keep:
                    continue
                child_key = _state_key(child)
                old_cost = best_cost_by_key.get(child_key)
                if old_cost is not None and float(child["cost"]) >= old_cost - 1.0e-9:
                    continue
                best_cost_by_key[child_key] = float(child["cost"])
                queued_by_layer[child_idx] = queued_by_layer.get(child_idx, 0) + 1
                _push_astar(child)
                pushed += 1

            expansions += 1
            if queue_keep > 0 and len(open_heap) > queue_keep:
                open_heap = heapq.nsmallest(queue_keep, open_heap)
                heapq.heapify(open_heap)

            if expansions % 20 == 0 or pushed == 0:
                checked_this = collision_checked - checked_before
                rejected_this = collision_rejected - rejected_before
                print(
                    "[ARM_DOOR][OPEN][ASTAR] "
                    f"expanded={expansions} layer={next_idx}/{target_count} "
                    f"base_candidates={base_count} accepted={len(children)} pushed={pushed} "
                    f"queue={len(open_heap)} collision_rejected={rejected_this}/{checked_this}"
                )

        if final_state is None:
            furthest_idx = 0
            if best_cost_by_key:
                furthest_idx = max(int(key[0]) for key in best_cost_by_key.keys())
            target = step_targets[max(0, min(target_count - 1, furthest_idx))]
            _raise_opening_failure(
                target=target,
                best_any=best_any_global,
                prefix=(
                    "opening A* failed "
                    f"(expanded={expansions}, queue={len(open_heap)}, furthest_layer={furthest_idx}/{target_count})"
                ),
            )
        print(
            "[ARM_DOOR][OPEN][ASTAR] done "
            f"expanded={expansions} final_cost={float(final_state['cost']):.3f} "
            f"final_base={[round(float(v), 4) for v in final_state['base_pose']]}"
        )

    if collision_checker is not None:
        print(
            "[ARM_DOOR][OPEN][COLLISION] candidate-aware selection "
            f"rejected={collision_rejected}/{collision_checked} FK-valid candidate(s)"
        )

    if final_state is None:
        raise RuntimeError("opening path search did not return a final state")
    best_path = [[float(v) for v in q] for q in final_state["path"]]
    best_base_poses = [[float(v) for v in pose] for pose in final_state["base_path"]]
    return list(ik.cspace_joint_names), list(ik.controlled_joint_names), best_path, alphas, desired_ee_poses, best_base_poses


def _publish_door_unlock(topic: str) -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("arm_door_unlock_publisher")
    pub = node.create_publisher(Bool, str(topic), 10)
    t_end = time.monotonic() + 0.5
    while rclpy.ok() and time.monotonic() < t_end and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)

    msg = Bool()
    msg.data = True
    for _ in range(3):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.03)
    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()


def _validate_opening_fk(
    args,
    *,
    arm: str,
    cspace_joint_names: Sequence[str],
    cspace_path: Sequence[Sequence[float]],
    door_alphas_rad: Sequence[float],
    desired_ee_poses: Sequence[tuple[Sequence[float], Sequence[float]]],
    base_poses: Sequence[Sequence[float]] | None,
    resolved_world_yml: str | None,
) -> None:
    from capstone_pkg.kinematics.curobo_ik import get_single_arm_ik
    import torch

    if not cspace_path:
        raise RuntimeError("opening FK validation path is empty")
    if len(cspace_path) != len(desired_ee_poses) or len(cspace_path) != len(door_alphas_rad):
        raise RuntimeError("opening FK validation inputs have mismatched lengths")
    if base_poses is not None and len(base_poses) != len(cspace_path):
        raise RuntimeError("opening FK validation base pose path length mismatch")

    opening_world_yml = _resolve_opening_world_yml(
        str(args.door_open_world_yml),
        resolved_world_yml,
    )
    ik = get_single_arm_ik(
        str(args.robot_yml),
        arm=arm,
        cpu=bool(args.cpu),
        world_yml=opening_world_yml,
    )

    records = []
    max_pos_err = 0.0
    max_rot_err = 0.0
    for idx, (q_cspace, alpha, desired_pose) in enumerate(
        zip(cspace_path, door_alphas_rad, desired_ee_poses)
    ):
        q_active = _extract_active_from_cspace(
            q_cspace,
            cspace_joint_names,
            ik.active_joint_names,
        )
        with torch.no_grad():
            kin = ik.solver.fk(
                torch.tensor([q_active], device=ik.device, dtype=torch.float32)
            )
        fk_xyz = [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]
        fk_quat = [float(v) for v in kin.ee_quaternion[0].detach().cpu().tolist()]
        base_pose = None if base_poses is None else [float(v) for v in base_poses[idx]]
        if base_pose is None:
            des_xyz = [float(v) for v in desired_pose[0]]
            des_quat = [float(v) for v in desired_pose[1]]
        else:
            des_xyz, des_quat = _world_pose_to_base_pose(
                desired_pose[0],
                desired_pose[1],
                base_pose,
            )
        pos_err = math.sqrt(sum((fk_xyz[i] - des_xyz[i]) ** 2 for i in range(3)))
        rot_err = _quat_angle_error_rad(fk_quat, des_quat)
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        records.append(
            {
                "index": int(idx),
                "alpha_deg": math.degrees(float(alpha)),
                "desired_xyz": des_xyz,
                "fk_xyz": fk_xyz,
                "base_pose_xyyaw": base_pose,
                "pos_err_m": float(pos_err),
                "rot_err_deg": math.degrees(float(rot_err)),
            }
        )

    print(
        "[ARM_DOOR][OPEN][FK] "
        f"waypoints={len(records)} "
        f"max_pos_err={max_pos_err:.4f}m "
        f"max_rot_err={math.degrees(max_rot_err):.2f}deg"
    )
    warn_pos = float(args.opening_fk_warn_pos_m)
    warn_rot = math.radians(float(args.opening_fk_warn_rot_deg))
    if max_pos_err > warn_pos or max_rot_err > warn_rot:
        print(
            "[ARM_DOOR][OPEN][FK][WARN] constraint error exceeds threshold: "
            f"pos>{warn_pos:.4f}m or rot>{math.degrees(warn_rot):.2f}deg"
        )
        if bool(args.opening_fk_fail_on_warn):
            raise RuntimeError(
                "opening FK constraint validation failed: "
                f"max_pos_err={max_pos_err:.4f}m "
                f"max_rot_err={math.degrees(max_rot_err):.2f}deg"
            )

    if str(args.opening_fk_log).strip():
        with open(str(args.opening_fk_log), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_pos_err_m": float(max_pos_err),
                    "max_rot_err_deg": math.degrees(float(max_rot_err)),
                    "records": records,
                },
                f,
                indent=2,
            )
        print(f"[ARM_DOOR][OPEN][FK] saved -> {args.opening_fk_log}")


def _cuboid_sphere_max_violation(
    spheres,
    cuboid: Mapping[str, Any],
    *,
    margin_m: float,
    device,
) -> float:
    pack = _pack_cuboids_for_sphere_check(
        {"cuboid": cuboid},
        device=device,
        dtype=spheres.dtype,
    )
    violation, _ = _cuboid_sphere_pack_max_violation(
        spheres,
        pack,
        margin_m=margin_m,
    )
    return float(violation)


def _validate_opening_dynamic_collision(
    args,
    *,
    cspace_path: Sequence[Sequence[float]],
    door_alphas_rad: Sequence[float],
    door_hinge_xyz: Sequence[float],
    base_poses: Sequence[Sequence[float]] | None,
    resolved_world_yml: str | None,
) -> None:
    from capstone_pkg.collision_check.collision import get_self_collision_checker
    import torch

    if not cspace_path:
        raise RuntimeError("opening dynamic collision validation path is empty")
    if len(cspace_path) != len(door_alphas_rad):
        raise RuntimeError("opening dynamic collision validation inputs have mismatched lengths")
    if base_poses is not None and len(base_poses) != len(cspace_path):
        raise RuntimeError("opening dynamic collision validation base pose path length mismatch")

    source_world_yml = _resolve_dynamic_opening_world_yml(args, resolved_world_yml)
    if source_world_yml is None:
        print("[ARM_DOOR][OPEN][COLLISION] dynamic opening collision disabled (world_yml=None)")
        return

    source_world = _load_world_yaml(str(source_world_yml))
    checker = get_self_collision_checker(
        str(args.robot_yml),
        cpu=bool(args.cpu),
        world_yml=None,
    )
    device = checker.tensor_args.device
    q_cspace = torch.tensor(cspace_path, device=device, dtype=torch.float32)

    self_result = checker.check_batch(q_cspace)
    q_active = checker._build_q_active_from_cspace(q_cspace)
    state = checker.robot_world_self_only.get_kinematics(q_active)
    x_sph = checker._extract_spheres(checker.robot_world_self_only, state, q_active)
    if x_sph.dim() == 4:
        spheres_batch = x_sph[:, 0, :, :]
    elif x_sph.dim() == 3:
        spheres_batch = x_sph
    else:
        raise RuntimeError(f"Unexpected sphere tensor shape: {tuple(x_sph.shape)}")

    ignore_names = [str(name) for name in getattr(args, "opening_dynamic_ignore", [])]
    margin_m = float(args.opening_dynamic_collision_margin_m)
    records: list[dict[str, Any]] = []
    max_world_violation = -float("inf")
    max_self_violation = -float("inf")
    worst_record: dict[str, Any] | None = None

    for idx, alpha in enumerate(door_alphas_rad):
        spheres = spheres_batch[idx].clone()
        base_pose = None if base_poses is None else [float(v) for v in base_poses[idx]]
        if base_pose is not None:
            spheres[:, :3] = _transform_points_by_base_pose(spheres[:, :3], base_pose)
        waypoint_max_world = -float("inf")
        waypoint_obstacle = ""
        worst_collision_alpha = float(alpha)
        for collision_alpha in _opening_collision_alpha_samples(args, float(alpha)):
            world = _make_opening_dynamic_world(
                source_world=source_world,
                alpha_rad=float(collision_alpha),
                door_hinge_xyz=door_hinge_xyz,
                ignore_names=ignore_names,
            )
            cuboids = world.get("cuboid", {}) or {}
            if not isinstance(cuboids, dict):
                raise RuntimeError("dynamic world cuboid entry must be a mapping")

            cuboid_pack = _pack_cuboids_for_sphere_check(
                cuboids,
                device=device,
                dtype=spheres.dtype,
            )
            violation, obstacle = _cuboid_sphere_pack_max_violation(
                spheres,
                cuboid_pack,
                margin_m=margin_m,
            )
            if violation > waypoint_max_world:
                waypoint_max_world = violation
                waypoint_obstacle = str(obstacle)
                worst_collision_alpha = float(collision_alpha)

        if waypoint_max_world == -float("inf"):
            waypoint_max_world = 0.0
        self_pen = float(self_result.d_self_max[idx].detach().cpu().item())
        max_world_violation = max(max_world_violation, waypoint_max_world)
        max_self_violation = max(max_self_violation, self_pen)
        record = {
            "index": int(idx),
            "alpha_deg": math.degrees(float(alpha)),
            "base_pose_xyyaw": base_pose,
            "max_world_violation_m": float(waypoint_max_world),
            "world_obstacle": waypoint_obstacle,
            "collision_alpha_deg": math.degrees(float(worst_collision_alpha)),
            "self_penetration_m": float(self_pen),
        }
        records.append(record)
        if waypoint_max_world > 0.0 or self_pen > 0.0:
            if worst_record is None:
                worst_record = record
            elif (
                max(waypoint_max_world, self_pen)
                > max(float(worst_record["max_world_violation_m"]), float(worst_record["self_penetration_m"]))
            ):
                worst_record = record

    print(
        "[ARM_DOOR][OPEN][COLLISION] "
        f"waypoints={len(records)} "
        f"max_world_violation={max_world_violation:.4f}m "
        f"max_self_penetration={max_self_violation:.4f}m "
        f"ignored={ignore_names} "
        f"lag_deg={float(getattr(args, 'door_open_collision_lag_deg', 0.0)):.1f}"
    )

    if str(args.opening_dynamic_collision_log).strip():
        with open(str(args.opening_dynamic_collision_log), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_world_yml": str(source_world_yml),
                    "ignored_collision_names": ignore_names,
                    "margin_m": float(margin_m),
                    "lag_deg": float(getattr(args, "door_open_collision_lag_deg", 0.0)),
                    "max_world_violation_m": float(max_world_violation),
                    "max_self_penetration_m": float(max_self_violation),
                    "records": records,
                },
                f,
                indent=2,
            )
        print(f"[ARM_DOOR][OPEN][COLLISION] saved -> {args.opening_dynamic_collision_log}")

    if worst_record is not None:
        raise RuntimeError(
            "dynamic opening collision failed: "
            f"idx={worst_record['index']} "
            f"alpha={float(worst_record['alpha_deg']):.1f}deg "
            f"world_violation={float(worst_record['max_world_violation_m']):.4f}m "
            f"obstacle={worst_record['world_obstacle']} "
            f"self_penetration={float(worst_record['self_penetration_m']):.4f}m"
        )


def _publish_real_opening_path_with_base(
    args,
    arm: str,
    joint_names,
    path,
    base_poses,
    desired_ee_poses=None,
) -> None:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import (
        _build_joint_trajectory,
        _command_qos,
        _command_stamp,
    )
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory

    publish_joint_names = [str(name) for name in joint_names]
    publish_path = [[float(v) for v in q] for q in path]
    publish_base_poses = [[float(v) for v in pose] for pose in base_poses]
    publish_arm_velocities: list[list[float]] = []
    publish_arm_accelerations: list[list[float]] = []
    publish_times: list[float] = []
    if len(publish_path) != len(publish_base_poses):
        raise RuntimeError("real opening base pose path length does not match arm path length")
    if len(publish_path) < 2:
        raise RuntimeError("real opening path must contain at least two waypoints for base cmd_vel")
    _print_base_path_summary("[ARM_DOOR][OPEN][REAL][BASE][RAW]", publish_base_poses)

    smooth_window = int(getattr(args, "door_open_base_publish_smooth_window", 0))
    if smooth_window > 1:
        old_base_poses = publish_base_poses
        publish_base_poses = _smooth_opening_base_poses(
            publish_base_poses,
            window=smooth_window,
        )
        changed_count = sum(
            1
            for old_pose, new_pose in zip(old_base_poses, publish_base_poses)
            if (
                math.hypot(new_pose[0] - old_pose[0], new_pose[1] - old_pose[1]) > 1.0e-6
                or abs(_wrap_pi(new_pose[2] - old_pose[2])) > 1.0e-6
            )
        )
        print(
            "[ARM_DOOR][OPEN][REAL][BASE] "
            f"smoothed base publish poses window={smooth_window} changed={changed_count}/{len(publish_base_poses)}"
        )
        _print_base_path_summary("[ARM_DOOR][OPEN][REAL][BASE][SMOOTH]", publish_base_poses)

    if bool(getattr(args, "door_open_wbc_qp", False)):
        if desired_ee_poses is None:
            raise RuntimeError("door_open_wbc_qp requires desired opening EE poses")
        if len(desired_ee_poses) != len(publish_path):
            raise RuntimeError("door_open_wbc_qp desired EE pose length does not match path length")
        from capstone_pkg.utils.config import LEFT_EE_FRAME, RIGHT_EE_FRAME, ROBOT_URDF
        from capstone_pkg.wbc.door_qp import DoorOpeningWBCQP, DoorWBCQPConfig

        ee_frame = LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME
        qp = DoorOpeningWBCQP(
            DoorWBCQPConfig(
                urdf_path=str(ROBOT_URDF),
                ee_frame=str(ee_frame),
                joint_limit_yml=str(args.joint_limit_yml),
                dt=max(1.0e-3, float(args.door_open_dt)),
                max_base_linear_mps=max(1.0e-6, float(args.real_base_max_linear_mps)),
                max_base_angular_rps=max(1.0e-6, float(args.real_base_max_angular_rps)),
                max_joint_velocity=max(1.0e-6, float(args.door_open_topp_arm_max_velocity)),
                kp_pos=float(args.door_open_wbc_qp_kp_pos),
                kp_rot=float(args.door_open_wbc_qp_kp_rot),
                task_weight=max(1.0e-9, float(args.door_open_wbc_qp_task_weight)),
                base_ref_weight=max(0.0, float(args.door_open_wbc_qp_base_ref_weight)),
                joint_ref_weight=max(0.0, float(args.door_open_wbc_qp_joint_ref_weight)),
                joint_reg_weight=max(0.0, float(args.door_open_wbc_qp_joint_reg_weight)),
                hard_task_constraint=bool(args.door_open_wbc_qp_hard_constraint),
                backend=str(args.door_open_wbc_qp_backend),
            )
        )
        result = qp.rollout(
            joint_names=publish_joint_names,
            joint_path=publish_path,
            base_poses=publish_base_poses,
            desired_ee_poses=desired_ee_poses,
        )
        publish_path = result.joint_path
        publish_base_poses = result.base_poses
        max_task_error = max(result.task_error_norms) if result.task_error_norms else 0.0
        print(
            "[ARM_DOOR][OPEN][REAL][WBC-QP] "
            f"backend={result.used_backend or str(args.door_open_wbc_qp_backend)} "
            f"hard_constraint={bool(args.door_open_wbc_qp_hard_constraint)} "
            f"waypoints={len(publish_path)} max_task_error={max_task_error:.6f}"
        )
        _print_base_path_summary("[ARM_DOOR][OPEN][REAL][BASE][WBC-QP]", publish_base_poses)

    if bool(getattr(args, "door_open_topp", True)):
        (
            publish_path,
            publish_base_poses,
            publish_arm_velocities,
            publish_arm_accelerations,
            publish_times,
        ) = _retime_opening_arm_base_path(args, publish_path, publish_base_poses)
        _print_base_path_summary("[ARM_DOOR][OPEN][REAL][BASE][TOPP]", publish_base_poses)
    elif bool(args.door_open_base_publish_interp):
        old_count = len(publish_path)
        publish_path, publish_base_poses = _densify_opening_publish_samples(
            publish_path,
            publish_base_poses,
            max_base_step_m=float(args.door_open_base_publish_step_m),
            max_base_step_yaw_rad=math.radians(float(args.door_open_base_publish_step_deg)),
        )
        if len(publish_path) != old_count:
            print(
                "[ARM_DOOR][OPEN][REAL][BASE] "
                f"densified publish samples {old_count}->{len(publish_path)} "
                f"(max_step={float(args.door_open_base_publish_step_m):.3f}m, "
                f"{float(args.door_open_base_publish_step_deg):.1f}deg)"
            )
        _print_base_path_summary("[ARM_DOOR][OPEN][REAL][BASE][INTERP]", publish_base_poses)

    dt = max(1.0e-3, float(args.door_open_dt))
    arm_topic = str(args.real_left_topic if arm == "left" else args.real_right_topic)
    base_topic = str(args.real_base_cmd_vel_topic)

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("arm_door_real_opening_publisher")
    arm_pub = node.create_publisher(
        JointTrajectory,
        arm_topic,
        _command_qos(
            reliability=str(getattr(args, "publish_reliability", "reliable")),
            durability=(
                "transient_local"
                if bool(getattr(args, "publish_transient_local", False))
                else str(getattr(args, "publish_durability", "volatile"))
            ),
            depth=int(getattr(args, "publish_qos_depth", 10)),
        ),
    )
    base_pub = node.create_publisher(Twist, base_topic, 10)

    wait_forever = float(args.publish_wait_subscriber_s) < 0.0
    t_end = None if wait_forever else time.monotonic() + max(0.0, float(args.publish_wait_subscriber_s))
    while rclpy.ok():
        arm_ok = arm_pub.get_subscription_count() > 0
        base_ok = base_pub.get_subscription_count() > 0
        if arm_ok and base_ok:
            break
        if t_end is not None and time.monotonic() >= t_end:
            break
        rclpy.spin_once(node, timeout_sec=0.05)

    missing = []
    if arm_pub.get_subscription_count() == 0:
        missing.append(arm_topic)
    if base_pub.get_subscription_count() == 0:
        missing.append(base_topic)
    if missing:
        msg = f"No subscribers detected for real opening publisher(s): {', '.join(missing)}"
        if bool(args.publish_require_subscriber):
            node.destroy_node()
            if owns_rclpy:
                rclpy.shutdown()
            raise RuntimeError(msg)
        node.get_logger().warning(msg + "; publishing anyway.")

    if publish_times:
        traj_msg = _build_timed_joint_trajectory(
            joint_names=publish_joint_names,
            positions=publish_path,
            velocities=publish_arm_velocities,
            accelerations=publish_arm_accelerations,
            times=publish_times,
        )
    else:
        traj_msg = _build_joint_trajectory(
            publish_path,
            publish_joint_names,
            dt=dt,
        )
    repeats = max(1, int(args.publish_repeat))
    duration_s = publish_times[-1] if publish_times else (len(publish_path) - 1) * dt
    print(
        "[ARM_DOOR][OPEN][REAL] "
        f"JointTrajectory -> {arm_topic}, base Twist -> {base_topic} "
        f"(dt={dt:.3f}s, waypoints={len(publish_path)}, duration={duration_s:.3f}s)"
    )
    start_delay_s = max(0.0, float(getattr(args, "start_delay_s", 0.2)))
    baseline = None
    if bool(getattr(args, "door_open_arm_republish_if_static", True)):
        baseline = _read_joint_sample_with_node(
            node,
            publish_joint_names,
            topic=str(args.joint_state_topic),
            wait_s=max(0.05, float(args.joint_state_wait_s)),
        )
        if baseline is None:
            print(
                "[ARM_DOOR][OPEN][REAL][ARM][WARN] "
                f"could not read baseline joint state on {args.joint_state_topic}; "
                "static-motion republish check disabled"
            )

    attempts = max(1, int(getattr(args, "door_open_arm_republish_attempts", 2)))
    motion_ok = baseline is None
    last_delta = 0.0
    for attempt in range(1, attempts + 1):
        for i in range(repeats):
            traj_msg.header.stamp = _command_stamp(
                node,
                delay_s=start_delay_s,
                zero_stamp=bool(getattr(args, "real_zero_trajectory_stamp", True)),
            )
            arm_pub.publish(traj_msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            if i + 1 < repeats:
                time.sleep(max(0.0, float(args.publish_period_s)))

        if baseline is None:
            break
        if start_delay_s > 0.0:
            time.sleep(start_delay_s)
        motion_ok, last_delta = _wait_for_joint_motion_with_node(
            node,
            publish_joint_names,
            baseline,
            topic=str(args.joint_state_topic),
            wait_s=max(0.0, float(args.door_open_arm_motion_check_s)),
            threshold_rad=max(0.0, float(args.door_open_arm_motion_threshold_rad)),
        )
        print(
            "[ARM_DOOR][OPEN][REAL][ARM] "
            f"publish_attempt={attempt}/{attempts} "
            f"motion_ok={motion_ok} max_delta={last_delta:.5f}rad"
        )
        if motion_ok:
            break
        if attempt < attempts:
            print("[ARM_DOOR][OPEN][REAL][ARM][WARN] arm still static; republishing trajectory")

    if baseline is not None and not motion_ok:
        print(
            "[ARM_DOOR][OPEN][REAL][ARM][WARN] "
            f"arm did not exceed motion threshold after {attempts} publish attempt(s); "
            "continuing with base command stream"
        )
    elif baseline is None and start_delay_s > 0.0:
        time.sleep(start_delay_s)

    max_linear = float(args.real_base_max_linear_mps)
    max_angular = float(args.real_base_max_angular_rps)
    cmd_period = 1.0 / max(1.0, float(args.real_base_cmd_rate_hz))
    clamped_count = 0
    for idx in range(1, len(publish_base_poses)):
        segment_dt = dt
        if publish_times:
            segment_dt = max(1.0e-3, float(publish_times[idx]) - float(publish_times[idx - 1]))
        vx, vy, wz, clamped = _base_pose_delta_to_body_twist(
            publish_base_poses[idx - 1],
            publish_base_poses[idx],
            dt=segment_dt,
            max_linear_mps=max_linear,
            max_angular_rps=max_angular,
        )
        clamped_count += 1 if clamped else 0
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        segment_end = time.monotonic() + segment_dt
        while rclpy.ok() and time.monotonic() < segment_end:
            base_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            remaining = segment_end - time.monotonic()
            if remaining > 0.0:
                time.sleep(min(cmd_period, remaining))

    stop_msg = Twist()
    stop_end = time.monotonic() + max(0.0, float(args.real_base_stop_duration_s))
    while rclpy.ok() and time.monotonic() < stop_end:
        base_pub.publish(stop_msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(min(0.05, max(0.001, dt)))

    if clamped_count:
        print(
            "[ARM_DOOR][OPEN][REAL][BASE][WARN] "
            f"clamped {clamped_count} cmd_vel segment(s); "
            f"limits=({max_linear:.3f}m/s, {max_angular:.3f}rad/s)"
        )

    keep_end = time.monotonic() + max(0.0, float(args.publish_keep_alive_s))
    while rclpy.ok() and time.monotonic() < keep_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()


def _publish_opening_path(
    args,
    arm: str,
    joint_names,
    path,
    door_alphas_rad,
    base_poses=None,
    desired_ee_poses=None,
) -> None:
    from capstone_pkg.planner.arm_rrt_common.path_publisher import publish_joint_path

    if args.publish_mode == "real":
        if base_poses is not None and bool(args.door_open_base_assist):
            _publish_real_opening_path_with_base(
                args,
                arm,
                joint_names,
                path,
                base_poses,
                desired_ee_poses=desired_ee_poses,
            )
            return
        _publish_real_path(args, arm, joint_names, path)
        return

    publish_joint_names = list(joint_names)
    publish_path = [[float(v) for v in q] for q in path]
    if bool(args.door_open_sync_hinge):
        if len(door_alphas_rad) != len(publish_path):
            raise RuntimeError("opening alpha path length does not match arm path length")
        publish_joint_names.append(str(args.door_hinge_joint))
        publish_path = [
            list(q) + [float(alpha)]
            for q, alpha in zip(publish_path, door_alphas_rad)
        ]

    if base_poses is not None and bool(args.door_open_base_assist):
        import rclpy
        from geometry_msgs.msg import Pose2D
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        if len(base_poses) != len(publish_path):
            raise RuntimeError("opening base pose path length does not match arm path length")
        publish_base_poses = [[float(v) for v in pose] for pose in base_poses]
        if bool(args.door_open_base_publish_interp):
            old_count = len(publish_path)
            publish_path, publish_base_poses = _densify_opening_publish_samples(
                publish_path,
                publish_base_poses,
                max_base_step_m=float(args.door_open_base_publish_step_m),
                max_base_step_yaw_rad=math.radians(float(args.door_open_base_publish_step_deg)),
            )
            if len(publish_path) != old_count:
                print(
                    "[ARM_DOOR][OPEN][BASE] "
                    f"densified publish samples {old_count}->{len(publish_path)} "
                    f"(max_step={float(args.door_open_base_publish_step_m):.3f}m, "
                    f"{float(args.door_open_base_publish_step_deg):.1f}deg)"
                )

        owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            owns_rclpy = True

        node = Node("arm_door_opening_base_joint_publisher")
        joint_pub = node.create_publisher(JointState, str(args.publish_topic), 10)
        base_pub = node.create_publisher(Pose2D, str(args.door_open_base_pose_topic), 10)

        t_end = time.monotonic() + max(0.0, float(args.publish_wait_subscriber_s))
        while rclpy.ok() and time.monotonic() < t_end:
            if joint_pub.get_subscription_count() > 0 and base_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(node, timeout_sec=0.02)

        print(
            f"[ARM_DOOR][OPEN] Publishing JointState+base opening path -> "
            f"{args.publish_topic}, {args.door_open_base_pose_topic} "
            f"(dt={float(args.door_open_dt):.3f}s, sync_hinge={bool(args.door_open_sync_hinge)})"
        )
        dt = max(0.0, float(args.door_open_dt))
        for q, base_pose in zip(publish_path, publish_base_poses):
            now = node.get_clock().now().to_msg()
            joint_msg = JointState()
            joint_msg.header.stamp = now
            joint_msg.name = [str(name) for name in publish_joint_names]
            joint_msg.position = [float(v) for v in q]
            joint_pub.publish(joint_msg)

            base_msg = Pose2D()
            base_msg.x = float(base_pose[0])
            base_msg.y = float(base_pose[1])
            base_msg.theta = float(base_pose[2])
            base_pub.publish(base_msg)

            rclpy.spin_once(node, timeout_sec=0.0)
            if dt > 0.0:
                time.sleep(dt)

        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        return

    print(
        f"[ARM_DOOR][OPEN] Publishing JointState opening path -> {args.publish_topic} "
        f"(dt={float(args.door_open_dt):.3f}s, sync_hinge={bool(args.door_open_sync_hinge)})"
    )
    publish_joint_path(
        publish_path,
        publish_joint_names,
        topic=str(args.publish_topic),
        dt=float(args.door_open_dt),
        wait_subscriber_s=float(args.publish_wait_subscriber_s),
    )


def _build_reverse_door_close_plan(
    *,
    arm: str,
    joint_names: Sequence[str],
    path: Sequence[Sequence[float]],
    door_alphas_rad: Sequence[float],
    base_poses: Sequence[Sequence[float]] | None,
    desired_ee_poses: Sequence[Any] | None,
    cspace_joint_names: Sequence[str],
    cspace_path: Sequence[Sequence[float]],
    world_yml: str | None,
) -> DoorClosePlan:
    if not cspace_path:
        raise RuntimeError("cannot build door close plan from empty cspace path")
    close_base_poses = (
        None
        if base_poses is None
        else [[float(v) for v in pose] for pose in reversed(base_poses)]
    )
    close_desired_ee_poses = (
        None
        if desired_ee_poses is None
        else [copy.deepcopy(pose) for pose in reversed(desired_ee_poses)]
    )
    return DoorClosePlan(
        arm=str(arm),
        joint_names=[str(name) for name in joint_names],
        path=[[float(v) for v in q] for q in reversed(path)],
        door_alphas_rad=[float(alpha) for alpha in reversed(door_alphas_rad)],
        base_poses=close_base_poses,
        desired_ee_poses=close_desired_ee_poses,
        close_end_cspace_joint_names=[str(name) for name in cspace_joint_names],
        close_end_q_cspace=[float(v) for v in cspace_path[0]],
        world_yml=None if world_yml is None else str(world_yml),
    )


def _run_arm_door_sequence(args: argparse.Namespace) -> int:
    from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
        build_active_joint_path,
        normalize_arm_name,
        plan_single_arm_motion,
    )
    from capstone_pkg.planner.arm_rrt_common.plot import (
        publish_joint_path_plot,
        save_ee_path_plot_3d_matplotlib,
        save_ee_path_plot_matplotlib,
    )
    from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
        _publish_world_collision_for_mujoco,
        _resolve_world_yml,
        build_single_arm_tbrrt_config,
    )
    from capstone_pkg.utils.config import LEFT_EE_FRAME, RIGHT_EE_FRAME

    _ = normalize_arm_name(args.arm)
    arm = "right"
    target_xyz = [float(v) for v in args.target_xyz]
    target_quat_xyzw = [float(v) for v in args.target_quat_xyzw]
    door_hinge_xyz = [
        float(v)
        for v in getattr(args, "_door_hinge_xyz", DOOR_HINGE_XYZ)
    ]
    handle_xyz = (
        [float(v) for v in args.handle_xyz]
        if args.handle_xyz is not None
        else list(target_xyz)
    )
    handle_quat_xyzw = (
        [float(v) for v in args.handle_quat_xyzw]
        if args.handle_quat_xyzw is not None
        else list(target_quat_xyzw)
    )

    resolved_world_yml = _resolve_world_yml(
        args,
        collision_models=_COLLISION_MODELS,
        default_world_yml=DOOR_COLLISION_YAML,
    )
    if resolved_world_yml is None:
        print("[ARM_DOOR] world collision disabled")
    else:
        print(f"[ARM_DOOR] world_yml={resolved_world_yml}")

    print(f"[ARM_DOOR] arm={arm}")
    print(f"[ARM_DOOR] door_model_xml={DOOR_MODEL_XML}")
    print(f"[ARM_DOOR] door_hinge_xyz={door_hinge_xyz}")
    print(f"[ARM_DOOR] target_xyz={target_xyz}")
    print(f"[ARM_DOOR] target_quat_xyzw={target_quat_xyzw}")
    print(f"[ARM_DOOR] handle_xyz={handle_xyz}")
    print(f"[ARM_DOOR] door_open_ee_radial_offset_m={float(args.door_open_handle_radial_offset_m):.3f}")
    print(f"[ARM_DOOR] handle_quat_xyzw={handle_quat_xyzw}")

    _publish_world_collision_for_mujoco(args, resolved_world_yml)

    try:
        plan = plan_single_arm_motion(
            robot_yml=str(args.robot_yml),
            arm=arm,
            target_xyz=target_xyz,
            target_quat_xyzw=target_quat_xyzw,
            world_yml=resolved_world_yml,
            cpu=bool(args.cpu),
            joint_state_topic=str(args.joint_state_topic),
            joint_state_wait_s=float(args.joint_state_wait_s),
            use_current_joint_state_start=bool(args.use_current_joint_state_start),
            step=float(args.step),
            max_iters=int(args.max_iters),
            goal_bias=float(args.goal_bias),
            connect_threshold=float(args.connect_threshold),
            planner_backend=str(args.planner_backend),
            joint_limit_yml=str(args.joint_limit_yml),
            ik_batch=int(args.ik_batch),
            ik_seed_noise_std=float(args.ik_seed_noise_std),
            ik_seed_random_seed=int(args.ik_seed),
            ik_goal_dedupe_tol=float(args.ik_goal_dedupe_tol),
            tbrrt_cfg=build_single_arm_tbrrt_config(args),
            tbrrt_block_k=int(args.tbrrt_block_k),
            spline_dt=float(args.publish_dt),
        )
    except Exception as exc:
        print(f"[ARM_DOOR][ERROR] planning failed: {exc}")
        return 1

    print(
        f"[ARM_DOOR] planned path: raw={len(plan.raw_path)} "
        f"spline={len(plan.spline_path)}"
    )
    print("[ARM_DOOR][AREA] a=0 approach/reach-to-handle completed")

    if args.save:
        with open(str(args.save), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "planner": "ARM_DOOR",
                    "arm": arm,
                    "target_xyz": target_xyz,
                    "target_quat_xyzw": target_quat_xyzw,
                    "handle_xyz": handle_xyz,
                    "handle_quat_xyzw": handle_quat_xyzw,
                    "world_yml": resolved_world_yml,
                    "cspace_joint_names": plan.cspace_joint_names,
                    "active_joint_names": plan.active_joint_names,
                    "q_start_cspace": plan.q_start_cspace,
                    "q_goal_cspace": plan.q_goal_cspace,
                    "path": plan.spline_path,
                },
                f,
                indent=2,
            )
        print(f"[ARM_DOOR] saved plan -> {args.save}")

    active_joint_names, active_path = build_active_joint_path(plan)
    plot_cspace_names = plan.cspace_joint_names
    plot_cspace_path = plan.spline_path
    plot_base_poses = None
    plot_stage = "reach"

    if args.publish_path:
        try:
            reach_publish_len = _publish_reach_path_with_left_zero(
                args,
                arm,
                active_joint_names,
                active_path,
                cspace_joint_names=plan.cspace_joint_names,
                q_start_cspace=plan.q_start_cspace,
                world_yml=resolved_world_yml,
            )
        except RuntimeError as exc:
            print(f"[ARM_DOOR][PUBLISH] {exc}")
            return 1
        print("[ARM_DOOR] publish done.")

        if args.close_gripper_after_path:
            if args.publish_mode == "real" and not args.real_use_action:
                traj_duration_s = max(
                    0.0,
                    float(reach_publish_len - 1) * float(args.publish_dt),
                )
                if traj_duration_s > 0.0:
                    print(
                        f"[ARM_DOOR][GRIPPER] Waiting for arm trajectory: "
                        f"{traj_duration_s:.2f}s"
                    )
                    time.sleep(traj_duration_s)
            if float(args.gripper_delay_s) > 0.0:
                print(
                    f"[ARM_DOOR][GRIPPER] Waiting additional delay: "
                    f"{float(args.gripper_delay_s):.2f}s"
                )
                time.sleep(float(args.gripper_delay_s))
            try:
                _publish_gripper_close(args, arm)
            except RuntimeError as exc:
                print(f"[ARM_DOOR][GRIPPER] {exc}")
                return 1
            print("[ARM_DOOR][GRIPPER] close done.")

            if args.open_door_after_grasp:
                print(
                    f"[ARM_DOOR][OPEN] building right-arm opening path "
                    f"to {float(args.door_open_angle_deg):.1f} deg "
                    f"({int(args.door_open_steps)} IK waypoint(s))"
                )
                try:
                    (
                        open_cspace_names,
                        open_active_names,
                        open_cspace_path,
                        open_alphas,
                        open_desired_ee_poses,
                        open_base_poses,
                    ) = (
                        _build_door_opening_cspace_path(
                            args,
                            arm=arm,
                            q_start_cspace=plan.q_goal_cspace,
                            grasp_ee_xyz=target_xyz,
                            grasp_ee_quat_xyzw=target_quat_xyzw,
                            closed_handle_xyz=handle_xyz,
                            closed_handle_quat_xyzw=handle_quat_xyzw,
                            door_hinge_xyz=door_hinge_xyz,
                            resolved_world_yml=resolved_world_yml,
                        )
                    )
                    if args.validate_opening_fk:
                        _validate_opening_fk(
                            args,
                            arm=arm,
                            cspace_joint_names=open_cspace_names,
                            cspace_path=open_cspace_path,
                            door_alphas_rad=open_alphas,
                            desired_ee_poses=open_desired_ee_poses,
                            base_poses=open_base_poses,
                            resolved_world_yml=resolved_world_yml,
                        )
                    if args.validate_opening_dynamic_collision:
                        _validate_opening_dynamic_collision(
                            args,
                            cspace_path=open_cspace_path,
                            door_alphas_rad=open_alphas,
                            door_hinge_xyz=door_hinge_xyz,
                            base_poses=open_base_poses,
                            resolved_world_yml=resolved_world_yml,
                        )
                    open_joint_names, open_path = _active_path_from_cspace(
                        open_cspace_names,
                        open_active_names,
                        open_cspace_path,
                    )
                    publish_open_path = open_path
                    publish_open_alphas = open_alphas
                    publish_open_base_poses = open_base_poses
                    if bool(args.door_open_base_assist) and open_base_poses:
                        if args.publish_mode == "real":
                            if bool(getattr(args, "post_open_base_enable", True)) and abs(
                                float(getattr(args, "post_open_base_forward_m", 0.0))
                            ) > 1.0e-9:
                                print(
                                    "[ARM_DOOR][OPEN][POST_BASE] skipped in real publish mode "
                                    "(release/regrasp handling is not implemented)"
                                )
                        else:
                            (
                                publish_open_path,
                                publish_open_alphas,
                                publish_open_base_poses,
                            ) = _append_post_open_base_forward_publish_samples(
                                args,
                                open_path,
                                open_alphas,
                                open_base_poses,
                            )
                    plot_cspace_names = open_cspace_names
                    plot_cspace_path = open_cspace_path
                    plot_base_poses = open_base_poses if open_base_poses else None
                    plot_stage = "opening"
                    if bool(args.door_open_base_assist) and publish_open_base_poses:
                        print(
                            "[ARM_DOOR][OPEN][BASE] "
                            f"final_pose_xyyaw={['%.3f' % v for v in publish_open_base_poses[-1]]}"
                        )
                        _print_base_path_summary(
                            "[ARM_DOOR][OPEN][BASE][PLANNED]",
                            publish_open_base_poses,
                        )
                except Exception as exc:
                    print(f"[ARM_DOOR][OPEN][ERROR] {exc}")
                    return 1

                print(f"[ARM_DOOR][OPEN] unlock topic -> {args.door_unlock_topic}")
                _publish_door_unlock(str(args.door_unlock_topic))
                try:
                    _publish_opening_path(
                        args,
                        arm,
                        open_joint_names,
                        publish_open_path,
                        publish_open_alphas,
                        publish_open_base_poses,
                        desired_ee_poses=open_desired_ee_poses,
                    )
                except RuntimeError as exc:
                    print(f"[ARM_DOOR][OPEN] {exc}")
                    return 1
                try:
                    _run_post_open_release_retreat_zero_ingress(
                        args,
                        cspace_joint_names=open_cspace_names,
                        q_cspace_end=open_cspace_path[-1],
                    )
                except RuntimeError as exc:
                    print(f"[ARM_DOOR][POST_OPEN][ERROR] {exc}")
                    return 1
                if bool(getattr(args, "post_open_release_ingress", True)):
                    args._door_close_plan = None
                    print(
                        "[ARM_DOOR][CLOSE] skipped reverse opening cache "
                        "because post-open release/ingress changes the robot state"
                    )
                else:
                    args._door_close_plan = _build_reverse_door_close_plan(
                        arm=arm,
                        joint_names=open_joint_names,
                        path=publish_open_path,
                        door_alphas_rad=publish_open_alphas,
                        base_poses=publish_open_base_poses,
                        desired_ee_poses=open_desired_ee_poses,
                        cspace_joint_names=open_cspace_names,
                        cspace_path=open_cspace_path,
                        world_yml=resolved_world_yml,
                    )
                    print("[ARM_DOOR][CLOSE] cached reverse opening path for post-picking door close")
                print("[ARM_DOOR][OPEN] done.")

    if args.plot_path:
        print(f"[ARM_DOOR] Publishing joint plot -> {args.plot_topic}")
        publish_joint_path_plot(
            active_path,
            active_joint_names,
            topic=str(args.plot_topic),
            frame_id=str(args.plot_frame),
            x_step=float(args.plot_x_step),
            y_scale=float(args.plot_y_scale),
            z_separation=float(args.plot_z_sep),
            marker_lifetime_s=float(args.plot_lifetime),
            keep_alive_s=float(args.plot_keep_alive),
        )

    if args.plot:
        ee_frame = LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME
        try:
            out_png = save_ee_path_plot_matplotlib(
                plot_cspace_path,
                plot_cspace_names,
                ee_frames=[(f"{arm} EE", ee_frame)],
                robot_yml=str(args.robot_yml),
                world_yml=resolved_world_yml,
                out_png=str(args.plot_output),
                prefix=f"arm_door_{arm}_{plot_stage}_path_ee",
                title=f"ARM_DOOR {arm} {plot_stage} End-Effector/Base Path",
                cpu=bool(args.cpu),
                base_poses=plot_base_poses,
            )
            print(f"[ARM_DOOR][PLOT] saved: {out_png}")
            try:
                out_png_3d = save_ee_path_plot_3d_matplotlib(
                    plot_cspace_path,
                    plot_cspace_names,
                    ee_frames=[(f"{arm} EE", ee_frame)],
                    robot_yml=str(args.robot_yml),
                    world_yml=resolved_world_yml,
                    out_png=str(args.plot_output),
                    prefix=f"arm_door_{arm}_{plot_stage}_path_ee_3d",
                    title=f"ARM_DOOR {arm} {plot_stage} End-Effector/Base Path 3D",
                    cpu=bool(args.cpu),
                    base_poses=plot_base_poses,
                )
                print(f"[ARM_DOOR][PLOT] saved 3D: {out_png_3d}")
            except Exception as exc:
                print(f"[ARM_DOOR][PLOT] 3D-only plot failed: {exc}")
        except Exception as exc:
            print(f"[ARM_DOOR][PLOT] task-space EE plot failed: {exc}")
            return 1

    return 0


class ArmDoorSequenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_door")
        from grasp_msgs.msg import ObjectAlign

        self._args = args
        self._armed = False
        self._active = False
        self._align_rx_seq = 0
        self._activation_align_seq = 0
        self._waiting_one_step_finish = False
        self._closing_after_one_step = False
        self._door_close_plan: DoorClosePlan | None = None

        self._run_startup_warmup()

        self._align_qos = _command_qos(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._start_qos = _command_qos(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self._finish_pub = self.create_publisher(
            Bool,
            str(args.arm_door_finish_topic),
            self._align_qos,
        )
        self._arm_picking_finish_pub = self.create_publisher(
            Bool,
            str(args.arm_picking_finish_topic),
            self._align_qos,
        )
        self._start_sub = self.create_subscription(
            Bool,
            str(args.arm_door_start_topic),
            self._on_start,
            self._start_qos,
        )
        self._align_sub = self.create_subscription(
            ObjectAlign,
            str(args.door_handle_align_topic),
            self._on_align,
            self._align_qos,
        )
        self._one_step_finish_sub = self.create_subscription(
            Bool,
            str(args.one_step_finish_topic),
            self._on_one_step_finish,
            self._start_qos,
        )
        self.get_logger().info(
            "ARM_DOOR ready: "
            f"start_sub={args.arm_door_start_topic}, "
            f"align_sub={args.door_handle_align_topic}, "
            f"finish_pub={args.arm_door_finish_topic}, "
            f"arm_picking_finish_pub={args.arm_picking_finish_topic}, "
            f"one_step_finish_sub={args.one_step_finish_topic}"
        )

    def _iter_startup_world_ymls(self) -> list[str | None]:
        from capstone_pkg.planner.arm_rrt_common.single_arm_runner import _resolve_world_yml

        candidates: list[str | None] = [
            _resolve_world_yml(
                self._args,
                collision_models=_COLLISION_MODELS,
                default_world_yml=DOOR_COLLISION_YAML,
            ),
            DOOR_COLLISION_YAML,
            None,
        ]
        out: list[str | None] = []
        seen: set[str] = set()
        for item in candidates:
            normalized = None if item in (None, "", "none", "None") else str(item)
            key = "<none>" if normalized is None else normalized
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out

    def _run_startup_warmup(self) -> None:
        if not bool(getattr(self._args, "startup_warmup", True)):
            self.get_logger().info("Startup warmup disabled")
            return

        warmup_iters = max(0, int(getattr(self._args, "startup_warmup_iters", 1)))
        configured_batch_size = getattr(self._args, "startup_warmup_batch_size", None)
        if configured_batch_size is None:
            configured_batch_size = getattr(self._args, "ik_batch", 100)
        warmup_batch_size = max(1, int(configured_batch_size))
        if warmup_iters <= 0:
            self.get_logger().info("Startup warmup skipped because startup_warmup_iters <= 0")
            return

        from capstone_pkg.collision_check.collision import get_self_collision_checker
        from capstone_pkg.kinematics.curobo_ik import (
            get_single_arm_ik,
            warmup_single_arm_ik_reachable,
        )

        worlds = self._iter_startup_world_ymls()
        t0 = time.monotonic()
        self.get_logger().info(
            "Starting ARM_DOOR warmup "
            f"(iters={warmup_iters}, batch_size={warmup_batch_size}, worlds={len(worlds)})"
        )
        for world_index, world_yml in enumerate(worlds):
            world_label = str(world_yml) if world_yml is not None else "none"
            self.get_logger().info(f"[warmup] world={world_label}")
            try:
                checker = get_self_collision_checker(
                    str(self._args.robot_yml),
                    cpu=bool(self._args.cpu),
                    world_yml=world_yml,
                )
                _ = checker.check_single([0.0 for _ in checker.cspace_names])
            except Exception as exc:
                self.get_logger().warning(f"[warmup] collision checker warmup skipped: {exc}")

            ik = get_single_arm_ik(
                str(self._args.robot_yml),
                arm="right",
                cpu=bool(self._args.cpu),
                world_yml=world_yml,
            )
            warmup_single_arm_ik_reachable(
                ik,
                iters=warmup_iters,
                batch_size=warmup_batch_size,
                noise_std=float(getattr(self._args, "ik_seed_noise_std", 0.25)),
                random_seed=int(getattr(self._args, "ik_seed", 0)) + world_index * 17,
            )
            self.get_logger().info(f"[warmup] ready arm=right world={world_label}")
        self.get_logger().info(
            f"ARM_DOOR warmup completed in {time.monotonic() - t0:.2f}s"
        )

    def _on_start(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        if self._active:
            self.get_logger().warning("Ignoring arm_door_start: sequence already active")
            return
        if self._waiting_one_step_finish or self._closing_after_one_step:
            self.get_logger().warning("Ignoring arm_door_start: waiting for post-door close")
            return
        if self._armed:
            self.get_logger().warning("Ignoring arm_door_start: already waiting for align result")
            return
        self._armed = True
        self._activation_align_seq = self._align_rx_seq
        self.get_logger().info(
            "Accepted arm_door_start; waiting for a new /door_handle_align_result"
        )

    def _on_align(self, msg) -> None:
        self._align_rx_seq += 1
        seq = self._align_rx_seq
        if not self._armed:
            return
        if seq <= self._activation_align_seq:
            return
        if self._active:
            self.get_logger().warning("Ignoring door_handle_align_result: sequence already active")
            return
        if not self._is_expected_align_msg(msg):
            return
        self._armed = False
        self._active = True
        self.get_logger().info("Accepted door_handle_align_result; starting ARM_DOOR sequence")
        try:
            result = self._process_align(msg)
            if result == 0:
                self._waiting_one_step_finish = (
                    bool(self._args.post_one_step_door_close)
                    and not bool(getattr(self._args, "post_open_release_ingress", True))
                )
                self._publish_finish()
                if self._waiting_one_step_finish:
                    if self._door_close_plan is None:
                        distance_m = max(0.0, float(self._args.post_one_step_reverse_distance_m))
                        self.get_logger().warning(
                            "Waiting for one_step_finish, but no reverse door close plan is cached; "
                            f"fallback will reverse base only for {distance_m:.3f}m"
                        )
                    else:
                        base_count = 0 if self._door_close_plan.base_poses is None else len(self._door_close_plan.base_poses)
                        self.get_logger().info(
                            "Waiting for one_step_finish before executing reverse door close path: "
                            f"arm_waypoints={len(self._door_close_plan.path)} base_waypoints={base_count}"
                        )
            else:
                self.get_logger().error(f"ARM_DOOR sequence failed with code={result}")
        except Exception as exc:
            self.get_logger().error(f"ARM_DOOR sequence failed: {exc}")
        finally:
            self._active = False

    def _on_one_step_finish(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        if not bool(getattr(self._args, "post_one_step_door_close", True)):
            return
        if not self._waiting_one_step_finish:
            return
        if self._closing_after_one_step:
            self.get_logger().warning("Ignoring one_step_finish: post-door close already active")
            return

        self._waiting_one_step_finish = False
        self._closing_after_one_step = True
        try:
            try:
                if self._door_close_plan is not None:
                    self._publish_post_one_step_door_close()
                else:
                    self._publish_post_one_step_reverse()
            except Exception as exc:
                self.get_logger().error(f"Post one_step door close failed: {exc}")
        finally:
            self._closing_after_one_step = False

    def _publish_post_one_step_door_close(self) -> None:
        plan = self._door_close_plan
        if plan is None:
            self._publish_post_one_step_reverse()
            return
        base_count = 0 if plan.base_poses is None else len(plan.base_poses)
        self.get_logger().info(
            "Received one_step_finish=True; executing reverse door close path "
            f"(arm_waypoints={len(plan.path)}, base_waypoints={base_count})"
        )
        _publish_opening_path(
            self._args,
            plan.arm,
            plan.joint_names,
            plan.path,
            plan.door_alphas_rad,
            plan.base_poses,
            desired_ee_poses=plan.desired_ee_poses,
        )
        self.get_logger().info("Post one_step reverse door close path complete")
        self._publish_post_close_release_and_reset(plan)
        self._door_close_plan = None
        _publish_post_one_step_base_reset(self._args)
        self.get_logger().info("Post one_step base yaw alignment/reverse complete")
        self._publish_arm_picking_finish()

    def _publish_post_close_release_and_reset(self, plan: DoorClosePlan | None = None) -> None:
        self.get_logger().info("Opening right gripper after door close")
        _publish_gripper_open(self._args, "right")
        self.get_logger().info("Planning and publishing right arm zero path after gripper open")
        if plan is None:
            _plan_and_publish_right_arm_zero(self._args)
        else:
            _plan_and_publish_right_arm_zero(
                self._args,
                start_cspace_joint_names=plan.close_end_cspace_joint_names,
                fallback_start_q_cspace=plan.close_end_q_cspace,
                world_yml=plan.world_yml,
            )
        self.get_logger().info("Right gripper open and right arm zero command complete")

    def _publish_post_one_step_reverse(self) -> None:
        self.get_logger().info(
            "Received one_step_finish=True; executing base yaw alignment/reverse fallback"
        )
        _publish_post_one_step_base_reset(self._args)
        self.get_logger().warning(
            "Post one_step base-only yaw alignment/reverse fallback complete"
        )
        self._publish_post_close_release_and_reset()
        self._publish_arm_picking_finish()

    def _publish_arm_picking_finish(self) -> None:
        msg = Bool()
        msg.data = True
        repeat = max(1, int(self._args.finish_publish_repeat))
        period_s = max(0.0, float(self._args.finish_publish_period_s))
        for idx in range(repeat):
            self._arm_picking_finish_pub.publish(msg)
            if idx + 1 < repeat and period_s > 0.0:
                time.sleep(period_s)
        self.get_logger().info(
            f"Published arm_picking_finish=True on {self._args.arm_picking_finish_topic} ({repeat}x)"
        )

    def _is_expected_align_msg(self, msg) -> bool:
        selected_arm = str(getattr(msg, "selected_arm", "")).strip().lower()
        if selected_arm and selected_arm != "right":
            self.get_logger().info(
                f"door_handle_align_result selected_arm={selected_arm!r}; forcing right arm"
            )
        shelf_type = str(getattr(msg, "shelf_type", "")).strip()
        if shelf_type and shelf_type != "shelf_1":
            self.get_logger().warning(
                f"Ignoring door_handle_align_result: shelf_type={shelf_type!r}"
            )
            return False
        return True

    def _process_align(self, msg) -> int:
        handle_center_xyz = self._extract_handle_center_xyz(msg)
        target_xyz, target_quat_xyzw = self._extract_target_pose(msg, handle_center_xyz)
        source_world_yml = DOOR_COLLISION_YAML
        shifted_world_yml, shifted_hinge_xyz, delta = _make_shifted_door_world_yaml(
            source_world_yml=source_world_yml,
            detected_handle_center_xyz=handle_center_xyz,
        )

        seq_args = copy.copy(self._args)
        seq_args.arm = "right"
        seq_args.world_yml = shifted_world_yml
        seq_args.collision_model = None
        seq_args.target_xyz = list(target_xyz)
        seq_args.target_quat_xyzw = list(target_quat_xyzw)
        seq_args.handle_xyz = list(handle_center_xyz)
        seq_args.handle_quat_xyzw = list(target_quat_xyzw)
        seq_args._door_hinge_xyz = list(shifted_hinge_xyz)

        self.get_logger().info(
            "Door handle target resolved: "
            f"handle_center={[round(v, 4) for v in handle_center_xyz]}, "
            f"target_xyz={[round(v, 4) for v in target_xyz]}, "
            f"world_delta={[round(v, 4) for v in delta]}, "
            f"world_yml={shifted_world_yml}"
        )
        result = _run_arm_door_sequence(seq_args)
        self._door_close_plan = (
            getattr(seq_args, "_door_close_plan", None)
            if result == 0
            else None
        )
        return result

    def _extract_handle_center_xyz(self, msg) -> list[float]:
        marker_position = getattr(msg, "marker_position", None)
        if marker_position is None:
            raise RuntimeError("ObjectAlign message has no marker_position")
        return [
            float(marker_position.x),
            float(marker_position.y),
            float(getattr(self._args, "detected_handle_target_z_m", DETECTED_HANDLE_TARGET_Z_M)),
        ]

    def _extract_target_pose(
        self,
        msg,
        handle_center_xyz: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        align_pose = getattr(msg, "align_pose", None)
        fixed_z = float(getattr(self._args, "detected_handle_target_z_m", DETECTED_HANDLE_TARGET_Z_M))
        target_xyz = [
            float(handle_center_xyz[0]) + float(self._args.detected_handle_target_x_offset_m),
            float(handle_center_xyz[1]),
            fixed_z,
        ]
        if align_pose is not None:
            pose = getattr(align_pose, "pose", align_pose)
            target_quat_xyzw = [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
            norm = math.sqrt(sum(v * v for v in target_quat_xyzw))
            if norm > 1.0e-9:
                return target_xyz, [v / norm for v in target_quat_xyzw]

        return target_xyz, list(HANDLE_QUAT_XYZW)

    def _publish_finish(self) -> None:
        msg = Bool()
        msg.data = True
        repeat = max(1, int(self._args.finish_publish_repeat))
        period_s = max(0.0, float(self._args.finish_publish_period_s))
        for idx in range(repeat):
            self._finish_pub.publish(msg)
            if idx + 1 < repeat and period_s > 0.0:
                time.sleep(period_s)
        self.get_logger().info(f"Published arm_door_finish=True ({repeat}x)")


def main_arm_door(argv: Sequence[str] | None = None) -> int:
    args = _build_arm_door_parser().parse_args(list(argv) if argv is not None else None)
    if bool(args.oneshot):
        return _run_arm_door_sequence(args)

    rclpy.init(args=None)
    node = ArmDoorSequenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_door(argv)


if __name__ == "__main__":
    raise SystemExit(main())

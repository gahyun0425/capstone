#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch

from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.constraint_projection.constraint import RigidConstraint
from capstone_pkg.constraint_projection.projection import ManifoldProjector
from capstone_pkg.kinematics.curobo_ik import SingleArmIK
from capstone_pkg.planner.ARM_CART.precompute import (
    _build_curobo_seed_batch,
    _joint_distance_metrics,
)
from capstone_pkg.planner.start_goal import (
    _auto_right_target_from_left,
    euler_rpy_deg_to_quat_wxyz,
)
from capstone_pkg.planner.tbrrt import TBRRTConfig
from capstone_pkg.planner.tbrrt.batch import plan_tbrrt_extcon_batch_conext
from capstone_pkg.utils.config import (
    CSPACE_JOINT_NAMES_14,
    JOINT_LIMIT,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    ROBOT_YAML,
)
from capstone_pkg.utils.joint_limit import load_joint_limits_torch


LEFT_ARM_Q_START = [
    0.4795008226882218,
    0.003163835375014135,
    -0.09580189389342043,
    -1.7581337305154305,
    1.5559958090367811,
    -0.09468736097722227,
    -0.3385783608491044,
]

RIGHT_ARM_Q_START = [
    0.5013840173654028,
    -0.0031758195999194916,
    0.09580189389342043,
    -1.7581576989652412,
    -1.556055730161308,
    -0.0947592663266544,
    0.3384542485467404,
]

Q_START_CSPACE = LEFT_ARM_Q_START + RIGHT_ARM_Q_START


@dataclass
class TargetPosePair:
    left_xyz: list[float]
    left_quat_wxyz: list[float]
    right_xyz: list[float]
    right_quat_wxyz: list[float]


@dataclass
class IKSolution:
    success: bool
    q_cspace: Optional[list[float]]
    score: float = float("-inf")
    continuity_max_abs: float = math.inf
    continuity_l2: float = math.inf
    seed_max_abs: float = math.inf
    seed_l2: float = math.inf
    seed_label: str = ""
    tried_candidates: int = 0
    valid_candidates: int = 0
    time_sec: float = 0.0


@dataclass
class BenchmarkRun:
    mode: str
    index: int
    ik_success: bool
    plan_success: bool
    ik_time_sec: float
    planning_time_sec: Optional[float]
    planner_stats_time_sec: Optional[float]
    path_len: Optional[int]
    score: Optional[float]
    extra: dict[str, Any]


@dataclass
class BenchmarkSummary:
    mode: str
    requested_runs: int
    ik_successes: int
    plan_calls: int
    plan_successes: int
    avg_planning_time_sec_all_calls: Optional[float]
    avg_planning_time_sec_success_only: Optional[float]


class BenchmarkContext:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.dtype = torch.float32
        self.world_yml = _normalize_optional_path(args.world_yml)

        self.checker = get_self_collision_checker(
            str(args.robot_yml),
            cpu=(device.type == "cpu"),
            world_yml=self.world_yml,
        )
        self.joint_limits = load_joint_limits_torch(
            str(args.joint_limit_yml),
            device=device,
            dtype=self.dtype,
        )
        self.left_ik = SingleArmIK(
            str(args.robot_yml),
            arm="left",
            cpu=(device.type == "cpu"),
            num_seeds=int(args.ik_num_seeds),
            world_yml=self.world_yml,
        )
        self.right_ik = SingleArmIK(
            str(args.robot_yml),
            arm="right",
            cpu=(device.type == "cpu"),
            num_seeds=int(args.ik_num_seeds),
            world_yml=self.world_yml,
        )

        q_ref = torch.tensor(Q_START_CSPACE, device=device, dtype=self.dtype)
        lock_z = bool(args.lock_z or args.planar_xy)
        self.constraint = RigidConstraint(
            robot_yml=str(args.robot_yml),
            left_ee=str(args.left_ee),
            right_ee=str(args.right_ee),
            q_ref=q_ref,
            device=device,
            dtype=self.dtype,
            mode="se3",
            rigid_orientation=bool(args.rigid_orientation),
            lock_z=lock_z,
            planar_xy=bool(args.planar_xy),
        )
        self.projector = ManifoldProjector(
            constraint=self.constraint,
            limits=self.joint_limits,
            max_iters=int(args.proj_iters),
            tol=float(args.proj_tol),
            fd_eps=float(args.proj_fd_eps),
        )


def _normalize_optional_path(value: object) -> Optional[str]:
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_target_from_args(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    if args.target is not None:
        values = [float(v) for v in args.target]
        return values[:3], values[3:]

    if args.target_left_xyz is not None or args.target_left_rpy_deg is not None:
        if args.target_left_xyz is None or args.target_left_rpy_deg is None:
            raise ValueError(
                "--target_left_xyz and --target_left_rpy_deg must be used together"
            )
        return (
            [float(v) for v in args.target_left_xyz],
            [float(v) for v in args.target_left_rpy_deg],
        )

    while True:
        raw = input("target x y z r p y (rpy deg): ").strip()
        toks = [tok for tok in raw.replace(",", " ").split() if tok]
        if len(toks) != 6:
            print("Please enter exactly 6 numbers.")
            continue
        values = [float(tok) for tok in toks]
        return values[:3], values[3:]


def _build_targets(
    *,
    args: argparse.Namespace,
    device: torch.device,
    left_xyz: Sequence[float],
    left_rpy_deg: Sequence[float],
) -> TargetPosePair:
    left_xyz_list = [float(v) for v in left_xyz]
    left_quat = euler_rpy_deg_to_quat_wxyz(
        float(left_rpy_deg[0]),
        float(left_rpy_deg[1]),
        float(left_rpy_deg[2]),
    )
    right_xyz, right_quat = _auto_right_target_from_left(
        robot_yml=str(args.robot_yml),
        left_ee=str(args.left_ee),
        right_ee=str(args.right_ee),
        q_start_cspace=list(Q_START_CSPACE),
        left_pos=left_xyz_list,
        left_quat_wxyz=left_quat,
        device=device,
    )
    return TargetPosePair(
        left_xyz=left_xyz_list,
        left_quat_wxyz=[float(v) for v in left_quat],
        right_xyz=[float(v) for v in right_xyz],
        right_quat_wxyz=[float(v) for v in right_quat],
    )


def _solve_plain_ik(
    ctx: BenchmarkContext,
    targets: TargetPosePair,
) -> IKSolution:
    t0 = time.perf_counter()
    left_out = ctx.left_ik.solve(
        xyz=targets.left_xyz,
        quat_wxyz=targets.left_quat_wxyz,
        q_start_cspace=list(Q_START_CSPACE),
    )
    if not left_out.success or left_out.q_cspace is None:
        return IKSolution(
            success=False,
            q_cspace=None,
            tried_candidates=1,
            time_sec=time.perf_counter() - t0,
        )

    right_out = ctx.right_ik.solve(
        xyz=targets.right_xyz,
        quat_wxyz=targets.right_quat_wxyz,
        q_start_cspace=list(left_out.q_cspace),
    )
    if not right_out.success or right_out.q_cspace is None:
        return IKSolution(
            success=False,
            q_cspace=None,
            tried_candidates=1,
            time_sec=time.perf_counter() - t0,
        )

    in_col, _, _ = ctx.checker.check_single(right_out.q_cspace)
    if in_col:
        return IKSolution(
            success=False,
            q_cspace=None,
            tried_candidates=1,
            time_sec=time.perf_counter() - t0,
        )

    _, l2 = _joint_distance_metrics(right_out.q_cspace, Q_START_CSPACE)
    return IKSolution(
        success=True,
        q_cspace=[float(v) for v in right_out.q_cspace],
        score=-float(l2),
        continuity_l2=float(l2),
        tried_candidates=1,
        valid_candidates=1,
        seed_label="q_start",
        time_sec=time.perf_counter() - t0,
    )


def _solve_scored_ik(
    ctx: BenchmarkContext,
    targets: TargetPosePair,
    *,
    run_index: int,
) -> IKSolution:
    args = ctx.args
    t0 = time.perf_counter()

    lower = ctx.joint_limits.lower.detach().cpu().numpy().astype(np.float64)
    upper = ctx.joint_limits.upper.detach().cpu().numpy().astype(np.float64)
    seed_batch = _build_curobo_seed_batch(
        anchor_q=Q_START_CSPACE,
        seed_candidates=[("q_start", Q_START_CSPACE)],
        num_trials=int(args.scored_ik_trials),
        noise_std=float(args.ik_seed_noise_std),
        random_seed=int(args.ik_seed) + int(run_index),
        lower=lower,
        upper=upper,
    )

    best: Optional[IKSolution] = None
    valid_candidates = 0
    for seed_label, q_seed in seed_batch:
        left_out = ctx.left_ik.solve(
            xyz=targets.left_xyz,
            quat_wxyz=targets.left_quat_wxyz,
            q_start_cspace=[float(v) for v in q_seed],
        )
        if not left_out.success or left_out.q_cspace is None:
            continue

        right_out = ctx.right_ik.solve(
            xyz=targets.right_xyz,
            quat_wxyz=targets.right_quat_wxyz,
            q_start_cspace=[float(v) for v in left_out.q_cspace],
        )
        if not right_out.success or right_out.q_cspace is None:
            continue

        in_col, _, _ = ctx.checker.check_single(right_out.q_cspace)
        if in_col:
            continue

        valid_candidates += 1
        continuity_max_abs, continuity_l2 = _joint_distance_metrics(
            right_out.q_cspace,
            Q_START_CSPACE,
        )
        seed_max_abs, seed_l2 = _joint_distance_metrics(
            right_out.q_cspace,
            q_seed,
        )
        candidate = IKSolution(
            success=True,
            q_cspace=[float(v) for v in right_out.q_cspace],
            score=-float(continuity_l2),
            continuity_max_abs=float(continuity_max_abs),
            continuity_l2=float(continuity_l2),
            seed_max_abs=float(seed_max_abs),
            seed_l2=float(seed_l2),
            seed_label=str(seed_label),
            tried_candidates=len(seed_batch),
            valid_candidates=valid_candidates,
        )

        cand_key = (
            candidate.continuity_max_abs,
            candidate.continuity_l2,
            candidate.seed_max_abs,
            candidate.seed_l2,
        )
        if best is None:
            best = candidate
        else:
            best_key = (
                best.continuity_max_abs,
                best.continuity_l2,
                best.seed_max_abs,
                best.seed_l2,
            )
            if cand_key < best_key:
                best = candidate

    elapsed = time.perf_counter() - t0
    if best is None:
        return IKSolution(
            success=False,
            q_cspace=None,
            tried_candidates=len(seed_batch),
            valid_candidates=0,
            time_sec=elapsed,
        )

    best.tried_candidates = len(seed_batch)
    best.valid_candidates = valid_candidates
    best.time_sec = elapsed
    return best


def _make_tbrrt_config(args: argparse.Namespace, run_index: int) -> TBRRTConfig:
    seed = None if int(args.seed) < 0 else int(args.seed) + int(run_index)
    return TBRRTConfig(
        step_size=float(args.step_size),
        goal_threshold=float(args.goal_threshold),
        EM=float(args.EM),
        E_conn=float(args.E_conn),
        ts_radius=float(args.ts_radius),
        p_uniform=float(args.p_uniform),
        goal_bias=float(args.goal_bias),
        connect_max_steps=int(args.connect_max_steps),
        escape_extend_steps=max(1, int(args.escape_extend_steps)),
        max_iters=int(args.max_iters),
        time_limit_sec=float(args.time_limit_sec),
        edge_step_q=float(args.edge_step_q),
        edge_max_steps=int(args.edge_max_steps),
        svd_tol=float(args.svd_tol),
        seed=seed,
    )


def _run_tbrrt_once(
    ctx: BenchmarkContext,
    *,
    mode: str,
    run_index: int,
    q_goal: Sequence[float],
) -> BenchmarkRun:
    cfg = _make_tbrrt_config(ctx.args, run_index)
    _sync_if_cuda(ctx.device)
    t0 = time.perf_counter()
    try:
        out = plan_tbrrt_extcon_batch_conext(
            q_start=list(Q_START_CSPACE),
            q_goals=[[float(v) for v in q_goal]],
            cfg=cfg,
            checker=ctx.checker,
            projector=ctx.projector,
            joint_limits=ctx.joint_limits,
            device=ctx.device,
        )
        _sync_if_cuda(ctx.device)
        planning_time = time.perf_counter() - t0
    except RuntimeError as exc:
        _sync_if_cuda(ctx.device)
        planning_time = time.perf_counter() - t0
        return BenchmarkRun(
            mode=mode,
            index=int(run_index),
            ik_success=True,
            plan_success=False,
            ik_time_sec=0.0,
            planning_time_sec=float(planning_time),
            planner_stats_time_sec=None,
            path_len=None,
            score=None,
            extra={
                "reason": "planner_exception",
                "exception": str(exc),
            },
        )

    path_len = len(out.path) if out.path is not None else None
    return BenchmarkRun(
        mode=mode,
        index=int(run_index),
        ik_success=True,
        plan_success=bool(out.success),
        ik_time_sec=0.0,
        planning_time_sec=float(planning_time),
        planner_stats_time_sec=float(out.stats.time_sec),
        path_len=path_len,
        score=None,
        extra=dict(out.stats.extra or {}),
    )


def _failed_ik_run(
    *,
    mode: str,
    run_index: int,
    ik: IKSolution,
) -> BenchmarkRun:
    return BenchmarkRun(
        mode=mode,
        index=int(run_index),
        ik_success=False,
        plan_success=False,
        ik_time_sec=float(ik.time_sec),
        planning_time_sec=None,
        planner_stats_time_sec=None,
        path_len=None,
        score=None,
        extra={
            "tried_candidates": int(ik.tried_candidates),
            "valid_candidates": int(ik.valid_candidates),
        },
    )


def _run_mode(
    ctx: BenchmarkContext,
    targets: TargetPosePair,
    *,
    mode: str,
) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    repeat = int(ctx.args.repeat)
    for run_index in range(repeat):
        if mode == "plain":
            ik = _solve_plain_ik(ctx, targets)
        elif mode == "scored":
            ik = _solve_scored_ik(ctx, targets, run_index=run_index)
        else:
            raise ValueError(f"unknown benchmark mode: {mode}")

        if not ik.success or ik.q_cspace is None:
            run = _failed_ik_run(mode=mode, run_index=run_index, ik=ik)
            runs.append(run)
            print(
                "[{}][{:02d}/{}] IK failed, ik_time={:.6f}s".format(
                    mode,
                    run_index + 1,
                    repeat,
                    float(ik.time_sec),
                )
            )
            continue

        run = _run_tbrrt_once(
            ctx,
            mode=mode,
            run_index=run_index,
            q_goal=ik.q_cspace,
        )
        run.ik_time_sec = float(ik.time_sec)
        run.score = float(ik.score)
        run.extra.update(
            {
                "ik_seed_label": str(ik.seed_label),
                "ik_tried_candidates": int(ik.tried_candidates),
                "ik_valid_candidates": int(ik.valid_candidates),
                "ik_continuity_l2": float(ik.continuity_l2),
                "ik_continuity_max_abs": float(ik.continuity_max_abs),
                "ik_seed_l2": float(ik.seed_l2),
                "ik_seed_max_abs": float(ik.seed_max_abs),
            }
        )
        runs.append(run)
        print(
            "[{}][{:02d}/{}] plan={} planning_time={:.6f}s "
            "stats_time={:.6f}s ik_time={:.6f}s score={:.6f} path_len={}".format(
                mode,
                run_index + 1,
                repeat,
                "success" if run.plan_success else "failed",
                float(run.planning_time_sec or 0.0),
                float(run.planner_stats_time_sec or 0.0),
                float(run.ik_time_sec),
                float(run.score or 0.0),
                run.path_len,
            )
        )

    return runs


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _summarize(mode: str, runs: Sequence[BenchmarkRun]) -> BenchmarkSummary:
    all_plan_times = [
        float(run.planning_time_sec)
        for run in runs
        if run.planning_time_sec is not None
    ]
    success_plan_times = [
        float(run.planning_time_sec)
        for run in runs
        if run.plan_success and run.planning_time_sec is not None
    ]
    return BenchmarkSummary(
        mode=mode,
        requested_runs=len(runs),
        ik_successes=sum(1 for run in runs if run.ik_success),
        plan_calls=len(all_plan_times),
        plan_successes=sum(1 for run in runs if run.plan_success),
        avg_planning_time_sec_all_calls=_mean_or_none(all_plan_times),
        avg_planning_time_sec_success_only=_mean_or_none(success_plan_times),
    )


def _print_summary(summary: BenchmarkSummary) -> None:
    avg_all = summary.avg_planning_time_sec_all_calls
    avg_success = summary.avg_planning_time_sec_success_only
    print("\n[{}] SUMMARY".format(summary.mode))
    print("  requested_runs={}".format(summary.requested_runs))
    print("  ik_successes={}".format(summary.ik_successes))
    print("  plan_calls={}".format(summary.plan_calls))
    print("  plan_successes={}".format(summary.plan_successes))
    print(
        "  avg_planning_time_all_calls={}".format(
            "n/a" if avg_all is None else "{:.6f}s".format(avg_all)
        )
    )
    print(
        "  avg_planning_time_success_only={}".format(
            "n/a" if avg_success is None else "{:.6f}s".format(avg_success)
        )
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run 30 TB-RRT benchmark trials with one plain cuRobo IK goal and "
            "one cart-style scored IK goal."
        )
    )
    ap.add_argument(
        "--target",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "R", "P", "YAW"),
        help="left EE target xyz(m) and rpy(deg)",
    )
    ap.add_argument("--target_left_xyz", type=float, nargs=3)
    ap.add_argument("--target_left_rpy_deg", type=float, nargs=3)
    ap.add_argument("--robot_yml", type=str, default=ROBOT_YAML)
    ap.add_argument("--world_yml", type=str, default=None)
    ap.add_argument("--joint_limit_yml", type=str, default=JOINT_LIMIT)
    ap.add_argument("--left_ee", type=str, default=LEFT_EE_FRAME)
    ap.add_argument("--right_ee", type=str, default=RIGHT_EE_FRAME)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--ik_num_seeds", type=int, default=20)
    ap.add_argument("--scored_ik_trials", type=int, default=24)
    ap.add_argument("--ik_seed_noise_std", type=float, default=0.25)
    ap.add_argument("--ik_seed", type=int, default=0)
    ap.add_argument("--rigid_orientation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lock_z", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--planar_xy", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--step_size", type=float, default=0.1)
    ap.add_argument("--goal_threshold", type=float, default=0.10)
    ap.add_argument("--EM", type=float, default=0.02)
    ap.add_argument("--E_conn", type=float, default=0.025)
    ap.add_argument("--ts_radius", type=float, default=1.1)
    ap.add_argument("--p_uniform", type=float, default=0.25)
    ap.add_argument("--goal_bias", type=float, default=0.15)
    ap.add_argument("--connect_max_steps", type=int, default=10)
    ap.add_argument("--escape_extend_steps", type=int, default=5)
    ap.add_argument("--max_iters", type=int, default=500000)
    ap.add_argument("--time_limit_sec", type=float, default=900.0)
    ap.add_argument("--proj_iters", type=int, default=60)
    ap.add_argument("--proj_tol", type=float, default=1e-3)
    ap.add_argument("--proj_fd_eps", type=float, default=1e-3)
    ap.add_argument("--edge_step_q", type=float, default=0.03)
    ap.add_argument("--edge_max_steps", type=int, default=128)
    ap.add_argument("--svd_tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--fail_if_start_in_collision", action="store_true")
    ap.add_argument("--output_json", type=str, default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be > 0")
    if int(args.scored_ik_trials) <= 0:
        raise ValueError("--scored_ik_trials must be > 0")

    requested_device = str(args.device)
    device = torch.device("cuda") if (
        requested_device == "cuda" and torch.cuda.is_available()
    ) else torch.device("cpu")

    left_xyz, left_rpy_deg = _parse_target_from_args(args)
    targets = _build_targets(
        args=args,
        device=device,
        left_xyz=left_xyz,
        left_rpy_deg=left_rpy_deg,
    )

    print("[benchmark] cspace_joint_names =", list(CSPACE_JOINT_NAMES_14))
    print("[benchmark] fixed q_start =", list(Q_START_CSPACE))
    print("[benchmark] device =", str(device))
    print("[benchmark] left_target_xyz =", targets.left_xyz)
    print("[benchmark] left_target_quat_wxyz =", targets.left_quat_wxyz)
    print("[benchmark] right_target_xyz =", targets.right_xyz)
    print("[benchmark] right_target_quat_wxyz =", targets.right_quat_wxyz)

    ctx = BenchmarkContext(args, device)
    in_col, d_self, d_world = ctx.checker.check_single(Q_START_CSPACE)
    print(
        "[benchmark] q_start collision: self={:.6f} world={:.6f} -> {}".format(
            float(d_self),
            float(d_world),
            "IN COLLISION" if in_col else "FREE",
        )
    )
    if in_col and bool(args.fail_if_start_in_collision):
        raise RuntimeError("fixed q_start is in collision")

    print("\n========== plain cuRobo IK + TB-RRT ==========")
    plain_runs = _run_mode(ctx, targets, mode="plain")
    plain_summary = _summarize("plain", plain_runs)
    _print_summary(plain_summary)

    print("\n========== scored IK + TB-RRT ==========")
    scored_runs = _run_mode(ctx, targets, mode="scored")
    scored_summary = _summarize("scored", scored_runs)
    _print_summary(scored_summary)

    if args.output_json:
        payload = {
            "target": {
                "left_xyz": targets.left_xyz,
                "left_rpy_deg": [float(v) for v in left_rpy_deg],
                "left_quat_wxyz": targets.left_quat_wxyz,
                "right_xyz": targets.right_xyz,
                "right_quat_wxyz": targets.right_quat_wxyz,
            },
            "q_start": list(Q_START_CSPACE),
            "summaries": {
                "plain": asdict(plain_summary),
                "scored": asdict(scored_summary),
            },
            "runs": {
                "plain": [asdict(run) for run in plain_runs],
                "scored": [asdict(run) for run in scored_runs],
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\n[benchmark] wrote JSON:", str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

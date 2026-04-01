#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import random

import numpy as np

from capstone_pkg.kinematics.curobo_ik import SingleArmIK
from capstone_pkg.utils.config import (
    ROBOT_YAML,
    ROBOT_URDF,
    WORLD_YAML,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    CSPACE_JOINT_NAMES_14,
)
from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.constraint_projection.bimanual_jacobian_compare_urdf import URDFModel


@dataclass
class ForceIKCandidate:
    q_cspace: List[float]
    score: float
    left_force_capacity: float
    right_force_capacity: float


@dataclass
class ForceCuroboIKOutput:
    success: bool
    q_cspace: Optional[List[float]]
    cspace_joint_names: List[str]
    score: float
    left_force_capacity: float
    right_force_capacity: float
    tried_candidates: int
    valid_candidates: int


class ForceCuroboIK:
    """
    1) cuRobo로 양팔 IK 후보를 여러 개 생성
    2) URDF Jacobian으로 전방 힘 점수를 평가
    3) 가장 좋은 후보를 시작점으로, 양팔 말단 pose를 거의 유지하는 null-space 방향으로
       force score를 더 올리는 local refinement 수행

    점수는 기본적으로
        score = (left + right) + balance_weight * min(left, right)
    를 사용하여 한쪽 팔만 좋은 해보다 양팔이 함께 좋은 해를 선호한다.
    """

    def __init__(
        self,
        robot_yml: str = ROBOT_YAML,
        urdf_path: str = ROBOT_URDF,
        *,
        world_yml: Optional[str] = WORLD_YAML,
        cpu: bool = False,
        num_seeds: int = 20,
        rotation_threshold: float = 0.05,
        position_threshold: float = 0.005,
        use_cuda_graph: bool = True,
    ):
        self.robot_yml = robot_yml
        self.urdf_path = urdf_path
        self.cpu = bool(cpu)
        self.cspace_joint_names = list(CSPACE_JOINT_NAMES_14)

        self.left_ik = SingleArmIK(
            robot_yml,
            arm="left",
            cpu=cpu,
            num_seeds=num_seeds,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            use_cuda_graph=use_cuda_graph,
            world_yml=world_yml,
        )
        self.right_ik = SingleArmIK(
            robot_yml,
            arm="right",
            cpu=cpu,
            num_seeds=num_seeds,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            use_cuda_graph=use_cuda_graph,
            world_yml=world_yml,
        )

        self._sc = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
        self.urdf_model = URDFModel(urdf_path)
        self._joint_limits = self._build_joint_limits()

    # ---------------------------
    # public API
    # ---------------------------
    def solve_max_forward_force(
        self,
        left_xyz: Sequence[float],
        left_quat_wxyz: Sequence[float],
        right_xyz: Sequence[float],
        right_quat_wxyz: Sequence[float],
        *,
        q_start_cspace: Optional[Sequence[float]] = None,
        forward_direction_base: Sequence[float] = (1.0, 0.0, 0.0),
        num_trials: int = 24,
        seed_noise_std: float = 0.25,
        random_seed: int = 0,
        balance_weight: float = 1.0,
        refine_best: bool = True,
        refine_top_k: int = 3,
        refinement_steps: int = 20,
        refinement_step_size: float = 0.12,
        gradient_eps: float = 1.0e-3,
        position_tolerance: float = 2.0e-3,
        rotation_tolerance_rad: float = 2.0e-2,
    ) -> ForceCuroboIKOutput:
        rng = random.Random(int(random_seed))
        valid: List[ForceIKCandidate] = []

        q0 = self._normalize_start_q(q_start_cspace)
        for trial_idx in range(int(num_trials)):
            q_seed = self._sample_seed(q0, rng=rng, sigma=float(seed_noise_std), trial_idx=trial_idx)
            cand = self._solve_candidate(
                left_xyz=list(left_xyz),
                left_quat_wxyz=list(left_quat_wxyz),
                right_xyz=list(right_xyz),
                right_quat_wxyz=list(right_quat_wxyz),
                q_seed=q_seed,
                forward_direction_base=forward_direction_base,
                balance_weight=balance_weight,
            )
            if cand is not None:
                valid.append(cand)

        if not valid:
            return ForceCuroboIKOutput(
                success=False,
                q_cspace=None,
                cspace_joint_names=list(self.cspace_joint_names),
                score=float("-inf"),
                left_force_capacity=0.0,
                right_force_capacity=0.0,
                tried_candidates=int(num_trials),
                valid_candidates=0,
            )

        valid.sort(key=lambda c: c.score, reverse=True)
        best = valid[0]

        if refine_best:
            target_left_T = self._target_T(left_xyz, left_quat_wxyz)
            target_right_T = self._target_T(right_xyz, right_quat_wxyz)
            for seed_cand in valid[: max(1, int(refine_top_k))]:
                refined = self._refine_candidate_pose_locked(
                    q_init=seed_cand.q_cspace,
                    target_left_T=target_left_T,
                    target_right_T=target_right_T,
                    forward_direction_base=forward_direction_base,
                    balance_weight=balance_weight,
                    steps=int(refinement_steps),
                    step_size=float(refinement_step_size),
                    gradient_eps=float(gradient_eps),
                    position_tolerance=float(position_tolerance),
                    rotation_tolerance_rad=float(rotation_tolerance_rad),
                )
                if refined is not None and refined.score > best.score:
                    best = refined

        return ForceCuroboIKOutput(
            success=True,
            q_cspace=list(best.q_cspace),
            cspace_joint_names=list(self.cspace_joint_names),
            score=float(best.score),
            left_force_capacity=float(best.left_force_capacity),
            right_force_capacity=float(best.right_force_capacity),
            tried_candidates=int(num_trials),
            valid_candidates=int(len(valid)),
        )

    # ---------------------------
    # candidate generation
    # ---------------------------
    def _solve_candidate(
        self,
        *,
        left_xyz: List[float],
        left_quat_wxyz: List[float],
        right_xyz: List[float],
        right_quat_wxyz: List[float],
        q_seed: List[float],
        forward_direction_base: Sequence[float],
        balance_weight: float,
    ) -> Optional[ForceIKCandidate]:
        left_out = self.left_ik.solve(left_xyz, left_quat_wxyz, q_start_cspace=q_seed)
        if not left_out.success or left_out.q_cspace is None:
            return None

        right_seed = list(left_out.q_cspace)
        right_out = self.right_ik.solve(right_xyz, right_quat_wxyz, q_start_cspace=right_seed)
        if not right_out.success or right_out.q_cspace is None:
            return None

        q = list(right_out.q_cspace)
        in_col, _, _ = self._sc.check_single(q)
        if in_col:
            return None

        left_cap, right_cap = self._compute_forward_force_capacities(q, forward_direction_base)
        score = self._combine_force_capacities(left_cap, right_cap, balance_weight=balance_weight)
        return ForceIKCandidate(
            q_cspace=q,
            score=score,
            left_force_capacity=float(left_cap),
            right_force_capacity=float(right_cap),
        )

    def _normalize_start_q(self, q_start_cspace: Optional[Sequence[float]]) -> List[float]:
        if q_start_cspace is None:
            return [0.0] * len(self.cspace_joint_names)
        q = [float(x) for x in q_start_cspace]
        if len(q) != len(self.cspace_joint_names):
            raise ValueError(
                f"q_start_cspace length mismatch: expected {len(self.cspace_joint_names)}, got {len(q)}"
            )
        return q

    def _sample_seed(self, q0: List[float], *, rng: random.Random, sigma: float, trial_idx: int) -> List[float]:
        if trial_idx == 0:
            return list(q0)
        return [float(v + rng.gauss(0.0, sigma)) for v in q0]

    # ---------------------------
    # Jacobian-based force scoring
    # ---------------------------
    def _compute_forward_force_capacities(
        self,
        q_cspace: Sequence[float],
        forward_direction_base: Sequence[float],
    ) -> Tuple[float, float]:
        q = np.asarray(q_cspace, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self.cspace_joint_names):
            raise ValueError(f"q dimension mismatch: {q.shape[0]}")

        d = np.asarray(forward_direction_base, dtype=np.float64).reshape(3)
        norm_d = float(np.linalg.norm(d))
        if norm_d < 1e-12:
            raise ValueError("forward_direction_base must be non-zero")
        d = d / norm_d

        _, Jw_left = self.urdf_model.fk_and_geometric_jacobian_world(
            LEFT_EE_FRAME,
            q,
            self.cspace_joint_names,
        )
        _, Jw_right = self.urdf_model.fk_and_geometric_jacobian_world(
            RIGHT_EE_FRAME,
            q,
            self.cspace_joint_names,
        )

        Jv_left = np.asarray(Jw_left[:3, :], dtype=np.float64)
        Jv_right = np.asarray(Jw_right[:3, :], dtype=np.float64)

        # same forward unit force at EE -> required joint effort magnitude
        tau_left = Jv_left.T @ d
        tau_right = Jv_right.T @ d

        eps = 1.0e-9
        left_capacity = 1.0 / (float(np.linalg.norm(tau_left)) + eps)
        right_capacity = 1.0 / (float(np.linalg.norm(tau_right)) + eps)
        return left_capacity, right_capacity

    def _combine_force_capacities(self, left_cap: float, right_cap: float, *, balance_weight: float) -> float:
        return float((left_cap + right_cap) + balance_weight * min(left_cap, right_cap))

    # ---------------------------
    # Pose-locked local refinement
    # ---------------------------
    def _refine_candidate_pose_locked(
        self,
        *,
        q_init: Sequence[float],
        target_left_T: np.ndarray,
        target_right_T: np.ndarray,
        forward_direction_base: Sequence[float],
        balance_weight: float,
        steps: int,
        step_size: float,
        gradient_eps: float,
        position_tolerance: float,
        rotation_tolerance_rad: float,
    ) -> Optional[ForceIKCandidate]:
        q = np.asarray(q_init, dtype=np.float64).copy()
        best = self._candidate_from_q(q, forward_direction_base, balance_weight)
        if best is None:
            return None

        for _ in range(max(0, steps)):
            grad = self._numerical_force_score_gradient(
                q,
                forward_direction_base=forward_direction_base,
                balance_weight=balance_weight,
                eps=gradient_eps,
            )
            J_task = self._stack_task_jacobian(q)
            J_pinv = np.linalg.pinv(J_task, rcond=1.0e-4)
            null_proj = np.eye(J_task.shape[1]) - J_pinv @ J_task
            dq_dir = null_proj @ grad
            nrm = float(np.linalg.norm(dq_dir))
            if nrm < 1.0e-10:
                break
            dq_dir = dq_dir / nrm

            accepted = False
            alpha = float(step_size)
            for _ls in range(6):
                q_try = self._clip_to_joint_limits(q + alpha * dq_dir)
                cand_try = self._candidate_from_q(q_try, forward_direction_base, balance_weight)
                if cand_try is None:
                    alpha *= 0.5
                    continue
                pos_err_l, rot_err_l, pos_err_r, rot_err_r = self._pose_errors(
                    q_try,
                    target_left_T=target_left_T,
                    target_right_T=target_right_T,
                )
                if (
                    pos_err_l <= position_tolerance
                    and pos_err_r <= position_tolerance
                    and rot_err_l <= rotation_tolerance_rad
                    and rot_err_r <= rotation_tolerance_rad
                    and cand_try.score > best.score + 1.0e-8
                ):
                    q = q_try
                    best = cand_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                continue

        return best

    def _candidate_from_q(
        self,
        q: Sequence[float],
        forward_direction_base: Sequence[float],
        balance_weight: float,
    ) -> Optional[ForceIKCandidate]:
        q_list = [float(x) for x in q]
        in_col, _, _ = self._sc.check_single(q_list)
        if in_col:
            return None
        left_cap, right_cap = self._compute_forward_force_capacities(q_list, forward_direction_base)
        score = self._combine_force_capacities(left_cap, right_cap, balance_weight=balance_weight)
        return ForceIKCandidate(
            q_cspace=q_list,
            score=float(score),
            left_force_capacity=float(left_cap),
            right_force_capacity=float(right_cap),
        )

    def _stack_task_jacobian(self, q: np.ndarray) -> np.ndarray:
        _, Jw_left = self.urdf_model.fk_and_geometric_jacobian_world(
            LEFT_EE_FRAME,
            q,
            self.cspace_joint_names,
        )
        _, Jw_right = self.urdf_model.fk_and_geometric_jacobian_world(
            RIGHT_EE_FRAME,
            q,
            self.cspace_joint_names,
        )
        return np.vstack([Jw_left, Jw_right])

    def _numerical_force_score_gradient(
        self,
        q: np.ndarray,
        *,
        forward_direction_base: Sequence[float],
        balance_weight: float,
        eps: float,
    ) -> np.ndarray:
        grad = np.zeros_like(q, dtype=np.float64)
        for i in range(q.shape[0]):
            dq = np.zeros_like(q)
            dq[i] = eps
            cp = self._candidate_from_q(self._clip_to_joint_limits(q + dq), forward_direction_base, balance_weight)
            cm = self._candidate_from_q(self._clip_to_joint_limits(q - dq), forward_direction_base, balance_weight)
            fp = cp.score if cp is not None else -1.0e12
            fm = cm.score if cm is not None else -1.0e12
            grad[i] = (fp - fm) / (2.0 * eps)
        return grad

    # ---------------------------
    # FK / pose error helpers
    # ---------------------------
    def _target_T(self, xyz: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._quat_wxyz_to_rot(quat_wxyz)
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return T

    def _pose_errors(
        self,
        q: Sequence[float],
        *,
        target_left_T: np.ndarray,
        target_right_T: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        q_arr = np.asarray(q, dtype=np.float64)
        cur_left_T, _ = self.urdf_model.fk_and_geometric_jacobian_world(
            LEFT_EE_FRAME,
            q_arr,
            self.cspace_joint_names,
        )
        cur_right_T, _ = self.urdf_model.fk_and_geometric_jacobian_world(
            RIGHT_EE_FRAME,
            q_arr,
            self.cspace_joint_names,
        )
        pos_err_l = float(np.linalg.norm(cur_left_T[:3, 3] - target_left_T[:3, 3]))
        pos_err_r = float(np.linalg.norm(cur_right_T[:3, 3] - target_right_T[:3, 3]))
        rot_err_l = self._rotation_angle(target_left_T[:3, :3], cur_left_T[:3, :3])
        rot_err_r = self._rotation_angle(target_right_T[:3, :3], cur_right_T[:3, :3])
        return pos_err_l, rot_err_l, pos_err_r, rot_err_r

    @staticmethod
    def _quat_wxyz_to_rot(quat_wxyz: Sequence[float]) -> np.ndarray:
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1.0e-12:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = q / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
        R_err = R_a.T @ R_b
        tr = float(np.trace(R_err))
        c = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
        return float(math.acos(c))

    # ---------------------------
    # Joint limits
    # ---------------------------
    def _build_joint_limits(self) -> dict[str, Tuple[float, float]]:
        limits: dict[str, Tuple[float, float]] = {}
        for jn in self.cspace_joint_names:
            j = self.urdf_model.joints.get(jn, None)
            if j is None:
                continue
            lo = -math.pi if j.limit_lower is None else float(j.limit_lower)
            hi = math.pi if j.limit_upper is None else float(j.limit_upper)
            limits[jn] = (lo, hi)
        return limits

    def _clip_to_joint_limits(self, q: Sequence[float]) -> np.ndarray:
        out = np.asarray(q, dtype=np.float64).copy()
        for i, jn in enumerate(self.cspace_joint_names):
            lo_hi = self._joint_limits.get(jn, None)
            if lo_hi is None:
                continue
            lo, hi = lo_hi
            out[i] = np.clip(out[i], lo, hi)
        return out

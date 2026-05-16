#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from capstone_pkg.utils.config import (
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    LEFT_JOINTS,
    RIGHT_JOINTS,
)
from capstone_pkg.collision_check.collision import get_self_collision_checker

@dataclass
class BimanualIKOutput:
    success: bool
    q_active: Optional[List[float]]     # cuRobo active joint order
    active_joint_names: List[str]
    q_cspace: Optional[List[float]]     # YAML cspace.joint_names order
    cspace_joint_names: List[str]


@dataclass
class SingleArmIKOutput:
    success: bool
    q_cspace: Optional[List[float]]


def _get_cspace_joint_names(robot_cfg_dict: Dict[str, Any]) -> List[str]:
    c1 = robot_cfg_dict.get("cspace", {}) or {}
    names = c1.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    kin = robot_cfg_dict.get("kinematics", {}) or {}
    c2 = kin.get("cspace", {}) or {}
    names = c2.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    return []


def _pose_from_xyz_quat_wxyz_batch(
    xyz_batch: List[List[float]],
    quat_wxyz_batch: List[List[float]],
    device: torch.device,
) -> Pose:
    """
    xyz_batch: (B,3) list
    quat_batch: (B,4) list (wxyz)
    """
    pos = torch.tensor(xyz_batch, dtype=torch.float32, device=device)   # (B,3)
    quat = torch.tensor(quat_wxyz_batch, dtype=torch.float32, device=device)  # (B,4)
    return Pose(pos, quat)


def _extract_q_from_result(result, b: int = 0) -> List[float]:
    """
    cuRobo IKSolver 결과에서 batch index b에 해당하는 q를 (D,) list로 뽑는다.
    """
    if not hasattr(result, "solution") or result.solution is None:
        raise RuntimeError("IK result.solution is None")

    sol = result.solution

    if isinstance(sol, torch.Tensor):
        if sol.dim() == 3:
            # (B, T, D) -> 마지막 timestep
            q = sol[b, -1]
        elif sol.dim() == 2:
            # (B, D)
            q = sol[b]
        elif sol.dim() == 1:
            # (D,) -> batch 없음
            q = sol
        else:
            raise RuntimeError(f"Unexpected solution tensor dim={sol.dim()} shape={tuple(sol.shape)}")

        return [float(x) for x in q.detach().cpu().tolist()]

    # list 형태도 batch 지원 (대충 버전별 대응)
    if isinstance(sol, list) and len(sol) > 0:
        # case1: sol[b]가 [D] 이거나 [T][D] 일 수 있음
        if isinstance(sol[0], list):
            sb = sol[b]
            if isinstance(sb[0], list):
                # (T, D)
                q = sb[-1]
                return [float(x) for x in q]
            # (D,)
            return [float(x) for x in sb]

    raise RuntimeError("Failed to extract q from IK result.solution")


def _map_active_q_to_cspace(
    q_active: List[float],
    active_joint_names: List[str],
    cspace_joint_names: List[str],
) -> List[float]:
    name_to_val = {n: v for n, v in zip(active_joint_names, q_active)}
    return [float(name_to_val.get(jn, 0.0)) for jn in cspace_joint_names]


def _map_cspace_q_to_active(
    q_cspace: List[float],
    cspace_joint_names: List[str],
    active_joint_names: List[str],
) -> List[float]:
    name_to_val = {n: v for n, v in zip(cspace_joint_names, q_cspace)}
    return [float(name_to_val.get(jn, 0.0)) for jn in active_joint_names]


def _build_seed_config_from_cspace_batch(
    q_cspace_batch: Optional[List[List[float]]],
    *,
    cspace_joint_names: List[str],
    active_joint_names: List[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if q_cspace_batch is None:
        return None

    q_active_batch: List[List[float]] = []
    for q_cspace in q_cspace_batch:
        q_active_batch.append(
            _map_cspace_q_to_active(
                q_cspace,
                cspace_joint_names,
                active_joint_names,
            )
        )

    return torch.tensor(q_active_batch, device=device, dtype=torch.float32).unsqueeze(1)


def _merge_active_q_to_cspace(
    q_active: List[float],
    active_joint_names: List[str],
    cspace_joint_names: List[str],
    *,
    q_base_cspace: Optional[List[float]] = None,
    update_joint_names: Optional[List[str]] = None,
) -> List[float]:
    if q_base_cspace is None:
        q_out = [0.0 for _ in cspace_joint_names]
    else:
        if len(q_base_cspace) != len(cspace_joint_names):
            raise ValueError(
                f"q_base_cspace length mismatch: expected {len(cspace_joint_names)}, got {len(q_base_cspace)}"
            )
        q_out = [float(v) for v in q_base_cspace]

    name_to_idx = {n: i for i, n in enumerate(cspace_joint_names)}
    update_set = set(update_joint_names) if update_joint_names is not None else None
    for joint_name, joint_value in zip(active_joint_names, q_active):
        if update_set is not None and joint_name not in update_set:
            continue
        idx = name_to_idx.get(joint_name, None)
        if idx is not None:
            q_out[idx] = float(joint_value)
    return q_out


def _ensure_cspace_defaults(robot_cfg: Dict[str, Any]) -> None:
    """
    IKSolver가 기대하는 kinematics.cspace 필드가 없거나 None/object이면 안전하게 채움.
    특히 null_space_weight가 object dtype으로 변하면 ArmBase init에서 터짐.
    """
    robot_cfg.setdefault("kinematics", {})
    kin = robot_cfg["kinematics"]
    kin.setdefault("cspace", {})
    cspace = kin["cspace"]

    joint_names = cspace.get("joint_names", None)
    if not isinstance(joint_names, list) or len(joint_names) == 0:
        raise RuntimeError("kinematics.cspace.joint_names가 없습니다. (또는 비어있음)")

    n = len(joint_names)

    def _to_float_list(x, default_val: float) -> List[float]:
        if not isinstance(x, list) or len(x) != n:
            return [float(default_val)] * n
        out: List[float] = []
        for v in x:
            if v is None:
                out.append(float(default_val))
            else:
                out.append(float(v))
        return out

    # retract_config
    cspace["retract_config"] = _to_float_list(cspace.get("retract_config", None), 0.0)

    # null_space_weight
    cspace["null_space_weight"] = _to_float_list(cspace.get("null_space_weight", None), 1.0)

    # cspace_distance_weight
    cspace["cspace_distance_weight"] = _to_float_list(cspace.get("cspace_distance_weight", None), 1.0)

    # 가끔 None이면 내부에서 더 터지는 애들
    if cspace.get("max_acceleration", None) is None:
        cspace["max_acceleration"] = 20.0
    if cspace.get("max_jerk", None) is None:
        cspace["max_jerk"] = 500.0


class FastBimanualIK:
    """
    ✅ demo_basic_ik 스타일 (초고속):
      - init에서 IKSolver 2개(Left/Right) 생성 1회
      - solve에서 left IK 1번 + right IK 1번 → joint merge
    """

    def __init__(
        self,
        robot_yml: str,
        *,
        left_ee: str = LEFT_EE_FRAME,
        right_ee: str = RIGHT_EE_FRAME,
        cpu: bool = False,
        num_seeds: int = 20,
        rotation_threshold: float = 0.05,
        position_threshold: float = 0.005,
        use_cuda_graph: bool = True,
        world_yml: Optional[str] = None,
    ):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self._sc = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)

        self.tensor_args = TensorDeviceType()
        if cpu:
            self.tensor_args = TensorDeviceType(device=torch.device("cpu"))

        cfg = load_yaml(robot_yml)
        robot_cfg_dict = cfg["robot_cfg"]

        self.cspace_joint_names = _get_cspace_joint_names(robot_cfg_dict)
        if not self.cspace_joint_names:
            raise RuntimeError("cspace joint_names를 YAML에서 찾을 수 없습니다.")

        # RobotConfig는 top-level cspace를 못 받음 -> kinematics.cspace로 이동
        robot_cfg_for_curobo = dict(robot_cfg_dict)
        if "cspace" in robot_cfg_for_curobo:
            robot_cfg_for_curobo.setdefault("kinematics", {})
            if "cspace" not in robot_cfg_for_curobo["kinematics"]:
                robot_cfg_for_curobo["kinematics"]["cspace"] = robot_cfg_for_curobo.pop("cspace")
            else:
                robot_cfg_for_curobo.pop("cspace", None)

        _ensure_cspace_defaults(robot_cfg_for_curobo)

        # base RobotConfig 생성 (검증)
        try:
            _ = RobotConfig.from_dict(robot_cfg_for_curobo, tensor_args=self.tensor_args)
        except TypeError:
            _ = RobotConfig.from_dict(robot_cfg_for_curobo)

        self._robot_cfg_for_curobo = robot_cfg_for_curobo
        self.device = self.tensor_args.device

        # solvers
        self.left_solver = self._build_solver(
            self._robot_cfg_for_curobo,
            ee_link=left_ee,
            num_seeds=num_seeds,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            use_cuda_graph=(use_cuda_graph and (not cpu)),
        )
        self.right_solver = self._build_solver(
            self._robot_cfg_for_curobo,
            ee_link=right_ee,
            num_seeds=num_seeds,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            use_cuda_graph=(use_cuda_graph and (not cpu)),
        )

        # ✅ active joint order는 "left_solver 기준"으로 정의
        self.active_joint_names = list(self.left_solver.kinematics.joint_names)

        # ✅ 이름->active index (merge에서 사용)
        self._active_name_to_idx = {n: i for i, n in enumerate(self.active_joint_names)}

        # left/right joint index (active 기준, 참고용)
        self.left_idx = [i for i, n in enumerate(self.active_joint_names) if "arm_l_" in n or n.startswith("arm_l")]
        self.right_idx = [i for i, n in enumerate(self.active_joint_names) if "arm_r_" in n or n.startswith("arm_r")]

        if not self.left_idx or not self.right_idx:
            raise RuntimeError(
                "left/right joint index를 자동으로 분리하지 못했습니다. "
                "active_joint_names naming(arm_l_/arm_r_)을 확인하세요."
            )

    def _build_solver(
        self,
        robot_cfg_for_curobo: Dict[str, Any],
        *,
        ee_link: str,
        num_seeds: int,
        rotation_threshold: float,
        position_threshold: float,
        use_cuda_graph: bool,
    ) -> IKSolver:

        robot_dict = dict(robot_cfg_for_curobo)
        robot_dict.setdefault("kinematics", {})
        robot_dict["kinematics"]["ee_link"] = ee_link

        try:
            robot_cfg_local = RobotConfig.from_dict(robot_dict, tensor_args=self.tensor_args)
        except TypeError:
            robot_cfg_local = RobotConfig.from_dict(robot_dict)

        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg_local,
            None,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=use_cuda_graph,
        )
        return IKSolver(ik_cfg)

    # ---------------------------
    # 핵심: solver joint order mismatch 대비한 "이름 기반 merge"
    # ---------------------------
    def _merge_by_name(
        self,
        *,
        q_base_active: List[float],
        q_left_sol: List[float],
        q_right_sol: List[float],
    ) -> List[float]:
        q_out = list(q_base_active)

        # solver의 joint_names 순서에 맞춰 name->value 맵 생성
        l_names = list(self.left_solver.kinematics.joint_names)
        r_names = list(self.right_solver.kinematics.joint_names)
        l_map = {n: float(v) for n, v in zip(l_names, q_left_sol)}
        r_map = {n: float(v) for n, v in zip(r_names, q_right_sol)}

        # left arm joints: left solver 결과로
        for n, v in l_map.items():
            if ("arm_l_" in n) or n.startswith("arm_l"):
                idx = self._active_name_to_idx.get(n, None)
                if idx is not None:
                    q_out[idx] = v

        # right arm joints: right solver 결과로
        for n, v in r_map.items():
            if ("arm_r_" in n) or n.startswith("arm_r"):
                idx = self._active_name_to_idx.get(n, None)
                if idx is not None:
                    q_out[idx] = v

        return q_out

    def solve(
        self,
        left_xyz: List[float],
        left_quat_wxyz: List[float],
        right_xyz: List[float],
        right_quat_wxyz: List[float],
        *,
        q_start_cspace: Optional[List[float]] = None,
    ) -> BimanualIKOutput:

        if q_start_cspace is not None:
            q_base = _map_cspace_q_to_active(
                q_start_cspace,
                self.cspace_joint_names,
                self.active_joint_names,
            )
            q_seed = _build_seed_config_from_cspace_batch(
                [list(q_start_cspace)],
                cspace_joint_names=self.cspace_joint_names,
                active_joint_names=self.active_joint_names,
                device=self.device,
            )
        else:
            q_base = [0.0 for _ in self.active_joint_names]
            q_seed = None

        # (B=1) batch로 감싸서 Pose 생성
        goal_left = _pose_from_xyz_quat_wxyz_batch([left_xyz], [left_quat_wxyz], self.device)
        goal_right = _pose_from_xyz_quat_wxyz_batch([right_xyz], [right_quat_wxyz], self.device)

        with torch.enable_grad():
            if q_seed is None:
                res_l = self.left_solver.solve_batch(goal_left)
            else:
                res_l = self.left_solver.solve_batch(goal_left, seed_config=q_seed)
            ok_l = bool(res_l.success[0].detach().cpu().item())
            if not ok_l:
                return BimanualIKOutput(False, None, self.active_joint_names, None, self.cspace_joint_names)
            q_l = _extract_q_from_result(res_l)

            if q_seed is None:
                res_r = self.right_solver.solve_batch(goal_right)
            else:
                res_r = self.right_solver.solve_batch(goal_right, seed_config=q_seed)
            ok_r = bool(res_r.success[0].detach().cpu().item())
            if not ok_r:
                return BimanualIKOutput(False, None, self.active_joint_names, None, self.cspace_joint_names)
            q_r = _extract_q_from_result(res_r)

        # 이름 기반 merge (order mismatch 안전)
        q_out = self._merge_by_name(q_base_active=q_base, q_left_sol=q_l, q_right_sol=q_r)

        q_cspace = _map_active_q_to_cspace(q_out, self.active_joint_names, self.cspace_joint_names)
        in_col, d_self, d_world = self._sc.check_single(q_cspace)
        if in_col:
            return BimanualIKOutput(False, None, self.active_joint_names, None, self.cspace_joint_names)

        return BimanualIKOutput(
            success=True,
            q_active=q_out,
            active_joint_names=self.active_joint_names,
            q_cspace=q_cspace,
            cspace_joint_names=self.cspace_joint_names,
        )


class SingleArmIK:
    def __init__(
        self,
        robot_yml: str,
        *,
        arm: str,  # "left" or "right"
        cpu: bool = False,
        num_seeds: int = 20,
        rotation_threshold: float = 0.05,
        position_threshold: float = 0.005,
        use_cuda_graph: bool = True,
        world_yml: Optional[str] = None,
    ):
        self.robot_yml = robot_yml
        self.arm = arm
        self.cpu = bool(cpu)

        if cpu:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = dev
        self.tensor_args = TensorDeviceType(device=dev)

        self.ee_link = LEFT_EE_FRAME if arm == "left" else RIGHT_EE_FRAME
        self.controlled_joint_names = list(LEFT_JOINTS if arm == "left" else RIGHT_JOINTS)

        cfg = load_yaml(robot_yml)
        robot_cfg_dict: Dict[str, Any] = cfg["robot_cfg"]

        self.cspace_joint_names = _get_cspace_joint_names(robot_cfg_dict)
        if not self.cspace_joint_names:
            raise RuntimeError("cspace joint_names를 YAML에서 찾을 수 없습니다.")

        robot_cfg_for_curobo = dict(robot_cfg_dict)
        if "cspace" in robot_cfg_for_curobo:
            robot_cfg_for_curobo.setdefault("kinematics", {})
            robot_cfg_for_curobo["kinematics"] = dict(robot_cfg_for_curobo["kinematics"])
            if "cspace" not in robot_cfg_for_curobo["kinematics"]:
                robot_cfg_for_curobo["kinematics"]["cspace"] = robot_cfg_for_curobo.pop("cspace")
            else:
                robot_cfg_for_curobo.pop("cspace", None)

        robot_cfg_for_curobo.setdefault("kinematics", {})
        robot_cfg_for_curobo["kinematics"] = dict(robot_cfg_for_curobo["kinematics"])
        robot_cfg_for_curobo["kinematics"]["ee_link"] = self.ee_link
        _ensure_cspace_defaults(robot_cfg_for_curobo)

        try:
            self.robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo, tensor_args=self.tensor_args)
        except TypeError:
            self.robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo)

        ik_cfg = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            self_collision_check=False,
            self_collision_opt=False,
            use_cuda_graph=(use_cuda_graph and (not cpu)),
        )
        self.solver = IKSolver(ik_cfg)

        self.sc = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
        self.active_joint_names = list(self.solver.kinematics.joint_names)
        self.active_controlled_joint_names = [
            joint_name for joint_name in self.active_joint_names if joint_name in self.controlled_joint_names
        ]
        if not self.active_controlled_joint_names:
            raise RuntimeError(
                f"{arm} arm joints were not found in solver active_joint_names: {self.active_joint_names}"
            )

    def solve(
        self,
        xyz: List[float],
        quat_wxyz: List[float],
        *,
        q_start_cspace: Optional[List[float]] = None,
    ) -> SingleArmIKOutput:
        outs = self.solve_batch(
            [list(xyz)],
            [list(quat_wxyz)],
            q_start_cspace=q_start_cspace,
        )
        if not outs:
            return SingleArmIKOutput(False, None)
        return outs[0]

    def solve_batch(
        self,
        xyz_batch: List[List[float]],
        quat_wxyz_batch: List[List[float]],
        *,
        q_start_cspace: Optional[List[float]] = None,
        q_seed_cspace_batch: Optional[List[List[float]]] = None,
    ) -> List[SingleArmIKOutput]:
        batch_size = len(xyz_batch)
        if len(quat_wxyz_batch) != batch_size:
            raise ValueError(
                f"quat_wxyz_batch length mismatch: expected {batch_size}, got {len(quat_wxyz_batch)}"
            )
        if batch_size <= 0:
            return []
        if q_seed_cspace_batch is not None and len(q_seed_cspace_batch) != batch_size:
            raise ValueError(
                "q_seed_cspace_batch length mismatch: "
                f"expected {batch_size}, got {len(q_seed_cspace_batch)}"
            )

        if q_seed_cspace_batch is not None:
            q_base_cspace_batch = [[float(v) for v in q] for q in q_seed_cspace_batch]
        elif q_start_cspace is not None:
            q_base_cspace_batch = [[float(v) for v in q_start_cspace] for _ in range(batch_size)]
        else:
            q_base_cspace_batch = None

        seed_config = _build_seed_config_from_cspace_batch(
            q_base_cspace_batch,
            cspace_joint_names=self.cspace_joint_names,
            active_joint_names=self.active_joint_names,
            device=self.device,
        )
        goal = _pose_from_xyz_quat_wxyz_batch(xyz_batch, quat_wxyz_batch, self.device)

        with torch.enable_grad():
            if seed_config is None:
                res = self.solver.solve_batch(goal)
            else:
                res = self.solver.solve_batch(goal, seed_config=seed_config)

        ok_all = res.success.detach().cpu().tolist()
        q_out_batch: List[Optional[List[float]]] = [None for _ in range(batch_size)]
        valid_indices: List[int] = []
        valid_q_batch: List[List[float]] = []

        for batch_idx in range(batch_size):
            if not bool(ok_all[batch_idx]):
                continue

            q_active = _extract_q_from_result(res, b=batch_idx)
            q_out = _merge_active_q_to_cspace(
                q_active,
                self.active_joint_names,
                self.cspace_joint_names,
                q_base_cspace=None if q_base_cspace_batch is None else q_base_cspace_batch[batch_idx],
                update_joint_names=self.active_controlled_joint_names,
            )
            q_out_batch[batch_idx] = q_out
            valid_indices.append(batch_idx)
            valid_q_batch.append(q_out)

        if valid_q_batch:
            batch_col = self.sc.check_batch(valid_q_batch)
            in_collision_all = batch_col.in_collision.detach().cpu().tolist()
            for local_idx, in_collision in enumerate(in_collision_all):
                if bool(in_collision):
                    q_out_batch[valid_indices[local_idx]] = None

        outs: List[SingleArmIKOutput] = []
        for q_out in q_out_batch:
            if q_out is None:
                outs.append(SingleArmIKOutput(False, None))
            else:
                outs.append(SingleArmIKOutput(True, list(q_out)))
        return outs


_SINGLE_ARM_IK_CACHE: dict[tuple, SingleArmIK] = {}
_SINGLE_ARM_IK_CACHE_LOCK = threading.Lock()


def get_single_arm_ik(
    robot_yml: str,
    *,
    arm: str,
    cpu: bool = False,
    num_seeds: int = 20,
    rotation_threshold: float = 0.05,
    position_threshold: float = 0.005,
    use_cuda_graph: bool = True,
    world_yml: Optional[str] = None,
) -> SingleArmIK:
    key = (
        str(robot_yml),
        str(arm),
        bool(cpu),
        int(num_seeds),
        float(rotation_threshold),
        float(position_threshold),
        bool(use_cuda_graph),
        None if world_yml in (None, "", "none", "None") else str(world_yml),
    )
    with _SINGLE_ARM_IK_CACHE_LOCK:
        cached = _SINGLE_ARM_IK_CACHE.get(key)
        if cached is None:
            cached = SingleArmIK(
                robot_yml,
                arm=arm,
                cpu=cpu,
                num_seeds=num_seeds,
                rotation_threshold=rotation_threshold,
                position_threshold=position_threshold,
                use_cuda_graph=use_cuda_graph,
                world_yml=world_yml,
            )
            _SINGLE_ARM_IK_CACHE[key] = cached
        return cached


def warmup_single_arm_ik_reachable(
    ik: SingleArmIK,
    *,
    iters: int = 1,
    batch_size: int = 32,
    noise_std: float = 0.25,
    random_seed: int = 0,
) -> None:
    if iters <= 0:
        return

    low_bounds = ik.solver.solver.safety_rollout.action_bound_lows.detach().cpu().numpy()
    high_bounds = ik.solver.solver.safety_rollout.action_bound_highs.detach().cpu().numpy()
    rng = np.random.default_rng(int(random_seed))

    for _ in range(int(iters)):
        q_active_base = ik.solver.sample_configs(1)
        kin = ik.solver.fk(q_active_base)
        base_active = q_active_base[0].detach().cpu().numpy()
        base_cspace = _map_active_q_to_cspace(
            [float(v) for v in base_active.tolist()],
            ik.active_joint_names,
            ik.cspace_joint_names,
        )

        goal_xyz = [float(v) for v in kin.ee_position[0].detach().cpu().tolist()]
        goal_quat = [float(v) for v in kin.ee_quaternion[0].detach().cpu().tolist()]
        q_seed_cspace_batch: List[List[float]] = [list(base_cspace)]
        for _seed_idx in range(max(0, int(batch_size) - 1)):
            noisy_active = base_active + rng.normal(loc=0.0, scale=float(noise_std), size=base_active.shape)
            noisy_active = np.clip(noisy_active, low_bounds, high_bounds)
            q_seed_cspace_batch.append(
                _map_active_q_to_cspace(
                    [float(v) for v in noisy_active.tolist()],
                    ik.active_joint_names,
                    ik.cspace_joint_names,
                )
            )

        _ = ik.solve_batch(
            [list(goal_xyz) for _ in range(len(q_seed_cspace_batch))],
            [list(goal_quat) for _ in range(len(q_seed_cspace_batch))],
            q_start_cspace=base_cspace,
            q_seed_cspace_batch=q_seed_cspace_batch,
        )

    try:
        _ = ik.sc.check_single([0.0 for _ in ik.cspace_joint_names])
    except Exception:
        pass

    if ik.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


_SOLVER_CACHE: dict[tuple, FastBimanualIK] = {}


def solve_batch_bimanual(
    self,
    left_xyz_batch: List[List[float]],
    left_quat_wxyz_batch: List[List[float]],
    right_xyz_batch: List[List[float]],
    right_quat_wxyz_batch: List[List[float]],
    *,
    q_start_cspace: Optional[List[float]] = None,
    q_seed_cspace_batch: Optional[List[List[float]]] = None,
    parallel_cuda_streams: bool = True,
) -> List[BimanualIKOutput]:
    """
    B개의 goal을 한 번에 처리:
        left_solver.solve_batch( B개 )
        right_solver.solve_batch( B개 )
    그리고 각 b에 대해 left/right 결과를 merge해서 bimanual q를 만든다.

    return: 길이 B 리스트
    """

    B = len(left_xyz_batch)
    assert len(left_quat_wxyz_batch) == B
    assert len(right_xyz_batch) == B
    assert len(right_quat_wxyz_batch) == B

    if q_seed_cspace_batch is not None and len(q_seed_cspace_batch) != B:
        raise ValueError(
            f"q_seed_cspace_batch length mismatch: expected {B}, got {len(q_seed_cspace_batch)}"
        )

    if q_seed_cspace_batch is not None:
        q_base_cspace_batch = [[float(v) for v in q] for q in q_seed_cspace_batch]
    elif q_start_cspace is not None:
        q_base_cspace_batch = [[float(v) for v in q_start_cspace] for _ in range(B)]
    else:
        q_base_cspace_batch = None

    if q_base_cspace_batch is not None:
        q_base_active_batch = [
            _map_cspace_q_to_active(
                q_cspace,
                self.cspace_joint_names,
                self.active_joint_names,
            )
            for q_cspace in q_base_cspace_batch
        ]
    else:
        q_base_active_batch = [[0.0 for _ in self.active_joint_names] for _ in range(B)]

    seed_config = _build_seed_config_from_cspace_batch(
        q_base_cspace_batch,
        cspace_joint_names=self.cspace_joint_names,
        active_joint_names=self.active_joint_names,
        device=self.device,
    )

    goal_left = _pose_from_xyz_quat_wxyz_batch(left_xyz_batch, left_quat_wxyz_batch, self.device)
    goal_right = _pose_from_xyz_quat_wxyz_batch(right_xyz_batch, right_quat_wxyz_batch, self.device)

    with torch.enable_grad():
        if (
            parallel_cuda_streams
            and (self.device.type == "cuda")
            and torch.cuda.is_available()
        ):
            s_l = torch.cuda.Stream()
            s_r = torch.cuda.Stream()

            with torch.cuda.stream(s_l):
                if seed_config is None:
                    res_l = self.left_solver.solve_batch(goal_left)
                else:
                    res_l = self.left_solver.solve_batch(goal_left, seed_config=seed_config)
            with torch.cuda.stream(s_r):
                if seed_config is None:
                    res_r = self.right_solver.solve_batch(goal_right)
                else:
                    res_r = self.right_solver.solve_batch(goal_right, seed_config=seed_config)

            torch.cuda.synchronize()
        else:
            if seed_config is None:
                res_l = self.left_solver.solve_batch(goal_left)
                res_r = self.right_solver.solve_batch(goal_right)
            else:
                res_l = self.left_solver.solve_batch(goal_left, seed_config=seed_config)
                res_r = self.right_solver.solve_batch(goal_right, seed_config=seed_config)

    ok_l = res_l.success.detach().cpu().tolist()
    ok_r = res_r.success.detach().cpu().tolist()

    # solver joint_names 기반 map 준비 (매 batch마다 재사용)
    l_names = list(self.left_solver.kinematics.joint_names)
    r_names = list(self.right_solver.kinematics.joint_names)
    active_name_to_idx = getattr(self, "_active_name_to_idx", {n: i for i, n in enumerate(self.active_joint_names)})

    outs: List[BimanualIKOutput] = []

    for b in range(B):
        if not bool(ok_l[b]) or not bool(ok_r[b]):
            outs.append(BimanualIKOutput(False, None, self.active_joint_names, None, self.cspace_joint_names))
            continue

        q_l = _extract_q_from_result(res_l, b=b)
        q_r = _extract_q_from_result(res_r, b=b)

        # 이름 기반 merge
        q_out = list(q_base_active_batch[b])

        l_map = {n: float(v) for n, v in zip(l_names, q_l)}
        r_map = {n: float(v) for n, v in zip(r_names, q_r)}

        for n, v in l_map.items():
            if ("arm_l_" in n) or n.startswith("arm_l"):
                idx = active_name_to_idx.get(n, None)
                if idx is not None:
                    q_out[idx] = v

        for n, v in r_map.items():
            if ("arm_r_" in n) or n.startswith("arm_r"):
                idx = active_name_to_idx.get(n, None)
                if idx is not None:
                    q_out[idx] = v

        q_cspace = _map_active_q_to_cspace(q_out, self.active_joint_names, self.cspace_joint_names)

        outs.append(
            BimanualIKOutput(
                success=True,
                q_active=q_out,
                active_joint_names=self.active_joint_names,
                q_cspace=q_cspace,
                cspace_joint_names=self.cspace_joint_names,
            )
        )

    return outs

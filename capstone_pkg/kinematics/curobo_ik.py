#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from capstone_pkg.utils.config import LEFT_EE_FRAME, RIGHT_EE_FRAME
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
        else:
            q_base = [0.0 for _ in self.active_joint_names]

        # (B=1) batch로 감싸서 Pose 생성
        goal_left = _pose_from_xyz_quat_wxyz_batch([left_xyz], [left_quat_wxyz], self.device)
        goal_right = _pose_from_xyz_quat_wxyz_batch([right_xyz], [right_quat_wxyz], self.device)

        with torch.enable_grad():
            res_l = self.left_solver.solve_batch(goal_left)
            ok_l = bool(res_l.success[0].detach().cpu().item())
            if not ok_l:
                return BimanualIKOutput(False, None, self.active_joint_names, None, self.cspace_joint_names)
            q_l = _extract_q_from_result(res_l)

            res_r = self.right_solver.solve_batch(goal_right)
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

    def solve(
        self,
        xyz: List[float],
        quat_wxyz: List[float],
        *,
        q_start_cspace: Optional[List[float]] = None,
    ) -> SingleArmIKOutput:
        if q_start_cspace is not None:
            q_base = _map_cspace_q_to_active(
                q_start_cspace,
                self.cspace_joint_names,
                self.active_joint_names,
            )
            # cuRobo expects seed_config shape: (batch, n_seed, dof)
            q_seed = torch.tensor([q_base], device=self.device, dtype=torch.float32).unsqueeze(1)
        else:
            q_seed = torch.zeros((1, 1, len(self.active_joint_names)), device=self.device, dtype=torch.float32)

        goal = _pose_from_xyz_quat_wxyz_batch([xyz], [quat_wxyz], self.device)

        with torch.enable_grad():
            res = self.solver.solve_batch(goal, seed_config=q_seed)

        ok = bool(res.success[0].detach().cpu().item())
        if not ok:
            return SingleArmIKOutput(False, None)

        q_active = _extract_q_from_result(res, b=0)

        if q_start_cspace is not None and len(q_start_cspace) == len(self.cspace_joint_names):
            q_out = list(q_start_cspace)
            name_to_idx = {n: i for i, n in enumerate(self.cspace_joint_names)}
            for n, v in zip(self.active_joint_names, q_active):
                idx = name_to_idx.get(n, None)
                if idx is not None:
                    q_out[idx] = float(v)
        else:
            q_out = _map_active_q_to_cspace(q_active, self.active_joint_names, self.cspace_joint_names)

        in_col, _, _ = self.sc.check_single(q_out)
        if in_col:
            return SingleArmIKOutput(False, None)

        return SingleArmIKOutput(True, q_out)

_SOLVER_CACHE: dict[tuple, FastBimanualIK] = {}


def solve_batch_bimanual(
    self,
    left_xyz_batch: List[List[float]],
    left_quat_wxyz_batch: List[List[float]],
    right_xyz_batch: List[List[float]],
    right_quat_wxyz_batch: List[List[float]],
    *,
    q_start_cspace: Optional[List[float]] = None,
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

    # q_base 준비(모든 batch에 동일하게 적용)
    if q_start_cspace is not None:
        q_base = _map_cspace_q_to_active(
            q_start_cspace,
            self.cspace_joint_names,
            self.active_joint_names,
        )
    else:
        q_base = [0.0 for _ in self.active_joint_names]

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
                res_l = self.left_solver.solve_batch(goal_left)
            with torch.cuda.stream(s_r):
                res_r = self.right_solver.solve_batch(goal_right)

            torch.cuda.synchronize()
        else:
            res_l = self.left_solver.solve_batch(goal_left)
            res_r = self.right_solver.solve_batch(goal_right)

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
        q_out = list(q_base)

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

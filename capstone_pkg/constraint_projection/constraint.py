# tb_rrt_pkg/tb_rrt_pkg/planner/backends/constraint_rigid.py
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Tuple, Literal, Any, Dict, List, Optional

import inspect
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from capstone_pkg.constraint_projection.bimanual_jacobian_compare_urdf import (
    BimanualConstraintJacobianBackend,
    BimanualConstraintJacobianBackendTorch,
)


# -----------------------------
# helpers
# -----------------------------
def _load_robot_cfg_dict(robot_yml: str) -> Dict[str, Any]:
    cfg = load_yaml(robot_yml)
    if isinstance(cfg, dict) and "robot_cfg" in cfg:
        return cfg["robot_cfg"]
    if isinstance(cfg, dict):
        return cfg
    raise ValueError(f"robot_yml must be a yaml dict: {robot_yml}")


def _get_cspace_joint_names(robot_cfg: Dict[str, Any]) -> List[str]:
    c1 = robot_cfg.get("cspace", {}) or {}
    names = c1.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    kin = robot_cfg.get("kinematics", {}) or {}
    c2 = (kin.get("cspace", {}) or {})
    names = c2.get("joint_names", None)
    if isinstance(names, list) and names:
        return [x for x in names if isinstance(x, str)]

    raise RuntimeError("YAML에서 cspace.joint_names를 찾을 수 없습니다.")


def _resolve_urdf_path(robot_cfg: Dict[str, Any]) -> str:
    kin = robot_cfg.get("kinematics", {}) or {}
    urdf_path = kin.get("urdf_path", None)
    if not isinstance(urdf_path, str) or not urdf_path:
        raise RuntimeError("robot_yml에 kinematics.urdf_path가 없습니다.")

    if os.path.isabs(urdf_path) and os.path.exists(urdf_path):
        return urdf_path

    asset_root = kin.get("asset_root_path", None)
    if isinstance(asset_root, str) and asset_root:
        cand = os.path.join(asset_root, urdf_path)
        if os.path.exists(cand):
            return cand

    if os.path.exists(urdf_path):
        return urdf_path

    raise FileNotFoundError(f"URDF not found: {urdf_path}")


def _make_robot_world_with_ee(
    robot_cfg_dict: Dict[str, Any],
    ee_link: str,
    tensor_args: TensorDeviceType,
) -> RobotWorld:
    """
    RobotWorld를 만들 때 kinematics.ee_link만 바꿔서 FK target을 지정.
    (현재 프로젝트 구조를 유지하면서도 안정적으로 FK 얻기 위함)
    """
    d = dict(robot_cfg_dict)
    d.pop("cspace", None)

    kin = dict(d.get("kinematics", {}))
    kin["ee_link"] = ee_link
    d["kinematics"] = kin

    sig = inspect.signature(RobotConfig.from_dict)
    if "tensor_args" in sig.parameters:
        robot_cfg = RobotConfig.from_dict(d, tensor_args=tensor_args)  # type: ignore
    else:
        robot_cfg = RobotConfig.from_dict(d)  # type: ignore

    rw_cfg = RobotWorldConfig.load_from_config(
        robot_config=robot_cfg,
        world_model=None,
        tensor_args=tensor_args,
        collision_activation_distance=0.0,
        self_collision_activation_distance=0.0,
    )
    return RobotWorld(rw_cfg)


# -----------------------------
# quaternion ops (wxyz)
# -----------------------------
def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_rotate_inv_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    v_local = R(q)^T * v
    q: (...,4) wxyz
    v: (...,3)
    return: (...,3)

    ✅ 속도 최적화 포인트:
    - rotmat 생성 + matmul 대신 quaternion rotate로 처리
    """
    w, x, y, z = q.unbind(-1)

    # normalize (안정성/수치 품질 위해)
    n = torch.sqrt(w * w + x * x + y * y + z * z).clamp_min(1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n

    qvec = torch.stack([x, y, z], dim=-1)  # (...,3)

    # t = 2 * cross(qvec, v)
    t = 2.0 * torch.cross(qvec, v, dim=-1)

    # inverse rotate: v - w*t + cross(qvec, t)
    return v - w.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


# -----------------------------
# FK wrapper (optimized mapping)
# -----------------------------
@dataclass
class BimanualFKRobotWorld:
    rw_L: RobotWorld
    rw_R: RobotWorld
    cspace_names: List[str]

    model_names_L: List[str]
    model_names_R: List[str]

    cspace_to_model_map_L: torch.Tensor  # (Dcspace,)
    cspace_to_model_map_R: torch.Tensor  # (Dcspace,)

    # ✅ precomputed valid indices (for fast vectorized mapping)
    valid_c_L: torch.Tensor  # (K,)
    valid_m_L: torch.Tensor  # (K,)
    valid_c_R: torch.Tensor  # (K,)
    valid_m_R: torch.Tensor  # (K,)

    device: torch.device
    dtype: torch.dtype

    def _cspace_to_model_fast(
        self,
        q_cspace: torch.Tensor,
        Dmodel: int,
        valid_c: torch.Tensor,
        valid_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_cspace: (N, Dcspace) or (Dcspace,)
        return q_model: (N, Dmodel)

        ✅ 속도 최적화 포인트:
        - 기존 for-loop + item() 제거
        - vectorized assignment 사용
        - dtype/device 변환도 필요할 때만 수행
        """
        if q_cspace.dim() == 1:
            q_cspace = q_cspace.view(1, -1)

        if q_cspace.device != self.device or q_cspace.dtype != self.dtype:
            q_cspace = q_cspace.to(device=self.device, dtype=self.dtype)

        N = q_cspace.shape[0]
        q_model = torch.zeros((N, Dmodel), device=self.device, dtype=self.dtype)

        # q_model[:, valid_m] = q_cspace[:, valid_c]
        q_model.index_copy_(dim=1, index=valid_m, source=q_cspace.index_select(dim=1, index=valid_c))
        return q_model

    @torch.no_grad()
    def fk_left_pose(self, q_cspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_model_L = self._cspace_to_model_fast(
            q_cspace=q_cspace,
            Dmodel=len(self.model_names_L),
            valid_c=self.valid_c_L,
            valid_m=self.valid_m_L,
        )
        stL = self.rw_L.get_kinematics(q_model_L)
        return stL.ee_position, stL.ee_quaternion

    @torch.no_grad()
    def fk_right_pose(self, q_cspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_model_R = self._cspace_to_model_fast(
            q_cspace=q_cspace,
            Dmodel=len(self.model_names_R),
            valid_c=self.valid_c_R,
            valid_m=self.valid_m_R,
        )
        stR = self.rw_R.get_kinematics(q_model_R)
        return stR.ee_position, stR.ee_quaternion

    @torch.no_grad()
    def fk_lr_pose(self, q_cspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return pL,qL,pR,qR
        """
        pL, qL = self.fk_left_pose(q_cspace)
        pR, qR = self.fk_right_pose(q_cspace)
        return pL, qL, pR, qR

    @torch.no_grad()
    def fk_lr_pos(self, q_cspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pL, _qL, pR, _qR = self.fk_lr_pose(q_cspace)
        return pL, pR


def build_bimanual_fk_robotworld(
    robot_yml: str,
    left_ee: str,
    right_ee: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BimanualFKRobotWorld:
    tensor_args = TensorDeviceType(device=device, dtype=dtype)
    robot_cfg_dict = _load_robot_cfg_dict(robot_yml)

    rw_L = _make_robot_world_with_ee(robot_cfg_dict, left_ee, tensor_args)
    rw_R = _make_robot_world_with_ee(robot_cfg_dict, right_ee, tensor_args)

    cspace_names = _get_cspace_joint_names(robot_cfg_dict)

    model_names_L = list(rw_L.kinematics.joint_names)
    model_names_R = list(rw_R.kinematics.joint_names)

    def make_map(model_names: List[str]) -> torch.Tensor:
        name_to_idx = {n: i for i, n in enumerate(model_names)}
        map_idx = [name_to_idx.get(n, -1) for n in cspace_names]
        return torch.tensor(map_idx, device=device, dtype=torch.long)

    def make_valid(map_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_c = torch.where(map_idx >= 0)[0]
        valid_m = map_idx[valid_c].long()
        return valid_c, valid_m

    map_L = make_map(model_names_L)
    map_R = make_map(model_names_R)

    valid_c_L, valid_m_L = make_valid(map_L)
    valid_c_R, valid_m_R = make_valid(map_R)

    return BimanualFKRobotWorld(
        rw_L=rw_L,
        rw_R=rw_R,
        cspace_names=cspace_names,
        model_names_L=model_names_L,
        model_names_R=model_names_R,
        cspace_to_model_map_L=map_L,
        cspace_to_model_map_R=map_R,
        valid_c_L=valid_c_L,
        valid_m_L=valid_m_L,
        valid_c_R=valid_c_R,
        valid_m_R=valid_m_R,
        device=device,
        dtype=dtype,
    )


# -----------------------------
# Rigid constraint
# -----------------------------
class RigidConstraint:
    """
    양팔 사이 rigid constraint
    - p_rel: left EE frame에서 본 right EE 위치
    - q_rel: left EE frame에서 본 right EE 회전 (quat)
    """

    def __init__(
        self,
        *,
        robot_yml: str,
        left_ee: str,
        right_ee: str,
        q_ref: torch.Tensor,  # (Dcspace,)
        device: torch.device,
        dtype: torch.dtype,
        mode: Literal["pos", "se3"] = "se3",
        rigid_orientation: bool = False,
        lock_z: bool = False,
        planar_xy: bool = False,
    ):
        if q_ref.ndim != 1:
            raise ValueError("q_ref must be (D,)")

        self.device = device
        self.dtype = dtype
        self.mode = mode
        self._analytic_backend_np: Optional[BimanualConstraintJacobianBackend] = None
        self._analytic_backend_torch: Optional[BimanualConstraintJacobianBackendTorch] = None
        # If True, keep the *absolute* (world-frame) orientation of the left EE
        # fixed to the one at q_ref. This prevents the bimanual pair from
        # rotating together (while still allowing translation).
        self.rigid_orientation = bool(rigid_orientation)
        # If True, keep the absolute z position of the left EE fixed to the
        # one at q_ref, so the motion stays on an x-y plane.
        self.lock_z = bool(lock_z)
        # If True, keep the absolute roll/pitch of the left EE fixed while
        # allowing yaw about world z. Intended to be used together with lock_z
        # for planar x-y motion with yaw-only rotation.
        self.planar_xy = bool(planar_xy)

        self.fk = build_bimanual_fk_robotworld(
            robot_yml=robot_yml,
            left_ee=left_ee,
            right_ee=right_ee,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            q0 = q_ref.to(device=device, dtype=dtype).view(1, -1)

            # FK
            pL, qL, pR, qR = self.fk.fk_lr_pose(q0)

            # Store absolute left-EE orientation at reference (world frame)
            if self.rigid_orientation:
                self.qL_ref = qL.squeeze(0).detach().clone().contiguous()
            if self.lock_z:
                self.pLz_ref = pL[:, 2].squeeze(0).detach().clone().contiguous()
            if self.planar_xy:
                world_z = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
                # R(q)^T * e_z is invariant to world-z yaw and depends only on roll/pitch.
                tilt_ref = _quat_rotate_inv_wxyz(qL, world_z.expand_as(pL))
                self.qL_world_z_local_ref_xy = tilt_ref[:, :2].squeeze(0).detach().clone().contiguous()

            # rotmat 대신 quaternion inverse rotate
            p_rel = _quat_rotate_inv_wxyz(qL, (pR - pL))

            self.p_rel_ref = p_rel.squeeze(0).detach().clone().contiguous()

            if self.mode == "pos":
                print("[RigidConstraint:pos] p_rel_ref:", self.p_rel_ref.detach().cpu().tolist())
                return

            q_rel = _quat_mul_wxyz(_quat_conj_wxyz(qL), qR)
            self.q_rel_ref = q_rel.squeeze(0).detach().clone().contiguous()

            print("[RigidConstraint:se3] p_rel_ref:", self.p_rel_ref.detach().cpu().tolist())
            self._try_enable_analytic_backend(
                robot_yml=robot_yml,
                left_ee=left_ee,
                right_ee=right_ee,
                q_ref=q_ref,
            )

    @property
    def has_analytic_jacobian(self) -> bool:
        return self._analytic_backend_np is not None or self._analytic_backend_torch is not None

    @property
    def has_analytic_torch_backend(self) -> bool:
        return self._analytic_backend_torch is not None

    def _try_enable_analytic_backend(
        self,
        *,
        robot_yml: str,
        left_ee: str,
        right_ee: str,
        q_ref: torch.Tensor,
    ) -> None:
        if self.mode != "se3" or self.rigid_orientation or self.lock_z or self.planar_xy:
            return

        try:
            robot_cfg = _load_robot_cfg_dict(robot_yml)
            urdf_path = _resolve_urdf_path(robot_cfg)
            q_ref_np = q_ref.detach().cpu().numpy().astype("float64", copy=False)
            self._analytic_backend_np = BimanualConstraintJacobianBackend(
                urdf_path=urdf_path,
                left_link=left_ee,
                right_link=right_ee,
                active_joints=self.fk.cspace_names,
                q_ref=q_ref_np,
            )
            self._analytic_backend_torch = BimanualConstraintJacobianBackendTorch(
                urdf_path=urdf_path,
                left_link=left_ee,
                right_link=right_ee,
                active_joints=self.fk.cspace_names,
                q_ref=q_ref_np,
            )
            print(f"[RigidConstraint:se3] analytic Jacobian enabled: {urdf_path}")
        except Exception as exc:
            self._analytic_backend_np = None
            self._analytic_backend_torch = None
            print(f"[RigidConstraint:se3] analytic Jacobian unavailable, fallback to FD: {exc}")

    @torch.no_grad()
    def _analytic_eval_np(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._analytic_backend_np is None:
            raise RuntimeError("analytic numpy Jacobian backend is not enabled")

        if q.dim() == 1:
            q = q.view(1, -1)

        out = self._analytic_backend_np.residual_and_jacobian(
            q.detach().cpu().numpy().astype("float64", copy=False)
        )
        h = torch.as_tensor(out.h, device=q.device, dtype=q.dtype)
        J = torch.as_tensor(out.J, device=q.device, dtype=q.dtype)
        return h, J

    @torch.no_grad()
    def _analytic_eval_torch(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._analytic_backend_torch is None:
            raise RuntimeError("analytic torch Jacobian backend is not enabled")
        return self._analytic_backend_torch.residual_and_jacobian(q)

    @torch.no_grad()
    def residual_torch(self, q: torch.Tensor) -> Optional[torch.Tensor]:
        if self._analytic_backend_torch is None:
            return None
        h, _ = self._analytic_eval_torch(q)
        return h

    @torch.no_grad()
    def jacobian_torch(self, q: torch.Tensor) -> Optional[torch.Tensor]:
        if self._analytic_backend_torch is None:
            return None
        _h, J = self._analytic_eval_torch(q)
        return J

    @torch.no_grad()
    def residual_and_jacobian_torch(self, q: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._analytic_backend_torch is None:
            return None
        return self._analytic_eval_torch(q)
    
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        """
        step2(tangent_space)에서 Jacobian을 만들 때 쓰는 residual vector.
        (B,m) 반환.
        """
        return self.h(q)

    @torch.no_grad()
    def h(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (N,Dcspace) cspace
        return:
          mode="pos" -> (N,3)
          mode="se3" -> (N,6)
          if lock_z=True: additionally appends absolute left-EE z residual (1)
          if planar_xy=True: additionally appends absolute left-EE roll/pitch residual (2),
                             while allowing yaw about world z
          if rigid_orientation=True: additionally appends absolute left-EE orientation residual (3)
        """
        if q.dim() == 1:
            q = q.view(1, -1)

        if self._analytic_backend_torch is not None:
            h, _ = self._analytic_eval_torch(q)
            return h
        if self._analytic_backend_np is not None:
            h, _ = self._analytic_eval_np(q)
            return h

        # FK
        pL, qL, pR, qR = self.fk.fk_lr_pose(q)

        # rotmat 대신 quaternion inverse rotate
        p_rel = _quat_rotate_inv_wxyz(qL, (pR - pL))
        dp = p_rel - self.p_rel_ref.view(1, 3)

        if self.mode == "pos":
            parts = [dp]

            if self.lock_z:
                dz = pL[..., 2:3] - self.pLz_ref.view(1, 1)
                parts.append(dz)

            if self.planar_xy:
                world_z = torch.zeros_like(pL)
                world_z[..., 2] = 1.0
                tilt_local = _quat_rotate_inv_wxyz(qL, world_z)
                dtilt = tilt_local[..., :2] - self.qL_world_z_local_ref_xy.view(1, 2)
                parts.append(dtilt)

            if not self.rigid_orientation:
                return torch.cat(parts, dim=-1)

            # Absolute orientation error (left EE)
            qL_ref = self.qL_ref.view(1, 4).expand_as(qL)
            q_err_abs = _quat_mul_wxyz(_quat_conj_wxyz(qL_ref), qL)

            # shortest path (w>=0)
            w = q_err_abs[..., 0:1]
            flip = (w < 0.0).to(q_err_abs.dtype)
            q_err_abs = q_err_abs * (1.0 - 2.0 * flip)

            dr_abs = 2.0 * q_err_abs[..., 1:4]
            parts.append(dr_abs)
            return torch.cat(parts, dim=-1)

        # relative rotation
        q_rel = _quat_mul_wxyz(_quat_conj_wxyz(qL), qR)

        # q_err = conj(q_rel_ref) * q_rel
        q_rel_ref = self.q_rel_ref.view(1, 4).expand_as(q_rel)
        q_err = _quat_mul_wxyz(_quat_conj_wxyz(q_rel_ref), q_rel)

        # shortest path (w>=0)
        w = q_err[..., 0:1]
        flip = (w < 0.0).to(q_err.dtype)
        q_err = q_err * (1.0 - 2.0 * flip)

        # small-angle approx for rotation error
        dr = 2.0 * q_err[..., 1:4]

        parts = [dp, dr]

        if self.lock_z:
            dz = pL[..., 2:3] - self.pLz_ref.view(1, 1)
            parts.append(dz)

        if self.planar_xy:
            world_z = torch.zeros_like(pL)
            world_z[..., 2] = 1.0
            tilt_local = _quat_rotate_inv_wxyz(qL, world_z)
            dtilt = tilt_local[..., :2] - self.qL_world_z_local_ref_xy.view(1, 2)
            parts.append(dtilt)

        if not self.rigid_orientation:
            return torch.cat(parts, dim=-1)

        # Absolute orientation error (left EE)
        qL_ref = self.qL_ref.view(1, 4).expand_as(qL)
        q_err_abs = _quat_mul_wxyz(_quat_conj_wxyz(qL_ref), qL)

        # shortest path (w>=0)
        w2 = q_err_abs[..., 0:1]
        flip2 = (w2 < 0.0).to(q_err_abs.dtype)
        q_err_abs = q_err_abs * (1.0 - 2.0 * flip2)

        dr_abs = 2.0 * q_err_abs[..., 1:4]
        parts.append(dr_abs)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def jacobian(self, q: torch.Tensor) -> Optional[torch.Tensor]:
        if q.dim() == 1:
            q = q.view(1, -1)

        if self._analytic_backend_torch is not None:
            _h, J = self._analytic_eval_torch(q)
            return J

        if self._analytic_backend_np is None:
            return None

        _h, J = self._analytic_eval_np(q)
        return J

    @torch.no_grad()
    def residual_norm(self, q: torch.Tensor) -> torch.Tensor:
        h = self.h(q)
        return torch.linalg.norm(h, dim=-1)
    
    @torch.no_grad()
    def residual(self, q: torch.Tensor) -> torch.Tensor:
        return self.h(q)

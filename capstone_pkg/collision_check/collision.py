#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import os
import yaml
import torch

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.types.robot import RobotConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.state import JointState

from capstone_pkg.utils.config import WORLD_YAML
from capstone_pkg.collision_check.attach_utils import extract_object_cuboids_from_mujoco_xml, merge_cuboids_to_aabb, make_box_cuboid
from capstone_pkg.collision_check.collision_link import ensure_collision_fields, add_connected_link_collision_ignores

def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> torch.Tensor:
    """Return (3,3) rotation matrix (torch) from quaternion wxyz."""
    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float64)
    q = q / torch.linalg.norm(q).clamp(min=1e-12)
    w, x, y, z = q.tolist()

    R = torch.tensor([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=torch.float64)
    return R


def _cuboid_corners_world(center_xyz, dims_xyz, quat_wxyz,
                          *, dims_are_half_extents: bool=False,
                          pose_is: str="center"):
    """
    dims_are_half_extents:
      - False: dims are full lengths (default)
      - True : dims are half extents
    pose_is:
      - "center" : pose xyz is box center (default, cuRobo convention)
      - "bottom" : pose xyz is bottom-center (z at bottom face center)
      - "corner" : pose xyz is min corner in local box frame (no rotation case)
    """
    cx, cy, cz = [float(v) for v in center_xyz]
    dx, dy, dz = [float(v) for v in dims_xyz]
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]

    if dims_are_half_extents:
        hx, hy, hz = dx, dy, dz
        full_dx, full_dy, full_dz = dx*2, dy*2, dz*2
    else:
        hx, hy, hz = dx*0.5, dy*0.5, dz*0.5
        full_dx, full_dy, full_dz = dx, dy, dz

    # pose 기준 보정(회전이 있는 경우 corner는 정확히 하려면 회전까지 고려해야 해서 디버그용으로만)
    if pose_is == "bottom":
        cz = cz + hz
    elif pose_is == "corner":
        cx = cx + hx
        cy = cy + hy
        cz = cz + hz

    corners_local = torch.tensor([
        [-hx, -hy, -hz],
        [+hx, -hy, -hz],
        [+hx, +hy, -hz],
        [-hx, +hy, -hz],
        [-hx, -hy, +hz],
        [+hx, -hy, +hz],
        [+hx, +hy, +hz],
        [-hx, +hy, +hz],
    ], dtype=torch.float64)

    R = _quat_wxyz_to_rotmat(qw, qx, qy, qz)
    t = torch.tensor([cx, cy, cz], dtype=torch.float64).view(1, 3)
    return (corners_local @ R.T) + t

def _apply_world_offset(world_model: Dict[str, Any], offset_xyz: Tuple[float, float, float]) -> Dict[str, Any]:
    ox, oy, oz = offset_xyz
    for k in ["cuboid", "sphere", "capsule", "cylinder", "mesh"]:
        if k not in world_model or not isinstance(world_model[k], dict):
            continue
        for name, item in world_model[k].items():
            if not isinstance(item, dict) or "pose" not in item:
                continue
            p = item["pose"]
            if isinstance(p, list) and len(p) >= 3:
                p[0] = float(p[0]) + ox
                p[1] = float(p[1]) + oy
                p[2] = float(p[2]) + oz
    return world_model


def _is_object_obstacle_name(name: str) -> bool:
    return name == "grasp_object" or name.startswith("grasp_object_") or name.startswith("grasp_object")


def _split_world_model_for_object_activation(
    world_model: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    base_world: Dict[str, Any] = {}
    object_world: Dict[str, Any] = {}
    object_names: List[str] = []

    for collider_type, collider_items in world_model.items():
        if not isinstance(collider_items, dict):
            base_world[collider_type] = collider_items
            continue

        base_items: Dict[str, Any] = {}
        object_items: Dict[str, Any] = {}
        for name, item in collider_items.items():
            if _is_object_obstacle_name(str(name)):
                object_items[name] = item
                object_names.append(str(name))
            else:
                base_items[name] = item

        if base_items:
            base_world[collider_type] = base_items
        if object_items:
            object_world[collider_type] = object_items

    return base_world, object_world, sorted(set(object_names))


def visualize_world_model_to_png(world_model: Dict[str, Any], out_path: str = "world_colliders.png") -> str:
    """
    world_model dict(정규화된 형태) 중 cuboid를 3D로 그리고 PNG로 저장.
    pose 포맷: [x,y,z,qw,qx,qy,qz]
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cub = world_model.get("cuboid", {}) or {}
    if not isinstance(cub, dict) or len(cub) == 0:
        print("[WORLD-VIZ] no cuboid found. skip:", out_path)
        return out_path

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # cuboid faces (indices of corners)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # side
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    mins = torch.tensor([+1e9, +1e9, +1e9], dtype=torch.float64)
    maxs = torch.tensor([-1e9, -1e9, -1e9], dtype=torch.float64)

    for name, item in cub.items():
        dims = item["dims"]
        pose = item["pose"]
        center = pose[0:3]
        quat = pose[3:7]  # (qw,qx,qy,qz)

        corners = _cuboid_corners_world(center, dims, quat)  # (8,3)
        mins = torch.minimum(mins, corners.min(dim=0).values)
        maxs = torch.maximum(maxs, corners.max(dim=0).values)

        poly3d = [[corners[idx].tolist() for idx in face] for face in faces]
        pc = Poly3DCollection(poly3d, alpha=0.25, linewidths=0.5)
        ax.add_collection3d(pc)

        # 이름 라벨(너무 많으면 지저분하니 필요없으면 주석)
        c = torch.tensor(center, dtype=torch.float64)
        ax.text(c[0].item(), c[1].item(), c[2].item(), str(name), fontsize=7)

    # 보기 좋게 축 범위/비율 맞추기
    cx = ((mins[0] + maxs[0]) * 0.5).item()
    cy = ((mins[1] + maxs[1]) * 0.5).item()
    cz = ((mins[2] + maxs[2]) * 0.5).item()
    span = float(torch.max(maxs - mins).item())
    span = max(span, 1e-3)

    ax.set_xlim(cx - span * 0.55, cx + span * 0.55)
    ax.set_ylim(cy - span * 0.55, cy + span * 0.55)
    ax.set_zlim(cz - span * 0.55, cz + span * 0.55)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("cuRobo world colliders (cuboids)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("[WORLD-VIZ] saved:", out_path)
    return out_path


# -----------------------------
# yaml utf-8 loader (중요!)
# -----------------------------
def _load_yaml_utf8(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_world_model(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    cuRobo 예시(world_collision_example) 기준:
      - top-level에 cuboid/mesh/sphere/capsule/cylinder 키가 오면 그대로 OK
    추가로, 혹시 world_cfg/colliders로 감싸진 포맷이 들어오면 풀어줌.
    """
    if not isinstance(raw, dict):
        raise ValueError("world yaml must be a dict at top-level")

    collider_keys = {"cuboid", "sphere", "capsule", "mesh", "cylinder"}
    if any(k in raw for k in collider_keys):
        return raw

    if "world_cfg" in raw and isinstance(raw["world_cfg"], dict):
        wc = raw["world_cfg"]
        if any(k in wc for k in collider_keys):
            return wc
        if "colliders" in wc and isinstance(wc["colliders"], dict):
            cc = wc["colliders"]
            if any(k in cc for k in collider_keys):
                return cc

    if "colliders" in raw and isinstance(raw["colliders"], dict):
        cc = raw["colliders"]
        if any(k in cc for k in collider_keys):
            return cc

    raise ValueError(
        "world yaml schema not recognized. "
        "Need top-level cuboid/mesh/sphere/capsule/cylinder (or world_cfg/colliders wrapping)."
    )


def _validate_world_model(model: Dict[str, Any]) -> None:
    """
    최소한 cuboid에 대해 dims/pose를 검사.
    pose 포맷: [x,y,z,qw,qx,qy,qz] (identity: [x,y,z,1,0,0,0])
    """
    if "cuboid" not in model:
        return

    cub = model["cuboid"]
    if not isinstance(cub, dict):
        raise ValueError("world_model['cuboid'] must be a dict: name -> {dims, pose}")

    for name, item in cub.items():
        if not isinstance(item, dict):
            raise ValueError(f"cuboid '{name}' must be a dict")
        if "dims" not in item or "pose" not in item:
            raise ValueError(f"cuboid '{name}' must have dims and pose")

        dims = item["dims"]
        pose = item["pose"]
        if not (isinstance(dims, list) and len(dims) == 3):
            raise ValueError(f"cuboid '{name}' dims must be len=3 list")
        if not (isinstance(pose, list) and len(pose) == 7):
            raise ValueError(f"cuboid '{name}' pose must be len=7 list")

        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        n = float(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1e-6:
            raise ValueError(f"cuboid '{name}' quaternion norm too small: {n}")


@dataclass
class SelfCollisionBatchResult:
    in_collision: torch.Tensor   # (N,) bool
    d_self_max: torch.Tensor     # (N,) float (penetration proxy: >0이면 충돌)
    d_world_max: torch.Tensor    # (N,) float (penetration proxy: >0이면 충돌)

    @property
    def pen_max(self) -> torch.Tensor:
        return torch.maximum(self.d_self_max, self.d_world_max)


class SelfCollisionChecker:
    """
    ✅ self + world 를 항상 같이 보는 checker

    제공 API:
      - check_batch(q) -> SelfCollisionBatchResult
      - get_collision_penetration(q) -> (N,)
      - get_collision_free_mask(q, margin=0.0) -> (N,) bool (True=free)
      - get_self_collision_mask(q) -> (N,) bool (호환용 alias, 실제로는 self+world)
    """

    def __init__(
        self,
        robot_yml: str,
        *,
        cpu: bool = False,
        world_yml: Optional[str] = None,
        debug_world_stats: bool = False,
    ):
        print("[SELF_COLLISION] LOADED FROM:", os.path.abspath(__file__))

        self.robot_yml = robot_yml
        self.world_yml = world_yml
        self.cpu = bool(cpu)
        self.debug_world_stats = bool(debug_world_stats)
        self.object_obstacle_names: List[str] = []
        self.base_world_model: Dict[str, Any] = {}
        self.object_world_model: Dict[str, Any] = {}
        self.robot_world_object_only: Optional[RobotWorld] = None
        self._object_joint_dist_fn_name: Optional[str] = None
        self._object_sphere_dist_fn_name: Optional[str] = None

        if self.cpu:
            dev = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                dev = torch.device(f"cuda:{idx}")
            else:
                dev = torch.device("cpu")
        self.tensor_args = TensorDeviceType(device=dev)

        # -----------------------------
        # 1) robot yaml 로드 + ignore 채우기
        # -----------------------------
        cfg = load_yaml(self.robot_yml)
        self.robot_cfg_dict = cfg["robot_cfg"]

        ensure_collision_fields(self.robot_cfg_dict)
        add_connected_link_collision_ignores(self.robot_cfg_dict, only_collision_links=True)

        robot_cfg_for_curobo = dict(self.robot_cfg_dict)
        robot_cfg_for_curobo.pop("cspace", None)

        try:
            robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo, self.tensor_args)
        except TypeError:
            robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo)

        # -----------------------------
        # 2) world yaml dict 로 직접 로드 (None이면 self collision만 사용)
        # -----------------------------
        if self.world_yml in (None, "", "none", "None"):
            world_model = {}
            self.world_yml = None
            print("[WORLD-DEBUG] world collision disabled (world_yml=None)")
        else:
            raw_world = _load_yaml_utf8(self.world_yml)
            world_model = _normalize_world_model(raw_world)
            _validate_world_model(world_model)
            world_model = _apply_world_offset(world_model, (0.0, 0.0, 0.0))
            visualize_world_model_to_png(world_model, out_path="world_colliders.png")

        # ✅ keep a copy for plotting/debug
        self.world_model = world_model
        self.base_world_model, self.object_world_model, self.object_obstacle_names = (
            _split_world_model_for_object_activation(world_model)
        )

        # -----------------------------
        # 3) RobotWorld 구성 (world_model=dict)
        # -----------------------------
        rw_cfg = RobotWorldConfig.load_from_config(
            robot_config=robot_cfg,
            world_model=self.base_world_model,
            tensor_args=self.tensor_args,
            collision_activation_distance=0.01,
            self_collision_activation_distance=0.02,
        )
        self.robot_world = RobotWorld(rw_cfg)
        self.robot_world_self_only = RobotWorld(rw_cfg)
        if self.object_world_model:
            rw_cfg_obj = RobotWorldConfig.load_from_config(
                robot_config=robot_cfg,
                world_model=self.object_world_model,
                tensor_args=self.tensor_args,
                collision_activation_distance=0.003,
                self_collision_activation_distance=0.02,
            )
            self.robot_world_object_only = RobotWorld(rw_cfg_obj)

        print("[WORLD-DEBUG] world_yml =", self.world_yml)
        print("[WORLD-DEBUG] object_obstacles =", self.object_obstacle_names)

        # (중요) cuRobo 버전에 따라 world_coll_checker / world_collision 속성이 없거나 None일 수 있음.
        # 그래서 '속성'으로 판정하지 말고 '실제 월드 거리 API 호출'이 되는지로 판정한다.
        if self.base_world_model:
            self._validate_world_distance_api(self.robot_world, world_label="base_world")
        if self.robot_world_object_only is not None:
            self._validate_world_distance_api(self.robot_world_object_only, world_label="object_world")


        # -----------------------------
        # 4) joint mapping (cspace -> RobotWorld joint order)
        # -----------------------------
        self.cspace_names = self.robot_cfg_dict.get("cspace", {}).get("joint_names", []) or []
        if not self.cspace_names:
            raise RuntimeError("[SelfCollisionChecker] cspace.joint_names not found in robot yaml")

        self.model_names = list(self.robot_world.kinematics.joint_names)
        name_to_idx = {n: i for i, n in enumerate(self.model_names)}

        self.active_indices: List[int] = []
        for jn in self.cspace_names:
            if jn not in name_to_idx:
                raise RuntimeError(
                    f"[SelfCollisionChecker] cspace joint '{jn}' not in RobotWorld joints.\n"
                    f"RobotWorld joints={self.model_names}"
                )
            self.active_indices.append(int(name_to_idx[jn]))
        self.active_indices_t = torch.tensor(self.active_indices, device=self.tensor_args.device, dtype=torch.long)

        # -----------------------------
        # 5) distance API 선택: joint 기반 우선
        # -----------------------------
        self._joint_dist_fn_name = self._pick_joint_distance_fn_name(self.robot_world)
        self._sphere_dist_fn_name = self._pick_sphere_distance_fn_name(self.robot_world)
        if self.robot_world_object_only is not None:
            self._object_joint_dist_fn_name = self._pick_joint_distance_fn_name(self.robot_world_object_only)
            self._object_sphere_dist_fn_name = self._pick_sphere_distance_fn_name(self.robot_world_object_only)

        print("[WORLD-DEBUG] joint dist fn =", self._joint_dist_fn_name)
        print("[WORLD-DEBUG] sphere dist fn =", self._sphere_dist_fn_name)
        if self.robot_world_object_only is not None:
            print("[WORLD-DEBUG] object joint dist fn =", self._object_joint_dist_fn_name)
            print("[WORLD-DEBUG] object sphere dist fn =", self._object_sphere_dist_fn_name)

    def _pick_joint_distance_fn_name(self, robot_world: RobotWorld) -> Optional[str]:
        for cand in [
            "get_world_self_collision_distance_from_joints",
            "get_world_collision_distance_from_joints",
        ]:
            if hasattr(robot_world, cand):
                return cand
        return None

    def _pick_sphere_distance_fn_name(self, robot_world: RobotWorld) -> Optional[str]:
        for cand in [
            "get_collision_distance",
            "get_world_collision_distance",
        ]:
            if hasattr(robot_world, cand):
                return cand
        return None

    def _validate_world_distance_api(self, robot_world: RobotWorld, *, world_label: str) -> None:
        q_test = torch.zeros(
            (1, len(robot_world.kinematics.joint_names)),
            device=self.tensor_args.device,
            dtype=torch.float32,
        )

        ok = False
        last_err = None
        for fn_name in [
            "get_world_self_collision_distance_from_joints",
            "get_world_collision_distance_from_joints",
        ]:
            if hasattr(robot_world, fn_name):
                try:
                    _ = getattr(robot_world, fn_name)(q_test)
                    print(f"[WORLD-DEBUG] world distance fn works ({world_label}):", fn_name)
                    ok = True
                    break
                except Exception as exc:
                    last_err = exc

        if not ok:
            raise RuntimeError(
                "World collision이 활성화되지 않았습니다. "
                "RobotWorld는 생성됐지만 월드 거리 API 호출이 실패했습니다.\n"
                f"world_label={world_label}\n"
                f"world_yml={self.world_yml}\n"
                f"last_error={repr(last_err)}\n"
                "✅ 확인할 것:\n"
                "  - world yaml 스키마/pose 포맷\n"
                "  - RobotWorldConfig.load_from_config(world_model=dict) 전달 여부\n"
            )

    def _build_q_active_from_cspace(self, q_cspace: torch.Tensor) -> torch.Tensor:
        if q_cspace.ndim != 2:
            raise ValueError("q_cspace must be (N,D)")
        if q_cspace.shape[1] != len(self.cspace_names):
            raise ValueError(f"q_cspace dim mismatch. got={q_cspace.shape[1]} expected={len(self.cspace_names)}")

        N = q_cspace.shape[0]
        q_full = torch.zeros((N, len(self.model_names)), dtype=torch.float32, device=self.tensor_args.device)
        q_full.index_copy_(1, self.active_indices_t, q_cspace.float())
        return q_full

    def _extract_spheres(self, robot_world: RobotWorld, state: Any, q_active: torch.Tensor) -> torch.Tensor:
        x_sph = None
        for name in ["robot_spheres", "link_spheres", "spheres"]:
            if hasattr(state, name):
                cand = getattr(state, name)
                if cand is not None:
                    x_sph = cand
                    break

        if x_sph is None:
            fk_out = robot_world.kinematics.forward(q_active)
            x_sph = fk_out[-1]

        if x_sph.dim() == 3:
            x_sph = x_sph.unsqueeze(1)
        elif x_sph.dim() != 4:
            raise RuntimeError(f"Unexpected sphere tensor shape: {tuple(x_sph.shape)}")
        return x_sph


    def _world_self_from_joints(
        self,
        q_active: torch.Tensor,
        *,
        robot_world: RobotWorld,
        fn_name: Optional[str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if fn_name is None:
            return None, None
        fn = getattr(robot_world, fn_name)
        out = fn(q_active)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]
        return out, None

    def _world_from_spheres(
        self,
        x_sph: torch.Tensor,
        *,
        robot_world: RobotWorld,
        fn_name: Optional[str],
    ) -> Optional[torch.Tensor]:
        if fn_name is None:
            return None
        fn = getattr(robot_world, fn_name)
        try:
            return fn(x_sph)
        except Exception:
            return None

    @torch.no_grad()
    def check_batch(self, q_cspace: torch.Tensor | List[List[float]]) -> SelfCollisionBatchResult:
        device = self.tensor_args.device
        if isinstance(q_cspace, list):
            q_cspace = torch.tensor(q_cspace, device=device, dtype=torch.float32)
        else:
            q_cspace = q_cspace.to(device=device, dtype=torch.float32)

        q_active = self._build_q_active_from_cspace(q_cspace)

        # (A) world는 attach된 robot_world 기준으로 계산
        d_world = None
        d_world_max = torch.zeros((q_active.shape[0],), device=device, dtype=torch.float32)

        if self.base_world_model:
            d_world_tmp, _ = self._world_self_from_joints(
                q_active,
                robot_world=self.robot_world,
                fn_name=self._joint_dist_fn_name,
            )
            if d_world_tmp is not None:
                d_world = d_world_tmp
                d_world_max = d_world.view(d_world.shape[0], -1).max(dim=1).values

        # (B) self는 "attach 안 된" robot_world_self_only 로만 계산
        state_self = self.robot_world_self_only.get_kinematics(q_active)
        x_sph_self = self._extract_spheres(self.robot_world_self_only, state_self, q_active)
        d_self = self.robot_world_self_only.get_self_collision_distance(x_sph_self)
        d_self_max = d_self.view(d_self.shape[0], -1).max(dim=1).values

        # (C) world fallback (spheres 기반)도 attach된 쪽으로만
        if d_world is None and self.base_world_model:
            state_att = self.robot_world.get_kinematics(q_active)
            x_sph_att = self._extract_spheres(self.robot_world, state_att, q_active)
            d_world_s = self._world_from_spheres(
                x_sph_att,
                robot_world=self.robot_world,
                fn_name=self._sphere_dist_fn_name,
            )
            if d_world_s is not None:
                d_world = d_world_s
                d_world_max = d_world.view(d_world.shape[0], -1).max(dim=1).values

        if self.robot_world_object_only is not None:
            d_obj_tmp, _ = self._world_self_from_joints(
                q_active,
                robot_world=self.robot_world_object_only,
                fn_name=self._object_joint_dist_fn_name,
            )
            d_obj_max = torch.zeros((q_active.shape[0],), device=device, dtype=torch.float32)
            if d_obj_tmp is not None:
                d_obj_max = d_obj_tmp.view(d_obj_tmp.shape[0], -1).max(dim=1).values
            else:
                state_obj = self.robot_world_object_only.get_kinematics(q_active)
                x_sph_obj = self._extract_spheres(self.robot_world_object_only, state_obj, q_active)
                d_obj_s = self._world_from_spheres(
                    x_sph_obj,
                    robot_world=self.robot_world_object_only,
                    fn_name=self._object_sphere_dist_fn_name,
                )
                if d_obj_s is not None:
                    d_obj_max = d_obj_s.view(d_obj_s.shape[0], -1).max(dim=1).values
            d_world_max = torch.maximum(d_world_max, d_obj_max)


        in_col = (d_self_max > 0.0) | (d_world_max > 0.0)

        if self.debug_world_stats:
            if d_world is None:
                print("[WORLD-DEBUG] d_world = None (API mismatch?)")
            else:
                flat = d_world.view(d_world.shape[0], -1)
                print("[WORLD-DEBUG] d_world stats: min", float(flat.min()), "max", float(flat.max()), "mean", float(flat.mean()))

        return SelfCollisionBatchResult(in_collision=in_col, d_self_max=d_self_max, d_world_max=d_world_max)

    @torch.no_grad()
    def get_collision_penetration(self, q_cspace: torch.Tensor) -> torch.Tensor:
        """
        (N,D) -> (N,)
        self/world 둘 중 큰 값을 penetration proxy로 사용.
        """
        out = self.check_batch(q_cspace)
        return out.pen_max

    @torch.no_grad()
    def get_collision_free_mask(self, q_cspace: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
        """
        True = collision-free (self+world)
        margin>0이면 보수적으로 더 멀리 떨어지게 필터링 가능.
        """
        pen = self.get_collision_penetration(q_cspace)
        return pen <= float(margin)

    # ✅ 호환용 이름인데, "self-only"로 쓰면 안 되므로 self+world로 반환하게 고정
    @torch.no_grad()
    def get_self_collision_mask(self, q_cspace: torch.Tensor) -> torch.Tensor:
        return self.get_collision_free_mask(q_cspace, margin=0.0)

    @torch.no_grad()
    def check_single(self, q_list: List[float]) -> Tuple[bool, float, float]:
        out = self.check_batch([q_list])
        return bool(out.in_collision[0].item()), float(out.d_self_max[0].item()), float(out.d_world_max[0].item())

    def _get_world_collision_obj(self, robot_world: RobotWorld):
        # cuRobo 버전에 따라 속성명이 다를 수 있어서 최대한 유연하게
        wc = getattr(robot_world, "world_coll_checker", None)
        if wc is None:
            wc = getattr(robot_world, "world_collision", None)
        return wc

    def set_world_obstacles_enabled(self, obstacle_names: List[str], enable: bool):
        base_wc = self._get_world_collision_obj(self.robot_world)
        obj_wc = None
        if self.robot_world_object_only is not None:
            obj_wc = self._get_world_collision_obj(self.robot_world_object_only)

        if base_wc is None and obj_wc is None:
            raise RuntimeError("World collision object not found in RobotWorld")

        last_err = None
        for nm in obstacle_names:
            target_wcs = []
            if nm in self.object_obstacle_names:
                if obj_wc is not None:
                    target_wcs.append(obj_wc)
            else:
                if base_wc is not None:
                    target_wcs.append(base_wc)

            if not target_wcs:
                if obj_wc is not None:
                    target_wcs.append(obj_wc)
                if base_wc is not None and base_wc not in target_wcs:
                    target_wcs.append(base_wc)

            ok = False
            for wc in target_wcs:
                try:
                    if hasattr(wc, "enable_obstacle"):
                        wc.enable_obstacle(nm, enable=enable)
                    elif hasattr(wc, "set_obstacle_enabled"):
                        wc.set_obstacle_enabled(nm, enable)
                    elif hasattr(wc, "enable"):
                        wc.enable(nm, enable)
                    else:
                        raise RuntimeError("No enable/disable API found on world collision object")
                    ok = True
                    break
                except Exception as e:
                    last_err = e
            if not ok:
                last_err = last_err or RuntimeError(f"Obstacle not found: {nm}")

        if last_err is not None:
            # 일부 이름이 없으면 여기로 올 수 있음
            raise RuntimeError(f"Failed to (en/dis)able some obstacles. last_err={repr(last_err)}")

    def attach_mujoco_object_to_robot(
        self,
        *,
        mujoco_xml_path: str,
        q_model_order: torch.Tensor,
        link_name: str = "attached_object",
        name_prefix: str = "att_",
        disable_in_world: bool = True,
    ):
        """
        1) world에서 동일 이름 obstacle disable(선택)
        2) object cuboid들을 robot에 attach
        """
        # 1) Cuboid 리스트 생성 (world pose 기준)
        cuboids = extract_object_cuboids_from_mujoco_xml(
            mujoco_xml_path, body_name="object", name_prefix=name_prefix
        )

        cuboids = merge_cuboids_to_aabb(cuboids, merged_name=f"{name_prefix}object_merged")

        # 2) world에서 disable
        if disable_in_world:
            names = [c.name for c in cuboids]
            self.set_world_obstacles_enabled(names, enable=False)

        # 3) attach
        kin = self.robot_world.kinematics  # 보통 CudaRobotModel 계열
        if not hasattr(kin, "attach_external_objects_to_robot"):
            raise RuntimeError("robot_world.kinematics has no attach_external_objects_to_robot()")

        # q는 model joint order로 들어가야 함 (SelfCollisionChecker가 이미 mapping 가지고 있음)
        js = JointState(
            position=q_model_order.view(1, -1),
            joint_names=list(kin.joint_names),
        )

        ok = kin.attach_external_objects_to_robot(
            joint_state=js,
            external_objects=cuboids,
            link_name=link_name,
            surface_sphere_radius=0.001,  # 너무 작으면 sphere 폭증/느려짐. 필요시 조정.
        )
        if not ok:
            raise RuntimeError("attach_external_objects_to_robot returned False")

    def attach_cuboid_to_robot(
        self,
        *,
        cuboid: Cuboid,
        q_model_order: torch.Tensor,
        link_name: str,
        disable_in_world: bool = True,
    ):
        """Attach a user-defined cuboid to the robot link for post-grasp planning/use."""
        if disable_in_world:
            try:
                self.set_world_obstacles_enabled([cuboid.name], enable=False)
            except Exception:
                pass

        kin = self.robot_world.kinematics
        if not hasattr(kin, "attach_external_objects_to_robot"):
            raise RuntimeError("robot_world.kinematics has no attach_external_objects_to_robot()")

        js = JointState(
            position=q_model_order.view(1, -1),
            joint_names=list(kin.joint_names),
        )
        ok = kin.attach_external_objects_to_robot(
            joint_state=js,
            external_objects=[cuboid],
            link_name=link_name,
            surface_sphere_radius=0.001,
        )
        if not ok:
            raise RuntimeError("attach_external_objects_to_robot returned False")

    def attach_box_object_to_robot(
        self,
        *,
        center_xyz: list[float] | tuple[float, float, float],
        dims_xyz: list[float] | tuple[float, float, float],
        q_model_order: torch.Tensor,
        link_name: str,
        object_name: str = "grasp_object",
        quat_wxyz: list[float] | tuple[float, float, float, float] | None = None,
        disable_in_world: bool = True,
    ):
        cuboid = make_box_cuboid(
            name=object_name,
            center_xyz=center_xyz,
            dims_xyz=dims_xyz,
            quat_wxyz=quat_wxyz,
        )
        self.attach_cuboid_to_robot(
            cuboid=cuboid,
            q_model_order=q_model_order,
            link_name=link_name,
            disable_in_world=disable_in_world,
        )



# ---------------------------------------------------------
# cache
# ---------------------------------------------------------
_CHECKER_CACHE: dict[tuple, SelfCollisionChecker] = {}


def get_self_collision_checker(robot_yml: str, *, cpu: bool = False, world_yml: Optional[str] = None) -> SelfCollisionChecker:
    key = (robot_yml, bool(cpu), world_yml)
    ck = _CHECKER_CACHE.get(key)
    if ck is None:
        ck = SelfCollisionChecker(robot_yml, cpu=cpu, world_yml=world_yml)
        _CHECKER_CACHE[key] = ck
    return ck

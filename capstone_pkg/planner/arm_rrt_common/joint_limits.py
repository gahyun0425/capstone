from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math

try:
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import load_yaml
    from curobo.types.robot import RobotConfig
except Exception:  # pragma: no cover
    torch = None
    TensorDeviceType = None
    load_yaml = None
    RobotConfig = None


@dataclass
class JointLimits:
    joint_names: List[str]
    lower: List[float]
    upper: List[float]


def _as_list(x) -> List[float]:
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


def load_joint_limits_from_curobo(robot_yml: str, *, cpu: bool = False) -> Optional[JointLimits]:
    """Best-effort joint limit extraction via cuRobo RobotConfig.
    If unavailable, returns None.
    """
    if load_yaml is None or RobotConfig is None:
        return None

    cfg = load_yaml(robot_yml)
    robot_cfg_dict = cfg.get("robot_cfg", {}) or {}

    # cuRobo expects cspace under kinematics in many versions.
    robot_cfg_for_curobo = dict(robot_cfg_dict)
    cspace = robot_cfg_for_curobo.pop("cspace", None)
    if cspace is not None:
        robot_cfg_for_curobo.setdefault("kinematics", {})
        robot_cfg_for_curobo["kinematics"].setdefault("cspace", {})
        robot_cfg_for_curobo["kinematics"]["cspace"].update(cspace)

    if TensorDeviceType is not None and torch is not None:
        tensor_args = TensorDeviceType(device=torch.device("cpu") if cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
        try:
            robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo, tensor_args)
        except TypeError:
            robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo)
    else:
        try:
            robot_cfg = RobotConfig.from_dict(robot_cfg_for_curobo)
        except Exception:
            return None

    kin = getattr(robot_cfg, "kinematics", None)
    joint_names = []
    if kin is not None:
        joint_names = list(getattr(kin, "joint_names", [])) or []
        # Try common attributes/methods for limits:
        for attr in ["joint_limits", "limits", "_joint_limits", "cspace_joint_limits"]:
            jl = getattr(kin, attr, None)
            if jl is not None:
                jl_l = _as_list(getattr(jl, "lower", None)) or (_as_list(jl[0]) if hasattr(jl, "__len__") else [])
                jl_u = _as_list(getattr(jl, "upper", None)) or (_as_list(jl[1]) if hasattr(jl, "__len__") else [])
                if jl_l and jl_u and len(jl_l) == len(jl_u):
                    return JointLimits(joint_names, jl_l, jl_u)

        if hasattr(kin, "get_joint_limits"):
            try:
                out = kin.get_joint_limits()
                # sometimes returns (lower, upper) or (2,D) tensor
                if isinstance(out, tuple) and len(out) == 2:
                    lower, upper = _as_list(out[0]), _as_list(out[1])
                    if lower and upper and len(lower) == len(upper):
                        return JointLimits(joint_names, lower, upper)
                if out is not None:
                    arr = _as_list(out)
                    # flatten length 2D
            except Exception:
                pass

    return None


def fallback_limits(joint_names: List[str]) -> JointLimits:
    # conservative generic limits
    lower = [-math.pi for _ in joint_names]
    upper = [ math.pi for _ in joint_names]
    return JointLimits(joint_names, lower, upper)


def get_joint_limits(robot_yml: str, joint_names: List[str], *, cpu: bool = False) -> JointLimits:
    jl = load_joint_limits_from_curobo(robot_yml, cpu=cpu)
    if jl is None or not jl.joint_names:
        return fallback_limits(joint_names)

    name_to_idx = {n:i for i,n in enumerate(jl.joint_names)}
    lower=[]; upper=[]
    for n in joint_names:
        i=name_to_idx.get(n, None)
        if i is None:
            lower.append(-math.pi); upper.append(math.pi)
        else:
            lower.append(float(jl.lower[i])); upper.append(float(jl.upper[i]))
    return JointLimits(joint_names, lower, upper)

#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple

import xml.etree.ElementTree as ET

from curobo.geom.types import Cuboid


def _get_float3(attr: str) -> Tuple[float, float, float]:
    vals = [float(x) for x in attr.strip().split()]
    assert len(vals) == 3, f"expected 3 floats, got {attr}"
    return (vals[0], vals[1], vals[2])


def extract_object_cuboids_from_mujoco_xml(
    mujoco_xml_path: str,
    *,
    body_name: str = "object",
    name_prefix: str = "att_",
    x_offset: float = 0.09, 
    z_offset: float = -0.12, 
) -> List[Cuboid]:
    """
    ffw_sg2_world.xml의 <body name="object"> 아래 box geom들을 Cuboid 리스트로 변환.
    - MuJoCo geom size는 half extent -> cuRobo dims는 full length(2*size).
    - pose는 world frame 기준 (body pos + geom pos), quat는 우선 identity로 둠(네 xml은 회전 없음).
    - ✅ y_offset 만큼 y축으로 이동해서 반환
    """
    tree = ET.parse(mujoco_xml_path)
    root = tree.getroot()

    body = None
    for b in root.iter("body"):
        if b.attrib.get("name", "") == body_name:
            body = b
            break
    if body is None:
        raise RuntimeError(f"body name='{body_name}' not found in {mujoco_xml_path}")

    body_pos = _get_float3(body.attrib.get("pos", "0 0 0"))

    cuboids: List[Cuboid] = []
    for g in body.iter("geom"):
        if g.attrib.get("type", "") != "box":
            continue

        gname = g.attrib.get("name", "unnamed")
        size = _get_float3(g.attrib.get("size", "0 0 0"))   # half extents
        gpos = _get_float3(g.attrib.get("pos", "0 0 0"))

        # dims = full lengths
        dims = [2.0 * size[0], 2.0 * size[1], 2.0 * size[2]]

        # ✅ world pose = body pos + geom pos + (y_offset on y)
        pose_xyz = [
            body_pos[0] + gpos[0] + float(x_offset),
            body_pos[1] + gpos[1],
            body_pos[2] + gpos[2] + float(z_offset),
        ]
        pose_wxyz = [1.0, 0.0, 0.0, 0.0]  # 네 xml에 quat 없음 → 회전 없음 가정

        cuboids.append(
            Cuboid(
                name=f"{name_prefix}{gname}",
                pose=pose_xyz + pose_wxyz,   # [x,y,z,w,x,y,z]
                dims=dims,
            )
        )

    if not cuboids:
        raise RuntimeError(f"no box geoms found under body '{body_name}'")
    return cuboids


def get_world_obstacle_names_for_object(xml_path: str, body_name: str = "object", prefix: str = "att_") -> List[str]:
    # extract와 동일 규칙의 이름 리스트
    tree = ET.parse(xml_path)
    root = tree.getroot()
    body = None
    for b in root.iter("body"):
        if b.attrib.get("name", "") == body_name:
            body = b
            break
    if body is None:
        return []
    names = []
    for g in body.iter("geom"):
        if g.attrib.get("type", "") == "box":
            gname = g.attrib.get("name", "unnamed")
            names.append(f"{prefix}{gname}")
    return names

import torch

def merge_cuboids_to_aabb(cuboids: List[Cuboid], *, merged_name: str = "att_object_merged") -> List[Cuboid]:
    # 모든 cuboid의 world corners를 모아 axis-aligned bounding box(AABB)로 합침
    mins = torch.tensor([+1e9, +1e9, +1e9], dtype=torch.float64)
    maxs = torch.tensor([-1e9, -1e9, -1e9], dtype=torch.float64)

    for c in cuboids:
        x, y, z, qw, qx, qy, qz = c.pose
        dx, dy, dz = c.dims

        # 회전은 지금 identity라고 가정(네 xml은 quat 없음)
        hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
        corners = torch.tensor([
            [x-hx, y-hy, z-hz],
            [x+hx, y-hy, z-hz],
            [x+hx, y+hy, z-hz],
            [x-hx, y+hy, z-hz],
            [x-hx, y-hy, z+hz],
            [x+hx, y-hy, z+hz],
            [x+hx, y+hy, z+hz],
            [x-hx, y+hy, z+hz],
        ], dtype=torch.float64)

        mins = torch.minimum(mins, corners.min(dim=0).values)
        maxs = torch.maximum(maxs, corners.max(dim=0).values)

    center = ((mins + maxs) * 0.5).tolist()
    dims = (maxs - mins).tolist()

    merged = Cuboid(
        name=merged_name,
        pose=[center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0],
        dims=[dims[0], dims[1], dims[2]],
    )
    return [merged]



def make_box_cuboid(
    *,
    name: str,
    center_xyz: list[float] | tuple[float, float, float],
    dims_xyz: list[float] | tuple[float, float, float],
    quat_wxyz: list[float] | tuple[float, float, float, float] | None = None,
) -> Cuboid:
    """Create a cuRobo Cuboid from a user-provided box center/size."""
    if quat_wxyz is None:
        quat_wxyz = [1.0, 0.0, 0.0, 0.0]
    return Cuboid(
        name=str(name),
        pose=[float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]),
              float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])],
        dims=[float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])],
    )

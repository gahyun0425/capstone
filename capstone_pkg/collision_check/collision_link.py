from __future__ import annotations
from typing import Any, Dict, Optional

import os
import xml.etree.ElementTree as ET


def _resolve_urdf_path(kin: Dict[str, Any]) -> Optional[str]:
    """urdf_path가 상대경로면 asset_root_path 기준으로 보정."""
    urdf_path = kin.get("urdf_path", None)
    if not isinstance(urdf_path, str) or not urdf_path:
        return None

    if os.path.isabs(urdf_path) and os.path.exists(urdf_path):
        return urdf_path

    asset_root = kin.get("asset_root_path", None)
    if isinstance(asset_root, str) and asset_root:
        cand = os.path.join(asset_root, urdf_path)
        if os.path.exists(cand):
            return cand

    if os.path.exists(urdf_path):
        return urdf_path
    return None


def _parse_urdf_connected_links(urdf_path: str) -> Dict[str, set]:
    """URDF의 모든 joint(parent-child)를 읽어서 인접(연결) 링크 그래프 생성."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return {}

    adj: Dict[str, set] = {}
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        pl = parent.attrib.get("link")
        cl = child.attrib.get("link")
        if not pl or not cl:
            continue
        adj.setdefault(pl, set()).add(cl)
        adj.setdefault(cl, set()).add(pl)
    return adj


def _merge_self_collision_ignore(kin: Dict[str, Any], ignore_map: Dict[str, set]) -> int:
    """kin['self_collision_ignore']에 ignore_map을 merge (양방향+중복제거). 추가된 항목 수 반환."""
    cur = kin.get("self_collision_ignore", {}) or {}
    if not isinstance(cur, dict):
        cur = {}

    added = 0

    def _add(a: str, b: str):
        nonlocal added
        cur.setdefault(a, [])
        if b not in cur[a]:
            cur[a].append(b)
            added += 1

    for a, bs in ignore_map.items():
        if not isinstance(a, str):
            continue
        for b in bs:
            if not isinstance(b, str):
                continue
            _add(a, b)
            _add(b, a)

    kin["self_collision_ignore"] = cur
    return added


def ensure_collision_fields(robot_dict: Dict[str, Any]) -> None:
    """버전 호환을 위해 kinematics의 collision 관련 키가 None이면 기본값 넣어줌."""
    kin = robot_dict.get("kinematics", {}) or {}
    if not isinstance(kin, dict):
        kin = {}

    if kin.get("self_collision_buffer", None) is None:
        kin["self_collision_buffer"] = {}
    if kin.get("self_collision_ignore", None) is None:
        kin["self_collision_ignore"] = {}

    robot_dict["kinematics"] = kin


def add_connected_link_collision_ignores(
    robot_dict: Dict[str, Any],
    *,
    only_collision_links: bool = True,
) -> int:
    """
    URDF parent-child로 연결된(인접) 링크끼리 self-collision ignore 등록.
    - only_collision_links=True면 collision_link_names에 있는 링크만 대상으로 제한.
    """
    kin = robot_dict.get("kinematics", {}) or {}
    if not isinstance(kin, dict):
        return 0

    urdf_path = _resolve_urdf_path(kin)
    if urdf_path is None:
        return 0

    adj = _parse_urdf_connected_links(urdf_path)
    if not adj:
        return 0

    col_links = kin.get("collision_link_names", []) or []
    col_set = set([x for x in col_links if isinstance(x, str)])

    ignore_map: Dict[str, set] = {}

    for a, nbrs in adj.items():
        if only_collision_links and col_set and a not in col_set:
            continue
        for b in nbrs:
            if only_collision_links and col_set and b not in col_set:
                continue
            ignore_map.setdefault(a, set()).add(b)

    added = _merge_self_collision_ignore(kin, ignore_map)
    robot_dict["kinematics"] = kin
    return added

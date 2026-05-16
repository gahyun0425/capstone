from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any

import yaml

DEFAULT_WORLD_COLLISION_TOPIC = "/mujoco/world_collision"
WORLD_COLLISION_PAYLOAD_TYPE = "capstone_world_collision_cuboids"


@dataclass(frozen=True)
class WorldCuboid:
    name: str
    dims: list[float]
    pose: list[float]


def _normalize_world_model(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("world yaml must be a dict at top-level")

    collider_keys = {"cuboid", "sphere", "capsule", "mesh", "cylinder"}
    if any(k in raw for k in collider_keys):
        return raw

    world_cfg = raw.get("world_cfg")
    if isinstance(world_cfg, dict):
        if any(k in world_cfg for k in collider_keys):
            return world_cfg
        colliders = world_cfg.get("colliders")
        if isinstance(colliders, dict) and any(k in colliders for k in collider_keys):
            return colliders

    colliders = raw.get("colliders")
    if isinstance(colliders, dict) and any(k in colliders for k in collider_keys):
        return colliders

    raise ValueError(
        "world yaml schema not recognized. "
        "Need cuboid at top-level or inside world_cfg/colliders."
    )


def _as_float_list(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must be a list of length {length}")
    return [float(v) for v in value]


def _normalize_quat_wxyz(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1.0e-9:
        raise ValueError("quaternion norm is too small")
    return [float(v) / n for v in q]


def load_world_cuboids(world_yml: str) -> list[WorldCuboid]:
    with open(world_yml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    world_model = _normalize_world_model(raw)
    cuboids = world_model.get("cuboid", {}) or {}
    if not isinstance(cuboids, dict):
        raise ValueError("world_model['cuboid'] must be a dict")

    out: list[WorldCuboid] = []
    for name, item in cuboids.items():
        if not isinstance(item, dict):
            raise ValueError(f"cuboid '{name}' must be a dict")
        dims = _as_float_list(item.get("dims"), length=3, label=f"cuboid '{name}' dims")
        pose = _as_float_list(item.get("pose"), length=7, label=f"cuboid '{name}' pose")
        if any(v <= 0.0 for v in dims):
            raise ValueError(f"cuboid '{name}' dims must be positive")
        pose = pose[:3] + _normalize_quat_wxyz(pose[3:7])
        out.append(WorldCuboid(name=str(name), dims=dims, pose=pose))
    return out


def make_world_collision_payload(world_yml: str) -> str:
    cuboids = load_world_cuboids(world_yml)
    payload = {
        "type": WORLD_COLLISION_PAYLOAD_TYPE,
        "source": os.path.abspath(world_yml),
        "cuboids": [
            {
                "name": c.name,
                "dims": list(c.dims),
                "pose": list(c.pose),
            }
            for c in cuboids
        ],
    }
    return json.dumps(payload, sort_keys=True)


def parse_world_collision_payload(data: str) -> tuple[str, list[WorldCuboid]]:
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("world collision payload must be a JSON object")
    if payload.get("type") != WORLD_COLLISION_PAYLOAD_TYPE:
        raise ValueError(f"unexpected world collision payload type: {payload.get('type')!r}")

    source = str(payload.get("source", ""))
    raw_cuboids = payload.get("cuboids", [])
    if not isinstance(raw_cuboids, list):
        raise ValueError("payload['cuboids'] must be a list")

    cuboids: list[WorldCuboid] = []
    for i, item in enumerate(raw_cuboids):
        if not isinstance(item, dict):
            raise ValueError(f"payload cuboid[{i}] must be a dict")
        name = str(item.get("name", f"cuboid_{i}"))
        dims = _as_float_list(item.get("dims"), length=3, label=f"payload cuboid '{name}' dims")
        pose = _as_float_list(item.get("pose"), length=7, label=f"payload cuboid '{name}' pose")
        if any(v <= 0.0 for v in dims):
            raise ValueError(f"payload cuboid '{name}' dims must be positive")
        pose = pose[:3] + _normalize_quat_wxyz(pose[3:7])
        cuboids.append(WorldCuboid(name=name, dims=dims, pose=pose))
    return source, cuboids


def publish_world_collision_yaml(
    world_yml: str,
    *,
    topic: str = DEFAULT_WORLD_COLLISION_TOPIC,
    wait_subscriber_s: float = 1.0,
    keep_alive_s: float = 0.5,
    node_name: str = "world_collision_publisher",
) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import String

    payload = make_world_collision_payload(world_yml)
    _, cuboids = parse_world_collision_payload(payload)

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    qos = QoSProfile(depth=1)
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    node = Node(node_name)
    pub = node.create_publisher(String, topic, qos)
    try:
        t_end = time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok() and time.monotonic() < t_end and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)

        msg = String()
        msg.data = payload
        for _ in range(3):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        keep_end = time.monotonic() + max(0.0, float(keep_alive_s))
        while rclpy.ok() and time.monotonic() < keep_end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()

    return len(cuboids)

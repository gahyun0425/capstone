from __future__ import annotations

import time
from typing import Sequence

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def publish_joint_path(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    topic: str = "/joint_states_cmd",
    dt: float = 0.1,
    wait_subscriber_s: float = 1.0,
) -> None:
    """Publish a joint-space path as JointState sequence to a ROS2 topic."""
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    n = len(joint_names)
    for i, q in enumerate(path):
        if len(q) != n:
            raise ValueError(f"path[{i}] length {len(q)} != len(joint_names) {n}")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("bidir_rrt_path_publisher")
    pub = node.create_publisher(JointState, topic, 10)

    # Give DDS graph discovery time so first samples are less likely to be dropped.
    t_end = time.monotonic() + max(0.0, float(wait_subscriber_s))
    while rclpy.ok() and time.monotonic() < t_end and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)

    for idx, q in enumerate(path):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(v) for v in q]
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        if idx + 1 < len(path):
            time.sleep(dt)

    # Publish goal once more to help the last command stick at receiver side.
    last = path[-1]
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(joint_names)
    msg.position = [float(v) for v in last]
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()

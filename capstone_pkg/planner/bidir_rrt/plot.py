from __future__ import annotations

import colorsys
import time
from typing import Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from capstone_pkg.utils.config import BASE_FRAME, ROBOT_YAML


def _joint_color(index: int, total: int) -> Tuple[float, float, float]:
    """Generate stable, vivid colors for each joint line."""
    if total <= 0:
        return 1.0, 1.0, 1.0
    hue = float(index % total) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return float(r), float(g), float(b)


def _duration_from_seconds(seconds: float) -> Duration:
    lifetime = Duration(sec=0, nanosec=0)
    if seconds > 0.0:
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        lifetime = Duration(sec=sec, nanosec=nanosec)
    return lifetime


def _compute_fk_points(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    robot_yml: str,
    base_frame: str,
    ee_frame: str,
    cpu: bool,
) -> list[Tuple[float, float, float]]:
    from capstone_pkg.kinematics.curobo_test_fk import compute_relative_link_path_from_cspace

    return compute_relative_link_path_from_cspace(
        path,
        joint_names,
        robot_yml=robot_yml,
        base_link=base_frame,
        ee_link=ee_frame,
        cpu=cpu,
    )


def _append_fk_plot_markers(
    markers: MarkerArray,
    fk_points: Sequence[Tuple[float, float, float]],
    *,
    frame_id: str,
    ee_frame_name: str,
    marker_lifetime_s: float,
    line_width: float,
) -> None:
    if not fk_points:
        return

    lifetime = _duration_from_seconds(marker_lifetime_s)
    marker_base_id = 100000

    fk_line = Marker()
    fk_line.header.frame_id = frame_id
    fk_line.ns = "bidir_rrt_fk_path"
    fk_line.id = marker_base_id
    fk_line.type = Marker.LINE_STRIP
    fk_line.action = Marker.ADD
    fk_line.pose.orientation.w = 1.0
    fk_line.frame_locked = True
    fk_line.scale.x = float(line_width)
    fk_line.color.r = 1.0
    fk_line.color.g = 0.95
    fk_line.color.b = 0.05
    fk_line.color.a = 1.0
    fk_line.lifetime = lifetime
    for x, y, z in fk_points:
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        fk_line.points.append(p)
    markers.markers.append(fk_line)

    start = Marker()
    start.header.frame_id = frame_id
    start.ns = "bidir_rrt_fk_path_endpoint"
    start.id = marker_base_id + 1
    start.type = Marker.SPHERE
    start.action = Marker.ADD
    start.pose.orientation.w = 1.0
    start.frame_locked = True
    start.scale.x = float(line_width * 3.0)
    start.scale.y = float(line_width * 3.0)
    start.scale.z = float(line_width * 3.0)
    start.color.r = 0.1
    start.color.g = 1.0
    start.color.b = 0.1
    start.color.a = 1.0
    start.lifetime = lifetime
    start.pose.position.x = float(fk_points[0][0])
    start.pose.position.y = float(fk_points[0][1])
    start.pose.position.z = float(fk_points[0][2])
    markers.markers.append(start)

    goal = Marker()
    goal.header.frame_id = frame_id
    goal.ns = "bidir_rrt_fk_path_endpoint"
    goal.id = marker_base_id + 2
    goal.type = Marker.SPHERE
    goal.action = Marker.ADD
    goal.pose.orientation.w = 1.0
    goal.frame_locked = True
    goal.scale.x = float(line_width * 3.0)
    goal.scale.y = float(line_width * 3.0)
    goal.scale.z = float(line_width * 3.0)
    goal.color.r = 1.0
    goal.color.g = 0.2
    goal.color.b = 0.2
    goal.color.a = 1.0
    goal.lifetime = lifetime
    goal.pose.position.x = float(fk_points[-1][0])
    goal.pose.position.y = float(fk_points[-1][1])
    goal.pose.position.z = float(fk_points[-1][2])
    markers.markers.append(goal)

    label = Marker()
    label.header.frame_id = frame_id
    label.ns = "bidir_rrt_fk_path_label"
    label.id = marker_base_id + 3
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.orientation.w = 1.0
    label.frame_locked = True
    label.scale.z = float(max(0.05, line_width * 8.0))
    label.color.r = 1.0
    label.color.g = 0.95
    label.color.b = 0.05
    label.color.a = 1.0
    label.text = f"FK EE path: {ee_frame_name}"
    label.lifetime = lifetime
    label.pose.position.x = float(fk_points[-1][0])
    label.pose.position.y = float(fk_points[-1][1])
    label.pose.position.z = float(fk_points[-1][2] + max(0.03, line_width * 2.0))
    markers.markers.append(label)


def _build_joint_plot_markers(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    frame_id: str,
    x_step: float,
    y_scale: float,
    z_separation: float,
    line_width: float,
    text_height: float,
    marker_lifetime_s: float,
) -> MarkerArray:
    markers = MarkerArray()
    lifetime = _duration_from_seconds(marker_lifetime_s)

    # Clear previous marker IDs on this topic so repeated runs stay clean.
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    total = len(joint_names)
    for joint_idx, joint_name in enumerate(joint_names):
        r, g, b = _joint_color(joint_idx, total)
        z = float(joint_idx) * float(z_separation)

        line = Marker()
        line.header.frame_id = frame_id
        line.ns = "bidir_rrt_joint_path"
        line.id = int(joint_idx)
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.frame_locked = True
        line.scale.x = float(line_width)
        line.color.r = r
        line.color.g = g
        line.color.b = b
        line.color.a = 1.0
        line.lifetime = lifetime

        for waypoint_idx, q in enumerate(path):
            p = Point()
            p.x = float(waypoint_idx) * float(x_step)
            p.y = float(q[joint_idx]) * float(y_scale)
            p.z = z
            line.points.append(p)

        markers.markers.append(line)

        label = Marker()
        label.header.frame_id = frame_id
        label.ns = "bidir_rrt_joint_path_label"
        label.id = int(total + joint_idx)
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.orientation.w = 1.0
        label.frame_locked = True
        label.scale.z = float(text_height)
        label.color.r = r
        label.color.g = g
        label.color.b = b
        label.color.a = 1.0
        label.text = joint_name
        label.lifetime = lifetime

        last_idx = max(0, len(path) - 1)
        label.pose.position.x = float(last_idx) * float(x_step) + 0.08
        label.pose.position.y = float(path[-1][joint_idx]) * float(y_scale)
        label.pose.position.z = z

        markers.markers.append(label)

    return markers


def publish_joint_path_plot(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    topic: str = "/bidir_rrt/joint_path_plot",
    frame_id: str = "map",
    x_step: float = 0.05,
    y_scale: float = 1.0,
    z_separation: float = 0.25,
    line_width: float = 0.02,
    text_height: float = 0.12,
    marker_lifetime_s: float = 0.0,
    wait_subscriber_s: float = 1.0,
    publish_repeat: int = 1,
    publish_period_s: float = 0.1,
    keep_alive_s: float = 5.0,
    fk_path: Sequence[Sequence[float]] | None = None,
    fk_joint_names: Sequence[str] | None = None,
    fk_robot_yml: str = ROBOT_YAML,
    fk_base_frame: str = BASE_FRAME,
    fk_ee_frame: str | None = None,
    fk_line_width: float = 0.025,
    fk_cpu: bool = False,
    fk_fail_silently: bool = True,
) -> None:
    """Publish joint-space waypoints as an RViz2 MarkerArray plot."""
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if x_step <= 0.0:
        raise ValueError("x_step must be > 0")
    if line_width <= 0.0:
        raise ValueError("line_width must be > 0")
    if text_height <= 0.0:
        raise ValueError("text_height must be > 0")
    if fk_line_width <= 0.0:
        raise ValueError("fk_line_width must be > 0")

    n = len(joint_names)
    for i, q in enumerate(path):
        if len(q) != n:
            raise ValueError(f"path[{i}] length {len(q)} != len(joint_names) {n}")

    wants_fk = (fk_path is not None) or (fk_joint_names is not None) or (fk_ee_frame is not None)
    if wants_fk:
        if not fk_path:
            raise ValueError("fk_path is empty or not provided")
        if not fk_joint_names:
            raise ValueError("fk_joint_names is empty or not provided")
        if not fk_ee_frame:
            raise ValueError("fk_ee_frame is empty or not provided")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    qos = QoSProfile(depth=1)
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    node = Node("bidir_rrt_plot_publisher")
    try:
        pub = node.create_publisher(MarkerArray, topic, qos)

        t_end = time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok() and time.monotonic() < t_end and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)

        markers = _build_joint_plot_markers(
            path,
            joint_names,
            frame_id=frame_id,
            x_step=x_step,
            y_scale=y_scale,
            z_separation=z_separation,
            line_width=line_width,
            text_height=text_height,
            marker_lifetime_s=marker_lifetime_s,
        )

        if wants_fk and fk_path is not None and fk_joint_names is not None and fk_ee_frame is not None:
            try:
                fk_points = _compute_fk_points(
                    fk_path,
                    fk_joint_names,
                    robot_yml=fk_robot_yml,
                    base_frame=fk_base_frame,
                    ee_frame=fk_ee_frame,
                    cpu=fk_cpu,
                )
                _append_fk_plot_markers(
                    markers,
                    fk_points,
                    frame_id=fk_base_frame,
                    ee_frame_name=fk_ee_frame,
                    marker_lifetime_s=marker_lifetime_s,
                    line_width=fk_line_width,
                )
            except Exception as exc:
                if fk_fail_silently:
                    node.get_logger().warning(f"FK plot disabled: {exc}")
                else:
                    raise

        repeat = max(1, int(publish_repeat))
        for idx in range(repeat):
            now = node.get_clock().now().to_msg()
            for marker in markers.markers:
                marker.header.stamp = now
            pub.publish(markers)
            rclpy.spin_once(node, timeout_sec=0.0)
            if idx + 1 < repeat:
                time.sleep(max(0.0, float(publish_period_s)))

        # Keep publisher alive for discovery / late RViz subscription.
        if keep_alive_s < 0.0:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            t_keep_end = time.monotonic() + max(0.0, float(keep_alive_s))
            while rclpy.ok() and time.monotonic() < t_keep_end:
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()

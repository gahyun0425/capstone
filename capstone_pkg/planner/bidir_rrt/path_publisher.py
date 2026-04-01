from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


@dataclass(frozen=True)
class JointTrajectoryCommand:
    endpoint: str
    joint_names: Sequence[str]
    path: Sequence[Sequence[float]]
    label: str = ""


def _duration_from_seconds(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def _command_qos(*, transient_local: bool) -> QoSProfile:
    qos = QoSProfile(depth=10)
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = (
        QoSDurabilityPolicy.TRANSIENT_LOCAL
        if transient_local
        else QoSDurabilityPolicy.VOLATILE
    )
    return qos


def _wait_for_matching_subscribers(
    node: Node,
    pub,
    *,
    topic: str,
    wait_subscriber_s: float,
) -> bool:
    wait_forever = float(wait_subscriber_s) < 0.0
    t_end = None if wait_forever else time.monotonic() + float(wait_subscriber_s)
    next_log_t = 0.0

    while rclpy.ok() and pub.get_subscription_count() == 0:
        now = time.monotonic()
        if next_log_t == 0.0 or now >= next_log_t:
            node.get_logger().info(
                f"Waiting for at least 1 matching subscription(s) on {topic}..."
            )
            next_log_t = now + 1.0

        if not wait_forever and now >= t_end:
            break

        rclpy.spin_once(node, timeout_sec=0.05)

    count = pub.get_subscription_count()
    if count > 0:
        node.get_logger().info(f"Matched {count} subscription(s) on {topic}.")
        return True

    node.get_logger().warning(
        f"No subscribers detected on {topic} after waiting {max(0.0, float(wait_subscriber_s)):.2f}s."
    )
    return False


def _validate_joint_path(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    dt: float,
) -> None:
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


def _build_joint_trajectory(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    dt: float,
) -> JointTrajectory:
    _validate_joint_path(path, joint_names, dt)

    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    first_point_offset_s = float(dt) if len(path) == 1 else 0.0

    for idx, q in enumerate(path):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        # Single-point trajectories need a positive duration; a point at t=0 can
        # arrive after its end time once stamped and processed by the controller.
        p.time_from_start = _duration_from_seconds(first_point_offset_s + float(idx) * float(dt))
        msg.points.append(p)

    return msg


def _future_stamp(node: Node, *, delay_s: float):
    if delay_s <= 0.0:
        return node.get_clock().now().to_msg()
    return (node.get_clock().now() + RclpyDuration(seconds=float(delay_s))).to_msg()


def _validate_commands(
    commands: Sequence[JointTrajectoryCommand],
    *,
    dt: float,
) -> list[JointTrajectoryCommand]:
    if not commands:
        raise ValueError("commands is empty")

    normalized: list[JointTrajectoryCommand] = []
    for i, cmd in enumerate(commands):
        if not isinstance(cmd.endpoint, str) or not cmd.endpoint:
            raise ValueError(f"commands[{i}].endpoint must be a non-empty string")
        _validate_joint_path(cmd.path, cmd.joint_names, dt)
        normalized.append(cmd)
    return normalized


def read_joint_positions_once(
    joint_names: Sequence[str],
    *,
    topic: str = "/joint_states",
    wait_s: float = 2.0,
) -> list[float]:
    """Read one fresh JointState sample containing all requested joints."""
    if not joint_names:
        raise ValueError("joint_names is empty")
    if wait_s <= 0.0:
        raise ValueError("wait_s must be > 0")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("bidir_rrt_joint_state_reader")
    latest_by_name: dict[str, float] = {}
    saw_message = False

    def _cb(msg: JointState) -> None:
        nonlocal saw_message
        saw_message = True
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for joint_name in joint_names:
            idx = name_to_idx.get(joint_name)
            if idx is None or idx >= len(msg.position):
                continue
            latest_by_name[joint_name] = float(msg.position[idx])

    sub = node.create_subscription(JointState, topic, _cb, 10)

    t_end = time.monotonic() + float(wait_s)
    while rclpy.ok() and time.monotonic() < t_end:
        missing = [n for n in joint_names if n not in latest_by_name]
        if not missing:
            result = [latest_by_name[n] for n in joint_names]
            del sub
            node.destroy_node()
            if owns_rclpy:
                rclpy.shutdown()
            return result
        rclpy.spin_once(node, timeout_sec=0.05)

    missing = [n for n in joint_names if n not in latest_by_name]
    del sub
    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()

    if not saw_message:
        raise RuntimeError(f"No JointState received on {topic} within {wait_s:.2f}s")
    raise RuntimeError(
        f"Timed out waiting for joints on {topic}; missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}"
    )


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


def publish_joint_trajectory(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    topic: str,
    dt: float = 0.1,
    wait_subscriber_s: float = 1.0,
    require_subscriber: bool = True,
    retry_until_subscriber: bool = True,
    publish_repeat: int = 2,
    publish_period_s: float = 0.05,
    wait_ack_s: float = 1.0,
    keep_alive_s: float = 0.5,
    transient_local: bool = False,
) -> None:
    """Publish joint-space waypoints as a JointTrajectory command."""
    _validate_joint_path(path, joint_names, dt)
    if wait_ack_s < 0.0:
        raise ValueError("wait_ack_s must be >= 0")
    if keep_alive_s < 0.0:
        raise ValueError("keep_alive_s must be >= 0")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("bidir_rrt_traj_publisher")
    pub = node.create_publisher(
        JointTrajectory,
        topic,
        _command_qos(transient_local=transient_local),
    )

    msg = _build_joint_trajectory(path, joint_names, dt=dt)

    initial_wait_s = 0.0 if (retry_until_subscriber and float(wait_subscriber_s) < 0.0) else wait_subscriber_s
    matched = _wait_for_matching_subscribers(
        node,
        pub,
        topic=topic,
        wait_subscriber_s=initial_wait_s,
    )
    if not matched and retry_until_subscriber:
        node.get_logger().warning(
            f"No matching subscribers on {topic}; re-publishing until one appears."
        )
        next_log_t = 0.0
        while rclpy.ok() and pub.get_subscription_count() == 0:
            now = time.monotonic()
            if next_log_t == 0.0 or now >= next_log_t:
                node.get_logger().info(
                    f"Still waiting for at least 1 matching subscription(s) on {topic}..."
                )
                next_log_t = now + 1.0

            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(max(0.05, float(publish_period_s)))

        if pub.get_subscription_count() > 0:
            node.get_logger().info(
                f"Matched {pub.get_subscription_count()} subscription(s) on {topic} after retry loop."
            )
            matched = True

    if not matched:
        message = f"No subscribers detected on {topic} after waiting {max(0.0, float(wait_subscriber_s)):.2f}s"
        if require_subscriber:
            raise RuntimeError(message)
        node.get_logger().warning(f"{message}; publishing anyway.")

    repeats = max(1, int(publish_repeat))
    ack_timeout = RclpyDuration(seconds=float(wait_ack_s))
    for i in range(repeats):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        if wait_ack_s > 0.0 and not pub.wait_for_all_acked(ack_timeout):
            node.get_logger().warning(
                f"Timed out waiting for DDS acknowledgements on {topic} after {wait_ack_s:.2f}s."
            )
        if i + 1 < repeats:
            time.sleep(max(0.0, float(publish_period_s)))

    t_keep_end = time.monotonic() + float(keep_alive_s)
    while rclpy.ok() and time.monotonic() < t_keep_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()


def publish_joint_trajectory_group(
    commands: Sequence[JointTrajectoryCommand],
    *,
    dt: float = 0.1,
    wait_subscriber_s: float = 1.0,
    require_subscriber: bool = True,
    retry_until_subscriber: bool = True,
    publish_repeat: int = 2,
    publish_period_s: float = 0.05,
    wait_ack_s: float = 1.0,
    keep_alive_s: float = 0.5,
    transient_local: bool = False,
    start_time_delay_s: float = 0.0,
) -> None:
    """Publish multiple JointTrajectory commands with a shared start timestamp."""
    normalized = _validate_commands(commands, dt=dt)
    if wait_ack_s < 0.0:
        raise ValueError("wait_ack_s must be >= 0")
    if keep_alive_s < 0.0:
        raise ValueError("keep_alive_s must be >= 0")
    if start_time_delay_s < 0.0:
        raise ValueError("start_time_delay_s must be >= 0")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("joint_trajectory_group_publisher")
    try:
        publishers = [
            node.create_publisher(
                JointTrajectory,
                cmd.endpoint,
                _command_qos(transient_local=transient_local),
            )
            for cmd in normalized
        ]
        messages = [
            _build_joint_trajectory(cmd.path, cmd.joint_names, dt=dt)
            for cmd in normalized
        ]

        unresolved = list(range(len(normalized)))
        initial_wait_s = (
            0.0
            if (retry_until_subscriber and float(wait_subscriber_s) < 0.0)
            else float(wait_subscriber_s)
        )
        wait_deadline = None if initial_wait_s < 0.0 else time.monotonic() + max(0.0, initial_wait_s)
        next_log_t = 0.0

        while rclpy.ok() and unresolved:
            unresolved = [
                idx for idx in unresolved if publishers[idx].get_subscription_count() == 0
            ]
            if not unresolved:
                break

            now = time.monotonic()
            if wait_deadline is not None and now >= wait_deadline:
                break

            if next_log_t == 0.0 or now >= next_log_t:
                waiting = ", ".join(normalized[idx].endpoint for idx in unresolved)
                node.get_logger().info(
                    f"Waiting for at least 1 matching subscription(s) on: {waiting}"
                )
                next_log_t = now + 1.0

            rclpy.spin_once(node, timeout_sec=0.05)

        if unresolved and retry_until_subscriber:
            waiting = ", ".join(normalized[idx].endpoint for idx in unresolved)
            node.get_logger().warning(
                f"No matching subscribers on {waiting}; re-publishing until all appear."
            )
            next_log_t = 0.0
            while rclpy.ok() and unresolved:
                now = time.monotonic()
                if next_log_t == 0.0 or now >= next_log_t:
                    still_waiting = ", ".join(normalized[idx].endpoint for idx in unresolved)
                    node.get_logger().info(
                        f"Still waiting for at least 1 matching subscription(s) on: {still_waiting}"
                    )
                    next_log_t = now + 1.0

                stamp = _future_stamp(node, delay_s=float(start_time_delay_s))
                for idx in unresolved:
                    messages[idx].header.stamp = stamp
                    publishers[idx].publish(messages[idx])
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(max(0.05, float(publish_period_s)))

                unresolved = [
                    idx for idx in unresolved if publishers[idx].get_subscription_count() == 0
                ]

            if not unresolved:
                matched = ", ".join(cmd.endpoint for cmd in normalized)
                node.get_logger().info(f"Matched subscriptions on {matched}.")

        if unresolved:
            waiting = ", ".join(normalized[idx].endpoint for idx in unresolved)
            message = (
                f"No subscribers detected on {waiting} after waiting "
                f"{max(0.0, float(wait_subscriber_s)):.2f}s"
            )
            if require_subscriber:
                raise RuntimeError(message)
            node.get_logger().warning(f"{message}; publishing anyway.")

        repeats = max(1, int(publish_repeat))
        ack_timeout = RclpyDuration(seconds=float(wait_ack_s))
        for i in range(repeats):
            stamp = _future_stamp(node, delay_s=float(start_time_delay_s))
            for msg in messages:
                msg.header.stamp = stamp
            for idx, pub in enumerate(publishers):
                pub.publish(messages[idx])
            rclpy.spin_once(node, timeout_sec=0.0)

            if wait_ack_s > 0.0:
                for idx, pub in enumerate(publishers):
                    if not pub.wait_for_all_acked(ack_timeout):
                        node.get_logger().warning(
                            f"Timed out waiting for DDS acknowledgements on "
                            f"{normalized[idx].endpoint} after {wait_ack_s:.2f}s."
                        )
            if i + 1 < repeats:
                time.sleep(max(0.0, float(publish_period_s)))

        t_keep_end = time.monotonic() + float(keep_alive_s)
        while rclpy.ok() and time.monotonic() < t_keep_end:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()


def send_joint_trajectory_action(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    action_name: str,
    dt: float = 0.1,
    wait_server_s: float = 2.0,
    wait_result_s: float = -1.0,
) -> None:
    """Send a JointTrajectory through FollowJointTrajectory action and wait for the result."""
    _validate_joint_path(path, joint_names, dt)
    if wait_server_s < 0.0:
        raise ValueError("wait_server_s must be >= 0")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("bidir_rrt_traj_action_client")
    client = ActionClient(node, FollowJointTrajectory, action_name)

    if not client.wait_for_server(timeout_sec=float(wait_server_s)):
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(f"FollowJointTrajectory action server not available: {action_name}")

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = _build_joint_trajectory(path, joint_names, dt=dt)

    send_goal_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(
        node,
        send_goal_future,
        timeout_sec=max(1.0, float(wait_server_s) + 1.0),
    )
    if not send_goal_future.done():
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(f"Timed out sending FollowJointTrajectory goal: {action_name}")

    goal_handle = send_goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(f"FollowJointTrajectory goal rejected: {action_name}")

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(
        node,
        result_future,
        timeout_sec=None if wait_result_s < 0.0 else float(wait_result_s),
    )
    if not result_future.done():
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(f"Timed out waiting for FollowJointTrajectory result: {action_name}")

    wrapped_result = result_future.result()
    result = wrapped_result.result if wrapped_result is not None else None
    if result is None:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(f"FollowJointTrajectory returned no result: {action_name}")

    if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
        detail = result.error_string.strip()
        suffix = f": {detail}" if detail else ""
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        raise RuntimeError(
            f"FollowJointTrajectory failed on {action_name} with error_code {result.error_code}{suffix}"
        )

    node.destroy_node()
    if owns_rclpy:
        rclpy.shutdown()


def send_joint_trajectory_action_group(
    commands: Sequence[JointTrajectoryCommand],
    *,
    dt: float = 0.1,
    wait_server_s: float = 2.0,
    wait_result_s: float = -1.0,
    start_time_delay_s: float = 0.0,
) -> None:
    """Send multiple FollowJointTrajectory goals with a shared start timestamp."""
    normalized = _validate_commands(commands, dt=dt)
    if wait_server_s < 0.0:
        raise ValueError("wait_server_s must be >= 0")
    if start_time_delay_s < 0.0:
        raise ValueError("start_time_delay_s must be >= 0")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    node = Node("joint_trajectory_group_action_client")
    try:
        clients = [
            ActionClient(node, FollowJointTrajectory, cmd.endpoint)
            for cmd in normalized
        ]

        unavailable = []
        for idx, client in enumerate(clients):
            if not client.wait_for_server(timeout_sec=float(wait_server_s)):
                unavailable.append(normalized[idx].endpoint)
        if unavailable:
            raise RuntimeError(
                "FollowJointTrajectory action server not available: "
                + ", ".join(unavailable)
            )

        common_stamp = _future_stamp(node, delay_s=float(start_time_delay_s))
        send_goal_futures = []
        for idx, cmd in enumerate(normalized):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = _build_joint_trajectory(cmd.path, cmd.joint_names, dt=dt)
            goal.trajectory.header.stamp = common_stamp
            send_goal_futures.append(clients[idx].send_goal_async(goal))

        send_deadline = time.monotonic() + max(1.0, float(wait_server_s) + 1.0)
        pending_send = set(range(len(send_goal_futures)))
        while rclpy.ok() and pending_send:
            pending_send = {
                idx for idx in pending_send if not send_goal_futures[idx].done()
            }
            if not pending_send:
                break
            if time.monotonic() >= send_deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.05)

        if pending_send:
            endpoints = ", ".join(normalized[idx].endpoint for idx in sorted(pending_send))
            raise RuntimeError(f"Timed out sending FollowJointTrajectory goal: {endpoints}")

        goal_handles = []
        rejected = []
        for idx, future in enumerate(send_goal_futures):
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                rejected.append(normalized[idx].endpoint)
            goal_handles.append(goal_handle)
        if rejected:
            raise RuntimeError(
                "FollowJointTrajectory goal rejected: " + ", ".join(rejected)
            )

        result_futures = [goal_handle.get_result_async() for goal_handle in goal_handles]
        result_deadline = None if wait_result_s < 0.0 else time.monotonic() + float(wait_result_s)
        pending_results = set(range(len(result_futures)))
        while rclpy.ok() and pending_results:
            pending_results = {
                idx for idx in pending_results if not result_futures[idx].done()
            }
            if not pending_results:
                break
            if result_deadline is not None and time.monotonic() >= result_deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.05)

        if pending_results:
            endpoints = ", ".join(normalized[idx].endpoint for idx in sorted(pending_results))
            raise RuntimeError(
                f"Timed out waiting for FollowJointTrajectory result: {endpoints}"
            )

        failures = []
        for idx, future in enumerate(result_futures):
            wrapped_result = future.result()
            result = wrapped_result.result if wrapped_result is not None else None
            if result is None:
                failures.append(
                    f"{normalized[idx].endpoint}: FollowJointTrajectory returned no result"
                )
                continue
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                detail = result.error_string.strip()
                suffix = f": {detail}" if detail else ""
                failures.append(
                    f"{normalized[idx].endpoint}: error_code {result.error_code}{suffix}"
                )

        if failures:
            raise RuntimeError(
                "FollowJointTrajectory failed on "
                + "; ".join(failures)
            )
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()

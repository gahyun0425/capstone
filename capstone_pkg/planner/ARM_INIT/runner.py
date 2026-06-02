from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from capstone_pkg.planner.arm_rrt_common.path_publisher import (
    JointTrajectoryCommand,
    publish_joint_trajectory_group,
    send_joint_trajectory_action_group,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_motion import (
    SingleArmMotionPlan,
    build_active_joint_path,
    execute_single_arm_motion,
    normalize_arm_name,
)
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import (
    _publish_world_collision_for_mujoco,
    _resolve_world_yml,
    build_single_arm_parser,
    build_single_arm_tbrrt_config,
)
from capstone_pkg.planner.arm_rrt_common.spline import spline_interpolate_path
from capstone_pkg.planner.arm_rrt_common.dual_arm_runner import (
    _load_stored_dual_trajectory,
)
from capstone_pkg.planner.tbrrt.batch.single_arm_batch_conext import (
    plan_single_arm_tbrrt_batch_conext,
)
from capstone_pkg.utils.config import CSPACE_JOINT_NAMES_14, LEFT_JOINTS, RIGHT_JOINTS

_DEFAULT_STORED_TRAJECTORY_JSON = (
    "/home/gaga/capstone_ws/src/capstone_pkg/data/arm_cart_picking_trajectory.json"
)
_ZERO_GOAL_TOL = 1.0e-4


def _command_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _joint_state_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _arm_joint_names(arm: str) -> list[str]:
    return list(LEFT_JOINTS if arm == "left" else RIGHT_JOINTS)


def _compute_max_abs_joint_error(a: Sequence[float], b: Sequence[float]) -> float:
    dof = min(len(a), len(b))
    if dof <= 0:
        raise ValueError("joint vectors are empty")
    return max(abs(float(a[idx]) - float(b[idx])) for idx in range(dof))


def _all_joints_zero(
    q_cspace: Sequence[float],
    *,
    tol: float = _ZERO_GOAL_TOL,
) -> bool:
    if not q_cspace:
        return False
    return max(abs(float(v)) for v in q_cspace) <= float(tol)


def _project_full_path_to_active(
    path: Sequence[Sequence[float]],
    *,
    active_joint_names: Sequence[str],
) -> list[list[float]]:
    name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}
    active_idx = [name_to_idx[name] for name in active_joint_names]
    return [[float(q[idx]) for idx in active_idx] for q in path]


def _build_stored_cart_path(
    current_q_cspace: Sequence[float],
    *,
    stored_trajectory_json: str,
    fallback_publish_dt: float,
) -> tuple[list[list[float]], dict, float]:
    stored_path, _left_path, _right_path, loaded_payload = _load_stored_dual_trajectory(
        str(stored_trajectory_json)
    )
    publish_dt = float(loaded_payload.get("publish_dt", fallback_publish_dt))
    profile_path = [[float(v) for v in row] for row in stored_path]
    profile_start = [float(v) for v in stored_path[0]]
    current_q = [float(v) for v in current_q_cspace]

    if _compute_max_abs_joint_error(current_q, profile_start) <= float(_ZERO_GOAL_TOL):
        combined_path = profile_path
    else:
        blend_path = spline_interpolate_path(
            [current_q, profile_start],
            dt=max(1.0e-3, float(publish_dt)),
        )
        combined_path = blend_path + profile_path[1:]

    return combined_path, loaded_payload, publish_dt


def build_parser() -> argparse.ArgumentParser:
    ap = build_single_arm_parser(
        default_world_yml=None,
        collision_models=None,
        default_collision_model=None,
    )
    ap.description = "ARM_INIT sequence worker"
    ap.add_argument("--arm_init_start_topic", default="/arm_init_start")
    ap.add_argument("--arm_init_finish_topic", default="/arm_init_finish")
    ap.add_argument("--finish_publish_repeat", type=int, default=1)
    ap.add_argument("--finish_publish_period_s", type=float, default=0.05)
    ap.add_argument("--stored_trajectory_json", default=_DEFAULT_STORED_TRAJECTORY_JSON)
    return ap


class ArmInitNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("arm_init")
        self._args = args
        self._busy = False
        self._lock = threading.Lock()
        self._joint_state_cv = threading.Condition()
        self._joint_state_by_name: dict[str, float] = {}
        self._resolved_world_yml = _resolve_world_yml(
            args,
            collision_models=None,
            default_world_yml=None,
        )

        self._finish_pub = self.create_publisher(
            Bool,
            str(args.arm_init_finish_topic),
            _command_qos(),
        )
        self._start_sub = self.create_subscription(
            String,
            str(args.arm_init_start_topic),
            self._start_callback,
            _command_qos(),
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            str(args.joint_state_topic),
            self._joint_state_callback,
            _joint_state_qos(),
        )
        self._joint_state_cmd_pub = self.create_publisher(
            JointState,
            str(args.publish_topic),
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        world_label = self._resolved_world_yml if self._resolved_world_yml is not None else "none"
        self.get_logger().info(
            "Listening on "
            f"{args.arm_init_start_topic}, publishing finish on {args.arm_init_finish_topic}, "
            f"world_yml={world_label}, stored_trajectory={args.stored_trajectory_json}"
        )

    def _start_callback(self, msg: String) -> None:
        try:
            arm = normalize_arm_name(msg.data)
        except ValueError:
            self.get_logger().warn(
                f"Ignoring arm_init_start with invalid arm={msg.data!r}"
            )
            return

        with self._lock:
            if self._busy:
                self.get_logger().warn(
                    f"Ignoring arm_init_start for arm={arm}: previous request still running"
                )
                return
            self._busy = True

        worker = threading.Thread(
            target=self._process_request,
            args=(arm,),
            daemon=True,
        )
        worker.start()

    def _joint_state_callback(self, msg: JointState) -> None:
        updates = {}
        for name, position in zip(list(msg.name), list(msg.position)):
            if isinstance(name, str):
                updates[str(name)] = float(position)
        if not updates:
            return
        with self._joint_state_cv:
            self._joint_state_by_name.update(updates)
            self._joint_state_cv.notify_all()

    def _wait_for_joint_sample(
        self,
        joint_names: Sequence[str],
        *,
        wait_s: float,
    ) -> list[float]:
        deadline = time.monotonic() + max(0.0, float(wait_s))
        with self._joint_state_cv:
            while rclpy.ok():
                missing = [name for name in joint_names if name not in self._joint_state_by_name]
                if not missing:
                    return [float(self._joint_state_by_name[name]) for name in joint_names]

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    known = list(self._joint_state_by_name.keys())
                    if not known:
                        raise RuntimeError(
                            f"No JointState received on {self._args.joint_state_topic} "
                            f"within {float(wait_s):.2f}s"
                        )
                    raise RuntimeError(
                        f"Timed out waiting for joints on {self._args.joint_state_topic}; "
                        f"missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}"
                    )
                self._joint_state_cv.wait(timeout=min(0.1, remaining))

        raise RuntimeError("rclpy shutdown while waiting for JointState")

    def _publish_finish(self) -> None:
        repeat = max(1, int(self._args.finish_publish_repeat))
        period_s = max(0.0, float(self._args.finish_publish_period_s))
        msg = Bool()
        msg.data = True
        for _ in range(repeat):
            self._finish_pub.publish(msg)
            if period_s > 0.0:
                time.sleep(period_s)

    def _wait_for_publisher_match(
        self,
        pub,
        topic: str,
        *,
        wait_subscriber_s: float,
    ) -> bool:
        deadline = None if float(wait_subscriber_s) < 0.0 else time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok():
            if pub.get_subscription_count() > 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _publish_joint_state_path(
        self,
        joint_names: Sequence[str],
        path: Sequence[Sequence[float]],
        *,
        topic: str,
        dt: float,
    ) -> None:
        matched = self._wait_for_publisher_match(
            self._joint_state_cmd_pub,
            topic,
            wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
        )
        if not matched:
            self.get_logger().warning(
                f"No subscriber detected on {topic} after waiting "
                f"{max(0.0, float(self._args.publish_wait_subscriber_s)):.2f}s"
            )

        for idx, q in enumerate(path):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [str(name) for name in joint_names]
            msg.position = [float(v) for v in q]
            self._joint_state_cmd_pub.publish(msg)
            if idx + 1 < len(path):
                time.sleep(max(0.0, float(dt)))

        if path:
            last = JointState()
            last.header.stamp = self.get_clock().now().to_msg()
            last.name = [str(name) for name in joint_names]
            last.position = [float(v) for v in path[-1]]
            self._joint_state_cmd_pub.publish(last)

    def _wait_until_joint_positions(
        self,
        joint_names: Sequence[str],
        target_positions: Sequence[float],
        *,
        wait_s: float,
        tolerance: float,
        poll_period_s: float,
    ) -> tuple[bool, list[float], float]:
        deadline = None if float(wait_s) < 0.0 else time.monotonic() + float(wait_s)
        latest_positions = [float("nan") for _ in joint_names]
        latest_max_abs_err = float("inf")
        with self._joint_state_cv:
            while rclpy.ok():
                missing = [name for name in joint_names if name not in self._joint_state_by_name]
                if not missing:
                    latest_positions = [
                        float(self._joint_state_by_name[name]) for name in joint_names
                    ]
                    latest_max_abs_err = max(
                        abs(float(latest_positions[idx]) - float(target_positions[idx]))
                        for idx in range(len(joint_names))
                    )
                    if latest_max_abs_err <= float(tolerance):
                        return True, latest_positions, latest_max_abs_err
                if deadline is not None and time.monotonic() >= deadline:
                    return False, latest_positions, latest_max_abs_err
                self._joint_state_cv.wait(timeout=max(0.01, float(poll_period_s)))

        raise RuntimeError("rclpy shutdown while waiting for joint arrival")

    def _run_single_arm_init(
        self,
        arm: str,
        q_start_cspace: Sequence[float],
    ) -> SingleArmMotionPlan | None:
        q_goal_cspace = [float(v) for v in q_start_cspace]
        active_joint_names = _arm_joint_names(arm)
        name_to_idx = {name: idx for idx, name in enumerate(CSPACE_JOINT_NAMES_14)}
        for joint_name in active_joint_names:
            q_goal_cspace[name_to_idx[joint_name]] = 0.0

        start_active = [float(q_start_cspace[name_to_idx[name]]) for name in active_joint_names]
        if max(abs(v) for v in start_active) <= float(_ZERO_GOAL_TOL):
            self.get_logger().info(
                f"[ARM_INIT] arm={arm} is already at zero goal within tol={_ZERO_GOAL_TOL:.1e}"
            )
            return None

        self.get_logger().info(
            f"[ARM_INIT] planning single-arm TB-RRT to zero pose for arm={arm}"
        )
        out = plan_single_arm_tbrrt_batch_conext(
            robot_yml=str(self._args.robot_yml),
            arm=arm,
            q_start=q_start_cspace,
            q_goals=[q_goal_cspace],
            world_yml=self._resolved_world_yml,
            cpu=bool(self._args.cpu),
            cfg=build_single_arm_tbrrt_config(self._args),
            joint_limit_yml=str(self._args.joint_limit_yml),
            block_k=int(self._args.tbrrt_block_k),
        )
        if not out.success or not out.path:
            raise RuntimeError(f"ARM_INIT single-arm TBRRT failed: {out.stats.extra}")

        self.get_logger().info(
            f"[ARM_INIT] arm={arm} path_len={len(out.path)} "
            f"iters={out.stats.iters} time={out.stats.time_sec:.3f}s"
        )
        return SingleArmMotionPlan(
            arm=arm,
            cspace_joint_names=list(CSPACE_JOINT_NAMES_14),
            active_joint_names=active_joint_names,
            q_start_cspace=[float(v) for v in q_start_cspace],
            q_goal_cspace=[float(v) for v in q_goal_cspace],
            raw_path=[[float(v) for v in row] for row in out.path],
            spline_path=[[float(v) for v in row] for row in out.path],
        )

    def _execute_single_arm_plan(self, plan: SingleArmMotionPlan) -> None:
        if str(self._args.publish_mode) == "real":
            execute_single_arm_motion(plan, self._args)
            return

        joint_names, active_path = build_active_joint_path(plan)
        self._publish_joint_state_path(
            joint_names,
            active_path,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
        )
        arrived, _positions, max_abs_err = self._wait_until_joint_positions(
            joint_names,
            active_path[-1],
            wait_s=max(
                2.0,
                float(max(0, len(active_path) - 1)) * float(self._args.publish_dt) + 2.0,
            )
            if float(getattr(self._args, "arrival_wait_s", -1.0)) < 0.0
            else float(getattr(self._args, "arrival_wait_s", -1.0)),
            tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
            poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
        )
        if not arrived:
            raise RuntimeError(
                f"single-arm init arrival failed for arm={plan.arm}: "
                f"max_abs_err={max_abs_err:.6f}"
            )

    def _publish_dual_cart_profile(
        self,
        q_path: Sequence[Sequence[float]],
    ) -> None:
        left_path = _project_full_path_to_active(
            q_path,
            active_joint_names=LEFT_JOINTS,
        )
        right_path = _project_full_path_to_active(
            q_path,
            active_joint_names=RIGHT_JOINTS,
        )

        if str(self._args.publish_mode) == "real":
            topic_commands = [
                JointTrajectoryCommand(
                    endpoint=str(self._args.real_left_topic),
                    joint_names=list(LEFT_JOINTS),
                    path=left_path,
                    label="left",
                ),
                JointTrajectoryCommand(
                    endpoint=str(self._args.real_right_topic),
                    joint_names=list(RIGHT_JOINTS),
                    path=right_path,
                    label="right",
                ),
            ]
            action_commands = [
                JointTrajectoryCommand(
                    endpoint=str(self._args.real_left_action),
                    joint_names=list(LEFT_JOINTS),
                    path=left_path,
                    label="left",
                ),
                JointTrajectoryCommand(
                    endpoint=str(self._args.real_right_action),
                    joint_names=list(RIGHT_JOINTS),
                    path=right_path,
                    label="right",
                ),
            ]

            if bool(self._args.real_use_action):
                try:
                    send_joint_trajectory_action_group(
                        action_commands,
                        dt=float(self._args.publish_dt),
                        wait_server_s=float(self._args.action_wait_server_s),
                        wait_result_s=float(self._args.action_wait_result_s),
                        start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
                    )
                    return
                except RuntimeError:
                    if not bool(self._args.real_action_fallback_to_topic):
                        raise

            publish_joint_trajectory_group(
                topic_commands,
                dt=float(self._args.publish_dt),
                wait_subscriber_s=float(self._args.publish_wait_subscriber_s),
                require_subscriber=bool(self._args.publish_require_subscriber),
                retry_until_subscriber=bool(self._args.publish_retry_until_subscriber),
                publish_repeat=int(self._args.publish_repeat),
                publish_period_s=float(self._args.publish_period_s),
                wait_ack_s=float(self._args.publish_wait_ack_s),
                keep_alive_s=float(self._args.publish_keep_alive_s),
                reliability=str(getattr(self._args, "publish_reliability", "best_effort")),
                durability=(
                    "transient_local"
                    if bool(getattr(self._args, "publish_transient_local", False))
                    else str(getattr(self._args, "publish_durability", "volatile"))
                ),
                qos_depth=int(getattr(self._args, "publish_qos_depth", 1)),
                start_time_delay_s=float(getattr(self._args, "start_delay_s", 0.2)),
            )
            return

        self._publish_joint_state_path(
            list(CSPACE_JOINT_NAMES_14),
            q_path,
            topic=str(self._args.publish_topic),
            dt=float(self._args.publish_dt),
        )

    def _wait_for_dual_profile_arrival(
        self,
        q_path: Sequence[Sequence[float]],
    ) -> None:
        left_goal = _project_full_path_to_active(
            [q_path[-1]],
            active_joint_names=LEFT_JOINTS,
        )[0]
        right_goal = _project_full_path_to_active(
            [q_path[-1]],
            active_joint_names=RIGHT_JOINTS,
        )[0]
        wait_s = max(
            2.0,
            float(max(0, len(q_path) - 1)) * float(self._args.publish_dt) + 2.0,
        )
        configured_wait_s = float(getattr(self._args, "arrival_wait_s", -1.0))
        if configured_wait_s >= 0.0:
            wait_s = configured_wait_s

        for arm_name, joint_names, goal in (
            ("left", LEFT_JOINTS, left_goal),
            ("right", RIGHT_JOINTS, right_goal),
        ):
            arrived, _positions, max_abs_err = self._wait_until_joint_positions(
                joint_names,
                goal,
                wait_s=wait_s,
                tolerance=float(getattr(self._args, "arrival_joint_tolerance", 0.05)),
                poll_period_s=float(getattr(self._args, "arrival_poll_s", 0.05)),
            )
            if not arrived:
                raise RuntimeError(
                    f"dual cart profile arrival failed for arm={arm_name}: "
                    f"max_abs_err={max_abs_err:.6f}"
                )

    def _process_request(self, arm: str) -> None:
        try:
            self.get_logger().info(f"Received arm_init_start arm={arm}")
            _publish_world_collision_for_mujoco(self._args, self._resolved_world_yml)

            q_start_cspace = self._wait_for_joint_sample(
                list(CSPACE_JOINT_NAMES_14),
                wait_s=float(self._args.joint_state_wait_s),
            )
            if _all_joints_zero(q_start_cspace):
                self.get_logger().info(
                    "[ARM_INIT] all joints are already at zero; skipping single-arm init "
                    "and starting cart profile immediately"
                )
            else:
                init_plan = self._run_single_arm_init(arm, q_start_cspace)
                if init_plan is not None:
                    self._execute_single_arm_plan(init_plan)

            q_after_init = self._wait_for_joint_sample(
                list(CSPACE_JOINT_NAMES_14),
                wait_s=float(self._args.joint_state_wait_s),
            )
            cart_path, loaded_trajectory, loaded_publish_dt = _build_stored_cart_path(
                q_after_init,
                stored_trajectory_json=str(self._args.stored_trajectory_json),
                fallback_publish_dt=float(self._args.publish_dt),
            )
            self._args.publish_dt = float(loaded_publish_dt)
            self.get_logger().info(
                "[ARM_INIT] starting dual-arm stored trajectory "
                f"file={self._args.stored_trajectory_json} "
                f"path_len={len(cart_path)}"
            )
            self._publish_dual_cart_profile(cart_path)
            self._wait_for_dual_profile_arrival(cart_path)

            if self._args.save:
                payload = {
                    "mode": "arm_init",
                    "arm": arm,
                    "world_yml": self._resolved_world_yml,
                    "q_start_cspace": [float(v) for v in q_start_cspace],
                    "q_after_init_cspace": [float(v) for v in q_after_init],
                    "stored_trajectory_json": str(self._args.stored_trajectory_json),
                    "stored_trajectory_mode": str(loaded_trajectory.get("mode", "")),
                    "combined_path": [[float(v) for v in row] for row in cart_path],
                    "left_path": _project_full_path_to_active(
                        cart_path,
                        active_joint_names=LEFT_JOINTS,
                    ),
                    "right_path": _project_full_path_to_active(
                        cart_path,
                        active_joint_names=RIGHT_JOINTS,
                    ),
                    "publish_dt": float(self._args.publish_dt),
                }
                with open(str(self._args.save), "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self.get_logger().info(f"[ARM_INIT] saved execution payload to {self._args.save}")

            self._publish_finish()
            self.get_logger().info(f"[ARM_INIT] completed arm={arm}")
        except Exception as exc:
            self.get_logger().error(f"arm_init failed for arm={arm}: {exc}")
        finally:
            with self._lock:
                self._busy = False


def main_arm_init(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    args, _unknown = build_parser().parse_known_args(argv_list)

    rclpy.init(args=argv_list)
    node = ArmInitNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ARM_INIT interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_init(argv)


if __name__ == "__main__":
    raise SystemExit(main())

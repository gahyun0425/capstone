from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping, Sequence


PHASE_APPROACH = 0
PHASE_OPENING = 1
PHASE_FRONT = 2


@dataclass(frozen=True)
class BaseDoorGraphConfig:
    steps: int
    goal_alpha_rad: float
    lambda_step_rad: float
    base_motion_start_alpha_rad: float
    closed_handle_xyz: tuple[float, float, float]
    door_hinge_xyz: tuple[float, float, float]
    start_base_xyyaw: tuple[float, float, float]
    opening_end_xyyaw: tuple[float, float, float]
    opening_span_xyyaw: tuple[float, float, float]
    front_goal_xyyaw: tuple[float, float, float]
    front_goal_tol_m: float
    front_goal_tol_yaw_rad: float
    xy_step_m: float
    yaw_step_rad: float
    max_expansions: int
    layer_keep: int
    max_candidates_per_layer: int
    bound_margin_m: float
    bound_margin_yaw_rad: float
    base_radius_m: float
    base_height_m: float
    collision_margin_m: float
    reach_shoulder_xyz: tuple[float, float, float]
    reach_min_m: float
    reach_max_m: float
    reach_z_min_m: float
    reach_z_max_m: float
    reach_nominal_m: float
    world: Mapping[str, Any] | None
    moving_collision_names: tuple[str, ...]
    ignore_collision_names: tuple[str, ...]


@dataclass(frozen=True)
class BaseDoorGraphPlan:
    base_poses: list[list[float]]
    door_alphas_rad: list[float]
    phases: list[int]
    cost: float
    expansions: int
    generated: int
    rejected_collision: int
    rejected_reach: int
    rejected_lambda_overlap: int


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _rotate_z(vec_xyz: Sequence[float], yaw_rad: float) -> list[float]:
    x, y, z = [float(v) for v in vec_xyz]
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return [c * x - s * y, s * x + c * y, z]


def _yaw_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in q_wxyz]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _handle_xyz_at_alpha(cfg: BaseDoorGraphConfig, alpha_rad: float) -> list[float]:
    hinge = list(cfg.door_hinge_xyz)
    closed = list(cfg.closed_handle_xyz)
    rel = [closed[i] - hinge[i] for i in range(3)]
    rel_rot = _rotate_z(rel, alpha_rad)
    return [hinge[i] + rel_rot[i] for i in range(3)]


def _world_point_to_base(point_xyz: Sequence[float], base_xyyaw: Sequence[float]) -> list[float]:
    bx, by, byaw = [float(v) for v in base_xyyaw]
    rel = [float(point_xyz[0]) - bx, float(point_xyz[1]) - by, float(point_xyz[2])]
    return _rotate_z(rel, -byaw)


def _assist_fraction(cfg: BaseDoorGraphConfig, alpha_rad: float) -> float:
    start = float(cfg.base_motion_start_alpha_rad)
    goal = max(start + 1.0e-6, float(cfg.goal_alpha_rad))
    return max(0.0, min(1.0, (float(alpha_rad) - start) / (goal - start)))


def _opening_center_span(
    cfg: BaseDoorGraphConfig,
    alpha_rad: float,
) -> tuple[list[float], list[float]]:
    frac = _assist_fraction(cfg, alpha_rad)
    center = [
        cfg.opening_end_xyyaw[0] * frac,
        cfg.opening_end_xyyaw[1] * frac,
        cfg.opening_end_xyyaw[2] * frac,
    ]
    span = [
        abs(cfg.opening_span_xyyaw[0]) * frac,
        abs(cfg.opening_span_xyyaw[1]) * frac,
        abs(cfg.opening_span_xyyaw[2]) * frac,
    ]
    return center, span


def _state_to_pose(state: tuple[int, int, int, int, int], cfg: BaseDoorGraphConfig) -> list[float]:
    ix, iy, iyaw, _, _ = state
    return [
        float(ix) * cfg.xy_step_m,
        float(iy) * cfg.xy_step_m,
        _wrap_pi(float(iyaw) * cfg.yaw_step_rad),
    ]


def _pose_to_indices(pose_xyyaw: Sequence[float], cfg: BaseDoorGraphConfig) -> tuple[int, int, int]:
    return (
        int(round(float(pose_xyyaw[0]) / cfg.xy_step_m)),
        int(round(float(pose_xyyaw[1]) / cfg.xy_step_m)),
        int(round(float(pose_xyyaw[2]) / cfg.yaw_step_rad)),
    )


def _alpha_from_idx(alpha_idx: int, cfg: BaseDoorGraphConfig) -> float:
    if cfg.steps <= 0:
        return 0.0
    return cfg.goal_alpha_rad * float(alpha_idx) / float(cfg.steps)


def _alpha_key(alpha_rad: float) -> int:
    return int(round(float(alpha_rad) * 1.0e6))


def _alpha_grid(cfg: BaseDoorGraphConfig) -> list[float]:
    goal = float(cfg.goal_alpha_rad)
    if abs(goal) <= 1.0e-12:
        return [0.0]
    step = max(1.0e-6, abs(float(cfg.lambda_step_rad)))
    n = max(1, int(math.ceil(abs(goal) / step)))
    values = [goal * float(i) / float(n) for i in range(n + 1)]
    values.extend(_alpha_from_idx(i, cfg) for i in range(cfg.steps + 1))
    out: list[float] = []
    seen: set[int] = set()
    for value in sorted(values):
        key = _alpha_key(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out


def _front_goal_distance(cfg: BaseDoorGraphConfig, pose_xyyaw: Sequence[float]) -> tuple[float, float]:
    dx = float(pose_xyyaw[0]) - cfg.front_goal_xyyaw[0]
    dy = float(pose_xyyaw[1]) - cfg.front_goal_xyyaw[1]
    dyaw = _wrap_pi(float(pose_xyyaw[2]) - cfg.front_goal_xyyaw[2])
    return math.hypot(dx, dy), abs(dyaw)


def _is_front_goal(cfg: BaseDoorGraphConfig, pose_xyyaw: Sequence[float]) -> bool:
    dist_xy, dist_yaw = _front_goal_distance(cfg, pose_xyyaw)
    return dist_xy <= cfg.front_goal_tol_m and dist_yaw <= cfg.front_goal_tol_yaw_rad


def _pose_in_bounds(
    cfg: BaseDoorGraphConfig,
    *,
    pose_xyyaw: Sequence[float],
    alpha_rad: float,
    phase: int,
) -> bool:
    if phase == PHASE_FRONT:
        open_center, open_span = _opening_center_span(cfg, cfg.goal_alpha_rad)
        xs = [
            cfg.start_base_xyyaw[0],
            open_center[0] - open_span[0],
            open_center[0] + open_span[0],
            cfg.front_goal_xyyaw[0],
        ]
        ys = [
            cfg.start_base_xyyaw[1],
            open_center[1] - open_span[1],
            open_center[1] + open_span[1],
            cfg.front_goal_xyyaw[1],
        ]
        yaws = [
            cfg.start_base_xyyaw[2],
            open_center[2] - open_span[2],
            open_center[2] + open_span[2],
            cfg.front_goal_xyyaw[2],
        ]
        x_ok = min(xs) - cfg.bound_margin_m <= float(pose_xyyaw[0]) <= max(xs) + cfg.bound_margin_m
        y_ok = min(ys) - cfg.bound_margin_m <= float(pose_xyyaw[1]) <= max(ys) + cfg.bound_margin_m
        yaw_ok = min(yaws) - cfg.bound_margin_yaw_rad <= float(pose_xyyaw[2]) <= max(yaws) + cfg.bound_margin_yaw_rad
        return x_ok and y_ok and yaw_ok

    center, span = _opening_center_span(cfg, alpha_rad)
    return (
        abs(float(pose_xyyaw[0]) - center[0]) <= span[0] + cfg.bound_margin_m
        and abs(float(pose_xyyaw[1]) - center[1]) <= span[1] + cfg.bound_margin_m
        and abs(_wrap_pi(float(pose_xyyaw[2]) - center[2])) <= span[2] + cfg.bound_margin_yaw_rad
    )


def _reach_cost(
    cfg: BaseDoorGraphConfig,
    *,
    pose_xyyaw: Sequence[float],
    alpha_rad: float,
) -> tuple[bool, float]:
    handle_base = _world_point_to_base(_handle_xyz_at_alpha(cfg, alpha_rad), pose_xyyaw)
    sx, sy, _ = cfg.reach_shoulder_xyz
    d_xy = math.hypot(handle_base[0] - sx, handle_base[1] - sy)
    if d_xy < cfg.reach_min_m or d_xy > cfg.reach_max_m:
        return False, 0.0
    if handle_base[2] < cfg.reach_z_min_m or handle_base[2] > cfg.reach_z_max_m:
        return False, 0.0
    return True, 2.0 * (d_xy - cfg.reach_nominal_m) ** 2


def _rotated_moving_cuboid_pose(
    pose_wxyz: Sequence[float],
    *,
    cfg: BaseDoorGraphConfig,
    alpha_rad: float,
) -> list[float]:
    hinge = list(cfg.door_hinge_xyz)
    xyz = [float(v) for v in pose_wxyz[:3]]
    quat = [float(v) for v in pose_wxyz[3:7]]
    rel = [xyz[i] - hinge[i] for i in range(3)]
    rel_rot = _rotate_z(rel, alpha_rad)
    yaw = _yaw_from_quat_wxyz(quat) + float(alpha_rad)
    half = 0.5 * yaw
    return [hinge[i] + rel_rot[i] for i in range(3)] + [math.cos(half), 0.0, 0.0, math.sin(half)]


def _base_collision_cost(
    cfg: BaseDoorGraphConfig,
    *,
    pose_xyyaw: Sequence[float],
    alpha_rad: float,
) -> tuple[bool, float]:
    if not cfg.world:
        return True, 0.0
    cuboids = cfg.world.get("cuboid", {}) if isinstance(cfg.world, Mapping) else {}
    if not isinstance(cuboids, Mapping):
        return True, 0.0

    min_clearance = float("inf")
    ignored = set(cfg.ignore_collision_names)
    moving = set(cfg.moving_collision_names)
    bx = float(pose_xyyaw[0])
    by = float(pose_xyyaw[1])
    base_radius = float(cfg.base_radius_m) + float(cfg.collision_margin_m)

    for name, cuboid in cuboids.items():
        if str(name) in ignored or not isinstance(cuboid, Mapping):
            continue
        dims = cuboid.get("dims")
        pose = cuboid.get("pose")
        if not isinstance(dims, Sequence) or not isinstance(pose, Sequence):
            continue
        if len(dims) != 3 or len(pose) != 7:
            continue

        pose_wxyz = [float(v) for v in pose]
        if str(name) in moving:
            pose_wxyz = _rotated_moving_cuboid_pose(pose_wxyz, cfg=cfg, alpha_rad=alpha_rad)
        cx, cy, cz = pose_wxyz[:3]
        half_x = 0.5 * float(dims[0])
        half_y = 0.5 * float(dims[1])
        half_z = 0.5 * float(dims[2])
        if cz - half_z > cfg.base_height_m or cz + half_z < 0.0:
            continue

        yaw = _yaw_from_quat_wxyz(pose_wxyz[3:7])
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        dx = bx - cx
        dy = by - cy
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        qx = abs(lx) - half_x
        qy = abs(ly) - half_y
        outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        signed_dist = outside + inside
        clearance = signed_dist - base_radius
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            return False, 0.0

    if min_clearance == float("inf"):
        return True, 0.0
    return True, 0.05 / max(0.02, min_clearance + 0.02)


LambdaEntry = tuple[float, float]
LambdaCache = dict[tuple[int, int, int], list[LambdaEntry]]
State = tuple[int, int, int, int, int]


def _lambda_for_pose(
    cfg: BaseDoorGraphConfig,
    *,
    pose_xyyaw: Sequence[float],
    pose_key: tuple[int, int, int],
    alpha_grid: Sequence[float],
    cache: LambdaCache,
) -> list[LambdaEntry]:
    # Chitta et al., ICRA 2010 Sec. IV-A: Lambda(s) is the set of door
    # angles that keep the handle reachable and the door/base collision-free.
    cached = cache.get(pose_key)
    if cached is not None:
        return cached

    entries: list[LambdaEntry] = []
    for alpha in alpha_grid:
        reachable, reach_cost = _reach_cost(cfg, pose_xyyaw=pose_xyyaw, alpha_rad=float(alpha))
        if not reachable:
            continue
        collision_free, collision_cost = _base_collision_cost(
            cfg,
            pose_xyyaw=pose_xyyaw,
            alpha_rad=float(alpha),
        )
        if not collision_free:
            continue
        entries.append((float(alpha), float(reach_cost) + float(collision_cost)))

    cache[pose_key] = entries
    return entries


def _lambda_for_state(
    state: State,
    cfg: BaseDoorGraphConfig,
    *,
    alpha_grid: Sequence[float],
    cache: LambdaCache,
) -> list[LambdaEntry]:
    pose = _state_to_pose(state, cfg)
    return _lambda_for_pose(
        cfg,
        pose_xyyaw=pose,
        pose_key=(state[0], state[1], state[2]),
        alpha_grid=alpha_grid,
        cache=cache,
    )


def _lambda_cost_at_alpha(entries: Sequence[LambdaEntry], alpha_rad: float) -> float | None:
    key = _alpha_key(alpha_rad)
    for alpha, cost in entries:
        if _alpha_key(alpha) == key:
            return float(cost)
    return None


def _lambda_overlap_cost(
    parent_entries: Sequence[LambdaEntry],
    child_entries: Sequence[LambdaEntry],
) -> float | None:
    parent_by_alpha = {_alpha_key(alpha): float(cost) for alpha, cost in parent_entries}
    best: float | None = None
    for alpha, child_cost in child_entries:
        parent_cost = parent_by_alpha.get(_alpha_key(alpha))
        if parent_cost is None:
            continue
        overlap_cost = 0.5 * (parent_cost + float(child_cost))
        if best is None or overlap_cost < best:
            best = overlap_cost
    return best


def _successor_actions(cfg: BaseDoorGraphConfig) -> list[tuple[int, int, int]]:
    actions: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            actions.append((dx, dy, 0))
    actions.extend([(0, 0, -1), (0, 0, 1)])
    actions.extend([(-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1)])
    return actions


def _successors(
    state: State,
    cfg: BaseDoorGraphConfig,
) -> list[tuple[State, float]]:
    ix, iy, iyaw, alpha_idx, phase = state
    out: list[tuple[State, float]] = []
    if phase == PHASE_APPROACH:
        next_alpha = min(cfg.steps, alpha_idx + 1)
        next_phase = PHASE_OPENING
    elif phase == PHASE_OPENING and alpha_idx < cfg.steps:
        next_alpha = alpha_idx + 1
        next_phase = PHASE_OPENING
    else:
        next_alpha = cfg.steps
        next_phase = PHASE_FRONT

    for dx, dy, dyaw in _successor_actions(cfg):
        if phase == PHASE_APPROACH and dx == 0 and dy == 0 and dyaw == 0:
            pass
        child = (ix + dx, iy + dy, iyaw + dyaw, next_alpha, next_phase)
        move_cost = math.hypot(float(dx) * cfg.xy_step_m, float(dy) * cfg.xy_step_m)
        turn_cost = 0.4 * abs(float(dyaw) * cfg.yaw_step_rad)
        alpha_cost = 0.02 if next_alpha != alpha_idx else 0.0
        phase_cost = 0.01 if next_phase != phase else 0.0
        out.append((child, move_cost + turn_cost + alpha_cost + phase_cost))
    return out


def _valid_state_cost(
    state: State,
    cfg: BaseDoorGraphConfig,
    *,
    alpha_grid: Sequence[float],
    cache: LambdaCache,
) -> tuple[bool, float, str]:
    pose = _state_to_pose(state, cfg)
    alpha = _alpha_from_idx(state[3], cfg)
    phase = state[4]
    if not _pose_in_bounds(cfg, pose_xyyaw=pose, alpha_rad=alpha, phase=phase):
        return False, 0.0, "bounds"
    reachable, reach_cost = _reach_cost(cfg, pose_xyyaw=pose, alpha_rad=alpha)
    if not reachable:
        return False, 0.0, "reach"
    collision_free, collision_cost = _base_collision_cost(cfg, pose_xyyaw=pose, alpha_rad=alpha)
    if not collision_free:
        return False, 0.0, "collision"
    lambda_entries = _lambda_for_state(state, cfg, alpha_grid=alpha_grid, cache=cache)
    target_cost = _lambda_cost_at_alpha(lambda_entries, alpha)
    if target_cost is None:
        return False, 0.0, "lambda"
    return True, float(target_cost), ""


def _heuristic(state: State, cfg: BaseDoorGraphConfig) -> float:
    pose = _state_to_pose(state, cfg)
    dist_xy, dist_yaw = _front_goal_distance(cfg, pose)
    alpha_remaining = max(0, cfg.steps - int(state[3]))
    return dist_xy + 0.4 * dist_yaw + 0.02 * float(alpha_remaining)


def _axis_samples(center: float, span: float, step: float) -> list[float]:
    span = abs(float(span))
    step = max(1.0e-6, abs(float(step)))
    if span <= 1.0e-9:
        return [float(center)]
    lo = float(center) - span
    hi = float(center) + span
    n = max(1, int(math.ceil((hi - lo) / step)))
    values = [lo + (hi - lo) * float(i) / float(n) for i in range(n + 1)]
    values.append(float(center))
    out: list[float] = []
    seen: set[int] = set()
    for value in values:
        key = int(round(float(value) / step))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out


def _layer_candidate_states(
    cfg: BaseDoorGraphConfig,
    *,
    alpha_idx: int,
) -> list[State]:
    alpha = _alpha_from_idx(alpha_idx, cfg)
    center, span = _opening_center_span(cfg, alpha)
    xs = _axis_samples(center[0], span[0], cfg.xy_step_m)
    ys = _axis_samples(center[1], span[1], cfg.xy_step_m)
    yaws = _axis_samples(center[2], span[2], cfg.yaw_step_rad)

    states: list[State] = []
    seen: set[State] = set()

    def _add(pose: Sequence[float], phase: int) -> None:
        ix, iy, iyaw = _pose_to_indices(pose, cfg)
        state = (ix, iy, iyaw, int(alpha_idx), int(phase))
        if state in seen:
            return
        seen.add(state)
        states.append(state)

    _add(center, PHASE_OPENING)
    for x in xs:
        for y in ys:
            for yaw in yaws:
                _add([x, y, yaw], PHASE_OPENING)

    if alpha_idx >= cfg.steps:
        _add(cfg.front_goal_xyyaw, PHASE_FRONT)

    def _rank(state: State) -> float:
        pose = _state_to_pose(state, cfg)
        dist_center = math.sqrt(
            (pose[0] - center[0]) ** 2
            + (pose[1] - center[1]) ** 2
            + _wrap_pi(pose[2] - center[2]) ** 2
        )
        dist_goal, yaw_goal = _front_goal_distance(cfg, pose)
        phase_bonus = -1.0 if state[4] == PHASE_FRONT else 0.0
        return dist_center + (0.25 * dist_goal + 0.1 * yaw_goal if alpha_idx >= cfg.steps else 0.0) + phase_bonus

    states.sort(key=_rank)
    cap = int(cfg.max_candidates_per_layer)
    if cap > 0 and len(states) > cap:
        front_states = [state for state in states if state[4] == PHASE_FRONT]
        kept = states[:cap]
        for state in front_states:
            if state not in kept:
                kept[-1] = state
                break
        states = kept
    return states


def _state_transition_cost(
    cfg: BaseDoorGraphConfig,
    parent: State,
    child: State,
) -> float:
    p0 = _state_to_pose(parent, cfg)
    p1 = _state_to_pose(child, cfg)
    dxy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    dyaw = abs(_wrap_pi(p1[2] - p0[2]))
    dalpha = abs(float(child[3] - parent[3])) / max(1.0, float(cfg.steps))
    phase_cost = 0.02 if child[4] != parent[4] else 0.0
    return dxy + 0.4 * dyaw + 0.02 * dalpha + phase_cost


def plan_base_door_graph(cfg: BaseDoorGraphConfig) -> BaseDoorGraphPlan:
    steps = max(2, int(cfg.steps))
    cfg = BaseDoorGraphConfig(**{**cfg.__dict__, "steps": steps})
    alpha_grid = _alpha_grid(cfg)
    lambda_cache: LambdaCache = {}
    start_idx = _pose_to_indices(cfg.start_base_xyyaw, cfg)
    start_state = (start_idx[0], start_idx[1], start_idx[2], 0, PHASE_APPROACH)
    ok, start_extra_cost, reason = _valid_state_cost(
        start_state,
        cfg,
        alpha_grid=alpha_grid,
        cache=lambda_cache,
    )
    if not ok:
        raise RuntimeError(f"base-door graph start state is invalid: {reason}")

    previous: list[tuple[float, State, list[State]]] = [
        (start_extra_cost, start_state, [])
    ]
    generated = 0
    expanded = 0
    rejected_collision = 0
    rejected_reach = 0
    rejected_lambda_overlap = 0

    for alpha_idx in range(1, cfg.steps + 1):
        candidates = _layer_candidate_states(cfg, alpha_idx=alpha_idx)
        next_layer: list[tuple[float, State, list[State]]] = []
        for child in candidates:
            generated += 1
            ok, state_cost, reason = _valid_state_cost(
                child,
                cfg,
                alpha_grid=alpha_grid,
                cache=lambda_cache,
            )
            if not ok:
                if reason == "collision":
                    rejected_collision += 1
                elif reason == "reach":
                    rejected_reach += 1
                continue

            best_item = None
            for parent_cost, parent_state, parent_path in previous:
                expanded += 1
                if expanded > cfg.max_expansions:
                    break
                overlap_cost = _lambda_overlap_cost(
                    _lambda_for_state(
                        parent_state,
                        cfg,
                        alpha_grid=alpha_grid,
                        cache=lambda_cache,
                    ),
                    _lambda_for_state(
                        child,
                        cfg,
                        alpha_grid=alpha_grid,
                        cache=lambda_cache,
                    ),
                )
                if overlap_cost is None:
                    rejected_lambda_overlap += 1
                    continue
                edge_cost = _state_transition_cost(cfg, parent_state, child)
                cost = parent_cost + edge_cost + state_cost + 0.2 * float(overlap_cost)
                if best_item is None or cost < best_item[0]:
                    best_item = (cost, child, parent_path + [child])
            if expanded > cfg.max_expansions:
                break
            if best_item is not None:
                next_layer.append(best_item)
        if expanded > cfg.max_expansions:
            break
        if not next_layer:
            raise RuntimeError(
                "base-door graph failed: "
                f"alpha_idx={alpha_idx}/{cfg.steps} generated={generated} "
                f"expanded={expanded} rejected_collision={rejected_collision} "
                f"rejected_reach={rejected_reach} "
                f"rejected_lambda_overlap={rejected_lambda_overlap}"
            )
        next_layer.sort(key=lambda item: item[0] + _heuristic(item[1], cfg))
        previous = next_layer[: max(1, int(cfg.layer_keep))]

    if not previous:
        raise RuntimeError(
            "base-door graph failed: "
            f"expanded={expanded} generated={generated} rejected_collision={rejected_collision} "
            f"rejected_reach={rejected_reach} rejected_lambda_overlap={rejected_lambda_overlap}"
        )

    goal_items = [
        item for item in previous
        if item[1][3] >= cfg.steps and _is_front_goal(cfg, _state_to_pose(item[1], cfg))
    ]
    if not goal_items:
        best_pose = _state_to_pose(previous[0][1], cfg)
        dist_xy, dist_yaw = _front_goal_distance(cfg, best_pose)
        raise RuntimeError(
            "base-door graph failed to reach fridge-front goal: "
            f"best_dist={dist_xy:.3f}m best_yaw={math.degrees(dist_yaw):.1f}deg "
            f"expanded={expanded} generated={generated} "
            f"rejected_collision={rejected_collision} rejected_reach={rejected_reach} "
            f"rejected_lambda_overlap={rejected_lambda_overlap}"
        )
    goal_items.sort(key=lambda item: item[0])
    best_cost, _, states = goal_items[0]

    base_poses: list[list[float]] = []
    alphas: list[float] = []
    phases: list[int] = []
    for state in states:
        base_poses.append(_state_to_pose(state, cfg))
        alphas.append(_alpha_from_idx(state[3], cfg))
        phases.append(state[4])

    return BaseDoorGraphPlan(
        base_poses=base_poses,
        door_alphas_rad=alphas,
        phases=phases,
        cost=best_cost,
        expansions=expanded,
        generated=generated,
        rejected_collision=rejected_collision,
        rejected_reach=rejected_reach,
        rejected_lambda_overlap=rejected_lambda_overlap,
    )

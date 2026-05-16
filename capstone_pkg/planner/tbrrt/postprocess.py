from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import List, Sequence

import torch

from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from capstone_pkg.constraint_projection.projection import ManifoldProjector

from .tree import Tree

@dataclass
class ShortcutSmoothingStats:
    input_len: int
    output_len: int
    attempts: int = 0
    accepted: int = 0
    projection_failures: int = 0
    point_collision_failures: int = 0
    edge_collision_failures: int = 0
    not_shorter: int = 0
    sampled_points: int = 0
    used_batch_projection: bool = False
    device: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplineInterpolationStats:
    input_len: int
    output_len: int
    success: bool
    projection_failures: int = 0
    point_collision_failures: int = 0
    edge_collision_failures: int = 0
    joint_limit_failures: int = 0
    fallback_used: bool = False
    used_batch_projection: bool = False
    device: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToppTrajectory:
    t: torch.Tensor
    q: torch.Tensor
    qdot: torch.Tensor
    qddot: torch.Tensor
    duration_sec: float
    input_len: int
    output_len: int
    max_abs_velocity: float
    max_abs_acceleration: float
    time_scale: float
    duration_capped: bool = False
    requested_max_duration_sec: float | None = None

    def stats_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "max_abs_velocity": self.max_abs_velocity,
            "max_abs_acceleration": self.max_abs_acceleration,
            "time_scale": self.time_scale,
            "duration_capped": self.duration_capped,
            "requested_max_duration_sec": self.requested_max_duration_sec,
        }


def path_to_list(path: torch.Tensor) -> List[List[float]]:
    return [[float(x) for x in row.tolist()] for row in path.detach().cpu()]


@torch.no_grad()
def extract_path(treeA: Tree, treeB: Tree, idxA: int, idxB: int) -> torch.Tensor:
    """Extract a full path (L,D) from start(root of A) to goal(root of B)."""
    pA = treeA.backtrack_path(idxA)
    pB = treeB.backtrack_path(idxB)
    pB_rev = torch.flip(pB, dims=[0])
    return torch.cat([pA, pB_rev], dim=0)


@torch.no_grad()
def lazy_project_path(
    path: torch.Tensor,
    *,
    projector: ManifoldProjector,
    edge_checker: EdgeCollisionChecker | None = None,
    residual_accept_tol: float | None = None,
) -> torch.Tensor:
    """Project the final extracted path onto the manifold."""
    if path.ndim != 2:
        raise ValueError("path must be (L,D)")

    q_out = []
    bad = []
    for i in range(int(path.shape[0])):
        pr_i = projector.project(path[i])
        if not pr_i.success:
            bad.append(i)
            q_out.append(path[i])
        else:
            q_out.append(pr_i.q_proj)
    q_proj = torch.stack(q_out, dim=0).contiguous()
    if bad:
        residual_max = float(projector.c.residual_norm(q_proj).max().item())
        accept_tol = None if residual_accept_tol is None else float(residual_accept_tol)
        accept_slack = 0.0 if accept_tol is None else max(1.0e-9, 1.0e-4 * accept_tol)
        if accept_tol is None or residual_max > accept_tol + accept_slack:
            raise RuntimeError(
                "Lazy projection failed at indices: {} "
                "(residual_max={:.9f}, accept_tol={})".format(
                    bad,
                    residual_max,
                    residual_accept_tol,
                )
            )
        print(
            (
                "[lazy_projection][WARN] projection failed at indices {} "
                "but using path with residual_max={:.6f} <= {:.6f}"
            ).format(
                bad,
                residual_max,
                accept_tol,
            )
        )

    if edge_checker is not None and q_proj.shape[0] >= 2:
        q0 = q_proj[:-1]
        q1 = q_proj[1:]
        out = edge_checker.check_edges_batch(q0, q1)
        if bool(out.edge_in_collision.any().item()):
            idx = int(out.edge_in_collision.to(torch.int32).nonzero(as_tuple=False)[0].item())
            raise RuntimeError(f"Edge collision after lazy projection at segment {idx}")

    return q_proj


def _expand_limit(value: float | torch.Tensor | Sequence[float], *, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=like.device, dtype=like.dtype).view(-1)
        if out.numel() == 1:
            out = out.expand(like.shape[-1])
        if out.numel() != like.shape[-1]:
            raise ValueError(f"limit length {out.numel()} != path dim {like.shape[-1]}")
        return out.clamp_min(1e-6)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = torch.as_tensor(list(value), device=like.device, dtype=like.dtype).view(-1)
        if out.numel() == 1:
            out = out.expand(like.shape[-1])
        if out.numel() != like.shape[-1]:
            raise ValueError(f"limit length {out.numel()} != path dim {like.shape[-1]}")
        return out.clamp_min(1e-6)
    return torch.full((like.shape[-1],), float(value), device=like.device, dtype=like.dtype).clamp_min(1e-6)


def _shortcut_steps(q0: torch.Tensor, q1: torch.Tensor, *, step_q: float, max_steps: int) -> int:
    if step_q <= 0.0:
        return 1
    max_delta = float((q1 - q0).abs().max().item())
    steps = int(math.ceil(max_delta / float(step_q)))
    return max(1, min(steps, int(max_steps)))


def _segment_steps(path: torch.Tensor, *, step_q: float, max_steps: int) -> list[int]:
    steps: list[int] = []
    for i in range(int(path.shape[0]) - 1):
        steps.append(_shortcut_steps(path[i], path[i + 1], step_q=step_q, max_steps=max_steps))
    return steps


@torch.no_grad()
def _project_shortcut_samples(
    samples: torch.Tensor,
    projector,
    *,
    use_batch_projection: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if use_batch_projection and hasattr(projector, "project_batch"):
        pr = projector.project_batch(samples)
        return pr.q_proj.contiguous(), pr.success_mask.bool(), True

    q_out = []
    success = []
    for i in range(int(samples.shape[0])):
        pr_i = projector.project(samples[i])
        q_out.append(pr_i.q_proj.view(-1))
        success.append(bool(pr_i.success))
    success_mask = torch.tensor(success, device=samples.device, dtype=torch.bool)
    return torch.stack(q_out, dim=0).contiguous(), success_mask, False


@torch.no_grad()
def _project_samples(
    samples: torch.Tensor,
    projector,
    *,
    use_batch_projection: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    return _project_shortcut_samples(
        samples,
        projector,
        use_batch_projection=use_batch_projection,
    )


@torch.no_grad()
def _try_projected_shortcut(
    *,
    q0: torch.Tensor,
    q1: torch.Tensor,
    projector,
    checker,
    edge_checker: EdgeCollisionChecker,
    step_q: float,
    max_steps: int,
    use_batch_projection: bool,
) -> tuple[torch.Tensor | None, str | None, int, bool]:
    steps = _shortcut_steps(q0, q1, step_q=float(step_q), max_steps=int(max_steps))
    alpha = torch.linspace(0.0, 1.0, steps + 1, device=q0.device, dtype=q0.dtype).view(-1, 1)
    samples = (1.0 - alpha) * q0.view(1, -1) + alpha * q1.view(1, -1)

    q_proj, success_mask, used_batch = _project_shortcut_samples(
        samples,
        projector,
        use_batch_projection=use_batch_projection,
    )
    if not bool(success_mask.all().item()):
        return None, "projection", int(samples.shape[0]), used_batch

    # Preserve the already validated endpoints exactly.
    q_proj[0] = q0
    q_proj[-1] = q1

    col_out = checker.check_batch(q_proj)
    if bool(col_out.in_collision.any().item()):
        return None, "point_collision", int(samples.shape[0]), used_batch

    if q_proj.shape[0] >= 2:
        edge_out = edge_checker.check_edges_batch(
            q_proj[:-1],
            q_proj[1:],
            step_q=float(step_q),
            max_steps=int(max_steps),
        )
        if bool(edge_out.edge_in_collision.any().item()):
            return None, "edge_collision", int(samples.shape[0]), used_batch

    return q_proj, None, int(samples.shape[0]), used_batch


@torch.no_grad()
def shortcut_smooth_path(
    path: torch.Tensor,
    *,
    projector,
    checker,
    edge_checker: EdgeCollisionChecker,
    iters: int,
    step_q: float,
    max_steps: int,
    min_skip: int = 1,
    seed: int | None = None,
    use_batch_projection: bool = True,
) -> tuple[torch.Tensor, ShortcutSmoothingStats]:
    """Constrained shortcut smoothing with projected, collision-checked samples.

    Each shortcut candidate is linearly sampled in joint space, projected as one
    batch when the projector supports it, then validated with batched point and
    edge collision checks. Accepted candidates replace the skipped segment with
    the projected sample sequence, not just the two endpoints.
    """

    if path.ndim != 2:
        raise ValueError("path must be (L,D)")

    smoothed = path.contiguous()
    stats = ShortcutSmoothingStats(
        input_len=int(path.shape[0]),
        output_len=int(path.shape[0]),
        used_batch_projection=False,
        device=str(path.device),
    )

    if int(iters) <= 0 or smoothed.shape[0] <= 2:
        return smoothed, stats

    rng = random.Random(seed)
    min_skip = max(1, int(min_skip))

    for _ in range(int(iters)):
        n = int(smoothed.shape[0])
        if n <= min_skip + 2:
            break

        i = rng.randrange(0, n - min_skip - 1)
        j = rng.randrange(i + min_skip + 1, n)
        original_segment_len = j - i + 1
        if original_segment_len <= 2:
            continue

        stats.attempts += 1
        replacement, reason, sampled_points, used_batch = _try_projected_shortcut(
            q0=smoothed[i],
            q1=smoothed[j],
            projector=projector,
            checker=checker,
            edge_checker=edge_checker,
            step_q=float(step_q),
            max_steps=int(max_steps),
            use_batch_projection=use_batch_projection,
        )
        stats.sampled_points += sampled_points
        stats.used_batch_projection = stats.used_batch_projection or used_batch

        if replacement is None:
            if reason == "projection":
                stats.projection_failures += 1
            elif reason == "point_collision":
                stats.point_collision_failures += 1
            elif reason == "edge_collision":
                stats.edge_collision_failures += 1
            continue

        if int(replacement.shape[0]) >= original_segment_len:
            stats.not_shorter += 1
            continue

        smoothed = torch.cat(
            [
                smoothed[:i],
                replacement,
                smoothed[j + 1 :],
            ],
            dim=0,
        ).contiguous()
        stats.accepted += 1
        stats.output_len = int(smoothed.shape[0])

    stats.output_len = int(smoothed.shape[0])
    return smoothed, stats


@torch.no_grad()
def cubic_spline_interpolate_and_validate_path(
    path: torch.Tensor,
    *,
    projector,
    checker,
    edge_checker: EdgeCollisionChecker,
    joint_limits=None,
    step_q: float,
    max_steps_per_segment: int,
    max_points: int,
    fallback_to_input: bool = True,
    use_batch_projection: bool = True,
) -> tuple[torch.Tensor, SplineInterpolationStats]:
    """Catmull-Rom/Hermite spline interpolation followed by projection and validation."""

    if path.ndim != 2:
        raise ValueError("path must be (L,D)")

    stats = SplineInterpolationStats(
        input_len=int(path.shape[0]),
        output_len=int(path.shape[0]),
        success=False,
        fallback_used=False,
        device=str(path.device),
    )

    if path.shape[0] <= 2:
        stats.success = True
        return path.contiguous(), stats

    steps_per_segment = _segment_steps(
        path,
        step_q=float(step_q),
        max_steps=max(1, int(max_steps_per_segment)),
    )
    point_count = 1 + sum(steps_per_segment)
    if int(max_points) > 0 and point_count > int(max_points):
        scale = float(max_points - 1) / float(max(1, point_count - 1))
        steps_per_segment = [max(1, int(round(s * scale))) for s in steps_per_segment]

    tangents = torch.empty_like(path)
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    tangents[1:-1] = 0.5 * (path[2:] - path[:-2])

    pieces = [path[0].view(1, -1)]
    for i, steps in enumerate(steps_per_segment):
        q0 = path[i]
        q1 = path[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        u = torch.linspace(0.0, 1.0, int(steps) + 1, device=path.device, dtype=path.dtype)[1:]
        u = u.view(-1, 1)
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        pieces.append(h00 * q0.view(1, -1) + h10 * m0.view(1, -1) + h01 * q1.view(1, -1) + h11 * m1.view(1, -1))

    samples = torch.cat(pieces, dim=0).contiguous()
    q_proj, success_mask, used_batch = _project_samples(
        samples,
        projector,
        use_batch_projection=use_batch_projection,
    )
    stats.used_batch_projection = bool(used_batch)
    if not bool(success_mask.all().item()):
        stats.projection_failures = int((~success_mask).sum().item())
        stats.fallback_used = bool(fallback_to_input)
        return (path.contiguous() if fallback_to_input else q_proj), stats

    q_proj[0] = path[0]
    q_proj[-1] = path[-1]

    if joint_limits is not None:
        below = q_proj < (joint_limits.lower.to(q_proj.device, q_proj.dtype) - 1e-5)
        above = q_proj > (joint_limits.upper.to(q_proj.device, q_proj.dtype) + 1e-5)
        if bool((below | above).any().item()):
            stats.joint_limit_failures = int((below | above).any(dim=1).sum().item())
            stats.fallback_used = bool(fallback_to_input)
            return (path.contiguous() if fallback_to_input else q_proj), stats

    col_out = checker.check_batch(q_proj)
    if bool(col_out.in_collision.any().item()):
        stats.point_collision_failures = int(col_out.in_collision.sum().item())
        stats.fallback_used = bool(fallback_to_input)
        return (path.contiguous() if fallback_to_input else q_proj), stats

    if q_proj.shape[0] >= 2:
        edge_out = edge_checker.check_edges_batch(
            q_proj[:-1],
            q_proj[1:],
            step_q=float(step_q),
            max_steps=int(max_steps_per_segment),
        )
        if bool(edge_out.edge_in_collision.any().item()):
            stats.edge_collision_failures = int(edge_out.edge_in_collision.sum().item())
            stats.fallback_used = bool(fallback_to_input)
            return (path.contiguous() if fallback_to_input else q_proj), stats

    stats.success = True
    stats.output_len = int(q_proj.shape[0])
    return q_proj.contiguous(), stats


def _finite_difference(q: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    qdot = torch.zeros_like(q)
    qddot = torch.zeros_like(q)
    if q.shape[0] >= 2:
        qdot[0] = (q[1] - q[0]) / dt
        qdot[-1] = (q[-1] - q[-2]) / dt
    if q.shape[0] >= 3:
        qdot[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
        qddot[1:-1] = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt * dt)
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]
    return qdot, qddot


@torch.no_grad()
def topp_retime_path(
    path: torch.Tensor,
    *,
    max_velocity: float | torch.Tensor | Sequence[float],
    max_acceleration: float | torch.Tensor | Sequence[float],
    output_dt: float,
    max_duration_sec: float | None = None,
    safety_scale: float = 1.05,
    max_iterations: int = 20,
) -> ToppTrajectory:
    """Discrete TOPP-style retiming under per-joint velocity and acceleration limits."""

    if path.ndim != 2:
        raise ValueError("path must be (L,D)")

    output_dt = max(1e-4, float(output_dt))
    requested_max_duration_sec = None
    if max_duration_sec is not None:
        requested_max_duration_sec = max(0.0, float(max_duration_sec))
    vmax = _expand_limit(max_velocity, like=path)
    amax = _expand_limit(max_acceleration, like=path)

    if path.shape[0] <= 1:
        t = torch.zeros((1,), device=path.device, dtype=path.dtype)
        qdot = torch.zeros_like(path)
        qddot = torch.zeros_like(path)
        return ToppTrajectory(
            t,
            path.contiguous(),
            qdot,
            qddot,
            0.0,
            int(path.shape[0]),
            int(path.shape[0]),
            0.0,
            0.0,
            1.0,
            False,
            requested_max_duration_sec,
        )

    def _build_trajectory(
        dt_segments_local: torch.Tensor,
        *,
        time_scale_local: float,
        duration_capped: bool,
    ) -> ToppTrajectory:
        t_knots_local = torch.cat(
            [
                torch.zeros((1,), device=path.device, dtype=path.dtype),
                torch.cumsum(dt_segments_local, dim=0),
            ]
        )
        duration_local = float(t_knots_local[-1].item())
        n_out_local = max(2, int(math.ceil(duration_local / output_dt)) + 1)
        t_out_local = torch.linspace(
            0.0,
            duration_local,
            n_out_local,
            device=path.device,
            dtype=path.dtype,
        )
        idx_local = torch.searchsorted(t_knots_local, t_out_local, right=True).clamp(
            min=1,
            max=int(t_knots_local.numel() - 1),
        )
        t0_local = t_knots_local[idx_local - 1]
        t1_local = t_knots_local[idx_local]
        alpha_local = ((t_out_local - t0_local) / (t1_local - t0_local).clamp_min(1e-6)).view(-1, 1)
        q_local = (1.0 - alpha_local) * path[idx_local - 1] + alpha_local * path[idx_local]
        qdot_local, qddot_local = _finite_difference(q_local, output_dt)
        return ToppTrajectory(
            t=t_out_local,
            q=q_local.contiguous(),
            qdot=qdot_local.contiguous(),
            qddot=qddot_local.contiguous(),
            duration_sec=duration_local,
            input_len=int(path.shape[0]),
            output_len=int(q_local.shape[0]),
            max_abs_velocity=float(qdot_local.abs().max().item()),
            max_abs_acceleration=float(qddot_local.abs().max().item()),
            time_scale=float(time_scale_local),
            duration_capped=bool(duration_capped),
            requested_max_duration_sec=requested_max_duration_sec,
        )

    dq = path[1:] - path[:-1]
    dt_segments = (dq.abs() / vmax.view(1, -1)).max(dim=1).values.clamp_min(output_dt)
    time_scale = 1.0

    for _ in range(max(1, int(max_iterations))):
        t_knots = torch.cat([torch.zeros((1,), device=path.device, dtype=path.dtype), torch.cumsum(dt_segments, dim=0)])
        duration = float(t_knots[-1].item())
        n_out = max(2, int(math.ceil(duration / output_dt)) + 1)
        t_out = torch.linspace(0.0, duration, n_out, device=path.device, dtype=path.dtype)

        idx = torch.searchsorted(t_knots, t_out, right=True).clamp(min=1, max=int(t_knots.numel() - 1))
        t0 = t_knots[idx - 1]
        t1 = t_knots[idx]
        alpha = ((t_out - t0) / (t1 - t0).clamp_min(1e-6)).view(-1, 1)
        q = (1.0 - alpha) * path[idx - 1] + alpha * path[idx]
        qdot, qddot = _finite_difference(q, output_dt)

        v_ratio = float((qdot.abs() / vmax.view(1, -1)).max().item())
        a_ratio = float((qddot.abs() / amax.view(1, -1)).max().item()) if q.shape[0] >= 3 else 0.0
        ratio = max(v_ratio, math.sqrt(max(0.0, a_ratio)), 1.0)
        if ratio <= 1.0 + 1e-3:
            if requested_max_duration_sec is not None and duration > requested_max_duration_sec > 0.0:
                compress_scale = requested_max_duration_sec / duration
                dt_segments = dt_segments * compress_scale
                time_scale *= compress_scale
                return _build_trajectory(
                    dt_segments,
                    time_scale_local=time_scale,
                    duration_capped=True,
                )
            return ToppTrajectory(
                t=t_out,
                q=q.contiguous(),
                qdot=qdot.contiguous(),
                qddot=qddot.contiguous(),
                duration_sec=duration,
                input_len=int(path.shape[0]),
                output_len=int(q.shape[0]),
                max_abs_velocity=float(qdot.abs().max().item()),
                max_abs_acceleration=float(qddot.abs().max().item()),
                time_scale=float(time_scale),
                duration_capped=False,
                requested_max_duration_sec=requested_max_duration_sec,
            )

        scale = float(ratio * safety_scale)
        time_scale *= scale
        dt_segments = dt_segments * scale

    if requested_max_duration_sec is not None:
        current_duration = float(dt_segments.sum().item())
        if current_duration > requested_max_duration_sec > 0.0:
            compress_scale = requested_max_duration_sec / current_duration
            dt_segments = dt_segments * compress_scale
            time_scale *= compress_scale
            return _build_trajectory(
                dt_segments,
                time_scale_local=time_scale,
                duration_capped=True,
            )
    return _build_trajectory(
        dt_segments,
        time_scale_local=time_scale,
        duration_capped=False,
    )

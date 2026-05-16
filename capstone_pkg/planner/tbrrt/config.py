from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TBRRTConfig:
    """Hyper-parameters for extcon TB-RRT."""

    # ---- basic stepping ----
    step_size: float = 0.15              # step length in configuration space
    goal_threshold: float = 0.10         # OMPL-style goal region threshold: distanceGoal(q) <= threshold
    min_progress_ratio: float = 0.05     # reject steps with progress < ratio * step_size
    min_separation_parent_ratio: float = 0.10  # reject steps too close to parent (< ratio * step_size)
    min_separation_tree_ratio: float = 0.05    # reject steps too close to any existing node (< ratio * step_size)

    # ---- constraint handling ----
    EM: float = 0.02                     # manifold approximation error threshold (create new TS)
    E_conn: float = 0.02                 # residual tolerance along connecting edges (sampling-based check)

    # ---- connection test (paper Sec. 3.8) ----
    conn_num_samples: int = 15        # number of samples along connection segment
    conn_tangent_tol: float = 0.05    # tolerance for ||J(q_end) v_hat|| at both ends

    # ---- tangent space sampling ----
    ts_radius: float = 0.5               # radius in tangent coordinates for sampling
    p_uniform: float = 0.15              # prob of sampling uniformly in joint limits
    goal_bias: float = 0.05              # prob of sampling the goal directly (classic RRT trick)

    # ---- TS selection bias (slightly success-leaning for batch_conext) ----
    ts_bias_volume: float = 1.0
    ts_bias_curvature: float = 1.0
    ts_bias_nodecount: float = 0.35
    ts_bias_collision: float = 1.5
    ts_curv_eps: float = 1e-3

    discard_overlap_max_tries: int = 10

    # ---- dynamic domain sizing ----
    dynamic_domain_enable: bool = True
    ts_domain_expand_ratio: float = 1.25
    ts_domain_shrink_ratio: float = 0.75
    ts_domain_expand_frac: float = 0.9
    ts_domain_shrink_frac: float = 0.4
    ts_domain_min: float = 0.04
    ts_domain_max: float = 3.0

    # ---- connect / loop control ----
    connect_max_steps: int = 64          # max Extend steps in Connect
    connect_stagnation_steps: int = 4    # consecutive low-progress CONNECT steps before retarget/escape
    connect_stagnation_progress_ratio: float = 0.10  # low-progress threshold = ratio * step_size
    connect_stagnation_escape: bool = True  # on stagnation, switch to ESCAPE instead of only retargeting
    failed_connection_cooldown: int = 8  # iterations to avoid the most recently failed opposite-tree target
    connect_bridge_enable: bool = True    # on not-reached CONNECT, try a short direct edge before retargeting
    connect_bridge_near_threshold: float = 0.0  # <=0 uses 3 * step_size
    connect_bridge_curvature_threshold: float = 2.5
    connect_bridge_max_attempts_per_iter: int = 4
    connect_bridge_reset_cooldown: int = 1
    connection_segment_precheck: bool = True  # reject direct A/B connector edge before full lazy path validation
    failed_connection_region_enable: bool = True  # skip endpoint pairs near previously lazy-failed connections
    failed_connection_region_radius: float = 0.05
    failed_connection_region_max: int = 2048
    failed_edge_region_enable: bool = True  # skip paths containing edge segments near previous edge-collision failures
    failed_edge_region_radius: float = 0.05
    failed_edge_region_max: int = 4096
    failed_edge_region_retarget_threshold: int = 4  # repeated region skips on one slot before retargeting
    failed_edge_region_escape_threshold: int = 16   # repeated region skips on one slot before ESCAPE
    failed_edge_region_cooldown: int = 8            # iterations to avoid the skipped opposite-tree target
    connection_lazy_prealloc_enable: bool = True
    connection_lazy_prealloc_max_candidates: int = 0  # <=0 uses B * block_K
    connection_lazy_prealloc_max_path_points: int = 512
    connection_lazy_edge_branch_ban_enable: bool = True
    connection_lazy_edge_subtree_ban_max_nodes: int = 256  # <=0 preserves full subtree-ban behavior
    escape_spawn_blocks: int = 5         # spawned blocks per trapped block in ESCAPE phase (batch_conext)
    escape_extend_steps: int = 1         # chained EXTEND steps in ESCAPE phase (batch_conext)
    escape_fuse_trees: bool = False      # fuse tree A/B ESCAPE work into one GPU batch in batch_conext
    max_iters: int = 5000000
    time_limit_sec: float = 60.0

    # ---- numerics ----
    svd_tol: float = 1e-6                # tolerance for rank estimation when building TS

    # ---- collision checking ----
    edge_step_q: float = 0.03            # interpolation step in joint space for edge collision
    edge_max_steps: int = 64

    # ---- post-processing ----
    shortcut_smoothing: bool = True
    shortcut_smoothing_iters: int = 80
    shortcut_smoothing_min_skip: int = 1
    spline_interpolation: bool = True
    spline_step_q: float = 0.03
    spline_max_steps_per_segment: int = 32
    spline_max_points: int = 2048
    spline_fallback_to_input: bool = True

    # ---- time parameterization ----
    topp_enable: bool = True
    topp_max_velocity: float | Sequence[float] = 1.0
    topp_max_acceleration: float | Sequence[float] = 2.0
    topp_output_dt: float = 0.2
    topp_max_duration_sec: float = 10.0
    topp_safety_scale: float = 1.05
    topp_max_iterations: int = 20

    # ---- misc ----
    seed: int | None = None

    # ---- heuristic toggles (planner-specific overrides can disable these) ----
    enable_halfspace: bool = True        # tangent half-space backtracking prevention
    enable_overlap_discard: bool = True  # overlapping TS sample discard (Sec. 3.5.2)

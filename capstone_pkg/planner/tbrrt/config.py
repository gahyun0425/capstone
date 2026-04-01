from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TBRRTConfig:
    """Hyper-parameters for extcon TB-RRT."""

    # ---- basic stepping ----
    step_size: float = 0.15              # step length in configuration space
    goal_threshold: float = 0.10         # OMPL-style goal region threshold: distanceGoal(q) <= threshold

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

    # ---- TS selection bias (paper Sec. 3.7 heuristics) ----
    # Note: these affect which tangent space is chosen when sampling in the bundle.
    # Larger values strengthen the effect.
    ts_bias_volume: float = 1.0          # legacy r^k term
    ts_bias_curvature: float = 1.0       # prefer low-curvature TS (proxy)
    ts_bias_nodecount: float = 1.0       # prefer less explored TS
    ts_curv_eps: float = 1e-3

    discard_overlap_max_tries: int = 20  # max resamples when overlap-prevention discards q_rand

    # ---- dynamic domain sizing of tangent spaces (paper Sec. 3.6 / Alg. 9) ----
    dynamic_domain_enable: bool = True
    ts_domain_expand_ratio: float = 1.2
    ts_domain_shrink_ratio: float = 0.8
    ts_domain_expand_frac: float = 0.9   # expand if ||q_new-q_root|| > frac * domain (and NOT projected)
    ts_domain_shrink_frac: float = 0.4   # shrink if ||q_new-q_root|| < frac * domain (and projected)
    ts_domain_min: float = 0.05
    ts_domain_max: float = 5.0
    # ---- connect / loop control ----
    connect_max_steps: int = 64          # max Extend steps in Connect
    escape_extend_steps: int = 1         # chained EXTEND steps in ESCAPE phase (batch_conext)
    max_iters: int = 5000000
    time_limit_sec: float = 60.0

    # ---- numerics ----
    svd_tol: float = 1e-6                # tolerance for rank estimation when building TS

    # ---- collision checking ----
    edge_step_q: float = 0.05            # interpolation step in joint space for edge collision
    edge_max_steps: int = 64

    # ---- misc ----
    seed: int | None = None

    # ---- heuristic toggles (planner-specific overrides can disable these) ----
    enable_halfspace: bool = True        # tangent half-space backtracking prevention
    enable_overlap_discard: bool = True  # overlapping TS sample discard (Sec. 3.5.2)

from __future__ import annotations
import torch
from typing import Optional, Any

from capstone_pkg.utils.joint_limit import JointLimitsTorch  # ✅ 너가 보여준 클래스
from .ts_bank import TSBank
from .stats import get_stats

@torch.no_grad()
def sample_uniform(joint_limits: Any, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    joint_limits:
      - JointLimitsTorch(lower, upper)  ✅
      - torch.Tensor (2,D) or (D,2)
    returns:
      - (D,) torch.Tensor
    """
    # ✅ JointLimitsTorch 지원
    if isinstance(joint_limits, JointLimitsTorch):
        lower, upper = joint_limits.lower, joint_limits.upper

    # ✅ Tensor 지원
    elif isinstance(joint_limits, torch.Tensor):
        jl = joint_limits
        if jl.ndim != 2:
            raise ValueError(f"joint_limits tensor must be 2D, got {jl.shape}")
        if jl.shape[0] == 2:
            lower, upper = jl[0], jl[1]
        elif jl.shape[1] == 2:
            lower, upper = jl[:, 0], jl[:, 1]
        else:
            raise ValueError(f"joint_limits must be (2,D) or (D,2), got {jl.shape}")

    else:
        raise TypeError(f"Unsupported joint_limits type: {type(joint_limits)}")

    # ✅ torch.rand는 size(tuple)만 받음 + generator None 안전 처리
    size = tuple(lower.shape)
    if generator is None:
        r = torch.rand(size, device=lower.device, dtype=lower.dtype)
    else:
        r = torch.rand(size, device=lower.device, dtype=lower.dtype, generator=generator)

    return lower + (upper - lower) * r


@torch.no_grad()
def sample_in_tangent_ball(
    bank: TSBank,
    *,
    # Optional target to enforce the "tangent half-space" heuristic (paper Sec. 3.7):
    # if provided, we flip the sampled tangent direction so it does not point
    # opposite to the local target direction in the tangent space.
    q_target: Optional[torch.Tensor] = None,
    enable_halfspace: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample q in a random tangent space within that TS's current domain.

    Sampling is uniform in a k-ball using the standard method:
    - sample direction from Normal
    - sample radius as u^(1/k)
    """
    ts_id = bank.sample_ts_id(generator=generator)
    stats = get_stats()
    stats.inc("tangent_samples", 1)
    ts = bank.get(ts_id)
    domain = float(bank.get_domain(ts_id))

    k = ts.dim
    if k == 0:
        return ts.root.clone()

    dev = ts.root.device
    dtype = ts.root.dtype

    # random direction in tangent coordinates
    v = torch.randn((k,), device=dev, dtype=dtype, generator=generator)
    v = v / torch.linalg.norm(v).clamp_min(1e-12)

    # --- Tangent half-space heuristic (backtracking prevention) ---
    # If q_target is provided (typically q_goal), compute the local target
    # direction in the tangent space and ensure v has non-negative dot with it.
    if enable_halfspace and (q_target is not None):
        # Map ambient direction into tangent coordinates.
        d_amb = (q_target - ts.root).to(device=dev, dtype=dtype)
        d_tan = ts.basis.T @ d_amb
        dn = torch.linalg.norm(d_tan).clamp_min(1e-12)
        d_tan = d_tan / dn
        # Flip sampled direction if it would backtrack.
        if float(torch.dot(v, d_tan).item()) < 0.0:
            stats.inc("halfspace_flips", 1)
            v = -v

    # random radius
    u = torch.rand((1,), device=dev, dtype=dtype, generator=generator)
    r = (u ** (1.0 / float(k))) * float(domain)

    a = v * r
    q = ts.root + ts.basis @ a
    return q


@torch.no_grad()
def sample_qrand(
    *,
    bank: TSBank,
    joint_limits: JointLimitsTorch,
    p_uniform: float,
    goal_bias: float,
    q_goal: torch.Tensor,
    enable_halfspace: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random configuration.

    - With prob goal_bias: return q_goal.
    - With prob p_uniform: sample uniform in joint limits.
    - Otherwise: sample in tangent bundle (random TS, tangent ball).
    """
    dev = q_goal.device
    stats = get_stats()
    stats.inc("qrand_draws", 1)
    u = float(torch.rand((1,), device=dev, generator=generator).item())
    if u < goal_bias:
        stats.inc("qrand_goal", 1)
        return q_goal.clone()
    if u < goal_bias + p_uniform:
        stats.inc("qrand_uniform", 1)
        return sample_uniform(joint_limits, generator=generator)
    stats.inc("qrand_tangent", 1)
    # Use goal direction for tangent half-space heuristic.
    q = sample_in_tangent_ball(bank, q_target=q_goal, enable_halfspace=enable_halfspace, generator=generator)
    return joint_limits.clamp(q)


@torch.no_grad()
def sample_uniform_batch(joint_limits: JointLimitsTorch, n: int, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Vectorized uniform sampling in joint limits.

    Args:
        joint_limits: JointLimitsTorch
        n: number of samples
    Returns:
        (n, D)
    """
    lower = joint_limits.lower
    upper = joint_limits.upper
    u = torch.rand((int(n), int(lower.numel())), device=lower.device, dtype=lower.dtype, generator=generator)
    return lower.view(1, -1) + u * (upper - lower).view(1, -1)


@torch.no_grad()
def pack_banks_for_sampling(banks: list[TSBank]):
    """Pack per-goal TSBank objects into a single padded tensor bundle.

    This is intended to be called OUTSIDE the main planning iteration loop.

    Returns a dict with keys:
      roots:  (B, Smax, D)
      basis:  (B, Smax, D, Kmax)
      dim:    (B, Smax)
      domain: (B, Smax)
      w:      (B, Smax)
      Smax, Kmax
    """
    packed = [bk.pack_tensors() for bk in banks]
    roots_list, basis_list, dim_list, domain_list, w_list = zip(*packed)

    device = roots_list[0].device
    dtype = roots_list[0].dtype
    B = len(banks)
    D = int(roots_list[0].shape[1])
    Smax = max(int(r.shape[0]) for r in roots_list)
    Kmax = max(int(b.shape[2]) for b in basis_list)

    def pad_S(x: torch.Tensor, S: int, fill: float = 0.0, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        out = torch.full((S,) + x.shape[1:], fill_value=fill, device=device, dtype=(out_dtype or x.dtype))
        out[: x.shape[0]] = x.to(dtype=(out_dtype or x.dtype))
        return out

    roots = torch.stack([pad_S(r, Smax, fill=0.0, out_dtype=dtype) for r in roots_list], dim=0)  # (B,Smax,D)
    # basis needs K padding
    basis_tmp = torch.stack([pad_S(b, Smax, fill=0.0, out_dtype=dtype) for b in basis_list], dim=0)  # (B,Smax,D,Kb)
    if int(basis_tmp.shape[3]) != Kmax:
        basis = torch.zeros((B, Smax, D, Kmax), device=device, dtype=dtype)
        basis[:, :, :, : basis_tmp.shape[3]] = basis_tmp
    else:
        basis = basis_tmp

    dim = torch.stack([pad_S(d.to(torch.long), Smax, fill=0.0, out_dtype=torch.long) for d in dim_list], dim=0)  # (B,Smax)
    domain = torch.stack([pad_S(dom.to(torch.float32), Smax, fill=1e-6, out_dtype=torch.float32) for dom in domain_list], dim=0)  # (B,Smax)
    w = torch.stack([pad_S(wi.to(torch.float32), Smax, fill=0.0, out_dtype=torch.float32) for wi in w_list], dim=0)  # (B,Smax)
    # ensure padded categories have zero prob
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return {
        "roots": roots,
        "basis": basis,
        "dim": dim,
        "domain": domain,
        "w": w,
        "Smax": Smax,
        "Kmax": Kmax,
    }


@torch.no_grad()
def sample_qrand_BK_packed(
    *,
    bank_packed: dict,
    joint_limits: JointLimitsTorch,
    q_start_proj: torch.Tensor,  # (B,D)
    q_goal_proj: torch.Tensor,   # (B,D)
    swapped: torch.Tensor,       # (B,K) bool
    active: torch.Tensor,        # (B,K) bool
    p_uniform: float,
    goal_bias: float,
    enable_halfspace: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized q_rand sampling over (B,K) using pre-packed TS tensors.

    Note: assumes bank_packed is built outside the iteration loop.
    """
    roots = bank_packed["roots"]
    basis = bank_packed["basis"]
    dim = bank_packed["dim"]
    domain = bank_packed["domain"]
    w = bank_packed["w"]
    Kmax = int(bank_packed["Kmax"])

    device = q_start_proj.device
    dtype = q_start_proj.dtype
    B, D = q_start_proj.shape
    K = int(swapped.shape[1])
    N = B * K

    # ---- flatten (B,K) -> (N,) / (N,D) ----
    b_idx = torch.arange(B, device=device).repeat_interleave(K)  # (N,)
    swapped_f = swapped.reshape(N)
    active_f = active.reshape(N)

    q_start_f = q_start_proj[b_idx]   # (N,D)
    q_goal_f0 = q_goal_proj[b_idx]    # (N,D)
    q_goal_for_bias = torch.where(swapped_f.view(N, 1), q_start_f, q_goal_f0)  # (N,D)

    # output default (inactive -> start)
    q_out = q_start_f.clone()
    if not bool(active_f.any().item()):
        return q_out.view(B, K, D)

    act_idx = torch.nonzero(active_f, as_tuple=False).squeeze(1)  # (M,)
    b_act = b_idx[act_idx]                                        # (M,)
    q_goal_act = q_goal_for_bias[act_idx]                         # (M,D)
    M = int(act_idx.numel())

    u = torch.rand((M,), device=device, dtype=torch.float32, generator=generator)
    mask_goal = u < float(goal_bias)
    mask_uniform = (u >= float(goal_bias)) & (u < float(goal_bias + p_uniform))
    mask_tangent = ~(mask_goal | mask_uniform)

    # goal
    if bool(mask_goal.any().item()):
        q_out[act_idx[mask_goal]] = q_goal_act[mask_goal]

    # uniform
    if bool(mask_uniform.any().item()):
        q_uni = sample_uniform_batch(joint_limits, int(mask_uniform.sum().item()), generator=generator)
        q_out[act_idx[mask_uniform]] = q_uni.to(dtype=dtype, device=device)

    # tangent
    if bool(mask_tangent.any().item()):
        tidx = act_idx[mask_tangent]
        b_t = b_act[mask_tangent]
        q_tgt = q_goal_act[mask_tangent]
        Mt = int(tidx.numel())

        # select TS per sample
        w_per = w[b_t]  # (Mt,Smax)
        ts_sel = torch.multinomial(w_per, num_samples=1, replacement=True, generator=generator).squeeze(1)  # (Mt,)

        root_sel = roots[b_t, ts_sel].to(dtype=dtype)            # (Mt,D)
        basis_sel = basis[b_t, ts_sel].to(dtype=dtype)           # (Mt,D,Kmax)
        k_sel = dim[b_t, ts_sel].to(torch.long)                  # (Mt,)
        dom_sel = domain[b_t, ts_sel].to(dtype=dtype)            # (Mt,)

        v = torch.randn((Mt, Kmax), device=device, dtype=dtype, generator=generator)
        mask_k = torch.arange(Kmax, device=device).view(1, Kmax) < k_sel.view(Mt, 1)
        v = v * mask_k
        v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-12)

        if enable_halfspace:
            d_amb = (q_tgt - root_sel)
            d_tan = torch.einsum("md,mdk->mk", d_amb, basis_sel) * mask_k
            d_tan = d_tan / torch.linalg.norm(d_tan, dim=1, keepdim=True).clamp_min(1e-12)
            flip = (v * d_tan).sum(dim=1) < 0
            v[flip] = -v[flip]

        u2 = torch.rand((Mt, 1), device=device, dtype=dtype, generator=generator)
        kf = k_sel.to(dtype).clamp_min(1.0).view(Mt, 1)
        r = (u2 ** (1.0 / kf)) * dom_sel.view(Mt, 1)
        a = v * r

        q_tan = root_sel + torch.einsum("mk,mdk->md", a, basis_sel)

        is0 = k_sel == 0
        if bool(is0.any().item()):
            q_tan[is0] = root_sel[is0]

        q_out[tidx] = q_tan

    q_out = joint_limits.clamp(q_out)
    return q_out.view(B, K, D)

from __future__ import annotations

from typing import Optional

import torch

from capstone_pkg.utils.joint_limit import JointLimitsTorch

from ..ts_bank import TSBank


@torch.no_grad()
def sample_uniform_batch(joint_limits: JointLimitsTorch, n: int, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    lower = joint_limits.lower
    upper = joint_limits.upper
    u = torch.rand((int(n), int(lower.numel())), device=lower.device, dtype=lower.dtype, generator=generator)
    return lower.view(1, -1) + u * (upper - lower).view(1, -1)


@torch.no_grad()
def pack_banks_for_sampling(banks: list[TSBank]):
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

    roots = torch.stack([pad_S(r, Smax, fill=0.0, out_dtype=dtype) for r in roots_list], dim=0)
    basis_tmp = torch.stack([pad_S(b, Smax, fill=0.0, out_dtype=dtype) for b in basis_list], dim=0)
    if int(basis_tmp.shape[3]) != Kmax:
        basis = torch.zeros((B, Smax, D, Kmax), device=device, dtype=dtype)
        basis[:, :, :, : basis_tmp.shape[3]] = basis_tmp
    else:
        basis = basis_tmp

    dim = torch.stack([pad_S(d.to(torch.long), Smax, fill=0.0, out_dtype=torch.long) for d in dim_list], dim=0)
    domain = torch.stack([pad_S(dom.to(torch.float32), Smax, fill=1e-6, out_dtype=torch.float32) for dom in domain_list], dim=0)
    w = torch.stack([pad_S(wi.to(torch.float32), Smax, fill=0.0, out_dtype=torch.float32) for wi in w_list], dim=0)
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
    q_start_proj: torch.Tensor,
    q_goal_proj: torch.Tensor,
    swapped: torch.Tensor,
    active: torch.Tensor,
    p_uniform: float,
    goal_bias: float,
    enable_halfspace: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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

    b_idx = torch.arange(B, device=device).repeat_interleave(K)
    swapped_f = swapped.reshape(N)
    active_f = active.reshape(N)

    q_start_f = q_start_proj[b_idx]
    q_goal_f0 = q_goal_proj[b_idx]
    q_goal_for_bias = torch.where(swapped_f.view(N, 1), q_start_f, q_goal_f0)

    q_out = q_start_f.clone()
    if not bool(active_f.any().item()):
        return q_out.view(B, K, D)

    act_idx = torch.nonzero(active_f, as_tuple=False).squeeze(1)
    b_act = b_idx[act_idx]
    q_goal_act = q_goal_for_bias[act_idx]
    M = int(act_idx.numel())

    u = torch.rand((M,), device=device, dtype=torch.float32, generator=generator)
    mask_goal = u < float(goal_bias)
    mask_uniform = (u >= float(goal_bias)) & (u < float(goal_bias + p_uniform))
    mask_tangent = ~(mask_goal | mask_uniform)

    if bool(mask_goal.any().item()):
        q_out[act_idx[mask_goal]] = q_goal_act[mask_goal]

    if bool(mask_uniform.any().item()):
        q_uni = sample_uniform_batch(joint_limits, int(mask_uniform.sum().item()), generator=generator)
        q_out[act_idx[mask_uniform]] = q_uni.to(dtype=dtype, device=device)

    if bool(mask_tangent.any().item()):
        tidx = act_idx[mask_tangent]
        b_t = b_act[mask_tangent]
        q_tgt = q_goal_act[mask_tangent]
        Mt = int(tidx.numel())

        w_per = w[b_t]
        ts_sel = torch.multinomial(w_per, num_samples=1, replacement=True, generator=generator).squeeze(1)

        root_sel = roots[b_t, ts_sel].to(dtype=dtype)
        basis_sel = basis[b_t, ts_sel].to(dtype=dtype)
        k_sel = dim[b_t, ts_sel].to(torch.long)
        dom_sel = domain[b_t, ts_sel].to(dtype=dtype)

        v = torch.randn((Mt, Kmax), device=device, dtype=dtype, generator=generator)
        mask_k = torch.arange(Kmax, device=device).view(1, Kmax) < k_sel.view(Mt, 1)
        v = v * mask_k
        v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-12)

        if enable_halfspace:
            d_amb = q_tgt - root_sel
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

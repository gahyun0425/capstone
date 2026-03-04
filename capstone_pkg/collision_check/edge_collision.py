from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import math
import torch
from capstone_pkg.utils.config import ROBOT_YAML, WORLD_YAML

from capstone_pkg.collision_check.collision import SelfCollisionChecker, get_self_collision_checker

TensorLike14 = Union[torch.Tensor, Sequence[float]]


@dataclass
class EdgeCollisionResult:
    """단일 edge 결과."""
    edge_in_collision: bool
    max_penetration: float
    first_collision_alpha: Optional[float] = None  # 0~1 (충돌이 처음 발생한 보간 비율)


@dataclass
class EdgeCollisionBatchResult:
    """여러 edge 결과."""
    edge_in_collision: torch.Tensor  # (M,) bool
    max_penetration: torch.Tensor  # (M,) float


class EdgeCollisionChecker:

    def __init__(
        self,
        robot_yml: str,
        *,
        cpu: bool = False,
        world_yml: Optional[str] = None,
        step_q: float = 0.05,
        max_steps: int = 64,
        point_chunk: int = 8192,
    ):
        self.robot_yml = robot_yml
        self.world_yml = world_yml
        self.cpu = bool(cpu)
        self.step_q = float(step_q)
        self.max_steps = int(max_steps)
        self.point_chunk = int(point_chunk)

        # self_collision.py의 캐시를 적극 활용
        self._checker: SelfCollisionChecker = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)

    @property
    def device(self) -> torch.device:
        return self._checker.tensor_args.device

    def _to_tensor_14(self, q: TensorLike14, *, device: torch.device) -> torch.Tensor:
        if isinstance(q, torch.Tensor):
            q_t = q
        else:
            q_t = torch.tensor(list(q), dtype=torch.float32)
        if q_t.ndim != 1:
            raise ValueError(f"q must be (14,), got shape={tuple(q_t.shape)}")
        if q_t.numel() != len(self._checker.cspace_names):
            raise ValueError(
                f"q dim mismatch. got={q_t.numel()} expected={len(self._checker.cspace_names)}"
            )
        return q_t.to(device=device, dtype=torch.float32)

    def _choose_steps(self, q0: torch.Tensor, q1: torch.Tensor, *, steps: Optional[int]) -> int:
        if steps is not None:
            s = int(steps)
            if s < 1:
                raise ValueError("steps must be >= 1")
            return min(s, self.max_steps)

        # 자동 steps: 최대 관절 변화량 기준
        max_delta = float((q1 - q0).abs().max().item())
        if self.step_q <= 0:
            return 1
        s = int(math.ceil(max_delta / self.step_q))
        s = max(1, s)
        return min(s, self.max_steps)

    def _interpolate_single(self, q0: torch.Tensor, q1: torch.Tensor, steps: int) -> torch.Tensor:
        """(steps+1, 14)"""
        t = torch.linspace(0.0, 1.0, steps + 1, device=q0.device, dtype=q0.dtype).unsqueeze(1)
        return (1.0 - t) * q0.unsqueeze(0) + t * q1.unsqueeze(0)

    @torch.no_grad()
    def check_edge(
        self,
        q0: TensorLike14,
        q1: TensorLike14,
        *,
        steps: Optional[int] = None,
        return_first_hit: bool = True,
    ) -> EdgeCollisionResult:

        dev = self.device
        q0_t = self._to_tensor_14(q0, device=dev)
        q1_t = self._to_tensor_14(q1, device=dev)
        s = self._choose_steps(q0_t, q1_t, steps=steps)

        qs = self._interpolate_single(q0_t, q1_t, s)  # (s+1,14)
        out = self._checker.check_batch(qs)

        in_col = out.in_collision  # (s+1,)
        max_pen = float(torch.maximum(out.d_self_max, out.d_world_max).max().item())

        if not bool(in_col.any().item()):
            return EdgeCollisionResult(edge_in_collision=False, max_penetration=max_pen, first_collision_alpha=None)

        if not return_first_hit:
            return EdgeCollisionResult(edge_in_collision=True, max_penetration=max_pen, first_collision_alpha=None)

        first_idx = int((in_col.to(torch.int32).cumsum(dim=0) == 1).nonzero(as_tuple=False)[0].item())
        alpha = float(first_idx / float(s)) if s > 0 else 0.0

        return EdgeCollisionResult(edge_in_collision=True, max_penetration=max_pen, first_collision_alpha=alpha)

    @torch.no_grad()
    def check_edges_batch(
        self,
        q0_batch: Union[torch.Tensor, Sequence[Sequence[float]]],
        q1_batch: Union[torch.Tensor, Sequence[Sequence[float]]],
        *,
        steps: Optional[int] = None,
        step_q: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> EdgeCollisionBatchResult:

        dev = self.device

        # 입력 정규화
        if isinstance(q0_batch, torch.Tensor):
            q0 = q0_batch.to(device=dev, dtype=torch.float32)
        else:
            q0 = torch.tensor(q0_batch, device=dev, dtype=torch.float32)

        if isinstance(q1_batch, torch.Tensor):
            q1 = q1_batch.to(device=dev, dtype=torch.float32)
        else:
            q1 = torch.tensor(q1_batch, device=dev, dtype=torch.float32)

        if q0.ndim != 2 or q1.ndim != 2:
            raise ValueError("q0_batch/q1_batch must be (M,14)")
        if q0.shape != q1.shape:
            raise ValueError(f"shape mismatch: q0={tuple(q0.shape)} q1={tuple(q1.shape)}")
        if q0.shape[1] != len(self._checker.cspace_names):
            raise ValueError(
                f"dim mismatch. got={q0.shape[1]} expected={len(self._checker.cspace_names)}"
            )

        M = q0.shape[0]

        # steps 결정
        if steps is None:
            _step_q = float(self.step_q if step_q is None else step_q)
            _max_steps = int(self.max_steps if max_steps is None else max_steps)
            if _step_q <= 0:
                s = 1
            else:
                max_delta = float((q1 - q0).abs().max().item())
                s = int(math.ceil(max_delta / _step_q))
                s = max(1, s)
            s = min(s, _max_steps)
        else:
            s = min(int(steps), int(self.max_steps if max_steps is None else max_steps))
            s = max(1, s)

        # (S,1,1)
        t = torch.linspace(0.0, 1.0, s + 1, device=dev, dtype=torch.float32).view(-1, 1, 1)
        # (S,M,14)
        qs = (1.0 - t) * q0.unsqueeze(0) + t * q1.unsqueeze(0)
        # (S*M,14)
        qs_flat = qs.reshape(-1, qs.shape[-1])

        # chunking으로 메모리/시간 관리
        in_col_flat = torch.zeros((qs_flat.shape[0],), device=dev, dtype=torch.bool)
        dmax_flat = torch.zeros((qs_flat.shape[0],), device=dev, dtype=torch.float32)

        for st in range(0, qs_flat.shape[0], self.point_chunk):
            ed = min(st + self.point_chunk, qs_flat.shape[0])
            out = self._checker.check_batch(qs_flat[st:ed])
            in_col_flat[st:ed] = out.in_collision
            dmax_flat[st:ed] = torch.maximum(out.d_self_max, out.d_world_max)

        # (S,M)
        in_col = in_col_flat.view(s + 1, M)
        dmax = dmax_flat.view(s + 1, M)

        edge_in_col = in_col.any(dim=0)  # (M,)
        edge_max_pen = dmax.max(dim=0).values  # (M,)

        return EdgeCollisionBatchResult(edge_in_collision=edge_in_col, max_penetration=edge_max_pen)


# ------------------------------------------------------------
# CLI (간단 테스트용)
# ------------------------------------------------------------

def _parse_csv_floats(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Edge collision check (self + world) (q0->q1).")
    ap.add_argument("--robot_yml", default=ROBOT_YAML)
    ap.add_argument("--world_yml", default=WORLD_YAML)
    ap.add_argument("--q0", required=True, help="comma-separated 14 floats")
    ap.add_argument("--q1", required=True, help="comma-separated 14 floats")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--step_q", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=64)

    args = ap.parse_args()

    q0 = _parse_csv_floats(args.q0)
    q1 = _parse_csv_floats(args.q1)

    ecc = EdgeCollisionChecker(
        args.robot_yml,
        world_yml=args.world_yml,
        cpu=args.cpu,
        step_q=args.step_q,
        max_steps=args.max_steps,
    )

    res = ecc.check_edge(q0, q1, steps=args.steps, return_first_hit=True)
    print(
        f"edge_in_collision={res.edge_in_collision} | max_penetration={res.max_penetration:.6e}"
        + ("" if res.first_collision_alpha is None else f" | first_hit_alpha={res.first_collision_alpha:.3f}")
    )


if __name__ == "__main__":
    main()

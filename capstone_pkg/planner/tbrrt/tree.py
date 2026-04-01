from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Tree:
    device: torch.device
    dtype: torch.dtype
    D: int
    capacity: int = 32768

    def __post_init__(self):
        # dataclass 생성 직후 자동 호출되는 초기화 훅.
        # 트리 노드들을 담을 텐서 버퍼(q, parent, ts_id 등)를 capacity 크기로 미리 할당한다.
        self.q = torch.empty((self.capacity, self.D), device=self.device, dtype=self.dtype)  # 노드 상태 q 저장 (N,D)
        self.parent = torch.full((self.capacity,), -1, device=self.device, dtype=torch.long)  # 각 노드의 부모 인덱스 (N,)
        self.ts_id = torch.zeros((self.capacity,), device=self.device, dtype=torch.long)  # 각 노드가 속한 tangent-space id (N,)
        self.size = 0  # 현재 트리에 들어있는 노드 개수

        # 아래 두 플래그는 “projection root” 관련 bookkeeping 용도:
        # - is_proj_root: 해당 노드가 projection으로 생성된 루트(특정 의미의 루트/앵커)인지
        # - is_parent_of_proj_root: 어떤 노드가 projection root의 부모인지 (추후 pruning/관리용으로 쓰기 좋음)
        self.is_proj_root = torch.zeros((self.capacity,), device=self.device, dtype=torch.bool)
        self.is_parent_of_proj_root = torch.zeros((self.capacity,), device=self.device, dtype=torch.bool)

    def __len__(self) -> int:
        # len(tree) 호출 시 현재 트리 노드 수 반환
        return int(self.size)

    def _ensure(self, n_add: int = 1):
        # 앞으로 n_add개의 노드를 추가할 때 capacity가 부족하면 버퍼를 2배씩 키우는 내부 함수.
        # (외부에서는 add_node가 호출하면서 자동으로 필요한 만큼 확장한다)
        need = self.size + n_add
        if need <= self.capacity:
            return

        # 필요한 크기를 만족할 때까지 2배씩 증가
        new_cap = self.capacity
        while new_cap < need:
            new_cap *= 2

        # 확장된 새 버퍼 생성
        q2 = torch.empty((new_cap, self.D), device=self.device, dtype=self.dtype)
        p2 = torch.full((new_cap,), -1, device=self.device, dtype=torch.long)
        t2 = torch.zeros((new_cap,), device=self.device, dtype=torch.long)

        # 기존 데이터 복사
        if self.size > 0:
            q2[: self.size] = self.q[: self.size]
            p2[: self.size] = self.parent[: self.size]
            t2[: self.size] = self.ts_id[: self.size]

        # projection 관련 플래그도 확장
        r2 = torch.zeros((new_cap,), device=self.device, dtype=torch.bool)
        pr2 = torch.zeros((new_cap,), device=self.device, dtype=torch.bool)
        if self.size > 0:
            r2[: self.size] = self.is_proj_root[: self.size]
            pr2[: self.size] = self.is_parent_of_proj_root[: self.size]

        # 새 버퍼로 교체
        self.q, self.parent, self.ts_id = q2, p2, t2
        self.is_proj_root, self.is_parent_of_proj_root = r2, pr2
        self.capacity = new_cap

    @torch.no_grad()
    def add_node(self, q: torch.Tensor, *, parent: int, ts_id: int, is_proj_root: bool = False) -> int:
        # 트리에 노드 1개를 추가하고, 추가된 노드의 인덱스를 반환한다.
        #
        # q: (D,) 모양의 상태 벡터
        # parent: 부모 노드 인덱스 (-1이면 루트)
        # ts_id: 이 노드가 속하는 tangent-space(또는 로컬 차트) 식별자
        # is_proj_root: 이 노드가 projection으로 생성된 “루트 성격의 노드”인지 표시하는 플래그
        if q.ndim != 1 or q.shape[0] != self.D:
            raise ValueError(f"q must be (D,), got {tuple(q.shape)}")

        # 버퍼 확장 필요 시 확장
        self._ensure(1)

        # 새 노드 인덱스는 현재 size 위치
        idx = int(self.size)

        # 데이터 기록
        self.q[idx] = q
        self.parent[idx] = int(parent)
        self.ts_id[idx] = int(ts_id)

        # projection root 플래그 기록 + 부모가 있다면 "projection root의 부모" 플래그도 기록
        self.is_proj_root[idx] = bool(is_proj_root)
        if bool(is_proj_root) and int(parent) >= 0:
            self.is_parent_of_proj_root[int(parent)] = True

        # 노드 수 증가
        self.size += 1
        return idx

    @torch.no_grad()
    def get_node(self, idx: int) -> torch.Tensor:
        # idx에 해당하는 노드의 q를 반환한다. (뷰일 수 있으니 호출 측에서 수정하면 원본에 영향 가능)
        return self.q[int(idx)]

    @torch.no_grad()
    def nearest(self, q_target: torch.Tensor, *, chunk: int = 8192) -> Tuple[int, float]:
        """Return (idx, dist) of nearest node in L2."""
        # 트리 내부 노드들 중 q_target과 L2 거리(유클리드)가 가장 가까운 노드를 찾는다.
        #
        # chunk: 한 번에 처리할 노드 개수. 노드가 많을 때 메모리 폭발을 막기 위해 슬라이스로 나눠 계산.
        # 반환: (가장 가까운 노드 인덱스, 그 거리 float)

        if self.size <= 0:
            raise RuntimeError("Tree is empty")
        if q_target.ndim != 1:
            raise ValueError("q_target must be (D,)")

        best_idx = 0
        best_d = float("inf")
        tgt = q_target.view(1, -1)  # (1,D)로 만들어 cdist 입력 형태 맞춤

        # 트리 노드들을 chunk 단위로 잘라서 torch.cdist로 거리 계산
        for st in range(0, self.size, chunk):
            ed = min(st + chunk, self.size)
            qs = self.q[st:ed]                 # (M,D)
            d = torch.cdist(qs, tgt).squeeze(1)  # (M,) 각 노드와 타겟의 거리
            v, i = torch.min(d, dim=0)          # 이 chunk에서 최소 거리와 인덱스
            dv = float(v.item())

            # 전역 최소 갱신
            if dv < best_d:
                best_d = dv
                best_idx = st + int(i.item())

        return best_idx, best_d

    @torch.no_grad()
    def backtrack_path(self, idx: int) -> torch.Tensor:
        """Return (L,D) path from root to idx."""
        # idx에서 루트까지 parent 포인터를 따라가며 경로를 복원(backtrack)한 뒤,
        # 루트 -> idx 순서로 (L,D) 텐서를 반환한다.
        #
        # 반환은 clone()을 하므로 원본 트리 버퍼와 분리된 텐서가 나온다.

        if idx < 0 or idx >= self.size:
            raise ValueError("idx out of range")

        # idx -> ... -> root(-1 직전)까지 인덱스 수집
        seq = []
        cur = int(idx)
        while cur != -1:
            seq.append(cur)
            cur = int(self.parent[cur].item())

        # 루트부터 시작하도록 뒤집기
        seq.reverse()

        # 인덱스 텐서로 q들을 모아 (L,D) 경로 생성
        return self.q[torch.tensor(seq, device=self.device, dtype=torch.long)].clone()

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GoalRegion:
    """OMPL-like goal region.

    A state is considered satisfied when distance_goal(state) <= threshold.
    """

    threshold: float

    def is_satisfied(self, q: torch.Tensor, *, return_distance: bool = False):
        dist = float(self.distance_goal(q).item())
        ok = dist <= float(self.threshold)
        if return_distance:
            return ok, dist
        return ok

    def distance_goal(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GoalStates(GoalRegion):
    """OMPL-like goal set with distanceGoal = min distance to any goal state."""

    def __init__(self, states: torch.Tensor, threshold: float):
        if states.ndim != 2:
            raise ValueError(f"states must be (N,D), got {tuple(states.shape)}")
        if states.shape[0] <= 0:
            raise ValueError("GoalStates requires at least one goal state")
        super().__init__(threshold=float(threshold))
        self.states = states.contiguous()
        self._sample_position = 0

    def clear(self) -> None:
        self.states = self.states[:0]
        self._sample_position = 0

    def has_states(self) -> bool:
        return int(self.states.shape[0]) > 0

    def get_state(self, index: int) -> torch.Tensor:
        return self.states[int(index)]

    def get_state_count(self) -> int:
        return int(self.states.shape[0])

    def add_state(self, q: torch.Tensor) -> None:
        q = q.view(1, -1).to(device=self.states.device, dtype=self.states.dtype)
        self.states = torch.cat([self.states, q], dim=0).contiguous()

    def distance_goal(self, q: torch.Tensor) -> torch.Tensor:
        q = q.view(1, -1)
        d = torch.cdist(self.states, q).squeeze(1)
        return d.min()

    def nearest_goal_index(self, q: torch.Tensor) -> int:
        q = q.view(1, -1)
        d = torch.cdist(self.states, q).squeeze(1)
        return int(torch.argmin(d).item())

    def nearest_goal_state(self, q: torch.Tensor) -> torch.Tensor:
        return self.states[self.nearest_goal_index(q)].clone()

    def sample_goal(self) -> torch.Tensor:
        idx = int(self._sample_position % max(1, self.get_state_count()))
        self._sample_position += 1
        return self.states[idx].clone()
